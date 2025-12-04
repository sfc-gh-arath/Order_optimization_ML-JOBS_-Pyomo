"""
Inventory Order Optimization using Pyomo
=========================================
This script optimizes the allocation of supplier orders to meet predicted demand
while minimizing total cost and avoiding over/under inventory situations.

Designed to run as a Snowflake ML Job.

Problem Statement:
- We have sales predictions (demand forecast) for the next 6 months
- We have multiple supplier orders available for each product
- Orders have 30 days delivery lead time
- Goal: Split orders optimally to meet demand without excess inventory

Optimization Model:
- Decision Variables: Quantity to order from each supplier order
- Objective: Minimize total procurement cost
- Constraints:
  - Meet total demand for each product
  - Don't exceed available quantity from each supplier
  - Respect minimum order quantities
  - Allow small buffer (safety stock) for demand variability
"""

import os
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeIntegers,
    SolverFactory,
    value,
    minimize,
)
from pyomo.opt import TerminationCondition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_snowflake(session) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load sales predictions and orders data from Snowflake tables."""
    logger.info("Loading data from Snowflake tables...")
    
    sales_df = session.table("SALES_PREDICTIONS").to_pandas()
    orders_df = session.table("ORDERS").to_pandas()
    
    # Standardize column names to lowercase for consistency
    sales_df.columns = sales_df.columns.str.lower()
    orders_df.columns = orders_df.columns.str.lower()
    
    logger.info(f"Loaded {len(sales_df)} sales prediction records")
    logger.info(f"Loaded {len(orders_df)} order records")
    
    return sales_df, orders_df


def prepare_demand_summary(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales predictions to get total demand per product."""
    demand_summary = sales_df.groupby('product_id').agg({
        'predicted_sales': 'sum',
        'product_name': 'first',
        'unit_price': 'first'
    }).reset_index()
    
    demand_summary.rename(columns={'predicted_sales': 'total_demand'}, inplace=True)
    
    logger.info("Demand summary by product:")
    for _, row in demand_summary.iterrows():
        logger.info(f"  {row['product_id']}: {row['total_demand']} units")
    
    return demand_summary


def build_optimization_model(
    orders_df: pd.DataFrame,
    demand_summary: pd.DataFrame,
    safety_stock_pct: float = 0.05
) -> ConcreteModel:
    """
    Build the Pyomo optimization model.
    
    Args:
        orders_df: DataFrame with available supplier orders
        demand_summary: DataFrame with total demand per product
        safety_stock_pct: Safety stock as percentage of demand (default 5%)
    
    Returns:
        Configured Pyomo ConcreteModel
    """
    logger.info("Building optimization model...")
    
    model = ConcreteModel(name="InventoryOrderOptimization")
    
    # Create index sets
    order_ids = orders_df['order_id'].tolist()
    product_ids = demand_summary['product_id'].tolist()
    
    # Create dictionaries for parameters
    order_to_product = orders_df.set_index('order_id')['product_id'].to_dict()
    order_quantity = orders_df.set_index('order_id')['quantity_available'].to_dict()
    order_cost = orders_df.set_index('order_id')['unit_cost'].to_dict()
    order_min_qty = orders_df.set_index('order_id')['min_order_qty'].to_dict()
    product_demand = demand_summary.set_index('product_id')['total_demand'].to_dict()
    
    # Decision Variables: quantity to order from each supplier
    # x[order_id] = quantity to order (integer, non-negative)
    model.x = Var(order_ids, domain=NonNegativeIntegers, bounds=lambda m, i: (0, order_quantity[i]))
    
    # Binary variable to track if an order is placed (for minimum quantity constraint)
    from pyomo.environ import Binary
    model.y = Var(order_ids, domain=Binary)
    
    # Objective: Minimize total procurement cost
    def objective_rule(m):
        return sum(m.x[i] * order_cost[i] for i in order_ids)
    
    model.objective = Objective(rule=objective_rule, sense=minimize)
    
    # Constraint 1: Meet demand for each product (with safety stock buffer)
    def demand_constraint_rule(m, p):
        relevant_orders = [oid for oid in order_ids if order_to_product[oid] == p]
        if not relevant_orders:
            return Constraint.Skip
        min_required = product_demand[p] * (1 + safety_stock_pct)
        return sum(m.x[oid] for oid in relevant_orders) >= min_required
    
    model.demand_constraint = Constraint(product_ids, rule=demand_constraint_rule)
    
    # Constraint 2: Don't exceed available quantity
    def supply_constraint_rule(m, oid):
        return m.x[oid] <= order_quantity[oid]
    
    model.supply_constraint = Constraint(order_ids, rule=supply_constraint_rule)
    
    # Constraint 3: Link binary variable to order quantity (if y=0, x must be 0)
    def link_constraint_upper_rule(m, oid):
        return m.x[oid] <= order_quantity[oid] * m.y[oid]
    
    model.link_upper = Constraint(order_ids, rule=link_constraint_upper_rule)
    
    # Constraint 4: If order is placed, meet minimum quantity
    def min_order_constraint_rule(m, oid):
        return m.x[oid] >= order_min_qty[oid] * m.y[oid]
    
    model.min_order_constraint = Constraint(order_ids, rule=min_order_constraint_rule)
    
    # Constraint 5: Don't over-order (max 120% of demand to prevent excess inventory)
    def max_inventory_rule(m, p):
        relevant_orders = [oid for oid in order_ids if order_to_product[oid] == p]
        if not relevant_orders:
            return Constraint.Skip
        max_allowed = product_demand[p] * 1.20  # Max 20% over demand
        return sum(m.x[oid] for oid in relevant_orders) <= max_allowed
    
    model.max_inventory = Constraint(product_ids, rule=max_inventory_rule)
    
    logger.info(f"Model built with {len(order_ids)} order variables and {len(product_ids)} product constraints")
    
    return model


def solve_model(model: ConcreteModel, solver_name: str = 'highs') -> Tuple[bool, str, float]:
    """
    Solve the optimization model.
    
    Args:
        model: Pyomo ConcreteModel to solve
        solver_name: Name of the solver to use (highs, glpk, cbc, or gurobi)
    
    Returns:
        Tuple of (success, status_message, solve_time)
    """
    logger.info(f"Attempting to solve model...")
    
    # Try available solvers in order of preference
    # HiGHS is preferred as it's pip-installable via 'highspy' package
    solvers_to_try = [
        ('appsi_highs', 'HiGHS (APPSI)'),      # Pyomo's APPSI interface to HiGHS
        ('highs', 'HiGHS'),                     # Direct HiGHS
        ('glpk', 'GLPK'),                       # GNU Linear Programming Kit
        ('cbc', 'CBC'),                         # COIN-OR Branch and Cut
        ('scip', 'SCIP'),                       # SCIP solver
    ]
    
    solver = None
    solver_used = None
    
    for solver_id, solver_desc in solvers_to_try:
        try:
            logger.info(f"Trying solver: {solver_desc} ({solver_id})...")
            solver = SolverFactory(solver_id)
            if solver is not None and solver.available():
                logger.info(f"✓ Solver {solver_desc} is available")
                solver_used = solver_desc
                break
            else:
                logger.info(f"  Solver {solver_desc} not available")
                solver = None
        except Exception as e:
            logger.info(f"  Solver {solver_desc} failed: {e}")
            solver = None
            continue
    
    if solver is None:
        # Last resort: try to use scipy's linprog via Pyomo (for LP relaxation)
        error_msg = (
            "No suitable MILP solver found. Please install one of:\n"
            "  - pip install highspy  (recommended, provides HiGHS solver)\n"
            "  - apt-get install glpk-utils (provides GLPK)\n"
            "  - apt-get install coinor-cbc (provides CBC)"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"Solving with {solver_used}...")
    start_time = time.time()
    
    try:
        results = solver.solve(model, tee=False)
    except Exception as e:
        logger.error(f"Solver execution failed: {e}")
        raise RuntimeError(f"Solver execution failed: {e}")
    
    solve_time = time.time() - start_time
    
    termination = results.solver.termination_condition
    
    if termination == TerminationCondition.optimal:
        logger.info(f"✓ Optimal solution found in {solve_time:.2f} seconds using {solver_used}")
        return True, "optimal", solve_time
    elif termination == TerminationCondition.feasible:
        logger.warning(f"Feasible (but not proven optimal) solution found in {solve_time:.2f} seconds")
        return True, "feasible", solve_time
    else:
        logger.error(f"Solver terminated with condition: {termination}")
        return False, str(termination), solve_time


def extract_results(
    model: ConcreteModel,
    orders_df: pd.DataFrame,
    run_id: str
) -> pd.DataFrame:
    """Extract optimization results into a DataFrame."""
    logger.info("Extracting optimization results...")
    
    results = []
    for _, row in orders_df.iterrows():
        order_id = row['order_id']
        allocated = int(round(value(model.x[order_id]) or 0))
        
        if allocated > 0:  # Only include orders with allocation
            results.append({
                'run_id': run_id,
                'run_timestamp': datetime.utcnow(),
                'order_id': order_id,
                'product_id': row['product_id'],
                'supplier_name': row['supplier_name'],
                'allocated_quantity': allocated,
                'unit_cost': row['unit_cost'],
                'total_cost': allocated * row['unit_cost'],
                'expected_delivery_date': row['expected_delivery_date']
            })
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"Allocated {len(results_df)} orders with total quantity: {results_df['allocated_quantity'].sum()}")
    
    return results_df


def create_summary(
    results_df: pd.DataFrame,
    demand_summary: pd.DataFrame,
    run_id: str,
    solver_status: str,
    solve_time: float
) -> pd.DataFrame:
    """Create summary metrics for the optimization run."""
    
    total_demand = demand_summary['total_demand'].sum()
    total_allocated = results_df['allocated_quantity'].sum() if len(results_df) > 0 else 0
    total_cost = results_df['total_cost'].sum() if len(results_df) > 0 else 0
    products_covered = results_df['product_id'].nunique() if len(results_df) > 0 else 0
    orders_used = len(results_df)
    
    summary = pd.DataFrame([{
        'run_id': run_id,
        'run_timestamp': datetime.utcnow(),
        'total_demand': int(total_demand),
        'total_allocated': int(total_allocated),
        'total_cost': float(total_cost),
        'products_covered': products_covered,
        'orders_used': orders_used,
        'solver_status': solver_status,
        'execution_time_seconds': round(solve_time, 3)
    }])
    
    logger.info("=" * 50)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total Demand:     {total_demand:,} units")
    logger.info(f"Total Allocated:  {total_allocated:,} units")
    logger.info(f"Coverage:         {(total_allocated/total_demand*100):.1f}%")
    logger.info(f"Total Cost:       ${total_cost:,.2f}")
    logger.info(f"Orders Used:      {orders_used}")
    logger.info(f"Solve Time:       {solve_time:.2f}s")
    logger.info("=" * 50)
    
    return summary


def save_results_to_snowflake(
    session,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame
):
    """Save optimization results back to Snowflake tables."""
    logger.info("Saving results to Snowflake...")
    
    if len(results_df) > 0:
        # Convert to Snowpark DataFrame and append to results table
        results_sp = session.create_dataframe(results_df)
        results_sp.write.mode("append").save_as_table("OPTIMIZED_ORDER_ALLOCATION")
        logger.info(f"Saved {len(results_df)} allocation records")
    
    # Save summary
    summary_sp = session.create_dataframe(summary_df)
    summary_sp.write.mode("append").save_as_table("OPTIMIZATION_SUMMARY")
    logger.info("Saved optimization summary")


def print_detailed_results(results_df: pd.DataFrame, demand_summary: pd.DataFrame):
    """Print detailed results by product."""
    logger.info("\nDETAILED ALLOCATION BY PRODUCT:")
    logger.info("-" * 70)
    
    for product_id in demand_summary['product_id'].unique():
        product_demand = demand_summary[demand_summary['product_id'] == product_id]['total_demand'].iloc[0]
        product_orders = results_df[results_df['product_id'] == product_id]
        product_allocated = product_orders['allocated_quantity'].sum() if len(product_orders) > 0 else 0
        product_cost = product_orders['total_cost'].sum() if len(product_orders) > 0 else 0
        
        logger.info(f"\n{product_id}:")
        logger.info(f"  Demand: {product_demand:,} | Allocated: {product_allocated:,} | Cost: ${product_cost:,.2f}")
        
        if len(product_orders) > 0:
            for _, order in product_orders.iterrows():
                logger.info(f"    - {order['order_id']} ({order['supplier_name']}): "
                          f"{order['allocated_quantity']:,} units @ ${order['unit_cost']:.2f}")


def main():
    """Main entry point for the optimization job."""
    logger.info("=" * 60)
    logger.info("INVENTORY ORDER OPTIMIZATION - Pyomo ML Job")
    logger.info("=" * 60)
    
    run_id = f"OPT-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    logger.info(f"Run ID: {run_id}")
    
    # Initialize Snowflake session
    # When running as ML Job, session is available via environment
    try:
        from snowflake.snowpark import Session
        from snowflake.snowpark.context import get_active_session
        
        # Try to get active session (available in Snowflake ML Jobs)
        try:
            session = get_active_session()
            logger.info("Using active Snowflake session from ML Job context")
        except Exception:
            # Fall back to creating session from environment variables
            logger.info("Creating Snowflake session from environment variables")
            session = Session.builder.configs({
                "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
                "user": os.environ.get("SNOWFLAKE_USER"),
                "password": os.environ.get("SNOWFLAKE_PASSWORD"),
                "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
                "database": os.environ.get("SNOWFLAKE_DATABASE", "PYOMO_ML_JOBS_POC"),
                "schema": os.environ.get("SNOWFLAKE_SCHEMA", "INVENTORY_OPTIMIZATION"),
            }).create()
        
        # Load data
        sales_df, orders_df = load_data_from_snowflake(session)
        
        # Prepare demand summary
        demand_summary = prepare_demand_summary(sales_df)
        
        # Build and solve optimization model
        model = build_optimization_model(orders_df, demand_summary, safety_stock_pct=0.05)
        success, solver_status, solve_time = solve_model(model)
        
        if success:
            # Extract and save results
            results_df = extract_results(model, orders_df, run_id)
            summary_df = create_summary(results_df, demand_summary, run_id, solver_status, solve_time)
            
            # Print detailed results
            print_detailed_results(results_df, demand_summary)
            
            # Save to Snowflake
            save_results_to_snowflake(session, results_df, summary_df)
            
            logger.info("\n✓ Optimization completed successfully!")
        else:
            logger.error(f"✗ Optimization failed: {solver_status}")
            raise RuntimeError(f"Optimization failed with status: {solver_status}")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Running in local mode with CSV files...")
        run_local_mode(run_id)


def run_local_mode(run_id: str):
    """Run optimization using local CSV files (for testing)."""
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    # Load data from CSV
    sales_df = pd.read_csv(os.path.join(data_dir, 'sales_predictions.csv'))
    orders_df = pd.read_csv(os.path.join(data_dir, 'orders.csv'))
    
    logger.info(f"Loaded {len(sales_df)} sales predictions from CSV")
    logger.info(f"Loaded {len(orders_df)} orders from CSV")
    
    # Prepare demand summary
    demand_summary = prepare_demand_summary(sales_df)
    
    # Build and solve
    model = build_optimization_model(orders_df, demand_summary, safety_stock_pct=0.05)
    success, solver_status, solve_time = solve_model(model)
    
    if success:
        results_df = extract_results(model, orders_df, run_id)
        summary_df = create_summary(results_df, demand_summary, run_id, solver_status, solve_time)
        
        print_detailed_results(results_df, demand_summary)
        
        # Save results locally
        output_dir = os.path.join(os.path.dirname(script_dir), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_csv(os.path.join(output_dir, f'allocation_{run_id}.csv'), index=False)
        summary_df.to_csv(os.path.join(output_dir, f'summary_{run_id}.csv'), index=False)
        
        logger.info(f"\nResults saved to {output_dir}/")
        logger.info("✓ Local optimization completed successfully!")
    else:
        logger.error(f"✗ Optimization failed: {solver_status}")


if __name__ == "__main__":
    main()

