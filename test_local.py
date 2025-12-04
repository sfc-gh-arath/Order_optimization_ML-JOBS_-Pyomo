#!/usr/bin/env python3
"""
Local Test Script
=================
Run the inventory optimization locally using CSV files to verify
the Pyomo model works before submitting to Snowflake ML Jobs.

Usage:
    python test_local.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inventory_optimization import (
    prepare_demand_summary,
    build_optimization_model,
    solve_model,
    extract_results,
    create_summary,
    print_detailed_results,
)

import pandas as pd
from datetime import datetime
import uuid


def main():
    print("=" * 60)
    print("LOCAL TEST - Inventory Order Optimization")
    print("=" * 60)
    
    # Generate run ID
    run_id = f"TEST-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    print(f"Run ID: {run_id}\n")
    
    # Load data from CSV
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    sales_df = pd.read_csv(os.path.join(data_dir, 'sales_predictions.csv'))
    orders_df = pd.read_csv(os.path.join(data_dir, 'orders.csv'))
    
    print(f"Loaded {len(sales_df)} sales predictions")
    print(f"Loaded {len(orders_df)} supplier orders\n")
    
    # Prepare demand summary
    demand_summary = prepare_demand_summary(sales_df)
    
    print("\nDemand Summary:")
    print(demand_summary.to_string(index=False))
    print()
    
    # Build optimization model
    model = build_optimization_model(
        orders_df=orders_df,
        demand_summary=demand_summary,
        safety_stock_pct=0.05  # 5% safety stock
    )
    
    # Solve
    success, solver_status, solve_time = solve_model(model)
    
    if not success:
        print(f"\n✗ Optimization failed: {solver_status}")
        sys.exit(1)
    
    # Extract results
    results_df = extract_results(model, orders_df, run_id)
    summary_df = create_summary(results_df, demand_summary, run_id, solver_status, solve_time)
    
    # Print detailed results
    print_detailed_results(results_df, demand_summary)
    
    # Save output
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f'allocation_{run_id}.csv')
    summary_file = os.path.join(output_dir, f'summary_{run_id}.csv')
    
    results_df.to_csv(results_file, index=False)
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n✓ Results saved to:")
    print(f"  - {results_file}")
    print(f"  - {summary_file}")
    
    print("\n" + "=" * 60)
    print("✓ LOCAL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    # Print final allocation summary
    print("\nFINAL ALLOCATION SUMMARY:")
    allocation_by_product = results_df.groupby('product_id').agg({
        'allocated_quantity': 'sum',
        'total_cost': 'sum'
    }).reset_index()
    
    allocation_by_product = allocation_by_product.merge(
        demand_summary[['product_id', 'total_demand']], 
        on='product_id'
    )
    allocation_by_product['coverage_pct'] = (
        allocation_by_product['allocated_quantity'] / 
        allocation_by_product['total_demand'] * 100
    ).round(1)
    
    print(allocation_by_product.to_string(index=False))


if __name__ == "__main__":
    main()

