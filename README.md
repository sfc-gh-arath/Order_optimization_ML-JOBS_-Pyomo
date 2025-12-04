# Snowflake ML Jobs - Pyomo Inventory Optimization POC

This project demonstrates running a **Pyomo optimization job** through **Snowflake ML Jobs** to solve an inventory order allocation problem.

## 🎯 Problem Statement

Given:
- **Sales predictions** for the next 6 months (demand forecast)
- **Supplier orders** available to place with different quantities, costs, and delivery dates
- **30-day lead time** for order delivery

**Goal:** Optimally split/allocate orders across suppliers to:
- Meet predicted demand for each product
- Minimize total procurement cost
- Avoid excess inventory (overstocking)
- Maintain a small safety stock buffer (5%)

## 📁 Project Structure

```
pyomo-mljobs/
├── README.md
├── requirements.txt              # Local development dependencies
├── requirements-job.txt          # ML Job environment dependencies
├── data/
│   ├── sales_predictions.csv     # Sample sales forecast data
│   └── orders.csv                # Sample supplier orders data
├── sql/
│   └── schema.sql                # Snowflake DDL for tables
├── src/
│   ├── inventory_optimization.py # Main Pyomo optimization script
│   └── submit_ml_job.py          # ML Job submission script
└── output/                       # Local test output (gitignored)
```

## 🚀 Quick Start

### 1. Set Up Snowflake Environment

```sql
-- Run the DDL script to create database, schema, and tables
-- See sql/schema.sql for complete DDL

CREATE DATABASE IF NOT EXISTS PYOMO_ML_JOBS_POC;
USE DATABASE PYOMO_ML_JOBS_POC;
CREATE SCHEMA IF NOT EXISTS INVENTORY_OPTIMIZATION;
USE SCHEMA INVENTORY_OPTIMIZATION;

-- Create compute pool for ML Jobs
CREATE COMPUTE POOL IF NOT EXISTS PYOMO_COMPUTE_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = 'CPU_X64_S'
    AUTO_SUSPEND_SECS = 300;

-- Create stage for ML Job files
CREATE STAGE IF NOT EXISTS ML_JOBS_STAGE
    DIRECTORY = (ENABLE = TRUE);
```

### 2. Load Sample Data

```sql
-- Upload CSV files to stage
PUT file:///path/to/data/sales_predictions.csv @ML_JOBS_STAGE/data/;
PUT file:///path/to/data/orders.csv @ML_JOBS_STAGE/data/;

-- Load data into tables
COPY INTO SALES_PREDICTIONS (date, product_id, product_name, predicted_sales, unit_price)
FROM @ML_JOBS_STAGE/data/sales_predictions.csv
FILE_FORMAT = (TYPE='CSV' SKIP_HEADER=1);

COPY INTO ORDERS 
FROM @ML_JOBS_STAGE/data/orders.csv
FILE_FORMAT = (TYPE='CSV' SKIP_HEADER=1);
```

### 3. Run Locally (for testing)

```bash
# Install dependencies (includes highspy solver)
pip install -r requirements.txt

# The HiGHS solver is included via 'highspy' package (pip-installable)
# No additional system packages needed!

# Alternative: Install GLPK if preferred
# On macOS:
# brew install glpk
# On Ubuntu/Debian:
# sudo apt-get install glpk-utils

# Run the optimization locally
python src/inventory_optimization.py
```

### 4. Submit as Snowflake ML Job

```bash
# Set environment variables
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_USER="your_user"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_WAREHOUSE="your_warehouse"
export SNOWFLAKE_DATABASE="PYOMO_ML_JOBS_POC"
export SNOWFLAKE_SCHEMA="INVENTORY_OPTIMIZATION"

# Submit the ML Job
python src/submit_ml_job.py
```

Or submit directly in Python:

```python
from snowflake.snowpark import Session
from snowflake.ml.jobs import submit_file

session = Session.builder.configs({...}).create()

job = submit_file(
    file_path="src/inventory_optimization.py",
    compute_pool="PYOMO_COMPUTE_POOL",
    stage_name="ML_JOBS_STAGE",
    session=session,
    pip_requirements=["pyomo>=6.7.0", "highspy>=1.5.0", "pandas>=2.0.0"],
)

job.wait()
print(job.get_logs())
```

## 📊 Sample Data

### Sales Predictions
| date | product_id | product_name | predicted_sales | unit_price |
|------|------------|--------------|-----------------|------------|
| 2026-01-01 | SKU001 | Widget Pro | 150 | 25.99 |
| 2026-01-01 | SKU002 | Gadget Plus | 200 | 45.50 |
| ... | ... | ... | ... | ... |

### Supplier Orders
| order_id | product_id | supplier_name | quantity_available | unit_cost | min_order_qty |
|----------|------------|---------------|-------------------|-----------|---------------|
| ORD001 | SKU001 | Acme Corp | 500 | 18.00 | 50 |
| ORD002 | SKU001 | Beta Supplies | 400 | 19.50 | 25 |
| ... | ... | ... | ... | ... | ... |

## 🔧 Optimization Model

The Pyomo model uses **Mixed-Integer Linear Programming (MILP)**:

### Decision Variables
- `x[order_id]`: Quantity to order from each supplier (integer)
- `y[order_id]`: Binary indicator if order is placed

### Objective
Minimize total procurement cost:
```
minimize Σ (x[i] × unit_cost[i])
```

### Constraints
1. **Meet demand**: Total orders for each product ≥ predicted demand × 1.05 (5% safety stock)
2. **Supply limits**: Don't exceed available quantity from each supplier
3. **Minimum orders**: If placing an order, must meet minimum order quantity
4. **Max inventory**: Don't over-order beyond 120% of demand

## 📈 Output

### OPTIMIZED_ORDER_ALLOCATION Table
| run_id | order_id | product_id | supplier_name | allocated_quantity | unit_cost | total_cost |
|--------|----------|------------|---------------|-------------------|-----------|------------|
| OPT-20251204-... | ORD001 | SKU001 | Acme Corp | 500 | 18.00 | 9000.00 |
| ... | ... | ... | ... | ... | ... | ... |

### OPTIMIZATION_SUMMARY Table
| run_id | total_demand | total_allocated | total_cost | solver_status | execution_time |
|--------|--------------|-----------------|------------|---------------|----------------|
| OPT-20251204-... | 6350 | 6668 | 147,845.50 | optimal | 0.45s |

## 🔍 Verify Results

```sql
-- Check optimization results
SELECT * FROM OPTIMIZED_ORDER_ALLOCATION 
ORDER BY run_timestamp DESC, product_id;

-- View summary
SELECT * FROM OPTIMIZATION_SUMMARY 
ORDER BY run_timestamp DESC;

-- Compare demand vs allocation
SELECT 
    o.product_id,
    SUM(s.predicted_sales) as total_demand,
    SUM(o.allocated_quantity) as total_allocated,
    SUM(o.total_cost) as total_cost
FROM OPTIMIZED_ORDER_ALLOCATION o
JOIN SALES_PREDICTIONS s ON o.product_id = s.product_id
WHERE o.run_id = (SELECT MAX(run_id) FROM OPTIMIZATION_SUMMARY)
GROUP BY o.product_id;
```

## 📚 References

- [Snowflake ML Jobs Documentation](https://docs.snowflake.com/en/developer-guide/snowflake-ml/ml-jobs/snowflake-ml-jobs)
- [Snowflake ML Jobs GitHub Samples](https://github.com/Snowflake-Labs/sf-samples/tree/main/samples/ml/ml_jobs)
- [Pyomo Documentation](https://pyomo.readthedocs.io/)
- [HiGHS Solver](https://highs.dev/) - High performance MILP solver (pip-installable via `highspy`)
- [GLPK Solver](https://www.gnu.org/software/glpk/) - Alternative solver (requires system install)

## 📝 License

MIT License

