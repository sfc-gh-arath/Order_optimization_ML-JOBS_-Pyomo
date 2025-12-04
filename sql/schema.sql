-- =============================================================================
-- Snowflake DDL Schema for Pyomo ML Jobs POC
-- Order Optimization for Inventory Management
-- =============================================================================

-- Create database and schema (if needed)
CREATE DATABASE IF NOT EXISTS PYOMO_ML_JOBS_POC;
USE DATABASE PYOMO_ML_JOBS_POC;

CREATE SCHEMA IF NOT EXISTS INVENTORY_OPTIMIZATION;
USE SCHEMA INVENTORY_OPTIMIZATION;

-- =============================================================================
-- Table: SALES_PREDICTIONS
-- Description: Contains monthly sales forecasts for each product
-- =============================================================================
CREATE OR REPLACE TABLE SALES_PREDICTIONS (
    date DATE NOT NULL COMMENT 'Month start date for the prediction',
    product_id VARCHAR(20) NOT NULL COMMENT 'Unique product identifier (SKU)',
    product_name VARCHAR(100) COMMENT 'Human-readable product name',
    predicted_sales INTEGER NOT NULL COMMENT 'Predicted unit sales for the month',
    unit_price DECIMAL(10,2) COMMENT 'Selling price per unit',
    PRIMARY KEY (date, product_id)
) COMMENT = 'Monthly sales predictions for inventory planning';

-- =============================================================================
-- Table: ORDERS
-- Description: Contains available supplier orders that can be placed
-- =============================================================================
CREATE OR REPLACE TABLE ORDERS (
    order_id VARCHAR(20) NOT NULL PRIMARY KEY COMMENT 'Unique order identifier',
    product_id VARCHAR(20) NOT NULL COMMENT 'Product identifier (SKU)',
    product_name VARCHAR(100) COMMENT 'Human-readable product name',
    supplier_name VARCHAR(100) COMMENT 'Supplier company name',
    quantity_available INTEGER NOT NULL COMMENT 'Maximum quantity available from supplier',
    unit_cost DECIMAL(10,2) NOT NULL COMMENT 'Cost per unit from this supplier',
    order_date DATE NOT NULL COMMENT 'Date the order would be placed',
    expected_delivery_date DATE NOT NULL COMMENT 'Expected delivery date (30 day lead time)',
    min_order_qty INTEGER DEFAULT 1 COMMENT 'Minimum order quantity requirement'
) COMMENT = 'Available supplier orders for inventory replenishment';

-- =============================================================================
-- Table: OPTIMIZED_ORDER_ALLOCATION
-- Description: Output table for the optimization results
-- =============================================================================
CREATE OR REPLACE TABLE OPTIMIZED_ORDER_ALLOCATION (
    run_id VARCHAR(50) NOT NULL COMMENT 'Unique identifier for the optimization run',
    run_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'When the optimization was executed',
    order_id VARCHAR(20) NOT NULL COMMENT 'Order identifier',
    product_id VARCHAR(20) NOT NULL COMMENT 'Product identifier',
    supplier_name VARCHAR(100) COMMENT 'Supplier name',
    allocated_quantity INTEGER NOT NULL COMMENT 'Optimized quantity to order',
    unit_cost DECIMAL(10,2) COMMENT 'Unit cost for this order',
    total_cost DECIMAL(12,2) COMMENT 'Total cost (allocated_quantity * unit_cost)',
    expected_delivery_date DATE COMMENT 'When the order will arrive',
    PRIMARY KEY (run_id, order_id)
) COMMENT = 'Results from the Pyomo optimization job';

-- =============================================================================
-- Table: OPTIMIZATION_SUMMARY
-- Description: Summary metrics for each optimization run
-- =============================================================================
CREATE OR REPLACE TABLE OPTIMIZATION_SUMMARY (
    run_id VARCHAR(50) NOT NULL PRIMARY KEY COMMENT 'Unique identifier for the optimization run',
    run_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP() COMMENT 'When the optimization was executed',
    total_demand INTEGER COMMENT 'Total forecasted demand across all products',
    total_allocated INTEGER COMMENT 'Total units allocated across all orders',
    total_cost DECIMAL(15,2) COMMENT 'Total cost of all allocated orders',
    products_covered INTEGER COMMENT 'Number of unique products covered',
    orders_used INTEGER COMMENT 'Number of orders with non-zero allocation',
    solver_status VARCHAR(50) COMMENT 'Solver termination condition',
    execution_time_seconds DECIMAL(10,3) COMMENT 'Time taken to solve the optimization'
) COMMENT = 'Summary metrics for optimization runs';

-- =============================================================================
-- Stage for ML Jobs
-- =============================================================================
CREATE OR REPLACE STAGE ML_JOBS_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for uploading ML job scripts and dependencies';

-- =============================================================================
-- File Format for CSV loading
-- =============================================================================
CREATE OR REPLACE FILE FORMAT CSV_FORMAT
    TYPE = 'CSV'
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    NULL_IF = ('NULL', 'null', '');

-- =============================================================================
-- Data Loading Commands (run after uploading CSVs to stage)
-- =============================================================================
/*
-- Upload CSV files to stage first:
-- PUT file://path/to/sales_predictions.csv @ML_JOBS_STAGE/data/;
-- PUT file://path/to/orders.csv @ML_JOBS_STAGE/data/;

-- Then load data:
COPY INTO SALES_PREDICTIONS (date, product_id, product_name, predicted_sales, unit_price)
FROM @ML_JOBS_STAGE/data/sales_predictions.csv
FILE_FORMAT = CSV_FORMAT;

COPY INTO ORDERS (order_id, product_id, product_name, supplier_name, quantity_available, unit_cost, order_date, expected_delivery_date, min_order_qty)
FROM @ML_JOBS_STAGE/data/orders.csv
FILE_FORMAT = CSV_FORMAT;
*/

-- =============================================================================
-- Compute Pool for ML Jobs (requires SYSADMIN or appropriate privileges)
-- =============================================================================
/*
CREATE COMPUTE POOL IF NOT EXISTS PYOMO_COMPUTE_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = 'CPU_X64_S'
    AUTO_SUSPEND_SECS = 300
    COMMENT = 'Compute pool for Pyomo optimization ML jobs';
*/

-- =============================================================================
-- Sample Queries for Verification
-- =============================================================================
/*
-- View total demand by product
SELECT 
    product_id,
    product_name,
    SUM(predicted_sales) as total_demand
FROM SALES_PREDICTIONS
GROUP BY product_id, product_name
ORDER BY total_demand DESC;

-- View available supply by product
SELECT 
    product_id,
    product_name,
    COUNT(*) as num_suppliers,
    SUM(quantity_available) as total_available,
    AVG(unit_cost) as avg_unit_cost
FROM ORDERS
GROUP BY product_id, product_name
ORDER BY product_id;

-- Compare demand vs supply
SELECT 
    d.product_id,
    d.product_name,
    d.total_demand,
    s.total_available,
    s.total_available - d.total_demand as surplus
FROM (
    SELECT product_id, product_name, SUM(predicted_sales) as total_demand
    FROM SALES_PREDICTIONS
    GROUP BY product_id, product_name
) d
JOIN (
    SELECT product_id, product_name, SUM(quantity_available) as total_available
    FROM ORDERS
    GROUP BY product_id, product_name
) s ON d.product_id = s.product_id
ORDER BY d.product_id;
*/

