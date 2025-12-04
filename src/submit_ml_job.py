"""
Snowflake ML Jobs Submission Script
====================================
This script demonstrates how to submit the Pyomo inventory optimization
as a Snowflake ML Job using the file-based approach.

Prerequisites:
- snowflake-ml-python >= 1.9.2
- A configured Snowflake compute pool
- ML Jobs stage created

Usage:
    python submit_ml_job.py
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_snowflake_session():
    """Create a Snowflake session from environment variables or config."""
    from snowflake.snowpark import Session
    
    # Try to get connection parameters from environment
    connection_params = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE", "PYOMO_ML_JOBS_POC"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA", "INVENTORY_OPTIMIZATION"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
    }
    
    # Remove None values
    connection_params = {k: v for k, v in connection_params.items() if v is not None}
    
    if not connection_params.get("account"):
        raise ValueError(
            "Snowflake connection parameters not found. "
            "Please set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and SNOWFLAKE_PASSWORD environment variables."
        )
    
    logger.info(f"Connecting to Snowflake account: {connection_params['account']}")
    session = Session.builder.configs(connection_params).create()
    
    return session


def submit_optimization_job(
    session,
    compute_pool: str = "PYOMO_COMPUTE_POOL",
    stage_name: str = "ML_JOBS_STAGE",
    wait_for_completion: bool = True
):
    """
    Submit the inventory optimization script as a Snowflake ML Job.
    
    Args:
        session: Snowflake session
        compute_pool: Name of the compute pool to use
        stage_name: Name of the stage for ML job files
        wait_for_completion: Whether to wait for job completion
    
    Returns:
        Job object with status and results
    """
    from snowflake.ml.jobs import submit_file
    
    # Path to the optimization script
    script_path = Path(__file__).parent / "inventory_optimization.py"
    
    if not script_path.exists():
        raise FileNotFoundError(f"Optimization script not found: {script_path}")
    
    logger.info(f"Submitting ML Job from: {script_path}")
    logger.info(f"Compute Pool: {compute_pool}")
    logger.info(f"Stage: {stage_name}")
    
    # Submit the job
    job = submit_file(
        file_path=str(script_path),
        compute_pool=compute_pool,
        stage_name=stage_name,
        session=session,
        # Specify pip packages to install in the job environment
        pip_requirements=[
            "pyomo>=6.7.0",
            "highspy>=1.5.0",  # HiGHS solver - pip installable MILP solver
            "pandas>=2.0.0",
        ],
    )
    
    logger.info(f"Job submitted successfully!")
    logger.info(f"Job ID: {job.id}")
    logger.info(f"Job Status: {job.status}")
    
    if wait_for_completion:
        logger.info("Waiting for job completion...")
        job.wait()
        
        logger.info(f"Final Status: {job.status}")
        
        # Get job output/logs
        try:
            output = job.get_logs()
            logger.info("Job Output:")
            print(output)
        except Exception as e:
            logger.warning(f"Could not retrieve job logs: {e}")
    
    return job


def submit_with_directory(
    session,
    compute_pool: str = "PYOMO_COMPUTE_POOL",
    stage_name: str = "ML_JOBS_STAGE",
):
    """
    Alternative: Submit the entire src directory as a job.
    Useful when you have multiple Python modules.
    """
    from snowflake.ml.jobs import submit_directory
    
    src_dir = Path(__file__).parent
    
    logger.info(f"Submitting ML Job from directory: {src_dir}")
    
    job = submit_directory(
        source_dir=str(src_dir),
        entrypoint="inventory_optimization.py",
        compute_pool=compute_pool,
        stage_name=stage_name,
        session=session,
        pip_requirements=[
            "pyomo>=6.7.0",
            "pandas>=2.0.0",
        ],
    )
    
    logger.info(f"Job ID: {job.id}")
    return job


def check_prerequisites(session):
    """Check if required Snowflake objects exist."""
    logger.info("Checking prerequisites...")
    
    # Check compute pool
    try:
        result = session.sql("SHOW COMPUTE POOLS LIKE 'PYOMO_COMPUTE_POOL'").collect()
        if not result:
            logger.warning("Compute pool 'PYOMO_COMPUTE_POOL' not found. Please create it first.")
            logger.info("Run: CREATE COMPUTE POOL PYOMO_COMPUTE_POOL MIN_NODES=1 MAX_NODES=3 INSTANCE_FAMILY='CPU_X64_S'")
            return False
        logger.info("✓ Compute pool exists")
    except Exception as e:
        logger.warning(f"Could not check compute pool: {e}")
    
    # Check stage
    try:
        result = session.sql("SHOW STAGES LIKE 'ML_JOBS_STAGE'").collect()
        if not result:
            logger.warning("Stage 'ML_JOBS_STAGE' not found. Creating...")
            session.sql("CREATE STAGE IF NOT EXISTS ML_JOBS_STAGE DIRECTORY = (ENABLE = TRUE)").collect()
            logger.info("✓ Stage created")
        else:
            logger.info("✓ Stage exists")
    except Exception as e:
        logger.warning(f"Could not check/create stage: {e}")
    
    # Check tables
    for table in ["SALES_PREDICTIONS", "ORDERS"]:
        try:
            count = session.table(table).count()
            logger.info(f"✓ Table {table} exists with {count} rows")
        except Exception as e:
            logger.warning(f"Table {table} not found or empty. Please load data first.")
            return False
    
    return True


def main():
    """Main entry point for job submission."""
    logger.info("=" * 60)
    logger.info("Snowflake ML Jobs - Pyomo Optimization Submission")
    logger.info("=" * 60)
    
    try:
        # Create session
        session = get_snowflake_session()
        logger.info("✓ Connected to Snowflake")
        
        # Check prerequisites
        if not check_prerequisites(session):
            logger.error("Prerequisites check failed. Please fix issues and retry.")
            sys.exit(1)
        
        # Submit the job
        job = submit_optimization_job(
            session=session,
            compute_pool="PYOMO_COMPUTE_POOL",
            stage_name="ML_JOBS_STAGE",
            wait_for_completion=True
        )
        
        if job.status == "COMPLETED":
            logger.info("\n✓ ML Job completed successfully!")
            logger.info("Check OPTIMIZED_ORDER_ALLOCATION and OPTIMIZATION_SUMMARY tables for results.")
        else:
            logger.error(f"\n✗ ML Job failed with status: {job.status}")
            sys.exit(1)
            
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Please install: pip install snowflake-ml-python>=1.9.2")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

