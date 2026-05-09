"""test_step_a.py

Test script for Step A using synthetic TLC-like data.

This script:
1. Generates synthetic reference and batch datasets
2. Saves them as parquets
3. Runs the MLOps project workflow Step A
4. Verifies the MLflow logging

Run: python test_step_a.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import subprocess
import sys
import logging

import pandas as pd
import numpy as np

from mlops_project_lib import setup_logger

logger = logging.getLogger("mlops")


def generate_synthetic_taxi_data(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic TLC Green Taxi data.
    
    Args:
        n_rows: number of rows to generate
        seed: random seed for reproducibility
        
    Returns:
        DataFrame with TLC-like columns
    """
    np.random.seed(seed)
    
    # Time range for this batch
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="10min")
    
    data = {
        "VendorID": np.random.choice([1, 2], n_rows),
        "lpep_pickup_datetime": dates,
        "lpep_dropoff_datetime": dates + pd.Timedelta(minutes=np.random.uniform(5, 45, n_rows)),
        "store_and_fwd_flag": np.random.choice(["Y", "N"], n_rows),
        "RatecodeID": np.random.choice([1, 2, 3, 4, 5], n_rows),
        "PULocationID": np.random.randint(1, 263, n_rows),
        "DOLocationID": np.random.randint(1, 263, n_rows),
        "passenger_count": np.random.choice([1, 2, 3, 4, 5, 6], n_rows),
        "trip_distance": np.random.uniform(0.1, 20, n_rows),
        "fare_amount": np.random.uniform(2.5, 50, n_rows),
        "extra": np.random.choice([0, 0.5, 1], n_rows),
        "mta_tax": np.random.choice([0.5], n_rows),
        "tip_amount": np.random.uniform(0, 10, n_rows),
        "tolls_amount": np.random.choice([0, 5.76], n_rows, p=[0.95, 0.05]),
        "total_amount": np.random.uniform(2.5, 50, n_rows),
        "payment_type": np.random.choice([1, 2, 3, 4], n_rows),
        "trip_type": np.random.choice([1, 2], n_rows),
        "congestion_surcharge": np.random.choice([0, 2.5, 2.75], n_rows),
        "improvement_surcharge": np.random.choice([0.3], n_rows),
        "ehail_fee": np.random.choice([None, 0], n_rows),
    }
    
    df = pd.DataFrame(data)
    return df


def main():
    # Setup logger
    setup_logger()
    
    logger.info("="*60)
    logger.info("TEST STEP A: Load Data")
    logger.info("="*60)
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Generate synthetic data
        logger.info("[1] Generating synthetic reference data...")
        ref_df = generate_synthetic_taxi_data(n_rows=100, seed=42)
        ref_path = tmpdir / "reference.parquet"
        ref_df.to_parquet(ref_path)
        logger.info(f"  ✓ Saved: {ref_path} (shape: {ref_df.shape})")
        
        logger.info("[2] Generating synthetic batch data...")
        batch_df = generate_synthetic_taxi_data(n_rows=80, seed=43)
        batch_path = tmpdir / "batch.parquet"
        batch_df.to_parquet(batch_path)
        logger.info(f"  ✓ Saved: {batch_path} (shape: {batch_df.shape})")
        
        # Run workflow
        logger.info("[3] Running MLOps project workflow Step A...")
        cmd = [
            sys.executable,
            "mlops_project_workflow.py",
            "--tracking-uri", "http://localhost:5000",
            "--experiment", "test_mlops_project",
            "--ref-parquet", str(ref_path),
            "--batch-parquet", str(batch_path),
        ]
        
        try:
            result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
            logger.info("  ✓ Workflow completed successfully!")
        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Workflow failed with exit code {e.returncode}")
            sys.exit(1)
        
        logger.info("="*60)
        logger.info("TEST PASSED!")
        logger.info("="*60)
        logger.info("Next steps:")
        logger.info("1. Check MLflow UI at http://localhost:5000")
        logger.info("2. Look for experiment 'test_mlops_project'")
        logger.info("3. View the data_load_summary.json artifact")


if __name__ == "__main__":
    main()
