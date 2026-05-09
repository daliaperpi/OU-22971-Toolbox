"""mlops_project_lib.py

Shared utilities for the MLOps project workflow.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd


def setup_logger(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("mlops")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    formatter = logging.Formatter( '%(asctime)s %(name)-10s %(levelname)-7s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.expanduser('~/mlops.log')
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log file {log_file}: {e}")
    
    return logger

# Valid range for numeric columns
RANGE_SPECS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("trip_distance", 0.0, 200.0),
    ("fare_amount", 0.0, 500.0),
    ("tip_amount", 0.0, 200.0),
    ("tolls_amount", 0.0, 200.0),
    ("total_amount", 0.0, 1000.0),
    ("passenger_count", 1.0, 10.0),
]

# Hard rule thresholds
HARD_RULES = {
    "max_missing_pct": 20.0,  # Allow max 20% missing per column
    "missing_thresholds": {
        "store_and_fwd_flag": 40.0,
        "RatecodeID": 40.0,
        "passenger_count": 40.0,
        "payment_type": 40.0,
        "trip_type": 40.0,
        "congestion_surcharge": 40.0,
    },
    "max_invalid_fraction": 0.005,
    "max_datetime_order_violations_fraction": 0.005,
    "max_tip_gt_total_fraction": 0.005,
    "required_columns": [
        "lpep_pickup_datetime",
        "lpep_dropoff_datetime",
        "trip_distance",
        "tip_amount",
        "PULocationID",
        "DOLocationID",
    ],
}


def load_taxi_table(path: Path) -> pd.DataFrame:
    """Load TLC Green Taxi data from parquet or CSV."""
    logger = logging.getLogger("mlops")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    logger.debug(f"Loading data from: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        try:
            df = pd.read_parquet(path)
            logger.debug(f"Loaded parquet file: {df.shape}")
        except ImportError as e:
            raise ImportError(
                "Parquet support missing. Install 'pyarrow' or 'fastparquet'. "
                "Example: conda install -c conda-forge pyarrow"
            ) from e
    elif suf == ".csv":
        df = pd.read_csv(path)
        logger.debug(f"Loaded CSV file: {df.shape}")
    else:
        raise ValueError(f"Unsupported file format: {suf}")

    # Convert datetime columns
    datetime_cols = ["lpep_pickup_datetime", "lpep_dropoff_datetime"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
