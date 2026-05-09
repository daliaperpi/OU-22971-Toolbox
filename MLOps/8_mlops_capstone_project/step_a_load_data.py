"""Step A: Load reference and batch data from TLC_data directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import mlflow

from mlops_project_lib import load_taxi_table
from mlflow_table_utils import log_table_with_markdown, log_step_header

logger = logging.getLogger("mlops_step_a")


def _dataset_profile(df: pd.DataFrame, label: str, source_path: Path) -> dict:
    pickup_min = None
    pickup_max = None
    if "lpep_pickup_datetime" in df.columns:
        pickup = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
        if pickup.notna().any():
            pickup_min = pickup.min().isoformat()
            pickup_max = pickup.max().isoformat()

    total_cells = max(len(df) * max(len(df.columns), 1), 1)
    missing_cells = int(df.isna().sum().sum())
    return {
        "dataset": label,
        "source_file": str(source_path.name),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": missing_cells,
        "missing_pct": float(100.0 * missing_cells / total_cells),
        "pickup_min": pickup_min,
        "pickup_max": pickup_max,
    }


def find_tlc_data_dir() -> Path:
    """Locate TLC_data directory from workspace root."""
    candidates = [Path("TLC_data"), Path("MLOps/TLC_data"), Path("../../TLC_data"),]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    
    raise FileNotFoundError(f"Could not find TLC_data directory. Searched: {candidates}")


def get_taxi_files(data_dir: Path, month: str, year: str = "2020") -> Path:
    """Get taxi data file for a specific month and year.
    """
    file_pattern = f"green_tripdata_{year}-{month}.parquet"
    file_path = data_dir / file_pattern
    
    if not file_path.exists():
        raise FileNotFoundError(f"Taxi data not found: {file_path}")
    
    return file_path


def load_taxi_batch(path: Path) -> pd.DataFrame:
    """Load and validate a single taxi data file.
    
    Handles:
    - Datetime parsing for pickup/dropoff
    - Basic shape and type validation
    
    Args:
        path: Path to parquet file
        
    Returns:
        DataFrame with loaded taxi data
    """
    df = load_taxi_table(path)
    
    # Ensure datetime columns are parsed
    for col in ["lpep_pickup_datetime", "lpep_dropoff_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    return df


def step_a_load_data(data_dir: Path, ref_month: str, batch_month: str) -> Dict:
    """Load reference and batch datasets from TLC_data.
    
    Step A: Load data
    - Load reference dataset by month
    - Load batch dataset by month
    - Log dataset info to MLflow
    
    Args:
        data_dir: Path to TLC_data directory
        ref_month: Reference dataset month (e.g., '01')
        batch_month: Batch dataset month (e.g., '04')
    
    Returns:
        Dictionary with loaded dataframes and metadata:
    """
    log_step_header(logger, "STEP A: Load data")
    
    # Get file paths
    ref_path = get_taxi_files(data_dir, ref_month)
    batch_path = get_taxi_files(data_dir, batch_month)
    
    logger.info(f"Loading reference: {ref_path}")
    ref_df = load_taxi_batch(ref_path)
    logger.info(f"  Reference shape: {ref_df.shape}")
    
    logger.info(f"Loading batch: {batch_path}")
    batch_df = load_taxi_batch(batch_path)
    logger.info(f"  Batch shape: {batch_df.shape}")
    
    # Log dataset info to MLflow
    mlflow.log_param("ref_month", ref_month)
    mlflow.log_param("batch_month", batch_month)
    mlflow.log_metric("ref_rows", len(ref_df))
    mlflow.log_metric("batch_rows", len(batch_df))

    # Log human-readable tables for MLflow review.
    summary_rows = [
        _dataset_profile(ref_df, "reference", ref_path),
        _dataset_profile(batch_df, "batch", batch_path),
    ]
    summary_df = pd.DataFrame(summary_rows)
    log_table_with_markdown(summary_df, "step_a/dataset_summary.json", "step_a/dataset_summary.md")

    schema_cols = sorted(set(ref_df.columns) | set(batch_df.columns))
    schema_rows = []
    for col in schema_cols:
        in_ref = col in ref_df.columns
        in_batch = col in batch_df.columns
        ref_dtype = str(ref_df[col].dtype) if in_ref else None
        batch_dtype = str(batch_df[col].dtype) if in_batch else None
        schema_rows.append(
            {
                "column": col,
                "in_reference": in_ref,
                "in_batch": in_batch,
                "ref_dtype": ref_dtype,
                "batch_dtype": batch_dtype,
                "dtype_match": bool(in_ref and in_batch and ref_dtype == batch_dtype),
            }
        )
    schema_df = pd.DataFrame(schema_rows)
    log_table_with_markdown(schema_df, "step_a/schema_compare.json", "step_a/schema_compare.md")

    missing_df = (
        batch_df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_frac"})
    )
    missing_df["missing_pct"] = (missing_df["missing_frac"] * 100.0).round(3)
    log_table_with_markdown(
        missing_df[["column", "missing_frac", "missing_pct"]],
        "step_a/batch_top_missing.json",
        "step_a/batch_top_missing.md",
    )

    logger.info("Step A completed successfully!")
  
    return {"reference_df": ref_df, "batch_df": batch_df,}
