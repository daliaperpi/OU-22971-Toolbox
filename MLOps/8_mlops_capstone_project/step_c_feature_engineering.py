"""Step C: Feature engineering - transform raw data to model-ready features.

Functions:
  - step_c_feature_engineering()
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List

import pandas as pd
import numpy as np
import mlflow
from mlflow_table_utils import log_table_with_markdown, log_step_header

logger = logging.getLogger("mlops")


def step_c_feature_engineering(data: Dict) -> Dict:
    """Step C: Feature engineering - transform raw data to model-ready features.
    
    Applies consistent transformations to both reference and batch:
    - Time features (hour, day_of_week, month, day_of_year)
    - Location features (normalized zone IDs)
    - Log/clip transforms for heavy-tailed numeric fields
    - Missing value imputation
    
    Args:
        data: Dictionary with 'reference_df' and 'batch_df' from Step B
        
    Returns:
        Dictionary with engineered features and feature specification:
        {
            "reference_features": pd.DataFrame,
            "batch_features": pd.DataFrame,
            "feature_spec": list[dict],
            "feature_names": list[str]
        }
    """
    log_step_header(logger, "STEP C: Feature engineering")
    
    ref_df = data["reference_df"].copy()
    batch_df = data["batch_df"].copy()

    def build_raw_preview(df: pd.DataFrame, n_rows: int = 8) -> pd.DataFrame:
        """Create a compact raw-data preview table for MLflow logging."""
        preferred_cols = [
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "fare_amount",
            "total_amount",
            "passenger_count",
            "payment_type",
            "congestion_surcharge",
            "tolls_amount",
            "trip_type",
            "tip_amount",
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        if not cols:
            cols = list(df.columns[:12])
        return df[cols].head(n_rows).reset_index(drop=True)

    def build_feature_preview(df: pd.DataFrame, n_rows: int = 8) -> pd.DataFrame:
        """Create a compact engineered-feature preview table for MLflow logging."""
        cols = list(df.columns[:12])
        return df[cols].head(n_rows).reset_index(drop=True)
    
    def engineer_features(df: pd.DataFrame, data_split: str = "reference") -> pd.DataFrame:
        """Apply feature engineering transformations to a dataframe.
        
        Features engineered:
        1. Time features from pickup datetime
        2. Location features (normalized zone IDs)
        3. Distance feature (log transform)
        4. Monetary fields (log transforms)
        5. Passenger count (binned)
        6. Trip type
        7. Payment type (one-hot encoded)
        8. Surcharge flags
        9. Trip duration
        
        Args:
            df: Input dataframe (reference or batch)
            data_split: Label for logging (reference or batch)
            
        Returns:
            Engineered features dataframe
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Time features from pickup datetime
        if "lpep_pickup_datetime" in df.columns:
            pickup_dt = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
            features["pickup_hour"] = pickup_dt.dt.hour.fillna(-1).astype(int)
            features["pickup_day_of_week"] = pickup_dt.dt.dayofweek.fillna(-1).astype(int)  # 0=Monday, 6=Sunday
            features["pickup_month"] = pickup_dt.dt.month.fillna(-1).astype(int)
            features["pickup_day_of_year"] = pickup_dt.dt.dayofyear.fillna(-1).astype(int)
        
        # 2. Location features - normalize zone IDs
        # NYC has ~260 zones; normalize to 0-1 range
        if "PULocationID" in df.columns:
            pu_max = 263  # Approximate max zone ID
            features["pu_location_norm"] = (df["PULocationID"].fillna(0) / pu_max).clip(0, 1).astype(float)
        
        if "DOLocationID" in df.columns:
            do_max = 263
            features["do_location_norm"] = (df["DOLocationID"].fillna(0) / do_max).clip(0, 1).astype(float)
        
        # 3. Distance feature - log transform heavy-tailed distribution
        if "trip_distance" in df.columns:
            # Replace negative/zero values before log
            distance = df["trip_distance"].fillna(0)
            distance = distance.clip(lower=0.1)  # Avoid log(0)
            features["log_trip_distance"] = np.log1p(distance).astype(float)
        
        # 4. Monetary fields - log transforms for fare and total
        # Do not include tip_amount-derived features, because tip_amount is the label.
        for col, feat_name in [
            ("fare_amount", "log_fare_amount"),
            ("total_amount", "log_total_amount"),
        ]:
            if col in df.columns:
                amount = df[col].fillna(0)
                amount = amount.clip(lower=0.1)  # Avoid log(0)
                features[feat_name] = np.log1p(amount).astype(float)
        
        # 5. Passenger count (binned to reduce cardinality)
        if "passenger_count" in df.columns:
            passenger = df["passenger_count"].fillna(1).astype(int).clip(1, 6)  # Cap at 6+
            features["passenger_count_binned"] = passenger.astype(int)
        
        # 6. Trip type (if available)
        if "trip_type" in df.columns:
            features["trip_type"] = df["trip_type"].fillna(0).astype(int)
        
        # 7. Payment type (one-hot encoded)
        if "payment_type" in df.columns:
            payment = df["payment_type"].fillna(0).astype(int)
            for payment_id in [1, 2, 3, 4]:  # Common payment types
                features[f"payment_type_{payment_id}"] = (payment == payment_id).astype(int)
        
        # 8. Surcharge flags
        if "congestion_surcharge" in df.columns:
            congestion = df["congestion_surcharge"].fillna(0)
            features["has_congestion_surcharge"] = (congestion > 0).astype(int)
        
        if "tolls_amount" in df.columns:
            tolls = df["tolls_amount"].fillna(0)
            features["has_tolls"] = (tolls > 0).astype(int)
        
        # 9. Trip duration in minutes
        if "lpep_pickup_datetime" in df.columns and "lpep_dropoff_datetime" in df.columns:
            pickup_dt = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
            dropoff_dt = pd.to_datetime(df["lpep_dropoff_datetime"], errors="coerce")
            duration = (dropoff_dt - pickup_dt).dt.total_seconds() / 60  # Convert to minutes
            duration = duration.fillna(0).clip(lower=0, upper=300)  # Cap at 5 hours
            features["trip_duration_minutes"] = duration.astype(float)
        
        logger.info(f"  {data_split}: {len(features)} rows x {len(features.columns)} features")
        return features
    
    # Engineer features for both reference and batch
    logger.info(f"  Engineering reference features...")
    ref_features = engineer_features(ref_df, "reference")
    
    logger.info(f"  Engineering batch features...")
    batch_features = engineer_features(batch_df, "batch")

    # Log before/after row previews so transformations are easy to inspect in MLflow.
    raw_ref_preview_df = build_raw_preview(ref_df)
    raw_batch_preview_df = build_raw_preview(batch_df)
    feat_ref_preview_df = build_feature_preview(ref_features)
    feat_batch_preview_df = build_feature_preview(batch_features)
    
    # Verify schema consistency
    if set(ref_features.columns) != set(batch_features.columns):
        logger.error("Schema mismatch between reference and batch features!")
        missing_in_batch = set(ref_features.columns) - set(batch_features.columns)
        extra_in_batch = set(batch_features.columns) - set(ref_features.columns)
        if missing_in_batch:
            logger.error(f"  Missing in batch: {missing_in_batch}")
        if extra_in_batch:
            logger.error(f"  Extra in batch: {extra_in_batch}")
        raise ValueError("Feature schema mismatch")
    
    # Create feature spec (for reproducibility)
    feature_spec = [
        {"name": col, "dtype": str(ref_features[col].dtype)}
        for col in sorted(ref_features.columns)
    ]

    feature_schema_rows = []
    for col in sorted(ref_features.columns):
        feature_schema_rows.append(
            {
                "feature": col,
                "dtype": str(ref_features[col].dtype),
                "ref_missing_frac": float(ref_features[col].isna().mean()),
                "batch_missing_frac": float(batch_features[col].isna().mean()),
            }
        )
    feature_schema_df = pd.DataFrame(feature_schema_rows)

    split_summary_df = pd.DataFrame(
        [
            {
                "split": "reference",
                "rows": int(len(ref_features)),
                "features": int(len(ref_features.columns)),
            },
            {
                "split": "batch",
                "rows": int(len(batch_features)),
                "features": int(len(batch_features.columns)),
            },
        ]
    )
    
    # Log feature spec to MLflow
    spec_json = json.dumps(feature_spec, indent=2)
    mlflow.log_text(spec_json, "feature_spec.json")
    mlflow.log_param("num_features", len(feature_spec))
    log_table_with_markdown(feature_schema_df, "step_c/feature_schema.json", "step_c/feature_schema.md")
    log_table_with_markdown(split_summary_df, "step_c/split_summary.json", "step_c/split_summary.md")
    log_table_with_markdown(raw_ref_preview_df, "step_c/preview_reference_before.json", "step_c/preview_reference_before.md")
    log_table_with_markdown(feat_ref_preview_df, "step_c/preview_reference_after.json", "step_c/preview_reference_after.md")
    log_table_with_markdown(raw_batch_preview_df, "step_c/preview_batch_before.json", "step_c/preview_batch_before.md")
    log_table_with_markdown(feat_batch_preview_df, "step_c/preview_batch_after.json", "step_c/preview_batch_after.md")
    
    logger.info(f"  Feature spec: {len(feature_spec)} features")
    logger.info(f"Step C completed successfully!")
    
    return {
        "reference_features": ref_features,
        "batch_features": batch_features,
        "feature_spec": feature_spec,
        "feature_names": list(ref_features.columns),
    }
