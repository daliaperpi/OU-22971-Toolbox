"""Step B: Integrity gate with hard rules and NannyML drift detection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
from datetime import datetime
import warnings

import pandas as pd
import mlflow
import nannyml as nml

from mlops_project_lib import HARD_RULES, RANGE_SPECS
from mlflow_table_utils import (
    log_bool_param_and_tag,
    log_decision_json,
    log_step_header,
    log_table_with_markdown,
)

logger = logging.getLogger("mlops_step_b")

SUPPLEMENTAL_EXPECTED_SCHEMA = {
    "ehail_fee": "object",
    "RatecodeID": "float64",
    "store_and_fwd_flag": "object",
    "trip_type": "float64",
    "payment_type": "float64",
    "passenger_count": "float64",
    "congestion_surcharge": "float64",
    "DOLocationID": "int64",
    "PULocationID": "int64",
    "lpep_pickup_datetime": "datetime64[us]",
    "lpep_dropoff_datetime": "datetime64[us]",
    "VendorID": "int64",
    "extra": "float64",
    "fare_amount": "float64",
    "trip_distance": "float64",
    "tolls_amount": "float64",
    "tip_amount": "float64",
    "mta_tax": "float64",
    "total_amount": "float64",
    "improvement_surcharge": "float64",
}

DOMAIN_SPECS = [
    ("store_and_fwd_flag", ["Y", "N"]),
    ("payment_type", [1, 2, 3, 4, 5, 6]),
    ("trip_type", [1, 2]),
    ("RatecodeID", [1, 2, 3, 4, 5, 6]),
]


def _expected_family(exp: str) -> str:
    exp = str(exp).strip().lower()
    if exp.startswith("datetime64"):
        return "datetime"
    if exp in {"object", "string"}:
        return "string"
    if exp.startswith("int") or exp.startswith("float") or exp in {"number", "numeric"}:
        return "numeric"
    if exp in {"bool", "boolean"}:
        return "bool"
    return "exact"


def _family_ok(actual_dtype, expected: str) -> bool:
    t = pd.api.types
    family = _expected_family(expected)
    if family == "datetime":
        return t.is_datetime64_any_dtype(actual_dtype)
    if family == "numeric":
        return t.is_numeric_dtype(actual_dtype)
    if family == "string":
        return t.is_object_dtype(actual_dtype) or t.is_string_dtype(actual_dtype) or t.is_categorical_dtype(actual_dtype)
    if family == "bool":
        return t.is_bool_dtype(actual_dtype)
    return str(actual_dtype) == str(expected)


def _run_additional_integrity_checks(batch_df: pd.DataFrame) -> Dict:
    """Supplemental Unit 6-style checks (additive; no gate behavior change)."""
    tables: Dict[str, pd.DataFrame] = {}
    metrics: Dict[str, float] = {}

    present_cols = set(batch_df.columns)
    expected_cols = set(SUPPLEMENTAL_EXPECTED_SCHEMA.keys())
    missing_expected = sorted(expected_cols - present_cols)
    extra_columns = sorted(present_cols - expected_cols)

    dtype_rows = []
    bad_family = 0
    bad_exact = 0
    for col, exp_dtype in SUPPLEMENTAL_EXPECTED_SCHEMA.items():
        if col not in batch_df.columns:
            continue
        actual_dtype = batch_df[col].dtype
        family_ok = _family_ok(actual_dtype, exp_dtype)
        exact_ok = str(actual_dtype) == str(exp_dtype)
        if not family_ok:
            bad_family += 1
        if not exact_ok:
            bad_exact += 1
        dtype_rows.append(
            {
                "column": col,
                "expected_dtype": str(exp_dtype),
                "actual_dtype": str(actual_dtype),
                "family_ok": bool(family_ok),
                "exact_match": bool(exact_ok),
            }
        )

    tables["schema_missing_expected"] = pd.DataFrame({"column": missing_expected}, columns=["column"])
    tables["schema_extra_columns"] = pd.DataFrame({"column": extra_columns}, columns=["column"])
    tables["schema_dtypes"] = pd.DataFrame(
        dtype_rows,
        columns=["column", "expected_dtype", "actual_dtype", "family_ok", "exact_match"],
    )
    metrics["schema_missing_cols"] = float(len(missing_expected))
    metrics["schema_extra_cols"] = float(len(extra_columns))
    metrics["schema_bad_family_dtypes"] = float(bad_family)
    metrics["schema_bad_exact_dtypes"] = float(bad_exact)

    missingness_df = pd.DataFrame(
        {
            "column": batch_df.columns,
            "dtype": batch_df.dtypes.astype(str).to_numpy(),
            "missing_frac": batch_df.isna().mean().to_numpy(),
            "missing_count": batch_df.isna().sum().to_numpy(),
            "n_unique": batch_df.nunique(dropna=False).to_numpy(),
        }
    ).sort_values("missing_frac", ascending=False, kind="stable")
    tables["missingness"] = missingness_df
    metrics["missing_frac_mean"] = float(batch_df.isna().mean().mean())
    metrics["missing_frac_max"] = float(batch_df.isna().mean().max())

    duplicate_rows = int(batch_df.duplicated().sum()) if len(batch_df) else 0
    duplicate_rows_frac = float(duplicate_rows / max(len(batch_df), 1))
    tables["duplicates"] = pd.DataFrame(
        [{"rows": int(len(batch_df)), "duplicate_rows": duplicate_rows, "duplicate_rows_frac": duplicate_rows_frac}]
    )
    metrics["duplicate_rows"] = float(duplicate_rows)
    metrics["duplicate_rows_frac"] = duplicate_rows_frac

    range_rows = []
    for col, min_val, max_val in RANGE_SPECS:
        if col not in batch_df.columns:
            continue
        values = pd.to_numeric(batch_df[col], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        bad = pd.Series(False, index=valid.index)
        if min_val is not None:
            bad |= valid < min_val
        if max_val is not None:
            bad |= valid > max_val
        range_rows.append(
            {
                "column": col,
                "lo": min_val,
                "hi": max_val,
                "bad_frac": float(bad.mean()),
                "min": float(valid.min()),
                "max": float(valid.max()),
            }
        )
    range_df = pd.DataFrame(range_rows).sort_values("bad_frac", ascending=False) if range_rows else pd.DataFrame(
        columns=["column", "lo", "hi", "bad_frac", "min", "max"]
    )
    tables["range_checks"] = range_df
    metrics["range_worst_bad_frac"] = float(range_df["bad_frac"].max()) if not range_df.empty else 0.0
    metrics["range_any_bad_cols"] = float((range_df["bad_frac"] > 0).sum()) if not range_df.empty else 0.0

    domain_rows = []
    for col, allowed in DOMAIN_SPECS:
        if col not in batch_df.columns:
            continue
        s = batch_df[col]
        bad = ~s.isna() & ~s.isin(set(allowed))
        domain_rows.append(
            {
                "column": col,
                "bad_frac": float(bad.mean()) if len(s) else 0.0,
                "bad_count": int(bad.sum()) if len(s) else 0,
                "n_unique": int(s.nunique(dropna=True)) if len(s) else 0,
            }
        )
    domain_df = pd.DataFrame(domain_rows).sort_values("bad_frac", ascending=False) if domain_rows else pd.DataFrame(
        columns=["column", "bad_frac", "bad_count", "n_unique"]
    )
    tables["domain_checks"] = domain_df
    metrics["domain_worst_bad_frac"] = float(domain_df["bad_frac"].max()) if not domain_df.empty else 0.0
    metrics["domain_any_bad_cols"] = float((domain_df["bad_count"] > 0).sum()) if not domain_df.empty else 0.0

    if "lpep_pickup_datetime" in batch_df.columns and "lpep_dropoff_datetime" in batch_df.columns:
        pickup = pd.to_datetime(batch_df["lpep_pickup_datetime"], errors="coerce")
        dropoff = pd.to_datetime(batch_df["lpep_dropoff_datetime"], errors="coerce")
        duration = (dropoff - pickup).dt.total_seconds() / 60.0
        duration_neg_frac = float((duration < 0).mean()) if len(duration) else 0.0
        duration_over_6h_frac = float((duration > 360).mean()) if len(duration) else 0.0
        duration_nan_frac = float(duration.isna().mean()) if len(duration) else 0.0
    else:
        duration_neg_frac = 0.0
        duration_over_6h_frac = 0.0
        duration_nan_frac = 0.0

    tables["datetime_checks"] = pd.DataFrame(
        [
            {"column": "duration_min", "check": "duration_negative", "bad_frac": duration_neg_frac},
            {"column": "duration_min", "check": "duration_over_6h", "bad_frac": duration_over_6h_frac},
            {"column": "duration_min", "check": "duration_nan", "bad_frac": duration_nan_frac},
        ]
    )
    metrics["duration_neg_frac"] = duration_neg_frac
    metrics["duration_over_6h_frac"] = duration_over_6h_frac
    metrics["duration_nan_frac"] = duration_nan_frac

    return {"tables": tables, "metrics": metrics}


def step_b_integrity_gate_hard_rules(batch_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Layer 1: Hard rules - fail-fast integrity checks on raw batch.
    
    Checks:
    - Required columns present
    - Missing value percentages
    - Value ranges for numeric columns
    - Datetime ordering
    - Logical consistency
    - Key columns have no None/NA
    
    Args:
        batch_df: Raw batch dataset
        
    Returns:
        Tuple of (passed: bool, failures: list[str])
    """
    failures = []
    
    # Check 1: Required columns present
    missing_cols = set(HARD_RULES["required_columns"]) - set(batch_df.columns)
    if missing_cols:
        failures.append(f"Missing required columns: {sorted(missing_cols)}")
    
    # Check 2: Missing value percentage (critical + configured columns)
    cols_to_check = set(HARD_RULES["required_columns"]) | set(HARD_RULES.get("missing_thresholds", {}).keys())
    for col in sorted(cols_to_check):
        if col not in batch_df.columns:
            continue
        missing_pct = 100.0 * batch_df[col].isna().sum() / len(batch_df)
        threshold = HARD_RULES.get("missing_thresholds", {}).get(col, HARD_RULES["max_missing_pct"])
        if missing_pct > threshold:
            failures.append(
                f"Column '{col}' has {missing_pct:.1f}% missing values "
                f"(threshold: {threshold}%)"
            )
    
    # Check 3: Value ranges for numeric columns
    for col, min_val, max_val in RANGE_SPECS:
        if col not in batch_df.columns:
            continue
        col_data = batch_df[col].dropna()
        if len(col_data) == 0:
            continue
            
        if min_val is not None and (col_data < min_val).any():
            invalid_count = (col_data < min_val).sum()
            invalid_fraction = invalid_count / len(batch_df)
            if invalid_fraction > HARD_RULES.get("max_invalid_fraction", 0.0):
                failures.append(
                    f"Column '{col}' has {invalid_count} values below minimum {min_val} "
                    f"({invalid_fraction:.2%} > {HARD_RULES.get('max_invalid_fraction', 0.0):.2%})"
                )
        
        if max_val is not None and (col_data > max_val).any():
            invalid_count = (col_data > max_val).sum()
            invalid_fraction = invalid_count / len(batch_df)
            if invalid_fraction > HARD_RULES.get("max_invalid_fraction", 0.0):
                failures.append(
                    f"Column '{col}' has {invalid_count} values above maximum {max_val} "
                    f"({invalid_fraction:.2%} > {HARD_RULES.get('max_invalid_fraction', 0.0):.2%})"
                )
    
    # Check 4: Datetime ordering (pickup < dropoff)
    if "lpep_pickup_datetime" in batch_df.columns and "lpep_dropoff_datetime" in batch_df.columns:
        pickup = pd.to_datetime(batch_df["lpep_pickup_datetime"], errors="coerce")
        dropoff = pd.to_datetime(batch_df["lpep_dropoff_datetime"], errors="coerce")
        invalid_dates = (pickup >= dropoff).sum()
        invalid_fraction = invalid_dates / len(batch_df)
        if invalid_fraction > HARD_RULES.get("max_datetime_order_violations_fraction", 0.0):
            failures.append(
                f"Found {invalid_dates} rows where pickup_datetime >= dropoff_datetime "
                f"({invalid_fraction:.2%} > {HARD_RULES.get('max_datetime_order_violations_fraction', 0.0):.2%})"
            )
    
    # Check 5: Logical consistency - tip_amount <= total_amount
    if "tip_amount" in batch_df.columns and "total_amount" in batch_df.columns:
        tip = batch_df["tip_amount"].dropna()
        total = batch_df["total_amount"].dropna()
        if len(tip) > 0 and len(total) > 0:
            # Compare aligned rows
            aligned_idx = batch_df.index[batch_df["tip_amount"].notna() & batch_df["total_amount"].notna()]
            invalid_tips = (batch_df.loc[aligned_idx, "tip_amount"] > batch_df.loc[aligned_idx, "total_amount"]).sum()
            invalid_fraction = invalid_tips / len(batch_df)
            if invalid_fraction > HARD_RULES.get("max_tip_gt_total_fraction", 0.0):
                failures.append(
                    f"Found {invalid_tips} rows where tip_amount > total_amount "
                    f"({invalid_fraction:.2%} > {HARD_RULES.get('max_tip_gt_total_fraction', 0.0):.2%})"
                )
    
    # Check 6: Key columns have no None/NA
    critical_cols = ["lpep_pickup_datetime", "lpep_dropoff_datetime", "trip_distance", "tip_amount"]
    for col in critical_cols:
        if col in batch_df.columns:
            na_count = batch_df[col].isna().sum()
            if na_count > 0:
                failures.append(f"Critical column '{col}' has {na_count} missing values")
    
    passed = len(failures) == 0
    return passed, failures


def step_b_integrity_gate_nannyml(ref_df: pd.DataFrame, batch_df: pd.DataFrame) -> Dict:
    """Layer 2: NannyML drift checks - soft gate using reference data as baseline.
    
    Uses Kolmogorov-Smirnov test to detect univariate drift on numeric columns.
    Falls back to simple checks if NannyML unavailable.
    
    Args:
        ref_df: Reference dataset (baseline)
        batch_df: Batch dataset (to check for drift)
        
    Returns:
        Dictionary with NannyML results and warnings:
        {
            "warnings": list[str],
            "drift_columns": list[str],
            "nannyml_used": bool
        }
    """
    results = {"warnings": [], "drift_columns": [], "nannyml_used": False}
    
    try:
        # Add metadata columns to support NannyML
        ref_data = ref_df.copy()
        batch_data = batch_df.copy()
        
        # Add timestamp column (required by NannyML)
        if "lpep_pickup_datetime" in ref_data.columns:
            ref_data["timestamp"] = pd.to_datetime(ref_data["lpep_pickup_datetime"], errors="coerce")
            batch_data["timestamp"] = pd.to_datetime(batch_data["lpep_pickup_datetime"], errors="coerce")
        else:
            ref_data["timestamp"] = pd.Timestamp.now()
            batch_data["timestamp"] = pd.Timestamp.now()
        
        # Add data split column (required by NannyML)
        ref_data["data_period"] = "reference"
        batch_data["data_period"] = "batch"
        
        # Combine for NannyML processing
        combined_data = pd.concat([ref_data, batch_data], ignore_index=True)
        
        # Determine columns and type treatment for NannyML.
        # Restrict to business-relevant columns and drop sparse/constant columns.
        monitored_candidates = [
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "total_amount",
            "passenger_count",
            "RatecodeID",
            "payment_type",
            "trip_type",
            "congestion_surcharge",
        ]
        candidate_cols = [c for c in monitored_candidates if c in combined_data.columns]

        usable_cols = []
        min_non_null_ratio = 0.05
        for col in candidate_cols:
            ref_non_null_ratio = ref_data[col].notna().mean()
            batch_non_null_ratio = batch_data[col].notna().mean()
            ref_unique = ref_data[col].dropna().nunique()
            batch_unique = batch_data[col].dropna().nunique()
            if (
                ref_non_null_ratio >= min_non_null_ratio
                and batch_non_null_ratio >= min_non_null_ratio
                and ref_unique > 1
                and batch_unique > 1
            ):
                usable_cols.append(col)

        numeric_cols = combined_data[usable_cols].select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = combined_data[usable_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        feature_cols = sorted(set(numeric_cols + categorical_cols))

        if not feature_cols:
            logger.warning("No valid feature columns found for NannyML drift detection")
            return results

        chunk_size = max(100, min(len(batch_data), 1000))

        detector = nml.drift.univariate.UnivariateDriftCalculator(
            column_names=feature_cols,
            treat_as_numerical=numeric_cols or None,
            treat_as_categorical=categorical_cols or None,
            timestamp_column_name="timestamp",
            continuous_methods=["kolmogorov_smirnov"],
            categorical_methods=["chi2"],
            chunk_size=chunk_size,
        )
        
        # Fit detector on reference data
        reference_subset = combined_data[combined_data["data_period"] == "reference"]
        with warnings.catch_warnings():
            # NannyML can emit noisy RuntimeWarnings for near-empty slices.
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
            detector.fit(reference_subset)

            # Detect drift in batch data
            drift_result = detector.calculate(batch_data)
        results_df = drift_result.to_df()

        drifted_cols = []
        if isinstance(results_df.columns, pd.MultiIndex):
            for col in feature_cols:
                alert_keys = [
                    (col, "kolmogorov_smirnov", "alert"),
                    (col, "chi2", "alert"),
                    (col, "jensen_shannon", "alert"),
                    (col, "wasserstein", "alert"),
                    (col, "hellinger", "alert"),
                    (col, "l_infinity", "alert"),
                ]
                for key in alert_keys:
                    if key in results_df.columns and results_df[key].fillna(False).astype(bool).any():
                        drifted_cols.append(col)
                        break

        for col in sorted(set(drifted_cols)):
            results["warnings"].append(f"Column '{col}': NannyML detected univariate drift")
            results["drift_columns"].append(col)
        
        results["nannyml_used"] = True
        logger.info(f"NannyML monitored columns: {feature_cols}")
        logger.info(f"NannyML drift detection complete. Drifted columns: {len(results['drift_columns'])}")
        
    except Exception as e:
        logger.warning(f"NannyML drift detection failed: {e}. Using fallback checks.")
        results["warnings"] = []
        results["drift_columns"] = []
        # Fall back to simple checks
        return _step_b_fallback_drift_checks(ref_df, batch_df)
    
    return results


def _step_b_fallback_drift_checks(ref_df: pd.DataFrame, batch_df: pd.DataFrame) -> Dict:
    """Fallback drift checks when NannyML is unavailable.
    
    Simple univariate checks based on missing values and unseen categories.
    
    Args:
        ref_df: Reference dataset
        batch_df: Batch dataset
        
    Returns:
        Dictionary with fallback drift results
    """
    results = {"warnings": [], "drift_columns": [], "nannyml_used": False}
    
    try:
        # Check for missing values spike
        numeric_cols = batch_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        for col in numeric_cols:
            ref_missing_pct = 100.0 * ref_df[col].isna().sum() / len(ref_df)
            batch_missing_pct = 100.0 * batch_df[col].isna().sum() / len(batch_df)
            missing_increase = batch_missing_pct - ref_missing_pct
            
            if missing_increase > 5.0:  # More than 5% increase
                results["warnings"].append(
                    f"Column '{col}': missing values increased from {ref_missing_pct:.1f}% to {batch_missing_pct:.1f}%"
                )
                results["drift_columns"].append(col)
        
        # Check for unseen categorical values
        categorical_cols = batch_df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            ref_cats = set(ref_df[col].dropna().unique())
            batch_cats = set(batch_df[col].dropna().unique())
            unseen = batch_cats - ref_cats
            
            if unseen:
                unseen_count = batch_df[col].isin(unseen).sum()
                results["warnings"].append(
                    f"Column '{col}': {unseen_count} rows with unseen categorical values"
                )
                results["drift_columns"].append(col)
        
        logger.info(f"Fallback drift checks complete. Warnings: {len(results['warnings'])}")
    except Exception as e:
        logger.warning(f"Fallback drift checks failed: {e}")
    
    return results


def step_b_integrity_gate(data: Dict) -> Dict:
    """Step B: Integrity gate - check batch data quality before proceeding.
    
    Two layers:
    - Layer 1 (hard rules): Fail-fast on critical data issues
    - Layer 2 (soft gate): NannyML drift checks (warnings only)
    
    Args:
        data: Dictionary with 'reference_df' and 'batch_df' from Step A
        
    Returns:
        Dictionary with gate results and decision:
    """
    log_step_header(logger, "STEP B: Integrity gate")
    
    ref_df = data["reference_df"]
    batch_df = data["batch_df"]
    
    # Layer 1: Hard rules
    logger.info("  Layer 1: Running hard rules...")
    hard_pass, hard_failures = step_b_integrity_gate_hard_rules(batch_df)
    
    if hard_pass:
        logger.info("  [PASS] Hard rules passed")
    else:
        logger.error(f"  [FAIL] Hard rules FAILED: {len(hard_failures)} violations")
        for failure in hard_failures:
            logger.error(f"    - {failure}")
    
    # Layer 2: NannyML soft gate (runs even if hard rules fail, for logging)
    logger.info("  Layer 2: Running NannyML drift checks...")
    nannyml_results = step_b_integrity_gate_nannyml(ref_df, batch_df)
    
    if nannyml_results["warnings"]:
        logger.warning(f"  [WARN] NannyML warnings: {len(nannyml_results['warnings'])}")
        for warning in nannyml_results["warnings"]:
            logger.warning(f"    - {warning}")
    else:
        logger.info("  [PASS] No NannyML warnings")
    
    # Log whether NannyML was actually used
    nannyml_method = "NannyML (KolmogorovSmirnov)" if nannyml_results.get("nannyml_used", False) else "Fallback checks"
    logger.info(f"  Drift detection method: {nannyml_method}")

    # Additive Unit 6-style diagnostics (does not change hard gate behavior).
    supplemental_checks = _run_additional_integrity_checks(batch_df)
    logger.info("  Supplemental checks complete (schema/missingness/duplicates/range/domain/datetime).")
    
    # Determine overall decision
    integrity_pass = hard_pass
    integrity_warn = len(nannyml_results["warnings"]) > 0
    
    decision = {
        "step": "B_integrity_gate",
        "timestamp": datetime.now().isoformat(),
        "hard_rules_passed": hard_pass,
        "hard_failures": hard_failures,
        "nannyml_warnings": nannyml_results["warnings"],
        "drift_columns": nannyml_results["drift_columns"],
        "additional_checks_summary": {
            "schema_missing_cols": supplemental_checks["metrics"].get("schema_missing_cols", 0.0),
            "schema_bad_family_dtypes": supplemental_checks["metrics"].get("schema_bad_family_dtypes", 0.0),
            "duplicate_rows_frac": supplemental_checks["metrics"].get("duplicate_rows_frac", 0.0),
            "missing_frac_max": supplemental_checks["metrics"].get("missing_frac_max", 0.0),
            "range_worst_bad_frac": supplemental_checks["metrics"].get("range_worst_bad_frac", 0.0),
            "domain_worst_bad_frac": supplemental_checks["metrics"].get("domain_worst_bad_frac", 0.0),
            "duration_over_6h_frac": supplemental_checks["metrics"].get("duration_over_6h_frac", 0.0),
        },
        "integrity_passed": integrity_pass,
        "integrity_warn": integrity_warn,
        "action": "proceed" if integrity_pass else "reject_batch",
    }
    
    # Log to MLflow
    log_bool_param_and_tag("integrity_passed", integrity_pass)
    log_bool_param_and_tag("integrity_warn", integrity_warn)

    # Log human-readable integrity/drift tables.
    hard_failures_df = pd.DataFrame(
        [{"failure_id": i + 1, "failure": msg} for i, msg in enumerate(hard_failures)]
    )
    log_table_with_markdown(
        hard_failures_df,
        "step_b/hard_failures.json",
        "step_b/hard_failures.md",
    )

    drift_warnings_df = pd.DataFrame(
        [{"warning_id": i + 1, "warning": msg} for i, msg in enumerate(nannyml_results["warnings"])]
    )
    log_table_with_markdown(
        drift_warnings_df,
        "step_b/drift_warnings.json",
        "step_b/drift_warnings.md",
    )

    drift_columns_df = pd.DataFrame(
        [{"column": col} for col in nannyml_results.get("drift_columns", [])]
    )
    log_table_with_markdown(
        drift_columns_df,
        "step_b/drift_columns.json",
        "step_b/drift_columns.md",
    )

    gate_summary_df = pd.DataFrame(
        [
            {
                "hard_rules_passed": hard_pass,
                "hard_failures_count": len(hard_failures),
                "nannyml_warnings_count": len(nannyml_results["warnings"]),
                "drift_columns_count": len(nannyml_results.get("drift_columns", [])),
                "drift_method": nannyml_method,
                "integrity_passed": integrity_pass,
                "integrity_warn": integrity_warn,
                "action": "proceed" if integrity_pass else "reject_batch",
            }
        ]
    )
    log_table_with_markdown(
        gate_summary_df,
        "step_b/gate_summary.json",
        "step_b/gate_summary.md",
    )

    for table_name, table_df in supplemental_checks["tables"].items():
        log_table_with_markdown(
            table_df,
            f"step_b/additional/{table_name}.json",
            f"step_b/additional/{table_name}.md",
        )

    for metric_name, metric_value in supplemental_checks["metrics"].items():
        if pd.notna(metric_value):
            mlflow.log_metric(f"step_b_extra_{metric_name}", float(metric_value))
    
    # Log decision.json
    log_decision_json(decision, "decision_step_b.json")
    
    logger.info(f"Step B completed: action={decision['action']}")
    
    return {
        "integrity_data": {
            "hard_pass": hard_pass,
            "hard_failures": hard_failures,
            "nannyml_results": nannyml_results,
        },
        "decision": decision,
    }
