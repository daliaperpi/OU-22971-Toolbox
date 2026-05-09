"""Step E: Evaluate champion model and decide whether retraining is needed."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from mlflow_table_utils import log_decision_json, log_step_header, log_table_with_markdown, set_bool_tag

logger = logging.getLogger("mlops_step_e")


def step_e_model_gate(
    data: Dict,
    rmse_degradation_pct_threshold: float = 5.0,
) -> Dict:
    """Evaluate champion on batch and decide retrain_needed.

    Baseline predictor is the mean reference tip_amount.
    """
    log_step_header(logger, "STEP E: Model gate (performance)")

    champion_model = data["champion_model"]
    batch_features = data["batch_features"]
    batch_df = data["batch_df"]
    reference_df = data["reference_df"]

    if "tip_amount" not in batch_df.columns:
        raise ValueError("Batch labels missing: tip_amount")

    y_true = batch_df["tip_amount"].fillna(0.0).to_numpy(dtype=float)
    y_pred_champion = champion_model.predict(batch_features)

    rmse_champion = float(root_mean_squared_error(y_true, y_pred_champion))
    mae_champion = float(mean_absolute_error(y_true, y_pred_champion))

    baseline_value = float(reference_df["tip_amount"].fillna(0.0).mean())
    y_pred_baseline = np.full(shape=len(y_true), fill_value=baseline_value, dtype=float)

    rmse_baseline = float(root_mean_squared_error(y_true, y_pred_baseline))
    mae_baseline = float(mean_absolute_error(y_true, y_pred_baseline))

    if rmse_baseline == 0:
        rmse_increase_pct = 0.0
    else:
        rmse_increase_pct = float(((rmse_champion - rmse_baseline) / rmse_baseline) * 100.0)

    retrain_needed = rmse_increase_pct > rmse_degradation_pct_threshold
    retrain_reason = (
        f"Champion RMSE is {rmse_increase_pct:.2f}% worse than baseline"
        if retrain_needed
        else f"Champion RMSE within threshold ({rmse_increase_pct:.2f}% vs {rmse_degradation_pct_threshold:.2f}%)"
    )

    mlflow.log_metric("rmse_champion", rmse_champion)
    mlflow.log_metric("mae_champion", mae_champion)
    mlflow.log_metric("rmse_baseline", rmse_baseline)
    mlflow.log_metric("mae_baseline", mae_baseline)
    mlflow.log_metric("rmse_increase_pct", rmse_increase_pct)
    mlflow.log_metric("rmse_degradation_pct_threshold", rmse_degradation_pct_threshold)

    set_bool_tag("retrain_recommended", retrain_needed)

    perf_df = pd.DataFrame(
        [
            {"model": "champion", "rmse": rmse_champion, "mae": mae_champion},
            {"model": "baseline", "rmse": rmse_baseline, "mae": mae_baseline},
        ]
    )
    gate_df = pd.DataFrame(
        [
            {
                "rmse_increase_pct": rmse_increase_pct,
                "threshold_pct": rmse_degradation_pct_threshold,
                "retrain_needed": retrain_needed,
                "reason": retrain_reason,
            }
        ]
    )
    log_table_with_markdown(perf_df, "step_e/performance_table.json", "step_e/performance_table.md")
    log_table_with_markdown(gate_df, "step_e/gate_summary.json", "step_e/gate_summary.md")

    decision = {
        "step": "E_model_gate",
        "timestamp": datetime.now().isoformat(),
        "rmse_champion": rmse_champion,
        "mae_champion": mae_champion,
        "rmse_baseline": rmse_baseline,
        "mae_baseline": mae_baseline,
        "rmse_increase_pct": rmse_increase_pct,
        "rmse_degradation_pct_threshold": rmse_degradation_pct_threshold,
        "retrain_needed": retrain_needed,
        "retrain_reason": retrain_reason,
        "action": "retrain" if retrain_needed else "keep_champion",
    }
    log_decision_json(decision, "decision_step_e.json")

    logger.info(f"Champion RMSE: {rmse_champion:.4f}")
    logger.info(f"Baseline RMSE: {rmse_baseline:.4f}")
    logger.info(f"RMSE increase pct: {rmse_increase_pct:.2f}%")
    logger.info(f"Retrain needed: {retrain_needed}")

    return {
        "performance": {
            "rmse_champion": rmse_champion,
            "mae_champion": mae_champion,
            "rmse_baseline": rmse_baseline,
            "mae_baseline": mae_baseline,
            "rmse_increase_pct": rmse_increase_pct,
        },
        "retrain_needed": retrain_needed,
        "retrain_reason": retrain_reason,
        "decision": decision,
    }
