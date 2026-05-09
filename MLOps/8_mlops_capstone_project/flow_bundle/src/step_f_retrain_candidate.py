"""Step F: Conditional retraining of a candidate model."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from mlflow_table_utils import log_decision_json, log_step_header, log_table_with_markdown

logger = logging.getLogger("mlops_step_f")


def step_f_retrain_candidate(
    data: Dict,
    fail_intentionally: bool = False,
    fail_once: bool = True,
) -> Dict:
    """Conditionally train and evaluate a candidate model.

    If retrain_needed is false, this step is a no-op with a logged decision.
    """
    log_step_header(logger, "STEP F: Retrain candidate (conditional)")

    retrain_needed = data["retrain_needed"]

    if fail_intentionally:
        active = mlflow.active_run()
        already_failed = False
        if active is not None and fail_once:
            tags = mlflow.get_run(active.info.run_id).data.tags
            already_failed = tags.get("intentional_step_f_failure_triggered", "false") == "true"

        if not already_failed:
            if active is not None:
                mlflow.set_tag("intentional_step_f_failure_triggered", "true")
                mlflow.set_tag("intentional_step_f_failure_at", datetime.now().isoformat())
            raise RuntimeError(
                "Intentional failure in Step F for robustness/resume demo."
            )

    if not retrain_needed:
        decision = {
            "step": "F_retrain_candidate",
            "timestamp": datetime.now().isoformat(),
            "retrain_executed": False,
            "action": "skip_retrain",
            "reason": data.get("retrain_reason", "Not requested by Step E"),
        }
        log_decision_json(decision, "decision_step_f.json")
        status_df = pd.DataFrame(
            [
                {
                    "retrain_needed": False,
                    "retrain_executed": False,
                    "action": "skip_retrain",
                    "reason": decision["reason"],
                }
            ]
        )
        log_table_with_markdown(status_df, "step_f/retrain_status.json", "step_f/retrain_status.md")
        logger.info("Retrain not needed. Skipping candidate training.")
        return {
            "retrain_executed": False,
            "candidate_model": None,
            "candidate_model_uri": None,
            "candidate_eval": None,
            "decision": decision,
        }

    ref_x = data["reference_features"]
    batch_x = data["batch_features"]
    ref_y = data["reference_df"]["tip_amount"].fillna(0.0).to_numpy(dtype=float)
    batch_y = data["batch_df"]["tip_amount"].fillna(0.0).to_numpy(dtype=float)

    train_x = np.concatenate([ref_x.to_numpy(), batch_x.to_numpy()], axis=0)
    train_y = np.concatenate([ref_y, batch_y], axis=0)

    candidate = HistGradientBoostingRegressor(
        max_iter=250,
        max_depth=12,
        learning_rate=0.07,
        random_state=42,
    )
    candidate.fit(train_x, train_y)

    y_pred = candidate.predict(batch_x)
    rmse_candidate = float(root_mean_squared_error(batch_y, y_pred))
    mae_candidate = float(mean_absolute_error(batch_y, y_pred))

    # Stability metric on reference slice (P3 guard input)
    ref_pred_candidate = candidate.predict(ref_x)
    rmse_candidate_reference = float(root_mean_squared_error(ref_y, ref_pred_candidate))

    champion_model = data["champion_model"]
    ref_pred_champion = champion_model.predict(ref_x)
    rmse_champion_reference = float(root_mean_squared_error(ref_y, ref_pred_champion))

    model_info = mlflow.sklearn.log_model(sk_model=candidate, artifact_path="candidate_model")

    # Register every candidate version, even if not promoted yet.
    model_name = data["model_name"]
    registered = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
    client = MlflowClient()
    client.set_model_version_tag(model_name, registered.version, "role", "candidate")
    client.set_model_version_tag(model_name, registered.version, "trained_on_batches", "reference+current_batch")
    client.set_model_version_tag(model_name, registered.version, "eval_batch_id", str(data.get("batch_month", "unknown")))
    client.set_model_version_tag(model_name, registered.version, "validation_status", "pending")
    client.set_model_version_tag(model_name, registered.version, "decision_reason", "candidate_trained_pending_promotion")

    mlflow.log_metric("rmse_candidate", rmse_candidate)
    mlflow.log_metric("mae_candidate", mae_candidate)
    mlflow.log_metric("rmse_candidate_reference", rmse_candidate_reference)
    mlflow.log_metric("rmse_champion_reference", rmse_champion_reference)
    mlflow.log_metric("candidate_train_rows", float(len(train_y)))

    candidate_metrics_df = pd.DataFrame(
        [
            {
                "candidate_version": str(registered.version),
                "rmse_candidate_batch": rmse_candidate,
                "mae_candidate_batch": mae_candidate,
                "rmse_candidate_reference": rmse_candidate_reference,
                "rmse_champion_reference": rmse_champion_reference,
                "reference_regression_pct": (
                    0.0
                    if rmse_champion_reference == 0
                    else ((rmse_candidate_reference - rmse_champion_reference) / rmse_champion_reference) * 100.0
                ),
            }
        ]
    )
    registration_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "candidate_version": str(registered.version),
                "role": "candidate",
                "validation_status": "pending",
                "eval_batch_id": str(data.get("batch_month", "unknown")),
            }
        ]
    )
    log_table_with_markdown(candidate_metrics_df, "step_f/candidate_metrics.json", "step_f/candidate_metrics.md")
    log_table_with_markdown(registration_df, "step_f/registration_summary.json", "step_f/registration_summary.md")

    decision = {
        "step": "F_retrain_candidate",
        "timestamp": datetime.now().isoformat(),
        "retrain_executed": True,
        "candidate_model_uri": model_info.model_uri,
        "candidate_version": str(registered.version),
        "rmse_candidate": rmse_candidate,
        "mae_candidate": mae_candidate,
        "rmse_candidate_reference": rmse_candidate_reference,
        "rmse_champion_reference": rmse_champion_reference,
        "action": "candidate_trained",
    }
    log_decision_json(decision, "decision_step_f.json")

    logger.info(f"Candidate RMSE: {rmse_candidate:.4f}")
    logger.info(f"Candidate MAE: {mae_candidate:.4f}")

    return {
        "retrain_executed": True,
        "candidate_model": candidate,
        "candidate_model_uri": model_info.model_uri,
        "candidate_version": str(registered.version),
        "candidate_eval": {
            "rmse_candidate": rmse_candidate,
            "mae_candidate": mae_candidate,
            "rmse_candidate_reference": rmse_candidate_reference,
            "rmse_champion_reference": rmse_champion_reference,
        },
        "decision": decision,
    }
