"""Step G: Candidate acceptance and champion promotion gate."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow_table_utils import log_decision_json, log_step_header, log_table_with_markdown, set_bool_tag

logger = logging.getLogger("mlops_step_g")


def step_g_promotion_gate(
    data: Dict,
    min_improvement: float = 0.01,
    max_reference_regression_pct: float = 5.0,
) -> Dict:
    """Promote candidate only if promotion criteria are met."""
    log_step_header(logger, "STEP G: Candidate acceptance (promotion gate)")

    retrain_executed = data["retrain_executed"]
    integrity_passed = data["integrity_passed"]

    decision = {
        "step": "G_promotion_gate",
        "timestamp": datetime.now().isoformat(),
        "min_improvement": min_improvement,
        "max_reference_regression_pct": max_reference_regression_pct,
        "promotion_recommended": False,
        "promotion_executed": False,
        "reason": "",
    }

    def log_gate_table() -> None:
        checks_df = pd.DataFrame(
            [
                {
                    "integrity_passed": integrity_passed,
                    "retrain_executed": retrain_executed,
                    "rmse_candidate": data.get("rmse_candidate"),
                    "rmse_champion": data.get("rmse_champion"),
                    "rmse_candidate_reference": data.get("rmse_candidate_reference"),
                    "rmse_champion_reference": data.get("rmse_champion_reference"),
                    "min_improvement": min_improvement,
                    "max_reference_regression_pct": max_reference_regression_pct,
                    "promotion_recommended": decision.get("promotion_recommended", False),
                    "promotion_executed": decision.get("promotion_executed", False),
                    "reason": decision.get("reason", ""),
                }
            ]
        )
        log_table_with_markdown(checks_df, "step_g/gate_checks.json", "step_g/gate_checks.md")

    if not integrity_passed:
        decision["reason"] = "Integrity gate failed. Promotion blocked."
        log_decision_json(decision, "decision_step_g.json")
        log_gate_table()
        set_bool_tag("promotion_recommended", False)
        return decision

    if not retrain_executed:
        decision["reason"] = "No retraining executed; no candidate to promote."
        log_decision_json(decision, "decision_step_g.json")
        log_gate_table()
        set_bool_tag("promotion_recommended", False)
        return decision

    rmse_candidate = data["rmse_candidate"]
    rmse_champion = data["rmse_champion"]
    rmse_candidate_reference = data.get("rmse_candidate_reference")
    rmse_champion_reference = data.get("rmse_champion_reference")
    model_name = data["model_name"]
    candidate_version = data.get("candidate_version")

    # P1: candidate evaluation validity
    if rmse_candidate is None or rmse_champion is None:
        decision["reason"] = "Missing evaluation metrics; promotion blocked."
        log_decision_json(decision, "decision_step_g.json")
        log_gate_table()
        set_bool_tag("promotion_recommended", False)
        return decision

    meets_improvement = rmse_candidate < rmse_champion * (1.0 - min_improvement)
    if not meets_improvement:
        decision["reason"] = (
            f"Candidate does not beat champion by {min_improvement * 100:.2f}%: "
            f"candidate={rmse_candidate:.4f}, champion={rmse_champion:.4f}"
        )
        log_decision_json(decision, "decision_step_g.json")
        log_gate_table()
        set_bool_tag("promotion_recommended", False)
        return decision

    # P3: stability check on reference slice
    if rmse_candidate_reference is None or rmse_champion_reference is None:
        decision["reason"] = "Missing reference-slice stability metrics; promotion blocked."
        log_decision_json(decision, "decision_step_g.json")
        log_gate_table()
        set_bool_tag("promotion_recommended", False)
        return decision

    max_allowed_reference_rmse = rmse_champion_reference * (1.0 + (max_reference_regression_pct / 100.0))
    if rmse_candidate_reference > max_allowed_reference_rmse:
        decision["reason"] = (
            f"Reference-slice regression too high: candidate_ref={rmse_candidate_reference:.4f}, "
            f"champion_ref={rmse_champion_reference:.4f}, "
            f"allowed_max={max_allowed_reference_rmse:.4f}"
        )
        log_decision_json(decision, "decision_step_g.json")
        log_gate_table()
        set_bool_tag("promotion_recommended", False)
        return decision

    client = MlflowClient()

    if candidate_version is None:
        decision["reason"] = "Candidate version missing; promotion blocked."
        log_decision_json(decision, "decision_step_g.json")
        log_gate_table()
        set_bool_tag("promotion_recommended", False)
        return decision

    try:
        previous = client.get_model_version_by_alias(model_name, "champion")
        client.set_model_version_tag(model_name, previous.version, "role", "previous_champion")
        client.set_model_version_tag(model_name, previous.version, "demoted_at", datetime.now().isoformat())
    except Exception:
        previous = None

    client.set_registered_model_alias(model_name, "champion", candidate_version)
    client.set_model_version_tag(model_name, candidate_version, "role", "champion")
    client.set_model_version_tag(model_name, candidate_version, "validation_status", "approved")
    client.set_model_version_tag(model_name, candidate_version, "promoted_at", datetime.now().isoformat())
    client.set_model_version_tag(model_name, candidate_version, "promotion_reason", "candidate_beats_champion_with_stability")

    decision["promotion_recommended"] = True
    decision["promotion_executed"] = True
    decision["reason"] = "Candidate beat champion by required margin and was promoted"
    decision["new_champion_version"] = str(candidate_version)

    set_bool_tag("promotion_recommended", True)
    mlflow.set_tag("new_champion_version", str(candidate_version))
    log_decision_json(decision, "decision_step_g.json")
    log_gate_table()

    logger.info(f"Promoted candidate to champion version {candidate_version}")
    return decision
