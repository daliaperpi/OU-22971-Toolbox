"""Step D: Load champion model from MLflow Model Registry.

If no champion exists, bootstrap by training an initial model on reference data,
registering it, and assigning the champion alias.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor
from mlflow_table_utils import log_decision_json, log_step_header, set_bool_tag

logger = logging.getLogger("mlops_step_d")


def _ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)
        logger.info(f"Created registered model: {model_name}")


def _get_champion_model_uri(client: MlflowClient, model_name: str) -> Optional[str]:
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        return f"models:/{model_name}@champion"
    except Exception:
        return None


def _bootstrap_champion(
    client: MlflowClient,
    model_name: str,
    reference_features,
    reference_labels,
) -> Dict:
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=10,
        learning_rate=0.08,
        random_state=42,
    )
    model.fit(reference_features, reference_labels)

    model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    model_uri = model_info.model_uri

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_registered_model_alias(model_name, "champion", mv.version)

    client.set_model_version_tag(model_name, mv.version, "role", "champion")
    client.set_model_version_tag(model_name, mv.version, "promotion_reason", "bootstrap")
    client.set_model_version_tag(model_name, mv.version, "validation_status", "approved")
    client.set_model_version_tag(model_name, mv.version, "promoted_at", datetime.now().isoformat())

    return {
        "bootstrap": True,
        "champion_model_uri": f"models:/{model_name}@champion",
        "champion_version": mv.version,
    }


def step_d_load_champion(data: Dict, model_name: str = "green_taxi_tip_model") -> Dict:
    """Step D: Load champion model or bootstrap if missing.

    Expects Step C output with engineered features and raw Step A data for labels.
    """
    log_step_header(logger, "STEP D: Load champion model")

    client = MlflowClient()
    _ensure_registered_model(client, model_name)

    champion_uri = _get_champion_model_uri(client, model_name)

    reference_features = data["reference_features"]
    reference_df = data["reference_df"]
    if "tip_amount" not in reference_df.columns:
        raise ValueError("Reference labels missing: tip_amount")

    reference_labels = reference_df["tip_amount"].fillna(0.0).to_numpy(dtype=float)

    if champion_uri is None:
        logger.info("No champion alias found. Bootstrapping initial champion model...")
        bootstrap_info = _bootstrap_champion(
            client=client,
            model_name=model_name,
            reference_features=reference_features,
            reference_labels=reference_labels,
        )
        champion_uri = bootstrap_info["champion_model_uri"]
        champion_version = bootstrap_info["champion_version"]
        bootstrap = True
    else:
        mv = client.get_model_version_by_alias(model_name, "champion")
        champion_version = mv.version
        bootstrap = False
        logger.info(f"Loaded champion alias -> version {champion_version}")

    champion_model = mlflow.pyfunc.load_model(champion_uri)

    mlflow.set_tag("model_name", model_name)
    mlflow.set_tag("champion_version", str(champion_version))
    set_bool_tag("champion_bootstrap", bootstrap)

    decision = {
        "step": "D_load_champion",
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "champion_version": str(champion_version),
        "bootstrap": bootstrap,
        "action": "load_champion",
    }
    log_decision_json(decision, "decision_step_d.json")

    return {
        "champion_model": champion_model,
        "champion_model_uri": champion_uri,
        "champion_version": str(champion_version),
        "model_name": model_name,
        "bootstrap": bootstrap,
    }
