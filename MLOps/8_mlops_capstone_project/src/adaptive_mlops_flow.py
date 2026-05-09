"""Adaptive MLOps flow without Metaflow.

Runs Steps A-G with MLflow-first observability and safe async work for tasks
that don't need to block the critical path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import tempfile
from urllib.parse import urlparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

# Silence Git detection warnings from MLflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops_project_lib import setup_logger
from nannyml_visualization import create_monitoring_report
from step_a_load_data import find_tlc_data_dir, step_a_load_data
from step_b_integrity_gate import step_b_integrity_gate
from step_c_feature_engineering import step_c_feature_engineering
from step_d_load_champion import step_d_load_champion
from step_e_model_gate import step_e_model_gate
from step_f_retrain_candidate import step_f_retrain_candidate
from step_g_promotion_gate import step_g_promotion_gate

logger = logging.getLogger("mlops")

STEP_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]
STEP_LABELS = {
    "A": "A_load_data",
    "B": "B_integrity_gate",
    "C": "C_feature_engineering",
    "D": "D_load_champion",
    "E": "E_model_gate",
    "F": "F_retrain_candidate",
    "G": "G_promotion_gate",
    "H": "H_inference_snapshot",
}
PREV_STEP_FOR_START = {
    "B": "A",
    "C": "B",
    "D": "C",
    "E": "D",
    "F": "E",
    "G": "F",
    "H": "G",
}


@dataclass
class StepEvent:
    step: str
    status: str
    started_at: str
    ended_at: str
    duration_sec: float
    note: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive MLOps flow: A->G with MLflow observability")
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="green_taxi_monitoring")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ref-month", default="01")
    p.add_argument("--batch-month", default="04")
    p.add_argument("--model-name", default="green_taxi_tip_model")
    p.add_argument("--rmse-degradation-pct", type=float, default=5.0)
    p.add_argument("--min-improvement", type=float, default=0.01)
    p.add_argument("--max-reference-regression-pct", type=float, default=5.0)
    p.add_argument("--fail-in-step-f", action="store_true")
    p.add_argument("--prediction-preview-rows", type=int, default=50)
    p.add_argument("--skip-monitoring-report", action="store_true")
    p.add_argument("--start-step", choices=STEP_ORDER, default="A", help="Start execution from a specific step (A-H)")
    p.add_argument("--state-dir", default=".flow_state", help="Directory for local step checkpoints used for resume")
    p.add_argument("--resume-state-id", default=None, help="Checkpoint state id (usually a prior run id) to resume from")
    p.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True, help="Save step checkpoints for future resume")
    return p.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monitoring_job(experiment_name: str, tracking_uri: str, output_dir: Path) -> Dict[str, Any]:
    """Run monitoring report generation in background and return file metadata."""
    try:
        report_files = create_monitoring_report(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            output_dir=output_dir,
        )
        return {"ok": True, "report_dir": str(output_dir), "report_files": report_files}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "report_dir": str(output_dir), "report_files": {}}


def _log_monitoring_artifacts(result: Dict[str, Any]) -> None:
    if not result.get("ok"):
        logger.warning(f"Monitoring report generation failed: {result.get('error', 'unknown error')}")
        mlflow.set_tag("monitoring_report_status", "failed")
        return

    report_dir = Path(result["report_dir"])
    report_files = result.get("report_files") or {}

    summary_file = report_dir / "REPORT_SUMMARY.md"
    if summary_file.exists():
        mlflow.log_artifact(str(summary_file), artifact_path="monitoring")

    preferred = ["warnings_over_time", "severity_scorecard", "integrity_summary"]
    selected_files = []
    for key in preferred:
        p = report_files.get(key)
        if p is not None:
            selected_files.append(Path(p))
    if not selected_files:
        selected_files = sorted(report_dir.glob("*.png"))[:3]

    max_bytes = 2 * 1024 * 1024
    for png_file in selected_files:
        if png_file.exists() and png_file.stat().st_size <= max_bytes:
            mlflow.log_artifact(str(png_file), artifact_path="monitoring_visualizations")

    mlflow.set_tag("monitoring_report_status", "ok")
    mlflow.log_metric("monitoring_visualizations_logged", float(len(selected_files)))


def _log_flow_graph_and_timeline(step_events: list[StepEvent]) -> None:
    timeline_df = pd.DataFrame([asdict(ev) for ev in step_events])
    mlflow.log_table(timeline_df, artifact_file="flow/step_timeline.json")

    flow_graph = """flowchart TD
    A[Step A Load Data] --> B[Step B Integrity Gate]
    B -->|pass| C[Step C Feature Engineering]
    B -->|fail| END[End]
    C --> D[Step D Load Champion]
    D --> E[Step E Model Gate]
    E --> F[Step F Conditional Retrain]
    F --> G[Step G Promotion Gate]
    G --> H[Step H Prediction Snapshot]
    H --> END
    """
    mlflow.log_text(flow_graph, "flow/flow_graph.mmd")


def _log_learning_chart(perf: Dict[str, float], candidate_eval: Optional[Dict[str, float]]) -> None:
    labels = ["baseline_rmse", "champion_rmse"]
    values = [float(perf.get("rmse_baseline", float("nan"))), float(perf.get("rmse_champion", float("nan")))]
    if candidate_eval and candidate_eval.get("rmse_candidate") is not None:
        labels.append("candidate_rmse")
        values.append(float(candidate_eval["rmse_candidate"]))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=["#6b7280", "#0ea5e9", "#22c55e"][: len(labels)])
    ax.set_ylabel("RMSE")
    ax.set_title("Model Learning Journey (This Run)")
    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    with tempfile.TemporaryDirectory(prefix="mlops_flow_") as td:
        out = Path(td) / "learning_journey.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        mlflow.log_artifact(str(out), artifact_path="flow_visualizations")
    plt.close(fig)


def _log_prediction_snapshot(
    model_name: str,
    batch_features: pd.DataFrame,
    batch_df: pd.DataFrame,
    preview_rows: int,
) -> None:
    model_uri = f"models:/{model_name}@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    y_pred = model.predict(batch_features)
    y_true = batch_df["tip_amount"].fillna(0.0).to_numpy(dtype=float)

    preview = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    preview["abs_error"] = (preview["y_true"] - preview["y_pred"]).abs()

    join_cols = [c for c in ["lpep_pickup_datetime", "PULocationID", "DOLocationID", "trip_distance"] if c in batch_df.columns]
    if join_cols:
        preview = pd.concat([batch_df[join_cols].reset_index(drop=True), preview], axis=1)

    preview = preview.head(max(1, preview_rows)).reset_index(drop=True)
    mlflow.log_table(preview, artifact_file="inference/prediction_preview.json")

    with tempfile.TemporaryDirectory(prefix="mlops_preds_") as td:
        full_path = Path(td) / "predictions.parquet"
        try:
            pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_parquet(full_path, index=False)
        except Exception:
            full_path = Path(td) / "predictions.csv"
            pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(full_path, index=False)
        mlflow.log_artifact(str(full_path), artifact_path="inference")


class AdaptiveMLOpsFlowRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.events: list[StepEvent] = []
        self.pipeline_data: Dict[str, Any] = {}
        self.state_root = Path(args.state_dir).resolve()

    def _run_step(self, name: str, fn, note: str = ""):
        started = _utc_now_iso()
        t0 = perf_counter()
        try:
            result = fn()
            status = "ok"
            return result
        except Exception:
            status = "failed"
            raise
        finally:
            ended = _utc_now_iso()
            self.events.append(
                StepEvent(
                    step=name,
                    status=status,
                    started_at=started,
                    ended_at=ended,
                    duration_sec=round(perf_counter() - t0, 3),
                    note=note,
                )
            )

    def _state_dir_for(self, state_id: str) -> Path:
        return self.state_root / state_id

    def _checkpoint_path(self, state_id: str, step_id: str) -> Path:
        return self._state_dir_for(state_id) / f"step_{step_id}.pkl"

    def _save_checkpoint(self, state_id: str, step_id: str, payload: Dict[str, Any]) -> None:
        if not self.args.save_checkpoints:
            return
        target_dir = self._state_dir_for(state_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(self._checkpoint_path(state_id, step_id), "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_checkpoint(self, state_id: str, step_id: str) -> Dict[str, Any]:
        checkpoint_file = self._checkpoint_path(state_id, step_id)
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for step {step_id}: {checkpoint_file}. "
                "Run once from an earlier step or provide a valid --resume-state-id."
            )
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)

    def _resolve_resume_state_id(self) -> str:
        if self.args.resume_state_id:
            return self.args.resume_state_id
        if not self.state_root.exists():
            raise FileNotFoundError(
                f"State directory does not exist: {self.state_root}. "
                "Provide --resume-state-id or run once from step A to create checkpoints."
            )
        candidates = [p for p in self.state_root.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint states found in {self.state_root}. "
                "Provide --resume-state-id or run once from step A."
            )
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return latest.name

    def _mark_skipped_steps(self, start_step: str, source_state_id: str) -> None:
        now = _utc_now_iso()
        for step_id in STEP_ORDER:
            if step_id == start_step:
                break
            self.events.append(
                StepEvent(
                    step=STEP_LABELS[step_id],
                    status="skipped",
                    started_at=now,
                    ended_at=now,
                    duration_sec=0.0,
                    note=f"Loaded from checkpoint state {source_state_id}",
                )
            )

    def run(self) -> str:
        tracking_uri = self.args.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        try:
            mlflow.set_experiment(self.args.experiment)
            client = MlflowClient()
            if client.get_experiment_by_name(self.args.experiment) is None:
                mlflow.create_experiment(self.args.experiment)
        except Exception as exc:
            parsed = urlparse(tracking_uri)
            if parsed.scheme in {"http", "https"}:
                fallback_dir = Path(__file__).resolve().parent / "mlruns_local"
                fallback_uri = fallback_dir.as_uri()
                logger.warning(
                    f"Could not connect to MLflow server at {tracking_uri}. "
                    f"Falling back to local store: {fallback_uri}. Error: {exc}"
                )
                mlflow.set_tracking_uri(fallback_uri)
                mlflow.set_experiment(self.args.experiment)
                client = MlflowClient()
                if client.get_experiment_by_name(self.args.experiment) is None:
                    mlflow.create_experiment(self.args.experiment)
                tracking_uri = fallback_uri
            else:
                raise

        data_dir = find_tlc_data_dir()
        logger.info(f"Using TLC_data directory: {data_dir}")

        source_state_id: Optional[str] = None
        if self.args.start_step != "A":
            source_state_id = self._resolve_resume_state_id()
            logger.info(f"Resuming from step {self.args.start_step} using checkpoint state: {source_state_id}")

        monitor_future: Optional[Future] = None
        executor: Optional[ThreadPoolExecutor] = None

        with mlflow.start_run(run_name=self.args.run_name) as run:
            run_state_id = run.info.run_id
            mlflow.set_tag("orchestrator", "adaptive_script_flow")
            mlflow.set_tag("flow_file", "adaptive_mlops_flow.py")
            mlflow.set_tag("flow_started_at", _utc_now_iso())
            mlflow.set_tag("tracking_uri", tracking_uri)
            mlflow.set_tag("flow_start_step", self.args.start_step)
            mlflow.set_tag("flow_state_id", run_state_id)
            mlflow.set_tag("flow_checkpoints_enabled", str(bool(self.args.save_checkpoints)).lower())
            if source_state_id is not None:
                mlflow.set_tag("flow_resume_source_state_id", source_state_id)
                self._mark_skipped_steps(self.args.start_step, source_state_id)

            data: Optional[Dict[str, Any]] = None
            integrity_result: Optional[Dict[str, Any]] = None
            promotion_result: Dict[str, Any] = {}

            if self.args.start_step != "A":
                prev_step = PREV_STEP_FOR_START[self.args.start_step]
                restored = self._load_checkpoint(source_state_id, prev_step)
                data = restored.get("data")
                integrity_result = restored.get("integrity_result")
                restored_pipeline = restored.get("pipeline_data")
                if isinstance(restored_pipeline, dict):
                    self.pipeline_data = restored_pipeline
                restored_promotion = restored.get("promotion_result")
                if isinstance(restored_promotion, dict):
                    promotion_result = restored_promotion

            if self.args.start_step == "A":
                data = self._run_step(
                    STEP_LABELS["A"],
                    lambda: step_a_load_data(data_dir, self.args.ref_month, self.args.batch_month),
                )
                self._save_checkpoint(run_state_id, "A", {"data": data})

            if self.args.start_step in {"A", "B"}:
                integrity_result = self._run_step(STEP_LABELS["B"], lambda: step_b_integrity_gate(data))
                self._save_checkpoint(run_state_id, "B", {"data": data, "integrity_result": integrity_result})
                if not integrity_result["decision"]["integrity_passed"]:
                    mlflow.set_tag("flow_terminated", "integrity_rejected")
                    _log_flow_graph_and_timeline(self.events)
                    return run.info.run_id

            if self.args.start_step in {"A", "B"} and not self.args.skip_monitoring_report:
                executor = ThreadPoolExecutor(max_workers=1)
                monitor_future = executor.submit(
                    _monitoring_job,
                    self.args.experiment,
                    tracking_uri,
                    Path("monitoring_reports"),
                )
                mlflow.set_tag("monitoring_report_mode", "async")

            if self.args.start_step in {"A", "B", "C"}:
                features_result = self._run_step(STEP_LABELS["C"], lambda: step_c_feature_engineering(data))

                self.pipeline_data = {
                    **data,
                    **features_result,
                    "integrity_passed": integrity_result["decision"]["integrity_passed"],
                    "integrity_warn": integrity_result["decision"]["integrity_warn"],
                    "ref_month": self.args.ref_month,
                    "batch_month": self.args.batch_month,
                }
                self._save_checkpoint(run_state_id, "C", {"pipeline_data": self.pipeline_data})

            if self.args.start_step in {"A", "B", "C", "D"}:
                champion_result = self._run_step(
                    STEP_LABELS["D"],
                    lambda: step_d_load_champion(data=self.pipeline_data, model_name=self.args.model_name),
                )
                self.pipeline_data.update(champion_result)
                self._save_checkpoint(run_state_id, "D", {"pipeline_data": self.pipeline_data})

            if self.args.start_step in {"A", "B", "C", "D", "E"}:
                perf_result = self._run_step(
                    STEP_LABELS["E"],
                    lambda: step_e_model_gate(
                        data=self.pipeline_data,
                        rmse_degradation_pct_threshold=self.args.rmse_degradation_pct,
                    ),
                )
                self.pipeline_data.update(perf_result)
                self._save_checkpoint(run_state_id, "E", {"pipeline_data": self.pipeline_data})

            if self.args.start_step in {"A", "B", "C", "D", "E", "F"}:
                retrain_result = self._run_step(
                    STEP_LABELS["F"],
                    lambda: step_f_retrain_candidate(
                        data=self.pipeline_data,
                        fail_intentionally=self.args.fail_in_step_f,
                        fail_once=True,
                    ),
                )
                self.pipeline_data.update(retrain_result)
                self._save_checkpoint(run_state_id, "F", {"pipeline_data": self.pipeline_data})

            if self.args.start_step in {"A", "B", "C", "D", "E", "F", "G"}:
                promotion_result = self._run_step(
                    STEP_LABELS["G"],
                    lambda: step_g_promotion_gate(
                        data={
                            "integrity_passed": self.pipeline_data["integrity_passed"],
                            "retrain_executed": self.pipeline_data["retrain_executed"],
                            "rmse_candidate": (self.pipeline_data["candidate_eval"] or {}).get("rmse_candidate")
                            if self.pipeline_data.get("candidate_eval")
                            else None,
                            "rmse_candidate_reference": (self.pipeline_data["candidate_eval"] or {}).get("rmse_candidate_reference")
                            if self.pipeline_data.get("candidate_eval")
                            else None,
                            "rmse_champion_reference": (self.pipeline_data["candidate_eval"] or {}).get("rmse_champion_reference")
                            if self.pipeline_data.get("candidate_eval")
                            else None,
                            "rmse_champion": self.pipeline_data["performance"]["rmse_champion"],
                            "model_name": self.pipeline_data["model_name"],
                            "candidate_model_uri": self.pipeline_data["candidate_model_uri"],
                            "candidate_version": self.pipeline_data.get("candidate_version"),
                        },
                        min_improvement=self.args.min_improvement,
                        max_reference_regression_pct=self.args.max_reference_regression_pct,
                    ),
                )
                self._save_checkpoint(
                    run_state_id,
                    "G",
                    {"pipeline_data": self.pipeline_data, "promotion_result": promotion_result},
                )

            _ = self._run_step(
                STEP_LABELS["H"],
                lambda: _log_prediction_snapshot(
                    model_name=self.pipeline_data["model_name"],
                    batch_features=self.pipeline_data["batch_features"],
                    batch_df=self.pipeline_data["batch_df"],
                    preview_rows=self.args.prediction_preview_rows,
                ),
                note="Logs prediction preview and full predictions artifact",
            )
            self._save_checkpoint(
                run_state_id,
                "H",
                {"pipeline_data": self.pipeline_data, "promotion_result": promotion_result},
            )

            if monitor_future is not None:
                monitor_result = monitor_future.result(timeout=240)
                _log_monitoring_artifacts(monitor_result)
            if executor is not None:
                executor.shutdown(wait=False)

            _log_learning_chart(self.pipeline_data["performance"], self.pipeline_data.get("candidate_eval"))
            _log_flow_graph_and_timeline(self.events)

            summary = {
                "promotion_recommended": promotion_result.get("promotion_recommended", False),
                "promotion_executed": promotion_result.get("promotion_executed", False),
                "retrain_executed": self.pipeline_data.get("retrain_executed", False),
                "integrity_warn": self.pipeline_data.get("integrity_warn", False),
                "start_step": self.args.start_step,
                "resume_state_id": source_state_id,
                "state_id": run_state_id,
                "finished_at": _utc_now_iso(),
            }
            mlflow.log_text(json.dumps(summary, indent=2), "flow/run_summary.json")
            mlflow.set_tag("flow_finished_at", _utc_now_iso())

            return run.info.run_id


def main() -> None:
    setup_logger()
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Adaptive MLOps Flow (non-Metaflow)")
    logger.info("=" * 70)

    runner = AdaptiveMLOpsFlowRunner(args)
    run_id = runner.run()
    logger.info("=" * 70)
    logger.info(f"Flow completed. Run ID: {run_id}")
    logger.info(f"Tracking URI used: {mlflow.get_tracking_uri()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
