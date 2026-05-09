from __future__ import annotations

from datetime import datetime
from time import perf_counter
from pathlib import Path
import os
import tempfile
import json

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Metaflow imports plugins eagerly; patch POSIX-only symbol for Windows import path.
if not hasattr(os, "O_NONBLOCK"):
    os.O_NONBLOCK = 0

import mlflow
from metaflow import FlowSpec, Parameter, current, step

from mlops_project_lib import setup_logger
from nannyml_visualization import create_monitoring_report
from step_a_load_data import find_tlc_data_dir, step_a_load_data
from step_b_integrity_gate import step_b_integrity_gate
from step_c_feature_engineering import step_c_feature_engineering
from step_d_load_champion import step_d_load_champion
from step_e_model_gate import step_e_model_gate
from step_f_retrain_candidate import step_f_retrain_candidate
from step_g_promotion_gate import step_g_promotion_gate


class MLOpsCapstoneFlow(FlowSpec):
    tracking_uri = Parameter("tracking-uri", default="http://localhost:5000")
    experiment = Parameter("experiment", default="green_taxi_monitoring")
    run_name = Parameter("run-name", default=None)
    ref_month = Parameter("ref-month", default="01")
    batch_month = Parameter("batch-month", default="04")
    model_name = Parameter("model-name", default="green_taxi_tip_model")
    rmse_degradation_pct = Parameter("rmse-degradation-pct", default=5.0, type=float)
    min_improvement = Parameter("min-improvement", default=0.01, type=float)
    max_reference_regression_pct = Parameter("max-reference-regression-pct", default=5.0, type=float)
    fail_in_step_f = Parameter("fail-in-step-f", default=False, type=bool)
    prediction_preview_rows = Parameter("prediction-preview-rows", default=50, type=int)
    skip_monitoring_report = Parameter("skip-monitoring-report", default=False, type=bool)

    def _record_step_event(self, step_name: str, started_at: str, duration_sec: float, status: str = "ok", note: str = "") -> None:
        self.step_events.append(
            {
                "step": step_name,
                "status": status,
                "started_at": started_at,
                "ended_at": datetime.now().isoformat(),
                "duration_sec": round(duration_sec, 3),
                "note": note,
            }
        )

    def _log_flow_graph_and_timeline(self) -> None:
        timeline_df = pd.DataFrame(self.step_events)
        if not timeline_df.empty:
            mlflow.log_table(timeline_df, artifact_file="flow/step_timeline.json")

        flow_graph = """flowchart TD
        A[Step A Load Data] --> B[Step B Integrity Gate]
        B -->|pass| M[Monitoring Report]
        B -->|fail| END[End]
        M --> C[Step C Feature Engineering]
        C --> D[Step D Load Champion]
        D --> E[Step E Model Gate]
        E --> F[Step F Conditional Retrain]
        F --> G[Step G Promotion Gate]
        G --> H[Step H Inference Snapshot]
        H --> END
        """
        mlflow.log_text(flow_graph, "flow/flow_graph.mmd")

    def _log_learning_chart(self) -> None:
        perf = self.pipeline_data.get("performance", {})
        candidate_eval = self.pipeline_data.get("candidate_eval")

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

        with tempfile.TemporaryDirectory(prefix="metaflow_flow_") as td:
            out = Path(td) / "learning_journey.png"
            fig.savefig(out, dpi=180, bbox_inches="tight")
            mlflow.log_artifact(str(out), artifact_path="flow_visualizations")
        plt.close(fig)

    def _log_prediction_snapshot(self) -> None:
        model_name = self.pipeline_data["model_name"]
        model_uri = f"models:/{model_name}@champion"
        model = mlflow.pyfunc.load_model(model_uri)

        batch_features = self.pipeline_data["batch_features"]
        batch_df = self.pipeline_data["batch_df"]
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

        preview = preview.head(max(1, int(self.prediction_preview_rows))).reset_index(drop=True)
        mlflow.log_table(preview, artifact_file="inference/prediction_preview.json")

        with tempfile.TemporaryDirectory(prefix="metaflow_preds_") as td:
            full_path = Path(td) / "predictions.parquet"
            try:
                pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_parquet(full_path, index=False)
            except Exception:
                full_path = Path(td) / "predictions.csv"
                pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(full_path, index=False)
            mlflow.log_artifact(str(full_path), artifact_path="inference")

    @step
    def start(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        self.logger = setup_logger()
        self.logger.info("=" * 70)
        self.logger.info("Metaflow MLOps Capstone Flow")
        self.logger.info("=" * 70)

        self.step_events = []
        self.flow_summary = {}

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)

        with mlflow.start_run(run_name=self.run_name) as run:
            self.mlflow_run_id = run.info.run_id
            mlflow.set_tag("orchestrator", "metaflow")
            mlflow.set_tag("metaflow_run_id", current.run_id)
            mlflow.set_tag("metaflow_flow", current.flow_name)
            mlflow.set_tag("metaflow_started_at", datetime.now().isoformat())
            mlflow.set_tag("flow_file", "metaflow_capstone_flow.py")
            mlflow.set_tag("flow_prediction_preview_rows", str(self.prediction_preview_rows))
            mlflow.set_tag("flow_skip_monitoring_report", str(bool(self.skip_monitoring_report)).lower())

        self.data_dir = find_tlc_data_dir()
        self.logger.info(f"Using TLC_data directory: {self.data_dir}")
        self._record_step_event("start", started_at, perf_counter() - t0)
        self.next(self.load_data)

    @step
    def load_data(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.data = step_a_load_data(self.data_dir, self.ref_month, self.batch_month)

        self._record_step_event("A_load_data", started_at, perf_counter() - t0)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.integrity_result = step_b_integrity_gate(self.data)

        self.integrity_passed = self.integrity_result["decision"]["integrity_passed"]
        self.integrity_warn = self.integrity_result["decision"]["integrity_warn"]

        if not self.integrity_passed:
            self.logger.error("Batch rejected by hard rules. Ending flow.")
            self.integrity_route = "end"
            step_note = "Batch rejected by hard integrity checks"
        else:
            self.integrity_route = "continue"
            step_note = "Batch passed integrity checks"

        self._record_step_event("B_integrity_gate", started_at, perf_counter() - t0, note=step_note)

        self.next({"continue": self.monitoring_report, "end": self.end}, condition="integrity_route")

    @step
    def monitoring_report(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            if self.skip_monitoring_report:
                mlflow.set_tag("monitoring_report_status", "skipped")
                note = "Monitoring report skipped by flag"
            else:
                try:
                    report_dir = Path("monitoring_reports")
                    report_files = create_monitoring_report(
                        experiment_name=self.experiment,
                        tracking_uri=self.tracking_uri,
                        output_dir=report_dir,
                    )

                    summary_file = report_dir / "REPORT_SUMMARY.md"
                    if summary_file.exists():
                        mlflow.log_artifact(str(summary_file), artifact_path="monitoring")

                    preferred = ["warnings_over_time", "severity_scorecard", "integrity_summary"]
                    selected_files = []
                    if isinstance(report_files, dict):
                        for key in preferred:
                            p = report_files.get(key)
                            if p is not None:
                                selected_files.append(Path(p))
                    if not selected_files:
                        selected_files = sorted(report_dir.glob("*.png"))[:3]

                    max_bytes = 2 * 1024 * 1024
                    logged_count = 0
                    for png_file in selected_files:
                        if png_file.exists() and png_file.stat().st_size <= max_bytes:
                            mlflow.log_artifact(str(png_file), artifact_path="monitoring_visualizations")
                            logged_count += 1

                    mlflow.log_metric("monitoring_visualizations_logged", float(logged_count))
                    mlflow.set_tag("monitoring_report_status", "ok")
                    note = f"Monitoring report logged ({logged_count} images)"
                except Exception as exc:
                    mlflow.set_tag("monitoring_report_status", "failed")
                    note = f"Monitoring report failed: {exc}"

        self._record_step_event("monitoring_report", started_at, perf_counter() - t0, note=note)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.features_result = step_c_feature_engineering(self.data)

        self.pipeline_data = {
            **self.data,
            **self.features_result,
            "integrity_passed": self.integrity_passed,
            "integrity_warn": self.integrity_warn,
            "ref_month": self.ref_month,
            "batch_month": self.batch_month,
        }

        self._record_step_event("C_feature_engineering", started_at, perf_counter() - t0)
        self.next(self.load_champion)

    @step
    def load_champion(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            champion_result = step_d_load_champion(
                data=self.pipeline_data,
                model_name=self.model_name,
            )

        self.pipeline_data.update(champion_result)
        self._record_step_event("D_load_champion", started_at, perf_counter() - t0)
        self.next(self.model_gate)

    @step
    def model_gate(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            perf_result = step_e_model_gate(
                data=self.pipeline_data,
                rmse_degradation_pct_threshold=self.rmse_degradation_pct,
            )

        self.pipeline_data.update(perf_result)
        self._record_step_event("E_model_gate", started_at, perf_counter() - t0)
        self.next(self.retrain_candidate)

    @step
    def retrain_candidate(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)

        if self.fail_in_step_f:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                run_tags = mlflow.get_run(self.mlflow_run_id).data.tags
                already_failed = run_tags.get("intentional_step_f_failure_triggered", "false") == "true"
                if not already_failed:
                    mlflow.set_tag("intentional_step_f_failure_triggered", "true")
                    mlflow.set_tag("intentional_step_f_failure_at", datetime.now().isoformat())
                    raise RuntimeError(
                        "Intentional failure in Step F for Metaflow resume demo. "
                        "Re-run with 'python metaflow_capstone_flow.py resume'."
                    )

        with mlflow.start_run(run_id=self.mlflow_run_id):
            retrain_result = step_f_retrain_candidate(
                data=self.pipeline_data,
                fail_intentionally=False,
                fail_once=True,
            )

        self.pipeline_data.update(retrain_result)
        self._record_step_event("F_retrain_candidate", started_at, perf_counter() - t0)
        self.next(self.promotion_gate)

    @step
    def promotion_gate(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.promotion_result = step_g_promotion_gate(
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
                min_improvement=self.min_improvement,
                max_reference_regression_pct=self.max_reference_regression_pct,
            )

            mlflow.set_tag("metaflow_finished_at", datetime.now().isoformat())

        self._record_step_event("G_promotion_gate", started_at, perf_counter() - t0)
        self.next(self.inference_snapshot)

    @step
    def inference_snapshot(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self._log_prediction_snapshot()

        self._record_step_event("H_inference_snapshot", started_at, perf_counter() - t0)
        self.next(self.end)

    @step
    def end(self):
        t0 = perf_counter()
        started_at = datetime.now().isoformat()

        if not hasattr(self, "promotion_result"):
            self.promotion_result = {
                "promotion_recommended": False,
                "promotion_executed": False,
                "reason": "Flow ended before promotion step",
            }

        self.flow_summary = {
            "promotion_recommended": bool(self.promotion_result.get("promotion_recommended", False)),
            "promotion_executed": bool(self.promotion_result.get("promotion_executed", False)),
            "retrain_executed": bool(self.pipeline_data.get("retrain_executed", False)) if hasattr(self, "pipeline_data") else False,
            "integrity_warn": bool(self.integrity_warn) if hasattr(self, "integrity_warn") else False,
            "finished_at": datetime.now().isoformat(),
        }

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_text(json.dumps(self.flow_summary, indent=2), "flow/run_summary.json")
            self._log_learning_chart()
            self._record_step_event("end", started_at, perf_counter() - t0)
            self._log_flow_graph_and_timeline()

        if hasattr(self, "promotion_result"):
            self.logger.info(f"Promotion recommended: {self.promotion_result['promotion_recommended']}")
            self.logger.info(f"Promotion executed: {self.promotion_result['promotion_executed']}")
        self.logger.info("Metaflow flow finished.")


if __name__ == "__main__":
    MLOpsCapstoneFlow()
