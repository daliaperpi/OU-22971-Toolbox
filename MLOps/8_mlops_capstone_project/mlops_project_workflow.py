"""mlops_project_workflow.py

MLOps Project: Manual monitoring + retraining + champion promotion workflow.

Workflow orchestrator that runs:
  Step A: Load reference and batch data
  Step B: Integrity gate (hard rules + drift detection + monitoring visualizations)
  Step C: Feature engineering
    Step D: Load champion model
    Step E: Model performance evaluation
    Step F: Conditional retraining
    Step G: Promotion logic

Example:
  python mlops_project_workflow.py --ref-month 01 --batch-month 04
  python mlops_project_workflow.py --tracking-uri http://localhost:5000 --experiment green_taxi_monitoring --ref-month 01 --batch-month 04

"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# Silence Git detection warnings from MLflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import mlflow
from mlflow.tracking import MlflowClient

from mlops_project_lib import setup_logger

# Import step modules
from step_a_load_data import find_tlc_data_dir, step_a_load_data
from step_b_integrity_gate import step_b_integrity_gate
from step_c_feature_engineering import step_c_feature_engineering
from step_d_load_champion import step_d_load_champion
from step_e_model_gate import step_e_model_gate
from step_f_retrain_candidate import step_f_retrain_candidate
from step_g_promotion_gate import step_g_promotion_gate
from nannyml_visualization import create_monitoring_report

logger = logging.getLogger("mlops")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="MLOps Workflow: Load data -> Integrity gate -> Feature engineering")
    p.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking server URI (default: http://localhost:5000)")
    p.add_argument("--experiment", default="green_taxi_monitoring", help="MLflow experiment name (default: green_taxi_monitoring)")
    p.add_argument("--run-name", default=None, help="MLflow run name (auto-generated if not provided)")
    p.add_argument("--ref-month", default="01", help="Reference dataset month as 2-digit string (default: 01)")
    p.add_argument("--batch-month", default="04", help="Batch dataset month as 2-digit string (default: 04)")
    p.add_argument("--model-name", default="green_taxi_tip_model", help="MLflow registered model name")
    p.add_argument("--rmse-degradation-pct", type=float, default=5.0, help="Retrain trigger threshold in percent")
    p.add_argument("--min-improvement", type=float, default=0.01, help="Minimum RMSE improvement required for promotion")
    p.add_argument("--max-reference-regression-pct", type=float, default=5.0, help="Maximum allowed reference-slice RMSE regression for promotion")
    p.add_argument("--fail-in-step-f", action="store_true", help="Intentionally fail in Step F (for failure/resume demo)")
    
    return p.parse_args()


def main():
    setup_logger()
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("MLOps Monitoring & Retraining Workflow")
    logger.info("=" * 70)
    
    # Locate data directory
    data_dir = find_tlc_data_dir()
    logger.info(f"Using TLC_data directory: {data_dir}")
    
    # Configure MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        logger.info(f"Creating experiment: {args.experiment}")
        mlflow.create_experiment(args.experiment)
    
    # Run workflow
    with mlflow.start_run(run_name=args.run_name):
        try:
            # Step A: Load data
            logger.info("")
            data = step_a_load_data(data_dir, args.ref_month, args.batch_month)
            logger.info(f"  Reference rows: {len(data['reference_df'])}")
            logger.info(f"  Batch rows: {len(data['batch_df'])}")
            
            # Step B: Integrity gate
            logger.info("")
            integrity_result = step_b_integrity_gate(data)
            
            # Check if batch passed integrity gate
            if not integrity_result["decision"]["integrity_passed"]:
                logger.error("Batch rejected by hard rules. Stopping workflow.")
                logger.error(f"Failures: {integrity_result['decision']['hard_failures']}")
                return
            
            logger.info("Batch passed integrity gate. Proceeding to next steps...")
            if integrity_result["decision"]["integrity_warn"]:
                logger.warning("Integrity warnings detected - see decision.json for details")
            
            # Generate monitoring report (visualization of drift across runs)
            logger.info("")
            logger.info("=" * 60)
            logger.info("Generating monitoring report with drift visualizations...")
            try:
                report_dir = Path("monitoring_reports")
                report_files = create_monitoring_report(
                    experiment_name=args.experiment,
                    tracking_uri=args.tracking_uri,
                    output_dir=report_dir
                )
                logger.info(f"Monitoring report saved to {report_dir.absolute()}")
                
                # Log report summary to MLflow
                summary_file = report_dir / "REPORT_SUMMARY.md"
                if summary_file.exists():
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary_text = f.read()
                    mlflow.log_text(summary_text, "monitoring_report_summary.md")
                
                # Log visualizations as artifacts
                for png_file in report_dir.glob("*.png"):
                    mlflow.log_artifact(str(png_file), "monitoring_visualizations")
                
                logger.info("Monitoring visualizations logged to MLflow artifacts")
            except Exception as e:
                logger.warning(f"Could not generate monitoring report: {e}")
            
            # Step C: Feature engineering
            logger.info("")
            features_result = step_c_feature_engineering(data)
            logger.info(f"  Reference features: {len(features_result['reference_features'])} rows x {len(features_result['feature_names'])} features")
            logger.info(f"  Batch features: {len(features_result['batch_features'])} rows x {len(features_result['feature_names'])} features")

            pipeline_data = {
                **data,
                **features_result,
                "integrity_passed": integrity_result["decision"]["integrity_passed"],
                "integrity_warn": integrity_result["decision"]["integrity_warn"],
                "ref_month": args.ref_month,
                "batch_month": args.batch_month,
            }

            # Step D: Load champion model (or bootstrap)
            logger.info("")
            champion_result = step_d_load_champion(
                data=pipeline_data,
                model_name=args.model_name,
            )
            pipeline_data.update(champion_result)

            # Step E: Performance gate
            logger.info("")
            perf_result = step_e_model_gate(
                data=pipeline_data,
                rmse_degradation_pct_threshold=args.rmse_degradation_pct,
            )
            pipeline_data.update(perf_result)

            # Step F: Conditional retrain
            logger.info("")
            retrain_result = step_f_retrain_candidate(
                data=pipeline_data,
                fail_intentionally=args.fail_in_step_f,
                fail_once=True,
            )
            pipeline_data.update(retrain_result)

            # Step G: Promotion gate
            logger.info("")
            promotion_result = step_g_promotion_gate(
                data={
                    "integrity_passed": pipeline_data["integrity_passed"],
                    "retrain_executed": pipeline_data["retrain_executed"],
                    "rmse_candidate": (pipeline_data["candidate_eval"] or {}).get("rmse_candidate") if pipeline_data.get("candidate_eval") else None,
                    "rmse_candidate_reference": (pipeline_data["candidate_eval"] or {}).get("rmse_candidate_reference") if pipeline_data.get("candidate_eval") else None,
                    "rmse_champion_reference": (pipeline_data["candidate_eval"] or {}).get("rmse_champion_reference") if pipeline_data.get("candidate_eval") else None,
                    "rmse_champion": pipeline_data["performance"]["rmse_champion"],
                    "model_name": pipeline_data["model_name"],
                    "candidate_model_uri": pipeline_data["candidate_model_uri"],
                    "candidate_version": pipeline_data.get("candidate_version"),
                },
                min_improvement=args.min_improvement,
                max_reference_regression_pct=args.max_reference_regression_pct,
            )

            logger.info(f"Promotion recommended: {promotion_result['promotion_recommended']}")
            logger.info(f"Promotion executed: {promotion_result['promotion_executed']}")
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("Workflow completed successfully!")
            logger.info("=" * 70)
            logger.info(f"View run in MLflow: {mlflow.get_artifact_uri()}")
            
        except Exception as e:
            logger.error(f"Workflow failed with exception: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    main()
