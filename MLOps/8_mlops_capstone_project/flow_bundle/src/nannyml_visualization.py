"""nannyml_visualization.py

Visualization utilities for nannyml drift warnings and errors.
Creates graphs showing drift detection results across monitoring runs.

Example:
    from nannyml_visualization import visualize_drift_warnings
    
    # From MLflow runs
    visualize_drift_warnings(
        experiment_name="green_taxi_monitoring",
        tracking_uri="http://localhost:5000",
        output_dir="monitoring_reports"
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def fetch_drift_results_from_mlflow(
    experiment_name: str,
    tracking_uri: str = "http://localhost:5000",
) -> pd.DataFrame:
    """Fetch all drift detection results from MLflow runs.
    
    Args:
        experiment_name: MLflow experiment name (e.g., 'green_taxi_monitoring')
        tracking_uri: MLflow tracking server URI
        
    Returns:
        DataFrame with columns: run_id, run_name, timestamp, num_warnings, 
                                drift_columns, hard_failures
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Get experiment
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")
    
    exp_id = exp.experiment_id
    runs = client.search_runs(experiment_ids=[exp_id])
    
    results = []
    skipped_missing_decision = 0
    for run in runs:
        run_id = run.info.run_id
        run_name = run.info.run_name or run_id[:8]
        
        try:
            # Try to load decision.json from artifacts
            artifacts = client.list_artifacts(run_id)
            decision_artifact = None
            for artifact in artifacts:
                if artifact.path == "decision_step_b.json":
                    decision_artifact = artifact
                    break
            
            if decision_artifact is None:
                skipped_missing_decision += 1
                logger.debug(f"Run {run_name}: No decision_step_b.json found")
                continue
            
            # Download and parse decision JSON
            local_path = client.download_artifacts(run_id, "decision_step_b.json")
            # The download_artifacts returns a directory, find the actual file
            decision_file = Path(local_path)
            if decision_file.is_dir():
                decision_file = decision_file / "decision_step_b.json"
            
            if not decision_file.exists():
                skipped_missing_decision += 1
                logger.debug(f"Run {run_name}: decision_step_b.json not found at {decision_file}")
                continue
            
            with open(decision_file, "r") as f:
                decision = json.load(f)
            
            # Extract data
            nannyml_warnings = decision.get("nannyml_warnings", [])
            drift_columns = decision.get("drift_columns", [])
            hard_failures = decision.get("hard_failures", [])
            
            results.append({
                "run_id": run_id,
                "run_name": run_name,
                "timestamp": decision.get("timestamp", ""),
                "num_warnings": len(nannyml_warnings),
                "num_hard_failures": len(hard_failures),
                "drift_columns": drift_columns,
                "hard_failures": hard_failures,
                "nannyml_warnings": nannyml_warnings,
                "integrity_passed": decision.get("integrity_passed", False),
            })
            
        except Exception as e:
            logger.warning(f"Error processing run {run_name}: {e}")
            continue

    if skipped_missing_decision > 0:
        logger.info(f"Skipped {skipped_missing_decision} runs without decision_step_b.json artifact")
    
    df = pd.DataFrame(results)
    
    # Convert timestamp to datetime
    if len(df) > 0 and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    
    return df


def plot_warnings_over_time(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 7)
) -> plt.Figure:
    """Plot severity trend with anomaly detection and statistical bounds.
    
    Shows:
    - Hard failures over time (red area)
    - Drift warnings over time (orange line)
    - Mean ± std deviation bands to detect anomalies
    
    Args:
        df: Results DataFrame from fetch_drift_results_from_mlflow
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if df.empty:
        raise ValueError("No data to plot")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(df))
    warnings = df["num_warnings"].values
    failures = df["num_hard_failures"].values
    
    # Calculate statistical bounds for anomaly detection
    warnings_mean = warnings.mean()
    warnings_std = warnings.std()
    upper_bound = warnings_mean + (2 * warnings_std)  # 2-sigma
    lower_bound = max(0, warnings_mean - (2 * warnings_std))
    
    # Plot hard failures (background)
    ax.bar(x, failures, alpha=0.2, label="Hard Failures", color="darkred", width=0.8)
    
    # Plot warnings with confidence bands
    ax.fill_between(x, lower_bound, upper_bound, alpha=0.2, color="orange", label="Normal Range (±2σ)")
    ax.axhline(warnings_mean, color="gray", linestyle="--", linewidth=2, label=f"Mean: {warnings_mean:.1f}", alpha=0.7)
    
    # Plot warnings line
    ax.plot(x, warnings, marker="o", linewidth=2.5, markersize=8, label="Drift Warnings", color="darkorange", zorder=5)
    
    # Highlight anomalies (values outside bounds)
    anomalies = (warnings > upper_bound) | (warnings < lower_bound)
    if anomalies.any():
        anomaly_x = [i for i, is_anomaly in enumerate(anomalies) if is_anomaly]
        anomaly_y = [warnings[i] for i in anomaly_x]
        ax.scatter(anomaly_x, anomaly_y, s=200, marker="*", color="red", zorder=10, label="Anomaly Detected", edgecolors="darkred", linewidth=2)
    
    # Format
    ax.set_xlabel("Monitoring Run (Time Order)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Severity (Number of Issues)", fontsize=12, fontweight="bold")
    ax.set_title("Data Quality Severity Trend with Anomaly Detection", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["run_name"], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)
    
    # Add value labels on points
    for i, (w, f) in enumerate(zip(warnings, failures)):
        if f > 0:
            ax.text(i, f - 1, f"{int(f)}", ha="center", va="top", fontsize=8, fontweight="bold", color="darkred")
        ax.text(i, w + 0.3, f"{int(w)}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="darkorange")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved to {output_path}")
    
    return fig


def plot_issue_severity_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot data quality scorecard showing severity of issues by column.
    
    Extracts failure messages to quantify issue severity for each column.
    Shows: frequency + severity ranking.
    
    Args:
        df: Results DataFrame from fetch_drift_results_from_mlflow
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if df.empty:
        raise ValueError("No data to plot")
    
    # Extract issues from hard failures
    issue_severity = {}
    
    for _, row in df.iterrows():
        hard_failures = row.get("hard_failures", [])
        if isinstance(hard_failures, str):
            hard_failures = [hard_failures]
        
        for failure in hard_failures:
            if not isinstance(failure, str):
                continue
            
            # Extract column name from failure message
            col_name = None
            if "Column '" in failure:
                col_name = failure.split("Column '")[1].split("'")[0]
            
            if col_name:
                if col_name not in issue_severity:
                    issue_severity[col_name] = {"count": 0, "examples": []}
                
                issue_severity[col_name]["count"] += 1
                if len(issue_severity[col_name]["examples"]) < 2:
                    issue_severity[col_name]["examples"].append(failure[:60] + "...")
    
    if not issue_severity:
        raise ValueError("No issue data found")
    
    # Sort by frequency
    sorted_issues = sorted(issue_severity.items(), key=lambda x: x[1]["count"], reverse=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cols = [col for col, _ in sorted_issues]
    counts = [data["count"] for _, data in sorted_issues]
    
    # Color by severity (more occurrences = darker red)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(cols)))
    
    bars = ax.barh(cols, counts, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels and severity rating
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        
        # Severity rating
        if count >= 10:
            severity = "CRITICAL"
            sev_color = "#e74c3c"
        elif count >= 5:
            severity = "HIGH"
            sev_color = "#f39c12"
        else:
            severity = "MEDIUM"
            sev_color = "#f1c40f"
        
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                f"{int(count)}x [{severity}]", 
                ha="left", va="center", fontweight="bold", fontsize=10)
    
    ax.set_xlabel("Frequency Across Runs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Column / Issue Type", fontsize=12, fontweight="bold")
    ax.set_title("Data Quality Issues - Severity Scorecard\n(Columns with most recurring problems)", 
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="black", label="CRITICAL (>=10 occurrences)"),
        Patch(facecolor="#f39c12", edgecolor="black", label="HIGH (5-9 occurrences)"),
        Patch(facecolor="#f1c40f", edgecolor="black", label="MEDIUM (<5 occurrences)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved to {output_path}")
    
    return fig


def plot_drift_columns_frequency(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot which columns have drift most frequently.
    
    Args:
        df: Results DataFrame from fetch_drift_results_from_mlflow
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if df.empty:
        raise ValueError("No data to plot")
    
    # Flatten drift columns across all runs
    all_drift_cols = []
    for drift_list in df["drift_columns"]:
        if isinstance(drift_list, list):
            all_drift_cols.extend(drift_list)
    
    if not all_drift_cols:
        raise ValueError("No drift columns detected in any runs")
    
    # Count frequency
    col_counts = pd.Series(all_drift_cols).value_counts().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    col_counts.plot(kind="barh", ax=ax, color="coral")
    
    ax.set_xlabel("Frequency (Number of Runs)", fontsize=12)
    ax.set_ylabel("Column Name", fontsize=12)
    ax.set_title("Most Frequently Drifted Columns", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved to {output_path}")
    
    return fig


def plot_drift_heatmap(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot heatmap of which columns drifted in which runs.
    
    Args:
        df: Results DataFrame from fetch_drift_results_from_mlflow
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if df.empty:
        raise ValueError("No data to plot")
    
    # Build matrix: runs x columns
    all_cols = set()
    for drift_list in df["drift_columns"]:
        if isinstance(drift_list, list):
            all_cols.update(drift_list)
    
    if not all_cols:
        raise ValueError("No drift columns detected in any runs")
    
    all_cols = sorted(list(all_cols))
    
    # Create binary matrix
    matrix = []
    for _, row in df.iterrows():
        drift_set = set(row["drift_columns"]) if isinstance(row["drift_columns"], list) else set()
        row_data = [1 if col in drift_set else 0 for col in all_cols]
        matrix.append(row_data)
    
    matrix = np.array(matrix)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        xticklabels=all_cols,
        yticklabels=df["run_name"].values,
        cmap="RdYlGn_r",
        cbar_kws={"label": "Drift Detected"},
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_title("Drift Detection Heatmap (Columns × Runs)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Monitoring Run", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved to {output_path}")
    
    return fig


def plot_integrity_summary(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """Plot summary of integrity check results with data quality scorecard.
    
    Args:
        df: Results DataFrame from fetch_drift_results_from_mlflow
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if df.empty:
        raise ValueError("No data to plot")
    
    # Count integrity status
    integrity_pass = (df["integrity_passed"] == True).sum()
    integrity_fail = (df["integrity_passed"] == False).sum()
    has_warnings = (df["num_warnings"] > 0).sum()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Left: Integrity pass/fail
    ax = axes[0]
    categories = ["Passed", "Failed"]
    counts = [integrity_pass, integrity_fail]
    colors = ["#2ecc71" if c > 0 else "#95a5a6" for c in counts]
    colors = ["#2ecc71" if i == 0 else "#e74c3c" for i in range(len(categories))]
    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    ax.set_ylabel("Number of Runs", fontsize=11, fontweight="bold")
    ax.set_title("Integrity Gate Results", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, str(int(count)), 
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    
    # Middle: Warnings breakdown
    ax = axes[1]
    no_warnings = len(df) - has_warnings
    categories = ["With Issues", "No Issues"]
    counts = [has_warnings, no_warnings]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    ax.set_ylabel("Number of Runs", fontsize=11, fontweight="bold")
    ax.set_title("Data Quality Detection", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, str(int(count)), 
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    
    # Right: Data Quality Score
    ax = axes[2]
    total_runs = len(df)
    total_issues = df["num_hard_failures"].sum() + df["num_warnings"].sum()
    avg_issues_per_run = total_issues / total_runs if total_runs > 0 else 0
    
    # Calculate quality score (0-100, lower is worse)
    quality_score = max(0, 100 - (avg_issues_per_run * 5))  # Each issue reduces score by 5
    
    # Color based on score
    if quality_score >= 80:
        color = "#2ecc71"
        status = "Good"
    elif quality_score >= 50:
        color = "#f39c12"
        status = "Moderate"
    else:
        color = "#e74c3c"
        status = "Poor"
    
    # Draw gauge
    ax.barh(["Quality"], [quality_score], color=color, alpha=0.8, edgecolor="black", linewidth=2, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Data Quality Score", fontsize=11, fontweight="bold")
    ax.set_title("Overall Data Quality Scorecard", fontsize=12, fontweight="bold")
    ax.text(quality_score / 2, 0, f"{quality_score:.0f}%", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="white")
    ax.text(quality_score + 3, 0, status, 
            ha="left", va="center", fontsize=11, fontweight="bold", color=color)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_yticks([])
    
    # Add reference zones
    ax.axvline(80, color="green", linestyle=":", alpha=0.5, linewidth=1)
    ax.axvline(50, color="orange", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(80, -0.25, "Good", fontsize=9, ha="center", color="green")
    ax.text(50, -0.25, "Moderate", fontsize=9, ha="center", color="orange")
    
    plt.suptitle("System Health Dashboard", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved to {output_path}")
    
    return fig


def create_monitoring_report(
    experiment_name: str,
    tracking_uri: str = "http://localhost:5000",
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Create comprehensive monitoring report with all visualizations.
    
    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking server URI
        output_dir: Directory to save report. If None, uses ./monitoring_reports/
        
    Returns:
        Dictionary with paths to generated report files
    """
    output_dir = Path(output_dir or "monitoring_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating monitoring report for experiment: {experiment_name}")
    
    # Fetch data
    df = fetch_drift_results_from_mlflow(experiment_name, tracking_uri)
    logger.info(f"Loaded {len(df)} runs from MLflow")
    
    if df.empty:
        logger.warning("No runs found. Cannot create report.")
        return {}
    
    report_files = {}
    
    # Generate visualizations
    try:
        fig = plot_warnings_over_time(df, output_dir / "01_warnings_over_time.png")
        report_files["warnings_over_time"] = output_dir / "01_warnings_over_time.png"
        plt.close(fig)
        logger.info("✓ Created warnings_over_time.png")
    except Exception as e:
        logger.warning(f"Could not create warnings_over_time: {e}")
    
    try:
        fig = plot_issue_severity_scorecard(df, output_dir / "02_issue_severity_scorecard.png")
        report_files["severity_scorecard"] = output_dir / "02_issue_severity_scorecard.png"
        plt.close(fig)
        logger.info("✓ Created issue_severity_scorecard.png")
    except Exception as e:
        logger.warning(f"Could not create issue_severity_scorecard: {e}")
    
    try:
        fig = plot_drift_columns_frequency(df, output_dir / "03_drift_columns_frequency.png")
        report_files["drift_frequency"] = output_dir / "03_drift_columns_frequency.png"
        plt.close(fig)
        logger.info("✓ Created drift_columns_frequency.png")
    except Exception as e:
        logger.warning(f"Could not create drift_columns_frequency: {e}")
    
    try:
        fig = plot_drift_heatmap(df, output_dir / "04_drift_heatmap.png")
        report_files["drift_heatmap"] = output_dir / "04_drift_heatmap.png"
        plt.close(fig)
        logger.info("✓ Created drift_heatmap.png")
    except Exception as e:
        logger.warning(f"Could not create drift_heatmap: {e}")
    
    try:
        fig = plot_integrity_summary(df, output_dir / "05_integrity_summary.png")
        report_files["integrity_summary"] = output_dir / "05_integrity_summary.png"
        plt.close(fig)
        logger.info("✓ Created integrity_summary.png")
    except Exception as e:
        logger.warning(f"Could not create integrity_summary: {e}")
    
    # Save raw data to CSV
    csv_path = output_dir / "monitoring_data.csv"
    # Flatten list columns for CSV
    df_export = df.copy()
    df_export["drift_columns"] = df_export["drift_columns"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )
    df_export["hard_failures"] = df_export["hard_failures"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else ""
    )
    df_export["nannyml_warnings"] = df_export["nannyml_warnings"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else ""
    )
    df_export.to_csv(csv_path, index=False)
    report_files["data"] = csv_path
    logger.info(f"✓ Saved raw data to {csv_path}")
    
    # Create summary report
    summary_path = output_dir / "REPORT_SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# NannyML Monitoring Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Experiment:** {experiment_name}\n\n")
        
        f.write(f"## Summary Statistics\n\n")
        f.write(f"- **Total Runs:** {len(df)}\n")
        f.write(f"- **Integrity Passed:** {(df['integrity_passed'] == True).sum()}\n")
        f.write(f"- **Integrity Failed:** {(df['integrity_passed'] == False).sum()}\n")
        f.write(f"- **Runs with Drift Warnings:** {(df['num_warnings'] > 0).sum()}\n")
        f.write(f"- **Average Issues per Run:** {(df['num_warnings'].mean() + df['num_hard_failures'].mean()):.1f}\n")
        f.write(f"- **Max Issues in Single Run:** {(df['num_warnings'].max() + df['num_hard_failures'].max())}\n\n")
        
        # Quality Score
        total_issues = df["num_hard_failures"].sum() + df["num_warnings"].sum()
        avg_issues = total_issues / len(df) if len(df) > 0 else 0
        quality_score = max(0, 100 - (avg_issues * 5))
        
        if quality_score >= 80:
            status = "GOOD"
        elif quality_score >= 50:
            status = "MODERATE"
        else:
            status = "POOR"
        
        f.write(f"## Data Quality Score\n\n")
        f.write(f"**Overall Score: {quality_score:.0f}/100** - {status}\n\n")
        
        # Top problematic columns
        issue_severity = {}
        for _, row in df.iterrows():
            hard_failures = row.get("hard_failures", [])
            if isinstance(hard_failures, str):
                hard_failures = [hard_failures]
            
            for failure in hard_failures:
                if not isinstance(failure, str):
                    continue
                
                col_name = None
                if "Column '" in failure:
                    col_name = failure.split("Column '")[1].split("'")[0]
                
                if col_name:
                    if col_name not in issue_severity:
                        issue_severity[col_name] = 0
                    issue_severity[col_name] += 1
        
        if issue_severity:
            f.write(f"## Action Items - Top Problematic Columns\n\n")
            f.write(f"Fix these columns first (sorted by impact):\n\n")
            
            for rank, (col, count) in enumerate(sorted(issue_severity.items(), key=lambda x: x[1], reverse=True)[:10], 1):
                if count >= 10:
                    priority = "[CRITICAL]"
                elif count >= 5:
                    priority = "[HIGH]"
                else:
                    priority = "[MEDIUM]"
                
                f.write(f"{rank}. **{col}** {priority}\n")
                f.write(f"   - Affected in {count} runs\n")
                f.write(f"   - Action: Review data pipeline and validation rules for this column\n\n")
        
        # Most frequently drifting columns
        all_drift_cols = []
        for drift_list in df["drift_columns"]:
            if isinstance(drift_list, list):
                all_drift_cols.extend(drift_list)
        
        if all_drift_cols:
            f.write(f"## Drift Detection - Most Frequent\n\n")
            col_counts = pd.Series(all_drift_cols).value_counts()
            for col, count in col_counts.head(5).items():
                f.write(f"- **{col}**: {count} occurrences\n")
            f.write(f"\n")
        
        f.write(f"## Generated Visualizations\n\n")
        f.write(f"1. **01_warnings_over_time.png** - Severity trend with anomaly detection\n")
        f.write(f"2. **02_issue_severity_scorecard.png** - Which columns have the worst issues\n")
        f.write(f"3. **03_drift_columns_frequency.png** - Most frequently drifted columns\n")
        f.write(f"4. **04_drift_heatmap.png** - Column-wise drift per run (matrix view)\n")
        f.write(f"5. **05_integrity_summary.png** - Overall system health dashboard\n\n")
        
        f.write(f"## Recommendations\n\n")
        if integrity_pass_count := (df['integrity_passed'] == True).sum():
            f.write(f"SUCCESS: {integrity_pass_count} batches passed validation - good progress\n\n")
        else:
            f.write(f"ALERT: No batches have passed validation - data pipeline needs investigation\n\n")
        
        f.write(f"- Start by fixing the CRITICAL columns identified above\n")
        f.write(f"- Check data collection and preprocessing scripts\n")
        f.write(f"- Consider relaxing thresholds if they are too strict\n")
        f.write(f"- Set up data quality checks at the source\n")
    
    report_files["summary"] = summary_path
    logger.info(f"✓ Report saved to {output_dir}")
    
    return report_files


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    try:
        files = create_monitoring_report(
            experiment_name="green_taxi_monitoring",
            tracking_uri="http://localhost:5000",
            output_dir="monitoring_reports"
        )
        
        print("\n✓ Monitoring report created successfully!")
        print(f"\nGenerated files:")
        for key, path in files.items():
            print(f"  {key}: {path}")
    except Exception as e:
        print(f"Error: {e}")
