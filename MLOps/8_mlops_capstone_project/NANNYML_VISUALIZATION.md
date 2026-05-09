## NannyML Drift Visualization

This module provides comprehensive visualization capabilities for monitoring nannyml drift detection results across your MLOps monitoring runs.

### Features

The `nannyml_visualization.py` module includes:

- **Warnings Over Time** - Track the trend of drift warnings and hard failures across monitoring runs
- **Drift Column Frequency** - Identify which columns are most problematic (drift most often)
- **Drift Heatmap** - Matrix view showing which columns drifted in which runs
- **Integrity Summary** - Overall statistics on pass/fail rates and drift detection

### Quick Start

#### Option 1: Generate Report via Script (Recommended)

```bash
# Basic usage (uses default MLflow URI and experiment)
python generate_monitoring_report.py

# With custom MLflow server
python generate_monitoring_report.py \
    --tracking-uri http://your-mlflow-server:5000 \
    --experiment green_taxi_monitoring \
    --output-dir my_reports
```

#### Option 2: Use in Your Code

```python
from nannyml_visualization import create_monitoring_report

# Create complete monitoring report with all visualizations
report_files = create_monitoring_report(
    experiment_name="green_taxi_monitoring",
    tracking_uri="http://localhost:5000",
    output_dir="monitoring_reports"
)

# Access individual files
print(f"Warnings over time: {report_files['warnings_over_time']}")
print(f"Drift heatmap: {report_files['drift_heatmap']}")
```

#### Option 3: Create Individual Visualizations

```python
from nannyml_visualization import (
    fetch_drift_results_from_mlflow,
    plot_warnings_over_time,
    plot_drift_columns_frequency,
    plot_drift_heatmap,
    plot_integrity_summary
)

# Fetch data
df = fetch_drift_results_from_mlflow("green_taxi_monitoring")

# Create specific plots
fig1 = plot_warnings_over_time(df, output_path="warnings.png")
fig2 = plot_drift_columns_frequency(df, output_path="columns.png")
fig3 = plot_drift_heatmap(df, output_path="heatmap.png")
fig4 = plot_integrity_summary(df, output_path="summary.png")
```

### Output Files

The `create_monitoring_report()` function generates:

1. **01_warnings_over_time.png**
   - Line plot showing warnings/failures trend
   - Useful for identifying problematic batches
   - Helps track improvements over time

2. **02_drift_columns_frequency.png**
   - Horizontal bar chart of most-problematic columns
   - Sorted by frequency of drift detection
   - Identifies chronic data issues

3. **03_drift_heatmap.png**
   - Matrix showing column × run drift patterns
   - Color-coded: red = drift, green = no drift
   - Easy to spot columns with consistent problems

4. **04_integrity_summary.png**
   - Two subplots: pass/fail rates + warnings breakdown
   - Quick overview of monitoring health
   - Shows proportion of runs with issues

5. **monitoring_data.csv**
   - Raw data exported from MLflow runs
   - Useful for further analysis or custom visualizations
   - Columns: run_id, run_name, timestamp, num_warnings, num_hard_failures, etc.

6. **REPORT_SUMMARY.md**
   - Markdown summary with statistics
   - Top problematic columns listed
   - Quick reference for monitoring health

### Integration with MLflow

The visualization module fetches data directly from MLflow artifact storage. It expects:

- **Experiment Name**: Your monitoring experiment (default: "green_taxi_monitoring")
- **Artifact**: Each run should have a `decision_step_b.json` containing drift results
  - This is automatically created by `mlops_project_workflow.py`

### Dependencies

Make sure you have matplotlib and seaborn installed:

```bash
pip install matplotlib seaborn
```

### Example Workflow

1. Run monitoring with `mlops_project_workflow.py`:
   ```bash
   python mlops_project_workflow.py --ref-month 01 --batch-month 04
   python mlops_project_workflow.py --ref-month 01 --batch-month 05
   python mlops_project_workflow.py --ref-month 01 --batch-month 06
   ```

2. Generate monitoring report:
   ```bash
   python generate_monitoring_report.py
   ```

3. View reports in `monitoring_reports/` directory

4. Share the `REPORT_SUMMARY.md` and images with stakeholders

### Customization

You can create custom plots by fetching the data and using matplotlib directly:

```python
from nannyml_visualization import fetch_drift_results_from_mlflow
import matplotlib.pyplot as plt

df = fetch_drift_results_from_mlflow("green_taxi_monitoring")

# Your custom visualization
plt.figure(figsize=(10, 6))
plt.plot(df["timestamp"], df["num_warnings"])
plt.xlabel("Date")
plt.ylabel("Number of Warnings")
plt.title("My Custom Drift Trend")
plt.savefig("custom_plot.png")
```
