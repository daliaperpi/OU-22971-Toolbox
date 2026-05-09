# Running the Workflow with Real NYC Green Taxi Data

## 1) Download monthly parquet files

Use the TLC dataset page:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Store files in the repository `TLC_data` directory with names like:

- `green_tripdata_2020-01.parquet`
- `green_tripdata_2020-02.parquet`
- `green_tripdata_2020-03.parquet`

For the capstone demo, keep at least 8 to 12 monthly files so you can show baseline, retrain/promotion, and failure/recovery runs.

## 2) Start MLflow UI

From repository root:

```powershell
cd "c:\Users\dperpign\OneDrive - Intel Corporation\Desktop\OU-22971-Toolbox-main"
.\.venv\Scripts\python.exe -m mlflow ui --host 127.0.0.1 --port 5000
```

## 3) Run the workflow

From repository root:

```powershell
cd "c:\Users\dperpign\OneDrive - Intel Corporation\Desktop\OU-22971-Toolbox-main"
.\.venv\Scripts\python.exe .\MLOps\8_mlops_capstone_project\mlops_project_workflow.py --ref-month 01 --batch-month 04
```

Try multiple runs by changing `--batch-month`.

## 4) Review results in MLflow

Open http://localhost:5000 and inspect:

- metrics: `rmse_champion`, `rmse_baseline`, `rmse_increase_pct`, `rmse_candidate`
- tags: `integrity_warn`, `retrain_recommended`, `promotion_recommended`
- artifacts: `decision_step_b.json` ... `decision_step_g.json`
- model registry: `green_taxi_tip_model` and alias `champion`
