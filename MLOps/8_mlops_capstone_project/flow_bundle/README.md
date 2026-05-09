# Flow Bundle (Copy-Only)

This folder is a copy-only portable package for the capstone flow.
Original files in the project were not moved.

## Structure

- src/: Python source files for script flow + Metaflow flow
- scripts/: Helper PowerShell launch scripts
- docs/: Copied README and design doc

## Setup

1. Create/activate a Python environment.
2. Install dependencies:

   pip install -r .\src\requirements.txt

## Run (no Metaflow)

powershell -ExecutionPolicy Bypass -File .\scripts\run_adaptive.ps1

## Run (Metaflow)

powershell -ExecutionPolicy Bypass -File .\scripts\run_metaflow.ps1
bash ./scripts/run_metaflow.sh

The shell script accepts optional year arguments:

```bash
bash ./scripts/run_metaflow.sh http://127.0.0.1:5000 01 04 2020 2022
```

## Start MLflow UI

powershell -ExecutionPolicy Bypass -File .\scripts\start_mlflow_ui.ps1
bash ./scripts/start_mlflow_ui.sh
