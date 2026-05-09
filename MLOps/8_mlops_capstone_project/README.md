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

## Start MLflow UI

powershell -ExecutionPolicy Bypass -File .\scripts\start_mlflow_ui.ps1
