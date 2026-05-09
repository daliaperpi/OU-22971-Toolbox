Param(
    [string]$TrackingUri = "http://127.0.0.1:5000",
    [string]$RefMonth = "01",
    [string]$BatchMonth = "04"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

python "$root\src\adaptive_mlops_flow.py" --tracking-uri $TrackingUri --ref-month $RefMonth --batch-month $BatchMonth
