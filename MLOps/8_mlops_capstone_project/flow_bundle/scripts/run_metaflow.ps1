Param(
    [string]$TrackingUri = "http://127.0.0.1:5000",
    [string]$RefMonth = "01",
    [string]$RefYear = "2020",
    [string]$BatchMonth = "04",
    [string]$BatchYear = "2020"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

python "$root\src\metaflow_capstone_flow.py" run --tracking-uri $TrackingUri --ref-month $RefMonth --ref-year $RefYear --batch-month $BatchMonth --batch-year $BatchYear
