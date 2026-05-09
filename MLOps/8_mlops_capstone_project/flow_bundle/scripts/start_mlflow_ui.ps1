
Param(
    [string]$MlflowHost = "127.0.0.1",
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"
mlflow ui --host $MlflowHost --port $Port
