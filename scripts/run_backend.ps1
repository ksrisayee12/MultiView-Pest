# MultiView-AgriHealth — Run Backend (PowerShell)
# Run from project root: .\scripts\run_backend.ps1

param(
    [int]$Port = 8000,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] .venv not found. Run .\scripts\setup_venv.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Starting MultiView-AgriHealth backend on port $Port..." -ForegroundColor Cyan
Write-Host "[INFO] API docs: http://localhost:$Port/docs" -ForegroundColor Cyan
Write-Host "[INFO] Health:   http://localhost:$Port/api/health" -ForegroundColor Cyan
Write-Host ""

Set-Location $PROJECT_ROOT

if ($NoReload) {
    & $venvPython -m uvicorn backend.app.main:app --host 0.0.0.0 --port $Port
} else {
    & $venvPython -m uvicorn backend.app.main:app --host 0.0.0.0 --port $Port --reload
}
