# MultiView-AgriHealth — Run Tests (PowerShell)
# Run from project root: .\scripts\run_tests.ps1

param(
    [string]$TestPath = "tests",
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] .venv not found. Run .\scripts\setup_venv.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  MultiView-AgriHealth — Test Runner" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PROJECT_ROOT

if ($Verbose) {
    & $venvPython -m pytest $TestPath -v 2>&1
} else {
    & $venvPython -m pytest $TestPath 2>&1
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "" 
    Write-Host "[PASS] All tests passed." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[FAIL] Some tests failed. See output above." -ForegroundColor Red
    exit 1
}
