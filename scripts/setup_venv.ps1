# MultiView-AgriHealth — Windows PowerShell Environment Setup
# Run from the project root directory: .\scripts\setup_venv.ps1

param(
    [switch]$Force   # Use -Force to overwrite an existing .venv
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  MultiView-AgriHealth — Python Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------------
# 1. Detect Python
# ------------------------------------------------------------------
$pythonExe = $null
$preferredVersions = @("python3.11", "python3.10", "python3.12", "python3")

foreach ($py in $preferredVersions) {
    try {
        $versionStr = & $py --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $versionStr -match "Python (\d+\.\d+)") {
            $pythonExe = $py
            break
        }
    } catch { }
}

# Fallback: check system python
if (-not $pythonExe) {
    try {
        $versionStr = & python --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $versionStr -match "Python 3") {
            $pythonExe = "python"
        }
    } catch { }
}

if (-not $pythonExe) {
    Write-Host "[ERROR] No compatible Python 3 installation found." -ForegroundColor Red
    Write-Host "  Install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

$detectedVersion = & $pythonExe --version 2>&1
Write-Host "[OK] Detected: $detectedVersion  ($pythonExe)" -ForegroundColor Green

# ------------------------------------------------------------------
# 2. Create .venv
# ------------------------------------------------------------------
$venvPath = Join-Path $PROJECT_ROOT ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

if (Test-Path $venvPath) {
    if ($Force) {
        Write-Host "[INFO] Removing existing .venv (--Force flag set)..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host "[INFO] .venv already exists at: $venvPath" -ForegroundColor Yellow
        Write-Host "       Use -Force to recreate it, or skip creation and upgrade packages." -ForegroundColor Yellow
    }
}

if (-not (Test-Path $venvPath)) {
    Write-Host "[...] Creating virtual environment at: $venvPath" -ForegroundColor Cyan
    & $pythonExe -m venv $venvPath --prompt "agrihealth"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# ------------------------------------------------------------------
# 3. Upgrade pip inside .venv
# ------------------------------------------------------------------
Write-Host "[...] Upgrading pip inside .venv..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] pip upgrade failed. Continuing..." -ForegroundColor Yellow
} else {
    Write-Host "[OK] pip upgraded." -ForegroundColor Green
}

# ------------------------------------------------------------------
# 4. Install requirements
# ------------------------------------------------------------------
$requirementsFile = Join-Path $PROJECT_ROOT "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "[...] Installing requirements from requirements.txt..." -ForegroundColor Cyan
    & $venvPython -m pip install -r $requirementsFile --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Dependency installation failed." -ForegroundColor Red
        Write-Host "  Check requirements.txt for version conflicts." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[OK] Runtime dependencies installed." -ForegroundColor Green
} else {
    Write-Host "[WARN] requirements.txt not found. Skipping." -ForegroundColor Yellow
}

$devRequirementsFile = Join-Path $PROJECT_ROOT "requirements-dev.txt"
if (Test-Path $devRequirementsFile) {
    Write-Host "[...] Installing dev requirements from requirements-dev.txt..." -ForegroundColor Cyan
    & $venvPython -m pip install -r $devRequirementsFile --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Dev dependency installation failed." -ForegroundColor Yellow
    } else {
        Write-Host "[OK] Dev dependencies installed." -ForegroundColor Green
    }
}

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
$installedPythonVersion = & $venvPython --version 2>&1
$installedPipVersion = & $venvPip --version 2>&1

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Python:    $installedPythonVersion" -ForegroundColor White
Write-Host "  Python exe: $venvPython" -ForegroundColor White
Write-Host "  Pip:       $installedPipVersion" -ForegroundColor White
Write-Host ""
Write-Host "  To activate:" -ForegroundColor Yellow
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  To run backend (after activation):" -ForegroundColor Yellow
Write-Host "    python -m uvicorn backend.app.main:app --reload --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "  To run backend (without activation):" -ForegroundColor Yellow
Write-Host "    .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "  To run tests:" -ForegroundColor Yellow
Write-Host "    .\scripts\run_tests.ps1" -ForegroundColor White
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
