@echo off
title MultiView-AgriHealth Launcher
cd /d "%~dp0"

echo ============================================================
echo   MULTIVIEW-AGRIHEALTH: ONE-CLICK APPLICATION LAUNCHER
echo ============================================================
echo.

:: 1. Check Python virtual environment
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please run scripts\setup_venv.ps1 first.
    pause
    exit /b 1
)

:: 2. Check Node / npm
where npm.cmd >nul 2>&1
if %ERRORLEVEL% neq 0 (
    where npm >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Node.js / npm not found on PATH.
        echo Please ensure Node.js is installed.
        pause
        exit /b 1
    )
)

echo [1/2] Starting Backend API Server (FastAPI on http://127.0.0.1:8000)...
start "MultiView-AgriHealth Backend (Port 8000)" cmd /k "cd /d ""%~dp0"" && title MultiView-AgriHealth Backend && .venv\Scripts\activate.bat && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting Frontend Dev Server (Vite on http://localhost:5173)...
start "MultiView-AgriHealth Frontend (Port 5173)" cmd /k "cd /d ""%~dp0frontend"" && title MultiView-AgriHealth Frontend && npm.cmd run dev"

echo.
echo ============================================================
echo   SERVERS LAUNCHED SUCCESSFULLY!
echo ============================================================
echo.
echo   * Web Application:     http://localhost:5173
echo   * Backend Swagger API: http://127.0.0.1:8000/docs
echo   * Model Status API:    http://127.0.0.1:8000/api/models/status
echo.
echo Both servers are now running in separate titled console windows.
echo Opening browser to http://localhost:5173...
echo To stop everything later, run stop_app.bat or close both windows.
echo.

echo Waiting for backend API server to finish initializing...
powershell -NoProfile -Command "for ($i=0; $i -lt 25; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/models/status' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch { Start-Sleep -Milliseconds 800 } }; exit 0"

echo Backend is ready! Opening browser to http://localhost:5173...
start http://localhost:5173
