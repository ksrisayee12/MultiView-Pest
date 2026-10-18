@echo off
title MultiView-AgriHealth Stopper
echo ============================================================
echo   STOPPING MULTIVIEW-AGRIHEALTH PROCESSES
echo ============================================================
echo.

echo [1/2] Terminating processes on Port 8000 (Backend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [2/2] Terminating processes on Port 5173 (Frontend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo ============================================================
echo   [OK] All MultiView-AgriHealth servers stopped cleanly.
echo ============================================================
echo.
timeout /t 2 /nobreak >nul
