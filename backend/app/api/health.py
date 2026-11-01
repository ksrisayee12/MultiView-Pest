"""
backend/app/api/health.py

System health check endpoint — used by the run_app.bat readiness probe
and frontend status polling.
"""

import time
import platform
import sys
from fastapi import APIRouter
from backend.database.mongodb import db_manager

router = APIRouter()

_STARTUP_TIME = time.time()


@router.get("/health", tags=["Health"])
def health_check():
    """
    Returns system readiness status. Used by the frontend and the run_app.bat
    readiness probe to confirm the backend is accepting requests.
    """
    uptime_seconds = round(time.time() - _STARTUP_TIME, 1)
    return {
        "status": "online",
        "system": "MultiView-AgriHealth Diagnostic API",
        "version": "2.0.0",
        "uptime_seconds": uptime_seconds,
        "database_connected": (db_manager.db is not None),
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "ready": True,
    }
