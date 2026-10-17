from fastapi import APIRouter
from backend.database.mongodb import db_manager

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "online",
        "system": "Multi-View Agricultural Health Assessment API",
        "version": "1.0.0",
        "database_connected": (db_manager.db is not None)
    }
