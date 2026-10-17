from fastapi import APIRouter, HTTPException
from backend.database.mongodb import db_manager

router = APIRouter()

@router.get("/scans")
def get_scans_history(limit: int = 50):
    scans = db_manager.get_scans(limit=limit)
    return {"scans": scans, "count": len(scans)}

@router.get("/scans/{scan_id}")
def get_scan_details(scan_id: str):
    scan = db_manager.get_scan_by_id(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    return scan
