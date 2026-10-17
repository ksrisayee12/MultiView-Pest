import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from backend.app.config.settings import settings

try:
    from pymongo import MongoClient
    mongo_available = True
except ImportError:
    mongo_available = False

class DatabaseManager:
    def __init__(self):
        self.db = None
        self.in_memory_scans: List[Dict[str, Any]] = []
        self._initialize()

    def _initialize(self):
        if mongo_available and settings.MONGODB_URI:
            try:
                self.client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
                # Quick ping check
                self.client.admin.command('ping')
                self.db = self.client[settings.DB_NAME]
                print(f"[MongoDB] Connected successfully to {settings.DB_NAME}")
            except Exception as e:
                print(f"[MongoDB] Atlas/Local connection failed ({e}). Operating in in-memory fallback mode.")
                self.db = None
        else:
            print("[MongoDB] PyMongo unavailable or MONGODB_URI not set. Operating in in-memory fallback mode.")

    def save_scan(self, scan_data: Dict[str, Any]) -> str:
        scan_data["created_at"] = scan_data.get("created_at") or datetime.utcnow().isoformat()
        scan_id = scan_data.get("scan_id") or f"scan_{int(datetime.utcnow().timestamp()*1000)}"
        scan_data["scan_id"] = scan_id

        if self.db is not None:
            try:
                self.db.scans.insert_one(scan_data.copy())
                return scan_id
            except Exception as e:
                print(f"[MongoDB] Insert error: {e}. Saving to in-memory store.")

        self.in_memory_scans.insert(0, scan_data)
        return scan_id

    def get_scans(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self.db is not None:
            try:
                scans = list(self.db.scans.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
                return scans
            except Exception as e:
                print(f"[MongoDB] Fetch error: {e}")

        return self.in_memory_scans[:limit]

    def get_scan_by_id(self, scan_id: str) -> Optional[Dict[str, Any]]:
        if self.db is not None:
            try:
                scan = self.db.scans.find_one({"scan_id": scan_id}, {"_id": 0})
                if scan:
                    return scan
            except Exception as e:
                print(f"[MongoDB] Fetch error: {e}")

        for scan in self.in_memory_scans:
            if scan.get("scan_id") == scan_id:
                return scan
        return None

db_manager = DatabaseManager()
