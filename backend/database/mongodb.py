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

    # -----------------------------------------------------------------------
    # Plant Session Persistence (Phase 3 Multi-View Support)
    # -----------------------------------------------------------------------

    def save_plant_session(self, session_data: Dict[str, Any]) -> str:
        """Store a new or updated plant session."""
        session_id = session_data["session_id"]
        if self.db is not None:
            try:
                self.db.plant_sessions.replace_one(
                    {"session_id": session_id},
                    session_data.copy(),
                    upsert=True,
                )
                return session_id
            except Exception as e:
                print(f"[MongoDB] Plant session upsert error: {e}. Falling back to in-memory.")

        # In-memory store
        if not hasattr(self, "in_memory_sessions"):
            self.in_memory_sessions: Dict[str, Dict[str, Any]] = {}
        self.in_memory_sessions[session_id] = session_data.copy()
        return session_id

    def get_plant_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a plant session by ID."""
        if self.db is not None:
            try:
                doc = self.db.plant_sessions.find_one({"session_id": session_id}, {"_id": 0})
                if doc:
                    return doc
            except Exception as e:
                print(f"[MongoDB] Plant session fetch error: {e}")

        if not hasattr(self, "in_memory_sessions"):
            self.in_memory_sessions = {}
        return self.in_memory_sessions.get(session_id)

    def list_plant_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent plant sessions."""
        if self.db is not None:
            try:
                docs = list(
                    self.db.plant_sessions.find({}, {"_id": 0})
                    .sort("created_at", -1)
                    .limit(limit)
                )
                return docs
            except Exception as e:
                print(f"[MongoDB] List plant sessions error: {e}")

        if not hasattr(self, "in_memory_sessions"):
            self.in_memory_sessions = {}
        sorted_sessions = sorted(
            self.in_memory_sessions.values(),
            key=lambda x: x.get("created_at", 0),
            reverse=True,
        )
        return sorted_sessions[:limit]

    def update_plant_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields in an existing plant session."""
        if self.db is not None:
            try:
                res = self.db.plant_sessions.update_one(
                    {"session_id": session_id},
                    {"$set": updates},
                )
                return res.matched_count > 0
            except Exception as e:
                print(f"[MongoDB] Update plant session error: {e}")

        if not hasattr(self, "in_memory_sessions"):
            self.in_memory_sessions = {}
        if session_id in self.in_memory_sessions:
            self.in_memory_sessions[session_id].update(updates)
            return True
        return False

    def delete_plant_session(self, session_id: str) -> bool:
        """Delete a plant session."""
        if self.db is not None:
            try:
                res = self.db.plant_sessions.delete_one({"session_id": session_id})
                self.db.plant_observations.delete_many({"session_id": session_id})
                return res.deleted_count > 0
            except Exception as e:
                print(f"[MongoDB] Delete plant session error: {e}")

        if not hasattr(self, "in_memory_sessions"):
            self.in_memory_sessions = {}
        if session_id in self.in_memory_sessions:
            del self.in_memory_sessions[session_id]
            if hasattr(self, "in_memory_observations") and session_id in self.in_memory_observations:
                del self.in_memory_observations[session_id]
            return True
        return False

    def save_observations(self, session_id: str, observations: List[Dict[str, Any]]) -> None:
        """Save frame observations for a session."""
        if self.db is not None:
            try:
                # Add session_id to each doc
                docs = [dict(obs, session_id=session_id) for obs in observations]
                if docs:
                    self.db.plant_observations.insert_many(docs)
                return
            except Exception as e:
                print(f"[MongoDB] Save observations error: {e}")

        if not hasattr(self, "in_memory_observations"):
            self.in_memory_observations: Dict[str, List[Dict[str, Any]]] = {}
        self.in_memory_observations.setdefault(session_id, []).extend(observations)

    def get_observations(self, session_id: str) -> List[Dict[str, Any]]:
        """Get frame observations for a session."""
        if self.db is not None:
            try:
                docs = list(self.db.plant_observations.find({"session_id": session_id}, {"_id": 0}))
                return docs
            except Exception as e:
                print(f"[MongoDB] Get observations error: {e}")

        if not hasattr(self, "in_memory_observations"):
            self.in_memory_observations = {}
        return self.in_memory_observations.get(session_id, [])


db_manager = DatabaseManager()

