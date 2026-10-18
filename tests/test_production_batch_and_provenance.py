"""
tests/test_production_batch_and_provenance.py

Comprehensive tests for:
  1. Multi-media batch upload to session (/plant-sessions/{id}/media/batch)
  2. Content-hash deduplication at upload time
  3. Multi-plant batch ingestion with strict sample boundary isolation (/plant-sessions/batch)
  4. Decision-support report export in Markdown and JSON formats (/plant-sessions/{id}/export)
  5. Grounded model provenance (SHA-256 hash, model version, timestamp)
"""

import io
import json
import pytest
import numpy as np
import cv2
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_registry import (
    get_disease_model_status,
    predict_disease,
)
from backend.database.mongodb import db_manager


@pytest.fixture
def client():
    return TestClient(app)


def _make_dummy_image_bytes(color=(0, 200, 0), text="test") -> bytes:
    """Generate in-memory encoded JPEG bytes."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[:] = color
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestProductionBatchAndProvenance:

    def test_model_registry_reports_sha256_provenance(self):
        """Verify model registry computes and reports artifact SHA-256 hash."""
        status = get_disease_model_status()
        assert status["status"] in ("READY", "MISSING_WEIGHTS", "NOT_CONFIGURED")
        if status["status"] == "READY":
            assert "sha256" in status
            assert len(status["sha256"]) == 64
            assert status["model_name"] == "tomato_disease_yolov8n"
            assert status["model_version"] == "1.0.0"

    def test_prediction_returns_provenance_metadata(self):
        """Verify individual predictions include full provenance."""
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_img[:] = (0, 180, 20)
        res = predict_disease(dummy_img)

        assert "prediction" in res
        assert "confidence" in res
        assert "provenance" in res
        prov = res["provenance"]
        assert "model_name" in prov
        assert "timestamp" in prov

    def test_batch_media_upload_to_session(self, client):
        """Verify uploading multiple files in a single batch request."""
        # 1. Create session
        res_create = client.post("/api/plant-sessions", json={"plant_id": "PLANT-BATCH-001", "crop_type": "tomato"})
        assert res_create.status_code == 201
        session_id = res_create.json()["session_id"]

        # 2. Upload multiple files at once
        img1 = _make_dummy_image_bytes(color=(10, 150, 10), text="view1")
        img2 = _make_dummy_image_bytes(color=(10, 160, 20), text="view2")
        meta = json.dumps({"leaf_front.jpg": "front", "leaf_back.jpg": "back"})

        files = [
            ("files", ("leaf_front.jpg", io.BytesIO(img1), "image/jpeg")),
            ("files", ("leaf_back.jpg", io.BytesIO(img2), "image/jpeg")),
        ]
        res_upload = client.post(
            f"/api/plant-sessions/{session_id}/media/batch",
            files=files,
            data={"metadata": meta},
        )
        assert res_upload.status_code == 201
        payload = res_upload.json()
        assert payload["added_count"] == 2
        assert len(payload["added_clips"]) == 2
        views = {c["view_label"] for c in payload["added_clips"]}
        assert "front" in views
        assert "back" in views

    def test_duplicate_content_hash_deduplication(self, client):
        """Verify identical files uploaded to same session are detected and skipped."""
        res_create = client.post("/api/plant-sessions", json={"plant_id": "PLANT-DEDUP-001", "crop_type": "tomato"})
        session_id = res_create.json()["session_id"]

        img1 = _make_dummy_image_bytes(color=(50, 120, 30), text="same_file")

        # First upload
        files1 = [("files", ("photo1.jpg", io.BytesIO(img1), "image/jpeg"))]
        res1 = client.post(f"/api/plant-sessions/{session_id}/media/batch", files=files1)
        assert res1.status_code == 201
        assert res1.json()["added_count"] == 1

        # Second upload with identical content
        files2 = [("files", ("photo1_dup.jpg", io.BytesIO(img1), "image/jpeg"))]
        res2 = client.post(f"/api/plant-sessions/{session_id}/media/batch", files=files2)
        assert res2.status_code == 201
        assert res2.json()["added_count"] == 0
        assert len(res2.json()["skipped_duplicates"]) == 1

    def test_multi_plant_batch_strict_isolation(self, client):
        """Verify multi-plant batch submission strictly segregates media across plant boundaries."""
        img_a1 = _make_dummy_image_bytes(color=(0, 200, 0), text="A1")
        img_a2 = _make_dummy_image_bytes(color=(0, 220, 0), text="A2")
        img_b1 = _make_dummy_image_bytes(color=(200, 0, 0), text="B1")

        manifest = json.dumps({
            "plantA_front.jpg": {"plant_id": "PLANT-ALPHA", "view_label": "front", "crop_type": "tomato"},
            "plantA_back.jpg": {"plant_id": "PLANT-ALPHA", "view_label": "back", "crop_type": "tomato"},
            "plantB_top.jpg": {"plant_id": "PLANT-BETA", "view_label": "top", "crop_type": "tomato"},
        })

        files = [
            ("files", ("plantA_front.jpg", io.BytesIO(img_a1), "image/jpeg")),
            ("files", ("plantA_back.jpg", io.BytesIO(img_a2), "image/jpeg")),
            ("files", ("plantB_top.jpg", io.BytesIO(img_b1), "image/jpeg")),
        ]

        res = client.post("/api/plant-sessions/batch", files=files, data={"batch_manifest": manifest})
        assert res.status_code == 201
        data = res.json()
        assert data["batch_count"] == 2

        sessions = data["sessions"]
        pids = {s["plant_id"] for s in sessions}
        assert "PLANT-ALPHA" in pids
        assert "PLANT-BETA" in pids

        # Verify isolation in database
        sess_a_id = next(s["session_id"] for s in sessions if s["plant_id"] == "PLANT-ALPHA")
        sess_b_id = next(s["session_id"] for s in sessions if s["plant_id"] == "PLANT-BETA")

        sess_a = db_manager.get_plant_session(sess_a_id)
        sess_b = db_manager.get_plant_session(sess_b_id)

        # Plant ALPHA has exactly 2 clips; Plant BETA has exactly 1 clip
        assert len(sess_a["clips"]) == 2
        assert len(sess_b["clips"]) == 1

        # Check filenames are strictly segregated
        a_files = {c["original_filename"] for c in sess_a["clips"]}
        b_files = {c["original_filename"] for c in sess_b["clips"]}
        assert a_files == {"plantA_front.jpg", "plantA_back.jpg"}
        assert b_files == {"plantB_top.jpg"}

    def test_report_export_endpoints(self, client):
        """Verify exporting session reports in JSON and Markdown formats."""
        # 1. Create and populate session
        res_create = client.post("/api/plant-sessions", json={"plant_id": "PLANT-EXPORT-001", "crop_type": "tomato"})
        session_id = res_create.json()["session_id"]

        img = _make_dummy_image_bytes(color=(10, 180, 10), text="leaf")
        files = [("files", ("leaf.jpg", io.BytesIO(img), "image/jpeg"))]
        client.post(f"/api/plant-sessions/{session_id}/media/batch", files=files)

        # 2. Test JSON export
        res_json = client.get(f"/api/plant-sessions/{session_id}/export?format=json")
        assert res_json.status_code == 200
        json_data = res_json.json()
        assert json_data["export_format"] == "json"
        assert "session" in json_data
        assert json_data["session"]["plant_id"] == "PLANT-EXPORT-001"

        # 3. Test Markdown export
        res_md = client.get(f"/api/plant-sessions/{session_id}/export?format=markdown")
        assert res_md.status_code == 200
        assert res_md.headers["content-type"].startswith("text/markdown")
        md_text = res_md.text
        assert "Plant Health Decision-Support Report: PLANT-EXPORT-001" in md_text
        assert "Model Provenance" in md_text
        assert "Important Notice" in md_text
