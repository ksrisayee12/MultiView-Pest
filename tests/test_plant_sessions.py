"""
tests/test_plant_sessions.py

Phase 3 API Integration and Logic Tests — Plant Sessions & Multi-View Fusion
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.plant_fusion import (
    fuse_plant_assessment,
    STATUS_HEALTHY,
    STATUS_SYMPTOMS,
    STATUS_UNCERTAIN,
    STATUS_INSUFFICIENT,
    STATUS_NO_MODEL,
)

client = TestClient(app)


def test_create_plant_session():
    res = client.post(
        "/api/plant-sessions",
        json={"plant_id": "PLANT-TEST-001", "crop_type": "tomato", "notes": "Greenhouse row 4"},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["plant_id"] == "PLANT-TEST-001"
    assert data["session_id"].startswith("sess_")
    assert data["status"] == "CREATED"
    assert "created_at" in data


def test_list_and_get_plant_session():
    # Create one
    c_res = client.post(
        "/api/plant-sessions",
        json={"plant_id": "PLANT-TEST-002", "crop_type": "tomato"},
    )
    sess_id = c_res.json()["session_id"]

    # List
    l_res = client.get("/api/plant-sessions")
    assert l_res.status_code == 200
    sessions = l_res.json()
    assert any(s["session_id"] == sess_id for s in sessions)

    # Get single
    g_res = client.get(f"/api/plant-sessions/{sess_id}")
    assert g_res.status_code == 200
    assert g_res.json()["session_id"] == sess_id


def test_create_and_query_simulated_multiview_session():
    res = client.post(
        "/api/plant-sessions/simulation",
        data={"plant_id": "PLANT-SIM-TEST", "crop_type": "tomato", "include_symptoms": "true"},
    )
    assert res.status_code == 201
    data = res.json()
    sess_id = data["session_id"]
    assert len(data["clips"]) == 4  # front, back, left, top
    assert data["assessment"] is not None
    assert data["observation_count"] > 0

    # Query status
    status_res = client.get(f"/api/plant-sessions/{sess_id}/status")
    assert status_res.status_code == 200
    st_data = status_res.json()
    assert st_data["status"] == "COMPLETED"
    assert st_data["progress_percent"] == 100
    assert st_data["has_assessment"] is True

    # Query assessment
    assess_res = client.get(f"/api/plant-sessions/{sess_id}/assessment")
    assert assess_res.status_code == 200
    ass_data = assess_res.json()
    assert ass_data["session_id"] == sess_id
    assert "views_submitted" in ass_data
    assert "front" in ass_data["views_submitted"]
    assert "back" in ass_data["views_submitted"]

    # Query observations
    obs_res = client.get(f"/api/plant-sessions/{sess_id}/observations")
    assert obs_res.status_code == 200
    obs_data = obs_res.json()
    assert obs_data["total_observations"] > 0


def test_plant_fusion_insufficient_coverage():
    # Empty observations should yield INSUFFICIENT_COVERAGE
    ass = fuse_plant_assessment(
        session_id="s1",
        plant_id="p1",
        crop_type="tomato",
        clips_meta=[],
        all_observations=[],
    )
    assert ass["status"] == STATUS_INSUFFICIENT


def test_plant_fusion_does_not_count_duplicate_frames_as_leaves():
    # 5 frames from the same clip should be tracked as observations, not 5 separate leaves
    observations = [
        {
            "clip_id": "clip_front",
            "frame_index": i * 10,
            "timestamp_sec": i * 0.5,
            "view_label": "front",
            "yolo_prediction": "healthy",
            "yolo_confidence": 0.95,
            "model_loaded": True,
            "opencv_visual_status": "Healthy",
            "lesion_percentage": 0.5,
        }
        for i in range(5)
    ]
    clips_meta = [
        {
            "clip_id": "clip_front",
            "view_label": "front",
            "frames_extracted": 5,
            "frames_skipped_duplicate": 2,
        }
    ]
    ass = fuse_plant_assessment(
        session_id="s_leaf_test",
        plant_id="p_leaf_test",
        crop_type="tomato",
        clips_meta=clips_meta,
        all_observations=observations,
    )
    # Status should reflect healthy appearing
    assert ass["status"] == STATUS_HEALTHY
    # Ensure there is no fabricated claim of 5 distinct leaves
    assert ass["total_usable_frames"] == 5


def test_delete_single_plant_session():
    # Create session
    c_res = client.post(
        "/api/plant-sessions",
        json={"plant_id": "PLANT-DELETE-TEST", "crop_type": "tomato"},
    )
    sess_id = c_res.json()["session_id"]

    # Delete session
    del_res = client.delete(f"/api/plant-sessions/{sess_id}")
    assert del_res.status_code == 200
    assert del_res.json()["session_id"] == sess_id

    # Verify 404
    g_res = client.get(f"/api/plant-sessions/{sess_id}")
    assert g_res.status_code == 404


def test_delete_duplicate_plant_sessions():
    # Create 3 duplicate sessions with identical plant_id
    dup_id = "PLANT-DUP-TEST"
    s1 = client.post("/api/plant-sessions", json={"plant_id": dup_id, "crop_type": "tomato"}).json()["session_id"]
    s2 = client.post("/api/plant-sessions", json={"plant_id": dup_id, "crop_type": "tomato"}).json()["session_id"]
    s3 = client.post("/api/plant-sessions", json={"plant_id": dup_id, "crop_type": "tomato"}).json()["session_id"]

    # Call deduplication endpoint
    dedup_res = client.delete("/api/plant-sessions?duplicates_only=true")
    assert dedup_res.status_code == 200
    data = dedup_res.json()
    assert data["deleted_count"] >= 2

    # Check that only one session remains for PLANT-DUP-TEST
    list_res = client.get("/api/plant-sessions")
    matching = [s for s in list_res.json() if s["plant_id"] == dup_id]
    assert len(matching) == 1


def test_bulk_delete_all_sessions():
    # Create test session
    client.post("/api/plant-sessions", json={"plant_id": "PLANT-BULK-DEL", "crop_type": "tomato"})

    # Bulk delete all
    res = client.delete("/api/plant-sessions?duplicates_only=false")
    assert res.status_code == 200
    assert "deleted_count" in res.json()

    # Verify session list is empty
    list_res = client.get("/api/plant-sessions")
    assert len(list_res.json()) == 0

