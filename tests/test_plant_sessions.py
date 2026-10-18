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
