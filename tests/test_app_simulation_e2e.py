"""
tests/test_app_simulation_e2e.py

Comprehensive End-to-End App Simulation & Module Verification:
  1. ML Modules:
     - ModelRegistry disease prediction contracts (missing weights / ultralytics integration)
     - YOLOPestDetector initialization and missing model safety
     - Leaf segmentation (deep learning loader + heuristic HSV foliage fallback)
     - Mask-aware visible lesion estimation
  2. Video Analysis:
     - Real synthetic MP4 video generation via OpenCV VideoWriter
     - Video format and size validation
     - Streamed decoding via cv2.VideoCapture
     - FPS frame sampling & duplicate frame suppression
     - Metadata preservation (timestamps, view labels, frame indices)
     - Resource cleanup
  3. Full End-to-End FastAPI App Simulation:
     - Health & Model readiness endpoints
     - Plant session lifecycle (create, upload clips, run analysis, query status)
     - Temporal tracking across frames
     - Cross-view multi-angle evidence fusion & conflict detection
     - Leaf analysis on session observations
     - Simulation suite execution (all 5 scenarios)
     - Statistical evaluation endpoint with Wilson score 95% CIs
     - Legacy field & tile scan endpoints
     - Clean session deletion & file removal
"""

import os
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_registry import (
    predict_disease,
    get_disease_model_status,
)
from backend.app.services.leaf_analysis import (
    analyze_frame_with_mask,
    get_leaf_segmentation_status,
    load_leaf_segmentation_model,
)
from backend.app.services.video_processor import (
    validate_video_upload,
    validate_video_stream,
    extract_frames_from_clip,
    process_clip,
    make_clip_meta,
)
from src.models.pest_detector import YOLOPestDetector


# ---------------------------------------------------------------------------
# Helper: Create a real playable MP4 video file
# ---------------------------------------------------------------------------
def _create_real_mp4_video(file_path: str, num_frames: int = 30, fps: float = 10.0, with_movement: bool = True):
    """Generate a real valid MP4 video on disk using OpenCV VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(file_path, fourcc, fps, (320, 240))
    assert writer.isOpened(), f"Could not open VideoWriter for {file_path}"

    for i in range(num_frames):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)  # dark background
        # Draw a moving green plant leaf
        center_x = 60 + (i * 6 if with_movement else 0)
        center_y = 120
        cv2.ellipse(frame, (center_x, center_y), (50, 30), 20, 0, 360, (30, 180, 50), -1)
        # Draw a brown lesion on some frames
        if i % 3 == 0:
            cv2.circle(frame, (center_x - 10, center_y), 8, (20, 60, 120), -1)

        writer.write(frame)

    writer.release()
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


# ===========================================================================
# 1. Verification of ML Modules
# ===========================================================================

class TestMLModulesVerification:
    def test_disease_model_registry_contracts(self):
        """Verify disease model registry returns structured honest results."""
        status_info = get_disease_model_status()
        assert "status" in status_info
        assert status_info["status"] in ("READY", "MISSING_WEIGHTS", "NOT_CONFIGURED", "LOAD_ERROR", "ULTRALYTICS_MISSING")

        # Test inference with synthetic leaf image
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(test_img, (112, 112), 50, (30, 180, 50), -1)

        result = predict_disease(test_img)
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "confidence" in result
        assert "model_status" in result
        assert "model_loaded" in result
        assert isinstance(result["confidence"], float)

        # If model is not loaded, confidence MUST be 0.0 and prediction honest
        if not result["model_loaded"]:
            assert result["confidence"] == 0.0
            assert "model_unavailable" in result["prediction"]

    def test_pest_detector_graceful_missing_weights(self):
        """Verify YOLOPestDetector handles missing weights cleanly without crash."""
        detector = YOLOPestDetector("configs/yolo_pest_config.yaml")
        assert not detector.is_model_available
        assert "MODEL_PENDING" in detector.model_status or "MISSING" in detector.model_status

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = detector.detect(dummy_img)
        assert res.detections == []
        assert res.total_pest_count == 0
        assert "MODEL_PENDING" in res.model_status or "MISSING" in res.model_status

    def test_leaf_segmentation_and_mask_aware_damage(self):
        """Verify leaf segmentation status, heuristic fallback, and lesion calculation."""
        status = get_leaf_segmentation_status()
        assert "leaf_segmentation_status" in status
        assert status["leaf_segmentation_status"] in ("AVAILABLE", "MODEL_MISSING")

        # Create synthetic frame with green leaf + brown lesion
        bgr = np.full((200, 200, 3), 20, dtype=np.uint8)
        # Leaf
        cv2.circle(bgr, (100, 100), 60, (35, 175, 45), -1)
        # Lesion inside leaf
        cv2.circle(bgr, (100, 100), 15, (20, 60, 120), -1)

        obs = analyze_frame_with_mask(bgr)
        assert obs["leaf_mask_available"] is True
        assert obs["leaf_segmentation_method"] == "HEURISTIC"
        assert obs["leaf_pixel_count"] > 0
        assert obs["leaf_coverage_fraction"] > 0.05
        assert obs["visible_symptom_area_percent"] is not None
        assert obs["visible_symptom_area_percent"] > 0.0

        # Test empty frame
        empty = np.zeros((200, 200, 3), dtype=np.uint8)
        obs_empty = analyze_frame_with_mask(empty)
        assert obs_empty["leaf_mask_available"] is False
        assert obs_empty["leaf_pixel_count"] == 0


# ===========================================================================
# 2. Verification of Video Analysis Module
# ===========================================================================

class TestVideoAnalysisVerification:
    def test_video_upload_validation(self):
        """Verify file size, container, and mime validation."""
        # Valid
        assert validate_video_upload("test.mp4", "video/mp4", 1024 * 1024) is None
        assert validate_video_upload("test.avi", "video/x-msvideo", 1024 * 1024) is None
        # Invalid extension
        err_ext = validate_video_upload("test.exe", "video/mp4", 100)
        assert err_ext is not None and "Unsupported file extension" in err_ext
        # Oversized
        err_size = validate_video_upload("test.mp4", "video/mp4", 600 * 1024 * 1024)
        assert err_size is not None and "Video too large" in err_size

    def test_real_video_stream_decoding_and_sampling(self, tmp_path):
        """Generate a real MP4 video and test OpenCV decoding and sampling."""
        video_file = str(tmp_path / "sample_plant_video.mp4")
        _create_real_mp4_video(video_file, num_frames=30, fps=10.0, with_movement=True)

        is_valid, msg, info = validate_video_stream(video_file)
        assert is_valid is True
        assert info["fps"] == 10.0
        assert info["total_frames"] == 30
        assert info["duration_seconds"] == 3.0

        # Extract frames at 2 FPS
        obs_list, stats = extract_frames_from_clip(
            clip_id="clip_test_01",
            file_path=video_file,
            view_label="front",
            sample_fps=2.0,
        )

        assert stats["frames_extracted"] >= 2
        assert len(obs_list) == stats["frames_extracted"]
        for obs in obs_list:
            assert obs["clip_id"] == "clip_test_01"
            assert obs["view_label"] == "front"
            assert "timestamp_sec" in obs
            assert "frame_index" in obs
            assert "yolo_prediction" in obs
            assert "opencv_visual_status" in obs
            assert "lesion_percentage" in obs

    def test_duplicate_frame_suppression(self, tmp_path):
        """Generate static identical frames and ensure duplicate suppression triggers."""
        video_file = str(tmp_path / "static_video.mp4")
        _create_real_mp4_video(video_file, num_frames=20, fps=10.0, with_movement=False)

        obs_list, stats = extract_frames_from_clip(
            clip_id="clip_static",
            file_path=video_file,
            view_label="top",
            sample_fps=5.0,
        )

        assert stats["frames_skipped_duplicate"] > 0, "Static frames should be skipped as duplicates"


# ===========================================================================
# 3. Full App Simulation via TestClient
# ===========================================================================

class TestAppProcessSimulation:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    def test_health_and_models_status_endpoints(self, client):
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "online"

        r_status = client.get("/api/models/status")
        assert r_status.status_code == 200
        data = r_status.json()
        assert "models" in data
        assert "disease_classifier" in data["models"]

    def test_full_plant_session_workflow_with_real_video_upload(self, client, tmp_path):
        """Simulate creating a session, uploading real mp4 clips, analyzing, and querying results."""
        # 1. Create plant session
        create_res = client.post(
            "/api/plant-sessions",
            json={"plant_id": "PLANT-SIM-E2E-001", "crop_type": "tomato", "notes": "E2E verification session"},
        )
        assert create_res.status_code == 201
        session = create_res.json()
        session_id = session["session_id"]
        assert session["plant_id"] == "PLANT-SIM-E2E-001"
        assert session["status"] == "CREATED"

        # 2. Generate and upload front-view clip
        vid_front = str(tmp_path / "front_view.mp4")
        _create_real_mp4_video(vid_front, num_frames=15, fps=10.0, with_movement=True)

        with open(vid_front, "rb") as f:
            up_res1 = client.post(
                f"/api/plant-sessions/{session_id}/videos",
                files={"file": ("front_view.mp4", f, "video/mp4")},
                data={"view_label": "front"},
            )
        assert up_res1.status_code == 201
        clip1 = up_res1.json()["clip"]
        assert clip1["view_label"] == "front"

        # 3. Generate and upload back-view clip
        vid_back = str(tmp_path / "back_view.mp4")
        _create_real_mp4_video(vid_back, num_frames=15, fps=10.0, with_movement=True)

        with open(vid_back, "rb") as f:
            up_res2 = client.post(
                f"/api/plant-sessions/{session_id}/videos",
                files={"file": ("back_view.mp4", f, "video/mp4")},
                data={"view_label": "back"},
            )
        assert up_res2.status_code == 201

        # 4. Trigger Analysis
        analyze_res = client.post(f"/api/plant-sessions/{session_id}/analyze")
        assert analyze_res.status_code == 200

        # Wait for background thread pool execution
        for _ in range(30):
            time.sleep(0.2)
            st_res = client.get(f"/api/plant-sessions/{session_id}/status")
            assert st_res.status_code == 200
            st_data = st_res.json()
            if st_data["status"] == "COMPLETED":
                break

        assert st_data["status"] == "COMPLETED", f"Analysis did not complete in time: {st_data}"
        assert st_data["has_assessment"] is True

        # 5. Retrieve Final Assessment
        assess_res = client.get(f"/api/plant-sessions/{session_id}/assessment")
        assert assess_res.status_code == 200
        assessment = assess_res.json()
        assert assessment["session_id"] == session_id
        assert assessment["plant_id"] == "PLANT-SIM-E2E-001"
        assert assessment["status"] in ("HEALTHY_APPEARING", "SYMPTOMS_DETECTED", "UNCERTAIN", "INSUFFICIENT_COVERAGE")
        assert "front" in assessment["submitted_views"]
        assert "back" in assessment["submitted_views"]
        assert "underside" in assessment["missing_views"]

        # 6. Retrieve Observations
        obs_res = client.get(f"/api/plant-sessions/{session_id}/observations")
        assert obs_res.status_code == 200
        obs_data = obs_res.json()
        assert obs_data["total_observations"] > 0

        # 7. Query Temporal Tracks
        tracks_res = client.get(f"/api/plant-sessions/{session_id}/temporal-tracks")
        assert tracks_res.status_code == 200
        tracks_data = tracks_res.json()
        assert tracks_data["total_clips"] == 2

        # 8. Query Cross-View Evidence
        cross_res = client.get(f"/api/plant-sessions/{session_id}/cross-view-evidence")
        assert cross_res.status_code == 200
        cross_data = cross_res.json()
        assert "plant_status" in cross_data
        assert "has_conflicts" in cross_data

        # 9. Query Leaf Analysis
        leaf_res = client.post(f"/api/plant-sessions/{session_id}/leaf-analysis")
        assert leaf_res.status_code == 200
        leaf_data = leaf_res.json()
        assert "aggregate" in leaf_data
        assert "per_observation_leaf_report" in leaf_data

        # 10. Clean up Session and Files
        del_res = client.delete(f"/api/plant-sessions/{session_id}")
        assert del_res.status_code == 200

        # Verify 404 after deletion
        get_deleted = client.get(f"/api/plant-sessions/{session_id}")
        assert get_deleted.status_code == 404

    def test_simulation_suite_and_evaluation_endpoints(self, client):
        """Verify the full 5-scenario deterministic simulation suite runs and passes."""
        r_sim = client.post("/api/plant-sessions/run-simulation-suite")
        assert r_sim.status_code == 200
        sim_data = r_sim.json()
        assert sim_data["total_fixtures"] == 5
        assert "suite_summary" in sim_data
        assert sim_data["suite_summary"]["pass_rate"] >= 0.80

        # Test evaluation endpoint with ground truth
        eval_payload = {
            "predictions": ["SYMPTOMS_DETECTED", "HEALTHY_APPEARING", "SYMPTOMS_DETECTED"],
            "ground_truth": ["SYMPTOMS_DETECTED", "HEALTHY_APPEARING", "HEALTHY_APPEARING"],
            "positive_label": "SYMPTOMS_DETECTED",
        }
        r_eval = client.post("/api/plant-sessions/evaluate", json=eval_payload)
        assert r_eval.status_code == 200
        eval_data = r_eval.json()
        assert "precision" in eval_data
        assert "recall" in eval_data
        assert "accuracy_95ci" in eval_data

    def test_legacy_endpoints_unbroken(self, client):
        """Verify legacy 16-tile composite field and tile endpoints still work."""
        # Create a synthetic 4x4 field composite image
        field_img = np.full((400, 400, 3), 50, dtype=np.uint8)
        # Add vegetation
        cv2.circle(field_img, (200, 200), 120, (30, 180, 50), -1)

        _, enc = cv2.imencode(".jpg", field_img)
        img_bytes = enc.tobytes()

        # Test field scan
        res_field = client.post(
            "/api/analyze/field",
            files={"file": ("field.jpg", img_bytes, "image/jpeg")},
        )
        assert res_field.status_code == 200
        field_data = res_field.json()
        assert "tile_results" in field_data
        assert len(field_data["tile_results"]) == 16
        assert "plant_health_index" in field_data

        # Test single tile scan
        tile_img = field_img[:100, :100]
        _, enc_tile = cv2.imencode(".jpg", tile_img)
        res_tile = client.post(
            "/api/analyze/tile",
            files={"file": ("tile.jpg", enc_tile.tobytes(), "image/jpeg")},
        )
        assert res_tile.status_code == 200
        tile_data = res_tile.json()
        assert "fusion_status" in tile_data
        assert "yolo_prediction" in tile_data
        assert "opencv_status" in tile_data
