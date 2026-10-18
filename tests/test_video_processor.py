"""
tests/test_video_processor.py

Phase 2 Unit Tests — Video Ingestion & Frame Extraction
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from backend.app.services.video_processor import (
    validate_video_upload,
    make_clip_meta,
    extract_frames_from_clip,
    process_clip,
    cleanup_clip_file,
    _MAX_VIDEO_BYTES,
)


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a short synthetic MP4 video file with OpenCV VideoWriter."""
    vpath = str(tmp_path / "test_plant_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 320x240, 10 fps, 10 frames = 1 second
    out = cv2.VideoWriter(vpath, fourcc, 10.0, (320, 240))

    # Generate 10 green plant-like frames
    for i in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Green foliage rectangle
        cv2.rectangle(frame, (50 + i * 2, 50), (270, 190), (35, 160 + i * 3, 40), -1)
        out.write(frame)
    out.release()
    yield vpath
    cleanup_clip_file(vpath)


def test_validate_video_upload_valid():
    err = validate_video_upload("clip.mp4", "video/mp4", 1024 * 1024)
    assert err is None


def test_validate_video_upload_oversized():
    err = validate_video_upload("clip.mp4", "video/mp4", _MAX_VIDEO_BYTES + 1000)
    assert err is not None
    assert "too large" in err.lower()


def test_validate_video_upload_invalid_extension():
    err = validate_video_upload("document.pdf", "application/pdf", 1024)
    assert err is not None
    assert "unsupported" in err.lower()


def test_make_clip_meta():
    meta = make_clip_meta(
        clip_id="c123",
        session_id="s456",
        original_filename="front.mp4",
        view_label="front",
        file_path="/tmp/front.mp4",
    )
    assert meta["clip_id"] == "c123"
    assert meta["session_id"] == "s456"
    assert meta["view_label"] == "front"
    assert meta["processing_status"] == "PENDING"
    assert meta["upload_status"] == "UPLOADED"


def test_extract_frames_from_synthetic_video(sample_video_path):
    obs, stats = extract_frames_from_clip(
        clip_id="clip_test_1",
        file_path=sample_video_path,
        view_label="front",
        sample_fps=5.0,
    )
    assert stats["frames_extracted"] > 0
    assert len(obs) == stats["frames_extracted"]
    first = obs[0]
    assert first["clip_id"] == "clip_test_1"
    assert first["view_label"] == "front"
    assert "timestamp_sec" in first
    assert "opencv_visual_status" in first
    assert "yolo_prediction" in first


def test_process_clip_wrapper(sample_video_path):
    clip_meta = make_clip_meta("c_wrap", "s_wrap", "clip.mp4", "top", sample_video_path)
    obs, updated = process_clip(clip_meta, sample_fps=4.0)

    assert updated["processing_status"] == "COMPLETED"
    assert updated["frames_extracted"] > 0
    assert updated["processed_at"] is not None
    assert len(obs) == updated["frames_extracted"]
