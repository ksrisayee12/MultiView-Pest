"""
backend/app/services/video_processor.py

Phase 2 — Video Ingestion Service

Handles video validation, frame extraction, sampling, duplicate suppression,
and temporary file management for multi-view plant analysis sessions.

Design principles:
  - Uses OpenCV VideoCapture (already a project dependency via cv2)
  - Streams frames — does NOT load the entire video into RAM
  - Configurable sampling rate (default: 2 fps)
  - Near-duplicate frame suppression via pixel-difference threshold
  - Releases VideoCapture on success AND on error
  - Returns structured metadata per extracted frame
  - Rejects corrupt, empty, or unsupported videos clearly
"""

import os
import uuid
import time
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_SAMPLE_FPS: float = 2.0       # extract up to N frames per second
_MAX_VIDEO_BYTES: int = 500 * 1024 * 1024   # 500 MB upload limit
_MAX_VIDEO_DURATION_SEC: float = 300.0      # 5 minutes max
_MAX_FRAMES_PER_CLIP: int = 600             # safety cap
_DUPLICATE_DIFF_THRESHOLD: float = 4.0     # mean absolute pixel diff (0-255)
_MIN_FRAME_DIM: int = 64                    # reject tiny frames

_ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "video/webm", "video/x-matroska",
    "video/3gpp", "image/jpeg", "image/png", "image/webp", "image/jpg",
}
_ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp", ".m4v",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
}

# Temp directory for uploaded clips
_UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))


# ---------------------------------------------------------------------------
# Data classes (plain dicts to avoid pydantic import here)
# ---------------------------------------------------------------------------

def make_clip_meta(
    clip_id: str,
    session_id: str,
    original_filename: str,
    view_label: str,
    file_path: str,
) -> Dict[str, Any]:
    """Create initial clip metadata dict."""
    return {
        "clip_id": clip_id,
        "session_id": session_id,
        "original_filename": original_filename,
        "view_label": view_label,
        "file_path": file_path,
        "upload_status": "UPLOADED",
        "processing_status": "PENDING",
        "video_valid": None,
        "error_message": None,
        "fps": None,
        "duration_seconds": None,
        "total_video_frames": None,
        "frames_extracted": 0,
        "frames_skipped_duplicate": 0,
        "sample_fps": _DEFAULT_SAMPLE_FPS,
        "created_at": time.time(),
        "processed_at": None,
    }


def make_frame_observation(
    clip_id: str,
    frame_index: int,
    timestamp_sec: float,
    view_label: str,
    frame_bgr: np.ndarray,
    crop_type: str = "tomato",
) -> Dict[str, Any]:
    """Create a frame observation dict — runs DIP, dual-backbone AI, and foliar perforation detection."""
    from backend.image_processing.visual_health import analyze_visual_health
    from backend.app.services.model_registry import predict_disease
    from backend.app.services.image_preflight_and_segmentation import detect_leaf_perforations_and_pest_damage

    dip = analyze_visual_health(frame_bgr)
    yolo = predict_disease(frame_bgr)
    
    pred = yolo["prediction"]
    conf = float(yolo["confidence"])
    
    # Run physical insect feeding hole and perforation detection on the frame
    is_pest_chewed, hole_boxes, hole_defol_pct = detect_leaf_perforations_and_pest_damage(frame_bgr)
    
    # Priority A: Pest chewing override if physical feeding perforations are verified
    known_fungal = {
        "early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot",
        "pepper_bacterial_spot", "potato_early_blight", "potato_late_blight",
        "leaf_mold", "spider_mites", "target_spot", "mosaic_virus", "yellow_leaf_curl"
    }
    is_confident_disease = (pred in known_fungal) and (conf >= 0.40)
    
    if is_pest_chewed and (not is_confident_disease or hole_defol_pct >= 25.0):
        pred = "pest_chewing_damage"
        conf = 0.88

    # Priority B: Crop-aware taxonomy adaptation
    c_str = (crop_type or "tomato").lower()
    if any(k in c_str for k in ("pepper", "bell_pepper", "chilli", "capsicum")):
        if "bacterial" in pred or pred in ("bacterial_spot", "pepper_bacterial_spot"):
            pred = "pepper_bacterial_spot"
        elif pred == "healthy":
            pred = "pepper_healthy"
    elif "potato" in c_str:
        if "early" in pred or pred == "early_blight":
            pred = "potato_early_blight"
        elif "late" in pred or pred == "late_blight":
            pred = "potato_late_blight"
        elif pred == "healthy":
            pred = "potato_healthy"

    return {
        "clip_id": clip_id,
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 3),
        "view_label": view_label,
        "yolo_prediction": pred,
        "yolo_confidence": conf,
        "yolo_model_status": yolo.get("model_status", "UNKNOWN"),
        "model_loaded": yolo["model_loaded"],
        "opencv_visual_status": dip["visual_status"],
        "greenness_index": dip["greenness_index"],
        "green_pixel_percentage": dip["green_pixel_percentage"],
        "lesion_percentage": max(dip["lesion_percentage"], hole_defol_pct if is_pest_chewed else 0.0),
        "is_pest_chewed": is_pest_chewed,
        "hole_boxes_count": len(hole_boxes),
        "defoliation_pct": hole_defol_pct,
        "frame_shape": list(frame_bgr.shape),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_video_upload(
    filename: str,
    content_type: Optional[str],
    file_size_bytes: int,
) -> Optional[str]:
    """
    Validate an uploaded video before saving. Returns an error string or None.
    Does NOT read the video stream — just checks metadata.
    """
    if file_size_bytes > _MAX_VIDEO_BYTES:
        mb = _MAX_VIDEO_BYTES // (1024 * 1024)
        return f"Video too large ({file_size_bytes // (1024*1024)} MB). Maximum is {mb} MB."

    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        return (
            f"Unsupported file extension '{ext}'. "
            f"Accepted: {', '.join(sorted(_ALLOWED_VIDEO_EXTENSIONS))}."
        )

    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        # Some browsers send generic binary type — allow it if extension is ok
        if ct not in _ALLOWED_VIDEO_TYPES and ct != "application/octet-stream":
            return f"Unsupported content type '{ct}'."

    return None


def validate_video_stream(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Open the saved video/image with OpenCV and validate it is readable.
    Returns (is_valid, error_message, video_info_dict).
    """
    ext = Path(file_path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        img = cv2.imread(file_path)
        if img is None or img.size == 0:
            return False, "OpenCV could not open the image file.", {}
        h, w = img.shape[:2]
        if min(h, w) < _MIN_FRAME_DIM:
            return False, f"Image resolution too small ({w}x{h}). Minimum: {_MIN_FRAME_DIM}px.", {}
        return True, "", {
            "fps": 1.0,
            "total_frames": 1,
            "duration_seconds": 1.0,
            "width": w,
            "height": h,
            "codec": "image",
            "is_image": True,
        }

    cap = None
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "OpenCV could not open the video file (corrupt or unsupported codec).", {}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 25.0

        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Try reading the first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, "Video appears to be empty (could not read any frames).", {}

        if frame.shape[0] < _MIN_FRAME_DIM or frame.shape[1] < _MIN_FRAME_DIM:
            return (
                False,
                f"Video frames too small ({width}×{height}). Minimum is {_MIN_FRAME_DIM}px.",
                {},
            )

        if duration_sec > _MAX_VIDEO_DURATION_SEC:
            return (
                False,
                f"Video too long ({duration_sec:.0f}s). Maximum is {_MAX_VIDEO_DURATION_SEC:.0f}s.",
                {},
            )

        return True, "", {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration_sec, 2),
            "width": width,
            "height": height,
        }
    except Exception as exc:
        return False, f"Error reading video: {exc}", {}
    finally:
        if cap is not None:
            cap.release()


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames_from_clip(
    clip_id: str,
    file_path: str,
    view_label: str,
    sample_fps: float = _DEFAULT_SAMPLE_FPS,
    progress_callback=None,
    crop_type: str = "tomato",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract sampled frames from a video clip and run analysis on each.

    Args:
        clip_id:           Unique clip identifier.
        file_path:         Path to the saved video file.
        view_label:        User-assigned view label (front/back/etc.).
        sample_fps:        Target frames per second to extract.
        progress_callback: Optional callable(processed, total) for progress updates.
        crop_type:         Crop type identifier for crop-specific diagnostic routing.

    Returns:
        (observations, stats) where:
          observations — list of frame observation dicts
          stats        — dict with frames_extracted, frames_skipped, timing
    """
    start_time = time.time()
    observations: List[Dict[str, Any]] = []
    frames_extracted = 0
    frames_skipped_dup = 0
    prev_gray: Optional[np.ndarray] = None

    ext = Path(file_path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        img = cv2.imread(file_path)
        if img is None:
            return [], {"error": "Cannot open image file", "frames_extracted": 0}
        obs = make_frame_observation(
            clip_id=clip_id,
            frame_index=0,
            timestamp_sec=0.0,
            view_label=view_label,
            frame_bgr=img,
            crop_type=crop_type,
        )
        return [obs], {
            "frames_extracted": 1,
            "frames_skipped_duplicate": 0,
            "total_video_frames_read": 1,
            "sample_fps": 1.0,
            "processing_time_sec": round(time.time() - start_time, 2),
        }

    cap = None
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return [], {"error": "Cannot open video", "frames_extracted": 0}

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if video_fps <= 0:
            video_fps = 25.0

        # Sample interval: every N frames to achieve target sample_fps
        sample_interval = max(1, int(video_fps / sample_fps))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Only process frames at sample interval
            if frame_idx % sample_interval == 0:
                # Duplicate detection — compare mean pixel diff with previous
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))
                    if diff < _DUPLICATE_DIFF_THRESHOLD:
                        frames_skipped_dup += 1
                        frame_idx += 1
                        continue

                prev_gray = gray
                timestamp_sec = frame_idx / video_fps

                obs = make_frame_observation(
                    clip_id=clip_id,
                    frame_index=frame_idx,
                    timestamp_sec=timestamp_sec,
                    view_label=view_label,
                    frame_bgr=frame,
                    crop_type=crop_type,
                )
                observations.append(obs)
                frames_extracted += 1

                if progress_callback:
                    progress_callback(frames_extracted, max(1, total_frames // sample_interval))

                if frames_extracted >= _MAX_FRAMES_PER_CLIP:
                    break

            frame_idx += 1

        elapsed = round(time.time() - start_time, 2)
        return observations, {
            "frames_extracted": frames_extracted,
            "frames_skipped_duplicate": frames_skipped_dup,
            "total_video_frames_read": frame_idx,
            "sample_fps": sample_fps,
            "processing_time_sec": elapsed,
        }

    except Exception as exc:
        return [], {"error": str(exc), "frames_extracted": frames_extracted}
    finally:
        if cap is not None:
            cap.release()


def process_clip(
    clip_meta: Dict[str, Any],
    sample_fps: float = _DEFAULT_SAMPLE_FPS,
    progress_callback=None,
    crop_type: str = "tomato",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process a clip metadata dict: extracts frames, runs DIP & YOLO (singleton),
    suppresses duplicates, and returns (observations, updated_clip_meta).
    """
    clip_id = clip_meta["clip_id"]
    file_path = clip_meta.get("file_path", "")
    view_label = clip_meta.get("view_label", "unknown")
    effective_crop = crop_type or clip_meta.get("crop_type", "tomato")

    observations, stats = extract_frames_from_clip(
        clip_id=clip_id,
        file_path=file_path,
        view_label=view_label,
        sample_fps=sample_fps,
        progress_callback=progress_callback,
        crop_type=effective_crop,
    )

    updated_meta = dict(clip_meta)
    if "error" in stats:
        updated_meta["processing_status"] = "ERROR"
        updated_meta["error_message"] = stats["error"]
    else:
        updated_meta["processing_status"] = "COMPLETED"
        updated_meta["frames_extracted"] = stats.get("frames_extracted", 0)
        updated_meta["frames_skipped_duplicate"] = stats.get("frames_skipped_duplicate", 0)
        updated_meta["sample_fps"] = stats.get("sample_fps", sample_fps)
        updated_meta["processed_at"] = time.time()

    return observations, updated_meta



# ---------------------------------------------------------------------------
# Temp file management
# ---------------------------------------------------------------------------

def save_upload_to_temp(
    content: bytes,
    original_filename: str,
    session_id: str,
    clip_id: str,
) -> str:
    """
    Save uploaded video bytes to a safe temporary path.
    Returns the absolute path to the saved file.
    Raises ValueError if the path cannot be created.
    """
    ext = Path(original_filename).suffix.lower() or ".mp4"
    dest_dir = _UPLOAD_DIR / session_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Safe filename: never use the uploaded filename directly
    safe_name = f"{clip_id}{ext}"
    dest_path = dest_dir / safe_name
    dest_path.write_bytes(content)
    return str(dest_path)


def cleanup_clip_file(file_path: str) -> None:
    """Remove a single uploaded clip file if it exists."""
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
    except Exception as exc:
        print(f"[VideoProcessor] Cleanup warning: {exc}")


def cleanup_session_files(session_id: str) -> None:
    """Remove all uploaded files for a session."""
    session_dir = _UPLOAD_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
