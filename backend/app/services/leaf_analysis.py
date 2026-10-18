"""
backend/app/services/leaf_analysis.py

Phase 4 — Leaf Detection, Segmentation & Mask-Aware Lesion Estimation

Design principles:
  - Modular interface: swappable between heuristic (GrabCut/contour) and
    deep-learning segmentation backends.
  - If no trained leaf-segmentation model is present, falls back to a
    conservative HSV-based foliage mask and marks availability clearly.
  - Lesion estimation is ALWAYS restricted to the leaf region (never whole frame).
  - Returns structured LeafObservation dicts with explicit availability flags.
  - Never fabricates masks or scores when data is insufficient.
  - Distinguishes pixel-area-based estimates from validated biological severity.

Status taxonomy for leaf_segmentation_status:
  AVAILABLE       — mask obtained from a loaded model
  HEURISTIC       — mask from HSV/GrabCut fallback (lower reliability)
  UNAVAILABLE     — no mask could be computed (tiny/corrupt frame)
  MODEL_MISSING   — segmentation model weights not loaded
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ---------------------------------------------------------------------------
# Leaf segmentation model — optional deep-learning backend
# ---------------------------------------------------------------------------
_SEG_MODEL = None
_SEG_STATUS = "MODEL_MISSING"  # Updated if weights are loaded


def load_leaf_segmentation_model(weights_path: Optional[str] = None) -> str:
    """
    Attempt to load a YOLO instance-segmentation model for leaf detection.
    If no weights are provided or file is absent, falls back to heuristic mode.

    Returns the resulting _SEG_STATUS string.
    """
    global _SEG_MODEL, _SEG_STATUS

    if weights_path is None:
        _SEG_STATUS = "MODEL_MISSING"
        return _SEG_STATUS

    from pathlib import Path
    p = Path(weights_path)
    if not p.exists():
        _SEG_STATUS = "MODEL_MISSING"
        return _SEG_STATUS

    try:
        from ultralytics import YOLO
        _SEG_MODEL = YOLO(str(p))
        _SEG_STATUS = "AVAILABLE"
        print(f"[LeafAnalysis] Leaf segmentation model loaded: {p}")
    except Exception as exc:
        _SEG_MODEL = None
        _SEG_STATUS = "MODEL_MISSING"
        print(f"[LeafAnalysis] Segmentation model load failed: {exc}")

    return _SEG_STATUS


def get_leaf_segmentation_status() -> Dict[str, Any]:
    """Return current leaf segmentation model availability."""
    return {
        "leaf_segmentation_status": _SEG_STATUS,
        "model_loaded": _SEG_STATUS == "AVAILABLE",
        "note": (
            "Leaf segmentation uses a YOLO instance-segmentation model. "
            "When model weights are absent, a heuristic HSV-based foliage mask is used. "
            "Heuristic masks are less precise and may include background, shadows, or adjacent objects."
        ) if _SEG_STATUS != "AVAILABLE" else
        "Leaf segmentation model is loaded and active.",
    }


# ---------------------------------------------------------------------------
# Heuristic foliage mask (HSV-based fallback)
# ---------------------------------------------------------------------------

def _heuristic_leaf_mask(bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Approximate leaf region using HSV green-range thresholding.
    Returns (binary_mask, 'HEURISTIC').
    Caveats: may include greenish backgrounds, shadows, or other plants nearby.
    """
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Broad green / yellowish-green foliage range
    lower = np.array([20, 25, 20])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological clean-up to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask, "HEURISTIC"


def _deep_leaf_mask(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Run YOLO instance-segmentation model to obtain leaf masks.
    Returns (combined_binary_mask, 'AVAILABLE') or (None, 'UNAVAILABLE').
    """
    if _SEG_MODEL is None:
        return None, "MODEL_MISSING"

    try:
        import cv2
        import numpy as np
        results = _SEG_MODEL.predict(bgr, verbose=False, conf=0.35)
        h, w = bgr.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)

        for result in results:
            if result.masks is None:
                continue
            for seg_mask in result.masks.data:
                seg_np = seg_mask.cpu().numpy()
                seg_resized = cv2.resize(
                    (seg_np * 255).astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
                combined = cv2.bitwise_or(combined, seg_resized)

        if np.count_nonzero(combined) < 50:
            return None, "UNAVAILABLE"

        return combined, "AVAILABLE"
    except Exception as exc:
        print(f"[LeafAnalysis] Segmentation inference error: {exc}")
        return None, "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Core per-frame leaf analysis
# ---------------------------------------------------------------------------

def analyze_frame_with_mask(
    frame_bgr: np.ndarray,
    clip_id: str = "",
    frame_index: int = 0,
    timestamp_sec: float = 0.0,
    view_label: str = "unknown",
) -> Dict[str, Any]:
    """
    Analyse one video frame:
      1. Obtain a leaf mask (deep model → heuristic fallback).
      2. Compute mask-aware lesion percentage (restricted to leaf region).
      3. Return structured observation with explicit availability flags.

    IMPORTANT: Never reports lesion estimates on the whole frame.
    Always documents which method produced the mask.
    """
    import cv2  # lazy import — cv2 may not be installed in all test environments
    import numpy as np

    if frame_bgr is None or frame_bgr.size == 0:
        return _empty_observation(clip_id, frame_index, timestamp_sec, view_label,
                                  reason="Empty or corrupt frame")

    h, w = frame_bgr.shape[:2]
    if h < 32 or w < 32:
        return _empty_observation(clip_id, frame_index, timestamp_sec, view_label,
                                  reason=f"Frame too small ({w}x{h})")

    # Step 1: Get leaf mask
    if _SEG_STATUS == "AVAILABLE":
        mask, mask_method = _deep_leaf_mask(frame_bgr)
        if mask is None:
            mask, mask_method = _heuristic_leaf_mask(frame_bgr)
    else:
        mask, mask_method = _heuristic_leaf_mask(frame_bgr)

    leaf_pixel_count = int(np.count_nonzero(mask))
    total_pixels = h * w

    if leaf_pixel_count < 100:
        # Mask too small to be meaningful
        return _empty_observation(clip_id, frame_index, timestamp_sec, view_label,
                                  reason="Leaf mask too small or absent")

    # Step 2: Mask-aware lesion estimation
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Lesion colors (brown necrosis + yellow chlorosis) — only within leaf mask
    lower_brown = np.array([0, 30, 20])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    lower_yellow = np.array([15, 100, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    symptom_raw = cv2.bitwise_or(brown_mask, yellow_mask)

    # Restrict to leaf region
    symptom_in_leaf = cv2.bitwise_and(symptom_raw, symptom_raw, mask=mask)
    symptom_pixels = int(np.count_nonzero(symptom_in_leaf))

    lesion_pct = round(min(100.0, (symptom_pixels / leaf_pixel_count) * 100.0), 2)

    # Step 3: Bounding box of the largest leaf contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    largest_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox = [int(x), int(y), int(x + bw), int(y + bh)]

    # Leaf coverage fraction
    leaf_coverage_fraction = round(leaf_pixel_count / total_pixels, 4)

    return {
        "clip_id": clip_id,
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 3),
        "view_label": view_label,
        # Mask info
        "leaf_mask_available": True,
        "leaf_segmentation_method": mask_method,
        "leaf_pixel_count": leaf_pixel_count,
        "leaf_coverage_fraction": leaf_coverage_fraction,
        # Lesion estimate — mask-restricted
        "symptom_pixel_count": symptom_pixels,
        "visible_symptom_area_percent": lesion_pct,
        "symptom_area_note": (
            "visible_symptom_area_percent is computed within the detected leaf region using HSV "
            "colour thresholds. It is a heuristic visual indicator, NOT a validated disease-severity score. "
            "Lighting, leaf age, cultivar variation, and camera white-balance affect this value."
        ),
        # Leaf bounding box (largest detected contour)
        "leaf_bbox": bbox,
        "leaf_contour_area_px": int(largest_area),
        # Quality flags
        "frame_shape": [h, w],
        "quality_flags": _quality_flags(frame_bgr, leaf_coverage_fraction, mask_method),
    }


def _quality_flags(
    bgr: np.ndarray,
    leaf_coverage: float,
    mask_method: str,
) -> List[str]:
    """Return a list of human-readable quality warnings for this frame."""
    flags = []
    if mask_method == "HEURISTIC":
        flags.append("leaf_mask_heuristic: mask is approximate — may include background pixels")
    if leaf_coverage < 0.05:
        flags.append("low_leaf_coverage: leaf occupies <5% of frame — estimate unreliable")
    if leaf_coverage > 0.95:
        flags.append("full_frame_leaf: entire frame may be leaf or background is absent")

    # Check exposure: mean brightness
    import cv2
    import numpy as np
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 30:
        flags.append("under_exposed: very dark frame — lesion detection may be unreliable")
    elif mean_brightness > 230:
        flags.append("over_exposed: very bright frame — colour-based lesion detection unreliable")

    return flags


def _empty_observation(
    clip_id: str,
    frame_index: int,
    timestamp_sec: float,
    view_label: str,
    reason: str = "Unknown",
) -> Dict[str, Any]:
    """Return a structured 'no data' observation with the reason documented."""
    return {
        "clip_id": clip_id,
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 3),
        "view_label": view_label,
        "leaf_mask_available": False,
        "leaf_segmentation_method": "UNAVAILABLE",
        "leaf_pixel_count": 0,
        "leaf_coverage_fraction": 0.0,
        "symptom_pixel_count": 0,
        "visible_symptom_area_percent": None,
        "symptom_area_note": f"Leaf analysis unavailable: {reason}",
        "leaf_bbox": None,
        "leaf_contour_area_px": 0,
        "frame_shape": [],
        "quality_flags": [f"analysis_unavailable: {reason}"],
    }


# ---------------------------------------------------------------------------
# Batch frame analysis convenience wrapper
# ---------------------------------------------------------------------------

def analyze_frames_batch(
    frames: List[Tuple[np.ndarray, str, int, float, str]],
) -> List[Dict[str, Any]]:
    """
    Batch-process multiple frames.

    Each element of 'frames' is a tuple:
        (frame_bgr, clip_id, frame_index, timestamp_sec, view_label)

    Returns a list of leaf observation dicts.
    """
    return [
        analyze_frame_with_mask(
            frame_bgr=f[0],
            clip_id=f[1],
            frame_index=f[2],
            timestamp_sec=f[3],
            view_label=f[4],
        )
        for f in frames
    ]
