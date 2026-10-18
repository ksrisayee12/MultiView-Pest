"""
backend/app/services/plant_fusion.py

Phase 3 — Plant-Level Multi-View Evidence Fusion Service

Aggregates evidence from multiple video clips (across multiple viewpoints)
for a single plant session, and produces a structured PlantAssessment.

Design principles:
  - Never counts repeated frames as independent leaf observations
  - Aggregates clip-level evidence, then combines across clips
  - Does NOT force cross-view leaf identity — tracks observations, not unique leaves
  - Reports coverage, uncertainty, and model availability honestly
  - Unobserved regions → unknown, NOT healthy
  - Returns structured status taxonomy (see STATUS_* constants)

Status taxonomy:
  HEALTHY_APPEARING      — evidence observed, no symptoms detected above threshold
  SYMPTOMS_DETECTED      — credible disease/symptom evidence found
  UNCERTAIN              — conflicting or insufficient confidence
  INSUFFICIENT_COVERAGE  — too few usable frames/views
  PROCESSING_ERROR       — analysis failed
  MODEL_UNAVAILABLE      — model weights missing
"""

import time
from typing import Any, Dict, List, Optional

# Status constants
STATUS_HEALTHY = "HEALTHY_APPEARING"
STATUS_SYMPTOMS = "SYMPTOMS_DETECTED"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_INSUFFICIENT = "INSUFFICIENT_COVERAGE"
STATUS_ERROR = "PROCESSING_ERROR"
STATUS_NO_MODEL = "MODEL_UNAVAILABLE"

# Thresholds (configurable)
_MIN_USABLE_FRAMES = 3          # need at least this many frames for an assessment
_SYMPTOM_CONF_THRESHOLD = 0.50  # min YOLO confidence to count as symptom evidence
_SYMPTOM_FRACTION_THRESHOLD = 0.25  # fraction of frames with symptoms to declare SYMPTOMS_DETECTED

_KNOWN_DISEASE_CLASSES = {"early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot"}
_HEALTHY_CLASS = "healthy"


def _get_pred_str(obs: Dict[str, Any]) -> str:
    val = obs.get("yolo_prediction")
    if isinstance(val, dict):
        return str(val.get("prediction", "UNKNOWN"))
    return str(val) if val is not None else "UNKNOWN"


def _get_conf_float(obs: Dict[str, Any]) -> float:
    conf = obs.get("yolo_confidence")
    if conf is None and isinstance(obs.get("yolo_prediction"), dict):
        conf = obs["yolo_prediction"].get("confidence")
    try:
        return float(conf) if conf is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def fuse_plant_assessment(
    session_id: str,
    plant_id: str,
    crop_type: str,
    clips_meta: List[Dict[str, Any]],
    all_observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate all frame observations across all clips and produce a
    plant-level assessment.

    Args:
        session_id:        Plant session identifier.
        plant_id:          Plant identifier (user-provided).
        crop_type:         Crop type string.
        clips_meta:        List of clip metadata dicts.
        all_observations:  All frame observation dicts from all clips.

    Returns:
        A plant assessment dict with status, evidence, coverage, and limitations.
    """
    now = time.time()

    if not all_observations:
        return _make_assessment(
            session_id, plant_id, crop_type,
            status=STATUS_INSUFFICIENT,
            limitations=["No frames were successfully extracted from any clip."],
            total_frames=0,
            clips_meta=clips_meta,
        )

    # ---------------------------------------------------------------
    # Step 1: Separate by clip and aggregate
    # ---------------------------------------------------------------
    by_clip: Dict[str, List[Dict]] = {}
    for obs in all_observations:
        cid = obs.get("clip_id", "unknown")
        by_clip.setdefault(cid, []).append(obs)

    clip_summaries = []
    for cid, frames in by_clip.items():
        clip_summaries.append(_summarize_clip(cid, frames))

    # ---------------------------------------------------------------
    # Step 2: Check model availability
    # ---------------------------------------------------------------
    all_model_unavailable = all(
        not f.get("model_loaded", False) for f in all_observations
    )

    submitted_views = sorted(list({
        m.get("view_label", "unknown") for m in clips_meta
        if m.get("view_label")
    } | {
        obs.get("view_label", "unknown") for obs in all_observations
        if obs.get("view_label")
    }))
    all_standard_views = {"front", "back", "left", "right", "top", "underside"}
    missing_views = sorted(all_standard_views - set(submitted_views))


    total_usable_frames = len(all_observations)

    # Minimum coverage check
    if total_usable_frames < _MIN_USABLE_FRAMES:
        status = STATUS_INSUFFICIENT
        limitations = [
            f"Only {total_usable_frames} usable frames extracted. "
            f"Minimum required: {_MIN_USABLE_FRAMES}.",
            "Try uploading higher-quality video clips.",
        ]
        return _make_assessment(
            session_id, plant_id, crop_type,
            status=status, limitations=limitations,
            total_frames=total_usable_frames,
            clips_meta=clips_meta,
            submitted_views=submitted_views,
            missing_views=missing_views,
            clip_summaries=clip_summaries,
        )

    # ---------------------------------------------------------------
    # Step 4: Evidence aggregation
    # ---------------------------------------------------------------
    disease_frames: List[Dict] = []
    symptom_area_values: List[float] = []
    opencv_abnormal_count = 0

    for obs in all_observations:
        pred = _get_pred_str(obs)
        conf = _get_conf_float(obs)
        if pred in _KNOWN_DISEASE_CLASSES and conf >= _SYMPTOM_CONF_THRESHOLD:
            disease_frames.append({"prediction": pred, "confidence": conf,
                                    "clip_id": obs.get("clip_id"),
                                    "timestamp_sec": obs.get("timestamp_sec"),
                                    "view_label": obs.get("view_label")})



        if obs.get("opencv_visual_status") == "abnormal":
            opencv_abnormal_count += 1

        lp = obs.get("lesion_percentage", 0.0)
        if lp > 0:
            symptom_area_values.append(lp)

    disease_fraction = len(disease_frames) / max(1, total_usable_frames)
    opencv_abnormal_fraction = opencv_abnormal_count / max(1, total_usable_frames)

    # Most common predicted disease class (if any)
    disease_class_counts: Dict[str, int] = {}
    for df in disease_frames:
        c = df["prediction"]
        disease_class_counts[c] = disease_class_counts.get(c, 0) + 1

    top_disease = max(disease_class_counts, key=disease_class_counts.get) if disease_class_counts else None
    avg_symptom_area = round(sum(symptom_area_values) / len(symptom_area_values), 2) if symptom_area_values else None

    # ---------------------------------------------------------------
    # Step 5: Determine status
    # ---------------------------------------------------------------
    limitations: List[str] = []

    if all_model_unavailable:
        # Only DIP available
        if opencv_abnormal_fraction >= _SYMPTOM_FRACTION_THRESHOLD:
            status = STATUS_UNCERTAIN
            limitations.append(
                "YOLO model weights are unavailable — only heuristic visual analysis was used. "
                "OpenCV detected visual anomalies in "
                f"{opencv_abnormal_count}/{total_usable_frames} frames. "
                "This does NOT confirm disease — it indicates visual abnormality only."
            )
        else:
            status = STATUS_HEALTHY
            limitations.append(
                "YOLO model weights are unavailable — only heuristic visual analysis was used. "
                "No significant visual anomalies detected, but model-based diagnosis was not possible."
            )
        limitations.append(
            "To enable AI disease classification, train the model and place weights at backend/models/yolo26_tomato.pt."
        )
    elif disease_fraction >= _SYMPTOM_FRACTION_THRESHOLD:
        status = STATUS_SYMPTOMS
        if opencv_abnormal_fraction < 0.10:
            # DIP and YOLO disagree — add uncertainty note
            limitations.append(
                f"YOLO predicted symptoms in {disease_fraction:.0%} of frames, "
                "but OpenCV visual analysis did not confirm significant abnormality. "
                "Results may need manual review."
            )
    elif disease_fraction > 0 and disease_fraction < _SYMPTOM_FRACTION_THRESHOLD:
        status = STATUS_UNCERTAIN
        limitations.append(
            f"Disease evidence found in only {disease_fraction:.0%} of frames — "
            "below the threshold for a confident diagnosis. More views may help."
        )
    else:
        status = STATUS_HEALTHY

    if missing_views:
        limitations.append(
            f"The following standard views were not submitted: {', '.join(missing_views)}. "
            "Unobserved plant regions remain UNKNOWN — not assumed healthy."
        )

    if avg_symptom_area is not None:
        limitations.append(
            "visible_symptom_area_percent is estimated from HSV color thresholds (whole-frame, "
            "no leaf mask). It is a heuristic visual indicator, NOT a validated disease-severity score."
        )

    # ---------------------------------------------------------------
    # Step 6: Build disease evidence list (de-duplicated across clips)
    # ---------------------------------------------------------------
    disease_evidence: List[Dict] = []
    for cls, count in sorted(disease_class_counts.items(), key=lambda x: -x[1]):
        # Average confidence for this class
        cls_confs = [df["confidence"] for df in disease_frames if df["prediction"] == cls]
        avg_conf = round(sum(cls_confs) / len(cls_confs), 3) if cls_confs else 0.0
        disease_evidence.append({
            "predicted_class": cls,
            "frame_count": count,
            "fraction_of_total": round(count / total_usable_frames, 3),
            "average_confidence": avg_conf,
            "note": (
                "Confidence represents model output, not biological certainty. "
                "Model is not validated on this dataset."
            ),
        })

    return _make_assessment(
        session_id, plant_id, crop_type,
        status=status,
        limitations=limitations,
        total_frames=total_usable_frames,
        clips_meta=clips_meta,
        submitted_views=submitted_views,
        missing_views=missing_views,
        clip_summaries=clip_summaries,
        disease_evidence=disease_evidence,
        visible_symptom_area_percent=avg_symptom_area,
        opencv_abnormal_fraction=round(opencv_abnormal_fraction, 3),
        disease_fraction=round(disease_fraction, 3),
        top_disease_class=top_disease,
        all_model_unavailable=all_model_unavailable,
        created_at=now,
    )


def _summarize_clip(clip_id: str, frames: List[Dict]) -> Dict[str, Any]:
    """Summarize evidence from a single clip's frames."""
    if not frames:
        return {"clip_id": clip_id, "frames": 0, "disease_frames": 0}

    disease_count = sum(
        1 for f in frames
        if _get_pred_str(f) in _KNOWN_DISEASE_CLASSES
        and _get_conf_float(f) >= _SYMPTOM_CONF_THRESHOLD
    )
    opencv_abn = sum(1 for f in frames if f.get("opencv_visual_status") == "abnormal")
    lesions = [f.get("lesion_percentage", 0.0) for f in frames if f.get("lesion_percentage", 0) > 0]
    view = frames[0].get("view_label", "unknown") if frames else "unknown"

    return {
        "clip_id": clip_id,
        "view_label": view,
        "total_frames": len(frames),
        "disease_frames": disease_count,
        "disease_fraction": round(disease_count / max(1, len(frames)), 3),
        "opencv_abnormal_frames": opencv_abn,
        "avg_lesion_percent": round(sum(lesions) / len(lesions), 2) if lesions else None,
    }


def _make_assessment(
    session_id: str,
    plant_id: str,
    crop_type: str,
    status: str,
    limitations: List[str],
    total_frames: int,
    clips_meta: List[Dict],
    submitted_views: Optional[List[str]] = None,
    missing_views: Optional[List[str]] = None,
    clip_summaries: Optional[List[Dict]] = None,
    disease_evidence: Optional[List[Dict]] = None,
    visible_symptom_area_percent: Optional[float] = None,
    opencv_abnormal_fraction: Optional[float] = None,
    disease_fraction: Optional[float] = None,
    top_disease_class: Optional[str] = None,
    all_model_unavailable: bool = False,
    created_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Construct the final assessment dict."""
    return {
        "session_id": session_id,
        "plant_id": plant_id,
        "crop_type": crop_type,
        "status": status,
        "top_disease_class": top_disease_class,
        "disease_evidence": disease_evidence or [],
        "visible_symptom_area_percent": visible_symptom_area_percent,
        "opencv_abnormal_fraction": opencv_abnormal_fraction,
        "disease_fraction": disease_fraction,
        "total_clips": len(clips_meta),
        "total_frames_processed": total_frames,
        "total_usable_frames": total_frames,
        "submitted_views": submitted_views or [],
        "views_submitted": submitted_views or [],
        "missing_views": missing_views or [],
        "clips_summary": clip_summaries or [],
        "model_available": not all_model_unavailable,
        "coverage_note": (
            "Coverage is described by submitted views. "
            "Unsubmitted view angles remain UNKNOWN, not assumed healthy. "
            "Physical plant coverage percentage is not quantified in this release."
        ),
        "limitations": limitations,
        "assessment_notes": [
            "This assessment is based on image evidence only — not a laboratory or expert diagnosis.",
            "Visible symptoms do not confirm the presence of disease with clinical certainty.",
            "Always consult a qualified agricultural specialist before applying treatments.",
        ],
        "created_at": created_at or time.time(),
    }

