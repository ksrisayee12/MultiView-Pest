"""
backend/app/api/plant_sessions.py

Phase 3 — Multi-View Plant Analysis API

Provides endpoints to:
  - Create and manage plant-level sessions (stable plant_id, session_id).
  - Upload multiple video clips per session with viewpoint metadata (front, back, left, right, top, underside, arbitrary).
  - Execute multi-view analysis asynchronously using a background thread/task.
  - Query analysis progress, frame observations, and fused plant assessments.
  - Generate deterministic simulated multi-view sessions for test/demo environments.
"""

import os
import uuid
import time
import asyncio
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from backend.app.schemas.plant_session import (
    CreateSessionRequest,
    ClipMetaResponse,
    PlantSessionResponse,
    SessionStatusResponse,
)
from backend.database.mongodb import db_manager
from backend.app.services.video_processor import (
    process_clip,
    make_clip_meta,
    cleanup_clip_file,
    _ALLOWED_VIDEO_TYPES,
    _ALLOWED_VIDEO_EXTENSIONS,
    _MAX_VIDEO_BYTES,
)
from backend.app.services.plant_fusion import (
    fuse_plant_assessment,
    STATUS_HEALTHY,
    STATUS_SYMPTOMS,
    STATUS_UNCERTAIN,
    STATUS_INSUFFICIENT,
    STATUS_ERROR,
    STATUS_NO_MODEL,
)

router = APIRouter()

# In-process active job tracker: session_id -> {status, stage, progress_percent, message, clips_total, clips_processed}
_JOBS: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_proc_")


def _get_or_create_job(session_id: str, plant_id: str, total_clips: int) -> Dict[str, Any]:
    if session_id not in _JOBS:
        _JOBS[session_id] = {
            "session_id": session_id,
            "plant_id": plant_id,
            "status": "QUEUED",
            "stage": "Initialized",
            "progress_percent": 0,
            "message": "Session ready for analysis",
            "clips_total": total_clips,
            "clips_processed": 0,
            "has_assessment": False,
            "error": None,
        }
    return _JOBS[session_id]


# ---------------------------------------------------------------------------
# Session Endpoints
# ---------------------------------------------------------------------------

@router.post("/plant-sessions", response_model=PlantSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_plant_session(req: CreateSessionRequest):
    """
    Create a new plant-analysis session with a stable plant_id and unique session_id.
    """
    plant_id = req.plant_id.strip()
    if not plant_id:
        raise HTTPException(status_code=400, detail="Plant ID must not be empty.")

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    now = time.time()

    session_data = {
        "session_id": session_id,
        "plant_id": plant_id,
        "crop_type": req.crop_type.strip().lower(),
        "notes": req.notes,
        "status": "CREATED",
        "created_at": now,
        "updated_at": now,
        "clips": [],
        "assessment": None,
        "observation_count": 0,
    }

    db_manager.save_plant_session(session_data)
    _get_or_create_job(session_id, plant_id, 0)

    return session_data


@router.get("/plant-sessions", response_model=List[PlantSessionResponse])
async def list_plant_sessions(limit: int = 50):
    """
    List recent plant-analysis sessions.
    """
    return db_manager.list_plant_sessions(limit=limit)


@router.get("/plant-sessions/{session_id}", response_model=PlantSessionResponse)
async def get_plant_session(session_id: str):
    """
    Retrieve session metadata, uploaded clips, and assessment.
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


@router.delete("/plant-sessions/{session_id}")
async def delete_plant_session(session_id: str):
    """
    Delete session, observations, and associated temporary files.
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    for clip in session.get("clips", []):
        fpath = clip.get("file_path")
        if fpath:
            cleanup_clip_file(fpath)

    db_manager.delete_plant_session(session_id)
    if session_id in _JOBS:
        del _JOBS[session_id]

    return {"message": f"Session '{session_id}' deleted successfully.", "session_id": session_id}


# ---------------------------------------------------------------------------
# Clip Upload Endpoint
# ---------------------------------------------------------------------------

@router.post("/plant-sessions/{session_id}/videos", status_code=status.HTTP_201_CREATED)
async def upload_clip_to_session(
    session_id: str,
    file: UploadFile = File(...),
    view_label: str = Form(default="unknown"),
):
    """
    Upload a single video clip to a plant session and tag it with viewpoint metadata.
    Accepted view labels: front, back, left, right, top, underside, unknown, arbitrary.
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    clean_view = view_label.strip().lower()
    valid_views = {"front", "back", "left", "right", "top", "underside", "unknown", "arbitrary"}
    if clean_view not in valid_views:
        clean_view = "unknown"

    content = await file.read()
    if len(content) > _MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Video file too large ({len(content)/(1024*1024):.1f} MB). Max allowed: {_MAX_VIDEO_BYTES//(1024*1024)} MB.",
        )

    # Extension validation
    fname = file.filename or "uploaded_clip.mp4"
    ext = os.path.splitext(fname)[1].lower()
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video extension '{ext}'. Accepted: {', '.join(sorted(_ALLOWED_VIDEO_EXTENSIONS))}",
        )

    # Save to upload temp directory
    upload_dir = os.path.join("data", "uploads", session_id)
    os.makedirs(upload_dir, exist_ok=True)
    clip_id = f"clip_{uuid.uuid4().hex[:8]}"
    saved_path = os.path.join(upload_dir, f"{clip_id}_{os.path.basename(fname)}")

    with open(saved_path, "wb") as f:
        f.write(content)

    clip_meta = make_clip_meta(
        clip_id=clip_id,
        session_id=session_id,
        original_filename=fname,
        view_label=clean_view,
        file_path=saved_path,
    )

    clips = session.get("clips", [])
    clips.append(clip_meta)
    db_manager.update_plant_session(session_id, {"clips": clips, "updated_at": time.time()})

    job = _get_or_create_job(session_id, session["plant_id"], len(clips))
    job["clips_total"] = len(clips)

    return {
        "message": f"Video clip uploaded successfully for view '{clean_view}'.",
        "clip": clip_meta,
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# Video Analysis Execution & Job Tracking
# ---------------------------------------------------------------------------

def _run_analysis_sync(session_id: str):
    """
    Synchronous worker executing in threadpool:
      1. Inspects all clips for the session
      2. For each clip, extracts sampled frames, runs DIP & YOLO (singleton)
      3. Aggregates all observations
      4. Runs plant_fusion to generate PlantAssessment
      5. Updates DB and job status
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        return

    job = _get_or_create_job(session_id, session["plant_id"], len(session.get("clips", [])))
    clips = session.get("clips", [])

    if not clips:
        job["status"] = "FAILED"
        job["stage"] = "No clips"
        job["message"] = "Cannot run analysis without at least one uploaded video clip."
        return

    job["status"] = "PROCESSING"
    job["stage"] = "Extracting frames & running vision models"
    job["progress_percent"] = 5

    all_observations: List[Dict[str, Any]] = []
    updated_clips: List[Dict[str, Any]] = []

    total_clips = len(clips)

    for i, clip in enumerate(clips):
        job["stage"] = f"Processing clip {i+1}/{total_clips} ({clip.get('view_label', 'view')})"
        job["message"] = f"Decoding and sampling {clip.get('original_filename')}..."

        fpath = clip.get("file_path")
        if not fpath or not os.path.exists(fpath):
            clip["processing_status"] = "ERROR"
            clip["error_message"] = "File not found on disk."
            updated_clips.append(clip)
            continue

        clip_obs, final_clip_meta = process_clip(clip, sample_fps=clip.get("sample_fps", 2.0))
        all_observations.extend(clip_obs)
        updated_clips.append(final_clip_meta)

        job["clips_processed"] = i + 1
        pct = 10 + int(70 * (i + 1) / total_clips)
        job["progress_percent"] = min(pct, 80)

    # Fusion stage
    job["stage"] = "Multi-view evidence fusion"
    job["progress_percent"] = 85
    job["message"] = "Aggregating cross-view observations, estimating coverage, applying confidence rules..."

    assessment = fuse_plant_assessment(
        session_id=session_id,
        plant_id=session.get("plant_id", "PLANT-001"),
        crop_type=session.get("crop_type", "tomato"),
        clips_meta=updated_clips,
        all_observations=all_observations,
    )

    # Save to db
    db_manager.save_observations(session_id, all_observations)
    db_manager.update_plant_session(
        session_id,
        {
            "clips": updated_clips,
            "assessment": assessment,
            "status": "COMPLETED",
            "observation_count": len(all_observations),
            "updated_at": time.time(),
        },
    )

    job["status"] = "COMPLETED"
    job["stage"] = "Assessment ready"
    job["progress_percent"] = 100
    job["message"] = f"Analysis complete. Fused {len(all_observations)} observations across {len(updated_clips)} view clips."
    job["has_assessment"] = True
    job["assessment"] = assessment


@router.post("/plant-sessions/{session_id}/analyze")
async def start_session_analysis(session_id: str, background_tasks: BackgroundTasks):
    """
    Trigger multi-view video processing for all clips in the session asynchronously.
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    clips = session.get("clips", [])
    if not clips:
        raise HTTPException(
            status_code=400,
            detail="Session has no uploaded clips. Please upload at least one video clip before analyzing.",
        )

    job = _get_or_create_job(session_id, session["plant_id"], len(clips))
    if job["status"] == "PROCESSING":
        return {
            "message": "Analysis is already running for this session.",
            "session_id": session_id,
            "status": job,
        }

    job["status"] = "QUEUED"
    job["stage"] = "Queued for processing"
    job["progress_percent"] = 2
    job["message"] = "Starting video decoders and vision inference..."

    # Dispatch to background thread pool
    _executor.submit(_run_analysis_sync, session_id)

    return {
        "message": f"Analysis queued for {len(clips)} clip(s).",
        "session_id": session_id,
        "job_status": job,
    }


@router.get("/plant-sessions/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Check the current progress and status of analysis for a session.
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    job = _JOBS.get(session_id)
    if not job:
        has_assessment = session.get("assessment") is not None
        return SessionStatusResponse(
            session_id=session_id,
            plant_id=session.get("plant_id", "PLANT-001"),
            status="COMPLETED" if has_assessment else session.get("status", "CREATED"),
            stage="Completed" if has_assessment else "Ready",
            progress_percent=100 if has_assessment else 0,
            message="Assessment available" if has_assessment else "Session created",
            clips_total=len(session.get("clips", [])),
            clips_processed=len(session.get("clips", [])) if has_assessment else 0,
            has_assessment=has_assessment,
            assessment=session.get("assessment"),
        )

    return SessionStatusResponse(
        session_id=session_id,
        plant_id=job["plant_id"],
        status=job["status"],
        stage=job["stage"],
        progress_percent=job["progress_percent"],
        message=job["message"],
        clips_total=job["clips_total"],
        clips_processed=job["clips_processed"],
        has_assessment=job["has_assessment"],
        assessment=job.get("assessment") or session.get("assessment"),
    )


@router.get("/plant-sessions/{session_id}/assessment")
async def get_session_assessment(session_id: str):
    """
    Retrieve the finalized multi-view plant assessment.
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    assessment = session.get("assessment")
    if not assessment:
        raise HTTPException(
            status_code=404,
            detail="Assessment not yet generated. Please call POST /api/plant-sessions/{session_id}/analyze first.",
        )
    return assessment


@router.get("/plant-sessions/{session_id}/observations")
async def get_session_observations(session_id: str):
    """
    Retrieve all frame-level observations across all clips for inspection.
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    observations = db_manager.get_observations(session_id)
    return {
        "session_id": session_id,
        "plant_id": session.get("plant_id"),
        "total_observations": len(observations),
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# Deterministic Multi-View Simulation Session
# ---------------------------------------------------------------------------

@router.post("/plant-sessions/simulation", status_code=status.HTTP_201_CREATED)
async def create_simulated_multiview_session(
    plant_id: str = Form(default="PLANT-SIM-001"),
    crop_type: str = Form(default="tomato"),
    include_symptoms: bool = Form(default=True),
):
    """
    Generate a deterministic synthetic multi-view session with 4 viewpoints
    (front, back, left, top) without requiring physical video upload.
    Clearly labeled as simulated.
    """
    session_id = f"sess_sim_{uuid.uuid4().hex[:8]}"
    now = time.time()

    viewpoints = ["front", "back", "left", "top"]
    sim_clips = []
    all_observations = []

    # Generate synthetic observations with realistic DIP metrics
    for v_idx, view in enumerate(viewpoints):
        clip_id = f"sim_clip_{view}"
        frames_extracted = 4

        clip_meta = {
            "clip_id": clip_id,
            "session_id": session_id,
            "original_filename": f"simulated_{view}_view.mp4",
            "view_label": view,
            "file_path": None,
            "upload_status": "SIMULATED",
            "processing_status": "COMPLETED",
            "video_valid": True,
            "fps": 30.0,
            "duration_seconds": 2.0,
            "total_video_frames": 60,
            "frames_extracted": frames_extracted,
            "frames_skipped_duplicate": 1,
            "sample_fps": 2.0,
            "created_at": now,
            "processed_at": now,
        }
        sim_clips.append(clip_meta)

        for f_idx in range(frames_extracted):
            # Synthetic symptom distribution: back view shows early blight
            has_symptom = include_symptoms and (view == "back" and f_idx >= 1)
            pred_class = "early_blight" if has_symptom else "healthy"
            conf = 0.82 if has_symptom else 0.91
            lesion_pct = 14.5 if has_symptom else 0.8

            obs = {
                "clip_id": clip_id,
                "frame_index": f_idx * 15,
                "timestamp_sec": round(f_idx * 0.5, 2),
                "view_label": view,
                "yolo_prediction": {
                    "prediction": pred_class,
                    "confidence": conf,
                    "class_id": 1 if has_symptom else 0,
                    "all_confidences": {pred_class: conf},
                    "model_status": "synthetic_simulation",
                    "model_loaded": True,
                },
                "dip_metrics": {
                    "health_status": "Diseased" if has_symptom else "Healthy",
                    "lesion_percentage": lesion_pct,
                    "green_ratio": 42.0 if has_symptom else 78.5,
                    "chlorosis_ratio": 12.0 if has_symptom else 1.2,
                    "necrosis_ratio": 8.5 if has_symptom else 0.4,
                    "vegetation_index": 0.35 if has_symptom else 0.72,
                },
                "is_duplicate": False,
                "model_loaded": True,
            }
            all_observations.append(obs)

    # Fuse assessment
    assessment = fuse_plant_assessment(
        session_id=session_id,
        plant_id=plant_id,
        crop_type=crop_type,
        clips_meta=sim_clips,
        all_observations=all_observations,
    )
    assessment["limitations"].append("NOTE: This session was generated using the deterministic multi-view simulator.")

    session_data = {
        "session_id": session_id,
        "plant_id": plant_id,
        "crop_type": crop_type,
        "notes": "Generated multi-view simulation session.",
        "status": "COMPLETED",
        "created_at": now,
        "updated_at": now,
        "clips": sim_clips,
        "assessment": assessment,
        "observation_count": len(all_observations),
    }

    db_manager.save_plant_session(session_data)
    db_manager.save_observations(session_id, all_observations)

    _JOBS[session_id] = {
        "session_id": session_id,
        "plant_id": plant_id,
        "status": "COMPLETED",
        "stage": "Simulation assessment ready",
        "progress_percent": 100,
        "message": f"Simulated session ready with {len(viewpoints)} viewpoints.",
        "clips_total": len(sim_clips),
        "clips_processed": len(sim_clips),
        "has_assessment": True,
        "assessment": assessment,
    }

    return session_data


# ===========================================================================
# Phase 4 — Leaf Analysis Status Endpoint
# ===========================================================================

@router.get(
    "/plant-sessions/leaf-segmentation-status",
    summary="Leaf segmentation model availability",
    tags=["Phase 4"],
)
def get_leaf_segmentation_status():
    """
    Returns the current leaf segmentation model availability.
    When weights are absent, heuristic HSV-based masking is used.
    """
    from backend.app.services.leaf_analysis import get_leaf_segmentation_status as _gls
    return _gls()


@router.post(
    "/plant-sessions/{session_id}/leaf-analysis",
    summary="Run Phase 4 leaf analysis on session observations",
    tags=["Phase 4"],
)
def run_leaf_analysis(session_id: str):
    """
    Returns per-observation leaf quality reports and aggregate summary.
    Visible symptom area is computed within the leaf region — NOT a disease severity score.
    """
    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    observations = db_manager.get_observations(session_id)
    if not observations:
        raise HTTPException(status_code=404, detail="No observations found for this session.")

    per_obs_report = []
    for obs in observations:
        leaf_coverage = obs.get("leaf_coverage_fraction", 0.0)
        method = obs.get("leaf_segmentation_method", "HEURISTIC")
        quality_flags: List[str] = obs.get("quality_flags", [])
        if not quality_flags:
            if method == "HEURISTIC":
                quality_flags.append("leaf_mask_heuristic: mask is approximate")
            if leaf_coverage < 0.05:
                quality_flags.append("low_leaf_coverage: leaf occupies <5% of frame")

        per_obs_report.append({
            "clip_id": obs.get("clip_id"),
            "frame_index": obs.get("frame_index"),
            "view_label": obs.get("view_label"),
            "leaf_mask_available": obs.get("leaf_mask_available", False),
            "leaf_segmentation_method": method,
            "leaf_coverage_fraction": leaf_coverage,
            "visible_symptom_area_percent": obs.get("visible_symptom_area_percent",
                                                     obs.get("lesion_percentage")),
            "quality_flags": quality_flags,
        })

    lesion_values = [
        r["visible_symptom_area_percent"]
        for r in per_obs_report
        if r["visible_symptom_area_percent"] is not None
    ]
    aggregate = {
        "total_observations": len(per_obs_report),
        "observations_with_leaf_mask": sum(1 for r in per_obs_report if r["leaf_mask_available"]),
        "mean_visible_symptom_area_percent": (
            round(sum(lesion_values) / len(lesion_values), 2) if lesion_values else None
        ),
        "max_visible_symptom_area_percent": (
            round(max(lesion_values), 2) if lesion_values else None
        ),
        "note": (
            "Visible symptom area is computed within the leaf region using HSV colour thresholds. "
            "It is a heuristic visual indicator — NOT a validated disease severity score."
        ),
    }

    return {
        "session_id": session_id,
        "per_observation_leaf_report": per_obs_report,
        "aggregate": aggregate,
    }


# ===========================================================================
# Phase 4 — Temporal Tracking Endpoint
# ===========================================================================

@router.get(
    "/plant-sessions/{session_id}/temporal-tracks",
    summary="Within-clip temporal leaf track summaries",
    tags=["Phase 4"],
)
def get_temporal_tracks(session_id: str):
    """
    Runs the within-clip temporal tracker over all frame observations.
    Returns unique track counts per clip to prevent double-counting.
    Track IDs are valid only within one clip (not across viewpoints).
    """
    from backend.app.services.temporal_tracker import track_clip_observations
    from collections import defaultdict

    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    observations = db_manager.get_observations(session_id)
    if not observations:
        return {"session_id": session_id, "clip_track_summaries": [], "note": "No observations."}

    clips_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        clips_dict[obs.get("clip_id", "unknown")].append(obs)

    clip_summaries = []
    for clip_id, clip_obs in clips_dict.items():
        view_label = clip_obs[0].get("view_label", "unknown") if clip_obs else "unknown"
        sorted_obs = sorted(clip_obs, key=lambda o: o.get("frame_index", 0))
        summary = track_clip_observations(clip_id, view_label, sorted_obs)
        clip_summaries.append(summary)

    return {
        "session_id": session_id,
        "total_clips": len(clip_summaries),
        "clip_track_summaries": clip_summaries,
    }


# ===========================================================================
# Phase 5 — Cross-View Evidence Endpoint
# ===========================================================================

@router.get(
    "/plant-sessions/{session_id}/cross-view-evidence",
    summary="Cross-view multi-angle evidence fusion report",
    tags=["Phase 5"],
)
def get_cross_view_evidence(session_id: str):
    """
    Aggregates frame observations across all viewpoints and returns a
    cross-view evidence report with per-view statuses, conflict detection,
    and plant-level fused status. Unobserved views are explicitly marked.
    """
    from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence

    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    observations = db_manager.get_observations(session_id)
    view_obs: Dict[str, List[Dict[str, Any]]] = {}
    for obs in observations:
        vl = obs.get("view_label", "unknown")
        view_obs.setdefault(vl, []).append(obs)

    declared = [c.get("view_label") for c in session.get("clips", [])]
    plant_id = session.get("plant_id", session_id)

    return aggregate_cross_view_evidence(
        session_id=session_id,
        plant_id=plant_id,
        view_observations=view_obs,
        declared_viewpoints=declared,
    )


# ===========================================================================
# Phase 6 — Simulation Suite Endpoint
# ===========================================================================

@router.post(
    "/plant-sessions/run-simulation-suite",
    summary="Run all deterministic simulation scenarios",
    tags=["Phase 6"],
)
def run_simulation_suite():
    """
    Runs the full deterministic simulation fixture suite.
    Tests cross-view fusion on synthetic scenarios without real video uploads.
    Returns full evaluation suite with pass/fail per scenario and caveats.
    """
    from backend.app.services.simulation_fixtures import run_all_simulation_scenarios
    return run_all_simulation_scenarios()


@router.post(
    "/plant-sessions/evaluate",
    summary="Evaluate predictions against ground truth labels",
    tags=["Phase 6"],
)
def evaluate_predictions(payload: Dict[str, Any]):
    """
    Evaluate plant status predictions against user-declared ground truth.
    Returns precision, recall, F1, accuracy with Wilson 95% confidence intervals.
    Ground truth labels are user-declared — not clinically validated.
    """
    from backend.app.services.evaluation import compute_plant_level_metrics

    predictions = payload.get("predictions", [])
    ground_truth = payload.get("ground_truth", [])
    positive_label = payload.get("positive_label", "SYMPTOMS_DETECTED")

    if not predictions or not ground_truth:
        raise HTTPException(status_code=400, detail="'predictions' and 'ground_truth' are required.")

    try:
        return compute_plant_level_metrics(predictions, ground_truth, positive_label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/plant-sessions/{session_id}/evaluation-scorecard",
    summary="Score a completed session against a declared ground truth",
    tags=["Phase 6"],
)
def session_evaluation_scorecard(session_id: str, ground_truth_status: str = ""):
    """
    Generates an evaluation scorecard comparing session assessment status
    against a user-provided ground truth label.
    Query param: ground_truth_status (e.g., "SYMPTOMS_DETECTED")
    """
    from backend.app.services.evaluation import score_simulation_scenario

    session = db_manager.get_plant_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if not ground_truth_status:
        raise HTTPException(
            status_code=400,
            detail="Provide 'ground_truth_status' as a query parameter."
        )

    observations = db_manager.get_observations(session_id)
    assessment = session.get("assessment", {})
    status = assessment.get("plant_status", "UNKNOWN")
    for obs in observations:
        obs["status"] = status

    return score_simulation_scenario(
        scenario_name=f"session_{session_id}",
        scenario_observations=observations if observations else [{"status": status}],
        expected_status=ground_truth_status,
    )
