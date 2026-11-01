"""
backend/app/api/analyze.py

Existing field/tile/closeup analysis API — LEGACY FIELD-TILE WORKFLOW.

Changes from Phase 1 stabilization:
  - Removed per-tile YOLO instantiation (was 16 loads per field scan).
  - Now calls model_registry.predict_disease() which uses a singleton.
  - Added upload size and MIME type validation.
  - Model path is now resolved absolutely, not relative to CWD.
  - Hard-coded accuracy claims have been moved to analytics.py (removed there too).
"""

import io
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.services.model_registry import predict_disease, get_disease_model_status
from backend.image_processing.visual_health import analyze_visual_health
from backend.fusion.fusion_engine import (
    calculate_plant_health_index,
    evaluate_tile_fusion,
    generate_priority_list,
)
from backend.recommendations.recommendations import get_recommendation
from backend.database.mongodb import db_manager
from simulation.tile_extractor import extract_16_tiles
from simulation.create_field_image import create_simulated_field

router = APIRouter()

# Upload limits
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
_ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp",
    "image/tiff", "image/webp",
}


def _validate_image_upload(file: UploadFile, content: bytes) -> None:
    """Raise HTTPException if upload exceeds size limit or is wrong MIME type."""
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large. Maximum allowed size is {_MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: '{content_type}'. Accepted: JPEG, PNG, BMP, TIFF, WebP.",
        )


def _decode_image(content: bytes, filename: str) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR array. Raises HTTPException on failure."""
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode image '{filename}'. File may be corrupt or not a supported image format.",
        )
    return img


def _analyze_tile(tile_crop: np.ndarray) -> Dict[str, Any]:
    """Run YOLO (singleton) + OpenCV + fusion on one tile."""
    yolo_res = predict_disease(tile_crop)
    opencv_res = analyze_visual_health(tile_crop)
    fusion_res = evaluate_tile_fusion(
        yolo_prediction=yolo_res["prediction"],
        yolo_confidence=yolo_res["confidence"],
        opencv_result=opencv_res,
    )
    return yolo_res, opencv_res, fusion_res


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@router.post("/analyze/field", tags=["Analysis"])
async def analyze_field(file: Optional[UploadFile] = File(None)):
    """
    Processes a 4×4 composite field image (16 tiles), runs YOLO + OpenCV on each tile,
    executes fusion, computes Plant Health Index, and stores scan results.

    If no file is uploaded, a simulated field image is used.
    This is the LEGACY FIELD-TILE workflow. Each tile is a spatial region of the field
    image, NOT a separate viewpoint of the same plant.
    """
    if file:
        content = await file.read()
        _validate_image_upload(file, content)
        field_img = _decode_image(content, file.filename or "upload")
        filename = file.filename or "upload"
    else:
        # Fallback: simulated composite
        f_path, _ = create_simulated_field()
        field_img = cv2.imread(f_path)
        if field_img is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate simulated field image.",
            )
        filename = "simulated_field_image.jpg"

    tiles = extract_16_tiles(field_img)
    tile_results: List[Dict[str, Any]] = []

    model_status = get_disease_model_status()

    for tile in tiles:
        tile_id = tile["tile_id"]
        tile_crop = tile["image"]

        yolo_res, opencv_res, fusion_res = _analyze_tile(tile_crop)
        recommendation = get_recommendation(
            fusion_res["confirmed_disease"] or yolo_res["prediction"]
        )

        tile_results.append(
            {
                "tile_id": tile_id,
                "grid_row": tile["grid_row"],
                "grid_col": tile["grid_col"],
                "bbox": tile["bbox"],
                "yolo_prediction": yolo_res["prediction"],
                "yolo_confidence": yolo_res["confidence"],
                "opencv_status": opencv_res["visual_status"],
                "greenness_index": opencv_res["greenness_index"],
                "green_pixel_percentage": opencv_res["green_pixel_percentage"],
                "lesion_percentage": opencv_res["lesion_percentage"],
                "fusion_status": fusion_res["fusion_status"],
                "confirmed_disease": fusion_res["confirmed_disease"],
                "severity": fusion_res["severity"],
                "requires_closeup": fusion_res["requires_closeup"],
                "recommendation": recommendation,
            }
        )

    health_index = calculate_plant_health_index(tile_results)
    priority_list = generate_priority_list(tile_results)

    scan_summary = {
        "crop": "Tomato",  # legacy field endpoint — crop type not yet configurable per scan
        "image_name": filename,
        "total_tiles": len(tile_results),
        "healthy_tiles": sum(1 for t in tile_results if t["fusion_status"] == "HEALTHY"),
        "confirmed_disease_tiles": sum(
            1 for t in tile_results if t["fusion_status"] == "CONFIRMED DISEASE"
        ),
        "disputed_tiles": sum(1 for t in tile_results if t["fusion_status"] == "DISPUTED"),
        "plant_health_index": health_index,
        "yolo_model_status": model_status["status"],
        "yolo_model_loaded": model_status["status"] == "READY",
        "tile_results": tile_results,
        "priority_list": priority_list,
    }

    scan_id = db_manager.save_scan(scan_summary)
    scan_summary["scan_id"] = scan_id
    return scan_summary


@router.post("/analyze/tile", tags=["Analysis"])
async def analyze_tile(file: UploadFile = File(...)):
    """Analyzes a single leaf tile image with YOLO + OpenCV + Fusion."""
    content = await file.read()
    _validate_image_upload(file, content)
    tile_img = _decode_image(content, file.filename or "tile")

    yolo_res, opencv_res, fusion_res = _analyze_tile(tile_img)
    recommendation = get_recommendation(
        fusion_res["confirmed_disease"] or yolo_res["prediction"]
    )

    return {
        "yolo_prediction": yolo_res["prediction"],
        "yolo_confidence": yolo_res["confidence"],
        "yolo_model_status": yolo_res.get("model_status", "UNKNOWN"),
        "yolo_model_loaded": yolo_res["model_loaded"],
        "opencv_status": opencv_res["visual_status"],
        "greenness_index": opencv_res["greenness_index"],
        "green_pixel_percentage": opencv_res["green_pixel_percentage"],
        "lesion_percentage": opencv_res["lesion_percentage"],
        "fusion_status": fusion_res["fusion_status"],
        "confirmed_disease": fusion_res["confirmed_disease"],
        "severity": fusion_res["severity"],
        "requires_closeup": fusion_res["requires_closeup"],
        "recommendation": recommendation,
    }


@router.post("/analyze/closeup", tags=["Analysis"])
async def analyze_closeup(
    file: UploadFile = File(...),
    tile_id: int = Form(...),
    scan_id: Optional[str] = Form(None),
):
    """
    Analyzes a close-up leaf image for a disputed tile and returns the resolution decision.
    """
    content = await file.read()
    _validate_image_upload(file, content)
    closeup_img = _decode_image(content, file.filename or "closeup")

    yolo_res, opencv_res, fusion_res = _analyze_tile(closeup_img)

    final_decision = fusion_res["fusion_status"]
    if final_decision == "DISPUTED":
        final_decision = "REQUIRES MANUAL REVIEW"

    recommendation = get_recommendation(
        fusion_res["confirmed_disease"] or yolo_res["prediction"]
    )

    return {
        "tile_id": tile_id,
        "scan_id": scan_id,
        "closeup_yolo_prediction": yolo_res["prediction"],
        "closeup_yolo_confidence": yolo_res["confidence"],
        "closeup_yolo_model_status": yolo_res.get("model_status", "UNKNOWN"),
        "closeup_opencv_status": opencv_res["visual_status"],
        "closeup_lesion_percentage": opencv_res["lesion_percentage"],
        "final_decision": final_decision,
        "confirmed_disease": fusion_res["confirmed_disease"],
        "severity": fusion_res["severity"],
        "recommendation": recommendation,
    }
