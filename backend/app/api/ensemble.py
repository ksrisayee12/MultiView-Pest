"""
backend/app/api/ensemble.py

API router for the Multi-Model Ensemble Diagnostic Engine:
  1. POST /api/ensemble/assess — Runs multi-model consensus assessment on uploaded foliar media
  2. GET /api/ensemble/models — Reports operational status and SHA-256 provenance of all ensemble models
"""

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.app.services.ensemble_engine import run_ensemble_assessment, _init_ensemble_models
from backend.app.services.model_registry import get_disease_model_status

router = APIRouter()


@router.get("/ensemble/models")
def get_ensemble_models_status():
    """
    Returns full provenance, SHA-256 hashes, and architecture metadata for all active models in the ensemble.
    """
    _init_ensemble_models()
    from backend.app.services.ensemble_engine import _SEVERITY_SHA256, _PEST_SHA256, _SEVERITY_MODEL, _PEST_MODEL
    yolo_status = get_disease_model_status()
    
    return {
        "ensemble_status": "READY",
        "models": {
            "model_1_yolo_detector": {
                "name": yolo_status.get("model_name", "tomato_disease_yolov8n"),
                "version": yolo_status.get("model_version", "1.0.0"),
                "status": yolo_status.get("status", "UNKNOWN"),
                "sha256": yolo_status.get("sha256", "ccc05ab88bb550567fb7a7a07c564d37e344e7e69e094da54481ea47c0f134b3"),
                "type": "YOLOv8n Object Detection",
                "purpose": "Localizes foliar lesions and identifies primary pathogen class"
            },
            "model_2_leaf_severity_net": {
                "name": "leaf_severity_net",
                "version": "1.0.0",
                "status": "READY" if _SEVERITY_MODEL is not None else "OFFLINE",
                "sha256": _SEVERITY_SHA256,
                "type": "Multi-Head Deep Convolutional Neural Network (PyTorch)",
                "purpose": "Estimates 4-tier foliar health stage and continuous lesion area % regression"
            },
            "model_3_pest_vector_net": {
                "name": "pest_vector_net",
                "version": "1.0.0",
                "status": "READY" if _PEST_MODEL is not None else "OFFLINE",
                "sha256": _PEST_SHA256,
                "type": "Deep Insect Classifier (PyTorch)",
                "purpose": "Monitors vectors (aphids, whiteflies, caterpillars) driving foliar pathogen spread"
            },
            "dip_optical_pipeline": {
                "name": "DIP Optical ExG Analyzer",
                "version": "2.0.0",
                "status": "READY",
                "type": "Digital Image Processing (Excess Green Index + Otsu Thresholding)",
                "purpose": "Independent physics-based color space verification of leaf necrosis"
            }
        }
    }


@router.post("/ensemble/assess")
async def assess_leaf_ensemble(
    file: UploadFile = File(...),
    crop_type: str = Form(default="tomato")
):
    """
    Submits an image for multi-model ensemble diagnosis.
    Returns:
      - Primary Diagnosis & Confidence
      - What Caused This (Pathogen Etiology & Environmental Drivers)
      - Actionable Cure (Cultural, Organic, Chemical, Prevention)
      - Multi-Model Consensus Agreement Score
      - Full Model Provenance
    """
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    nparr = np.frombuffer(content, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    try:
        report = run_ensemble_assessment(img_bgr, crop_type=crop_type)
        return report
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-model ensemble analysis failed: {str(exc)}"
        )
