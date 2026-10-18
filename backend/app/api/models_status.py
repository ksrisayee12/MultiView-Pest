"""
backend/app/api/models_status.py

Model Readiness Status API — reports which models are available and which are
missing weights, allowing the frontend to show meaningful status before analysis.
"""

from fastapi import APIRouter
from backend.app.services.model_registry import get_disease_model_status

router = APIRouter()


@router.get("/models/status", tags=["Models"])
def get_models_status():
    """
    Returns readiness status for all configured models.

    Possible status values for each model:
      READY             — weights found and loaded successfully
      MISSING_WEIGHTS   — .pt file does not exist at the configured path
      LOAD_ERROR        — file exists but model failed to load
      NOT_CONFIGURED    — no weights path configured
      ULTRALYTICS_MISSING — ultralytics package not installed
      NOT_INITIALIZED   — status not yet checked (first call will resolve this)

    The 'action' field in each model's meta explains what is needed to make it ready.
    """
    disease_status = get_disease_model_status()

    models = {
        "disease_classifier": {
            **disease_status,
            "description": "Tomato disease classification model (YOLO classification task).",
            "intended_classes": [
                "healthy",
                "early_blight",
                "late_blight",
                "septoria_leaf_spot",
            ],
            "note": (
                "This model has not been trained yet. "
                "See notebooks/YOLO26_PlantVillage_Training.ipynb for training instructions. "
                "Once trained, copy the .pt file to the configured weights_path."
            ),
        },
        "pest_detector": {
            "model_name": "pest_detector",
            "status": "MISSING_WEIGHTS",
            "description": "IP102 pest detection model (YOLO detection task).",
            "weights_path": "models/pest/yolo26n_ip102_pest.pt",
            "meta": {
                "reason": "No trained pest detection weights available.",
                "action": (
                    "Train using scripts/train_yolo_pest.py after placing the IP102 dataset "
                    "in data/raw/pest/. See configs/yolo_pest_config.yaml for configuration."
                ),
            },
        },
        "leaf_segmenter": {
            "model_name": "leaf_segmenter",
            "status": "NOT_CONFIGURED",
            "description": "Leaf instance segmentation model (planned — Phase 4).",
            "weights_path": None,
            "meta": {
                "reason": "Leaf segmentation is planned for Phase 4 of the upgrade.",
                "action": "No action required yet.",
            },
        },
    }

    all_ready = all(m["status"] == "READY" for m in models.values())
    any_ready = any(m["status"] == "READY" for m in models.values())

    return {
        "overall_status": "ALL_READY" if all_ready else ("PARTIAL" if any_ready else "NO_MODELS_READY"),
        "models": models,
        "note": (
            "Analysis will still run when models are unavailable — "
            "YOLO results will show 'model_unavailable' and only OpenCV heuristic analysis will be used."
        ),
    }
