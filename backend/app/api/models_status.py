"""
backend/app/api/models_status.py

Model Readiness Status API — reports which models are available and which are
missing weights, allowing the frontend to show meaningful status before analysis.
"""

from pathlib import Path
from fastapi import APIRouter
from backend.app.services.model_registry import get_disease_model_status

router = APIRouter()

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "backend" / "models"


def _check_file_status(path: Path) -> str:
    """Return READY if file exists and is non-empty, else MISSING_WEIGHTS."""
    if path.exists() and path.stat().st_size > 0:
        return "READY"
    return "MISSING_WEIGHTS"


@router.get("/models/status", tags=["Models"])
def get_models_status():
    """
    Returns readiness status for all configured models.

    Possible status values for each model:
      READY             — weights found and loaded successfully
      MISSING_WEIGHTS   — .pth / .pt file does not exist at the configured path
      LOAD_ERROR        — file exists but model failed to load (check startup logs)
      NOT_CONFIGURED    — no weights path configured
      NOT_INITIALIZED   — status not yet checked (first call will resolve this)

    The 'action' field in each model's meta explains what is needed to make it ready.
    """
    disease_status = get_disease_model_status()

    # Dynamically check auxiliary model weights
    severity_path = _MODELS_DIR / "leaf_severity_net.pth"
    pest_vec_path = _MODELS_DIR / "pest_vector_net.pth"

    severity_status = _check_file_status(severity_path)
    pest_vec_status = _check_file_status(pest_vec_path)

    models = {
        "disease_classifier": {
            **disease_status,
            "description": (
                "AgriHealthNet Dual-Backbone Ensemble (MobileNetV3 + ResNet18). "
                "Trained on real PlantVillage dataset across Tomato, Pepper, and Potato crops."
            ),
            "intended_classes": [
                "healthy", "pepper_healthy", "potato_healthy",
                "early_blight", "late_blight", "septoria_leaf_spot",
                "bacterial_spot", "yellow_leaf_curl", "leaf_mold",
                "spider_mites", "target_spot", "mosaic_virus",
                "pepper_bacterial_spot", "potato_early_blight",
                "potato_late_blight", "pest_chewing_damage",
                "unclassified_foliar_damage",
            ],
            "supported_crops": ["tomato", "bell_pepper", "chilli", "potato", "general"],
            "note": (
                "Dual-backbone fusion: 0.45×MobileNetV3 + 0.55×ResNet18 logits → Softmax. "
                "Test-Time Augmentation (TTA) with 4-angle averaging. "
                "Crop-aware arbitration routes non-tomato classes to crop-specific labels."
            ),
        },
        "leaf_severity_net": {
            "model_name": "leaf_severity_net",
            "status": severity_status,
            "description": "LeafSeverityNet CNN — predicts severity tier (HEALTHY / MILD / MODERATE / SEVERE) and lesion area %.",
            "weights_path": str(severity_path),
            "meta": {
                "reason": "Auxiliary deep severity estimator for lesion grade scoring.",
                "action": (
                    "Run scripts/train_auxiliary_models.py to train. "
                    "Outputs leaf_severity_net.pth to backend/models/."
                ),
            },
        },
        "pest_vector_net": {
            "model_name": "pest_vector_net",
            "status": pest_vec_status,
            "description": "PestVectorNet CNN — classifies insect vector type: Aphid, Armyworm, Flea Beetle, Cutworm, Whitefly, or NONE.",
            "weights_path": str(pest_vec_path),
            "meta": {
                "reason": "Auxiliary insect vector surveillance classifier.",
                "action": (
                    "Run scripts/train_auxiliary_models.py to train. "
                    "Outputs pest_vector_net.pth to backend/models/."
                ),
            },
        },
        "pest_perforation_detector": {
            "model_name": "pest_perforation_detector",
            "status": "READY",
            "description": "OpenCV-based physical hole & defoliation detector. Identifies caterpillar/beetle feeding perforations without ML weights.",
            "weights_path": None,
            "meta": {
                "reason": "Pure computer vision — no weights needed.",
                "action": "Always active. Uses contour hierarchy analysis and convex-hull defoliation measurement.",
            },
        },
        "dip_pipeline": {
            "model_name": "dip_pipeline",
            "status": "READY",
            "description": "Digital Image Processing pipeline — ExG vegetation index, HSV necrotic lesion masking, blur/exposure preflight.",
            "weights_path": None,
            "meta": {
                "reason": "Deterministic OpenCV pipeline — always available.",
                "action": "Always active. No training required.",
            },
        },
    }

    all_ready = all(m["status"] == "READY" for m in models.values())
    any_ready = any(m["status"] == "READY" for m in models.values())

    return {
        "api_version": "2.0.0",
        "overall_status": "ALL_READY" if all_ready else ("PARTIAL" if any_ready else "NO_MODELS_READY"),
        "models": models,
        "note": (
            "Analysis runs even when auxiliary model weights are missing — "
            "the dual-backbone classifier and OpenCV DIP pipeline are always active. "
            "LeafSeverityNet and PestVectorNet enhance accuracy when weights are present."
        ),
    }
