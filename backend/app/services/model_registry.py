"""
backend/app/services/model_registry.py

Model Registry: loads disease classifier once at startup (singleton pattern),
reports readiness, and provides a safe inference interface.

Model statuses:
  READY             — weights found and loaded successfully
  MISSING_WEIGHTS   — .pt file does not exist at configured path
  LOAD_ERROR        — file exists but YOLO failed to load it
  NOT_CONFIGURED    — no path configured
  ULTRALYTICS_MISSING — ultralytics package not installed
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Resolve paths relative to THIS file (project_root/backend/app/services/)
_SERVICES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICES_DIR.parent.parent.parent  # → project root

# Default model path — can be overridden via MODEL_WEIGHTS_PATH env var
_DEFAULT_WEIGHTS_REL = os.path.join("backend", "models", "yolo26_tomato.pt")
_WEIGHTS_PATH = Path(os.getenv("MODEL_WEIGHTS_PATH", str(_PROJECT_ROOT / _DEFAULT_WEIGHTS_REL)))

# ---------------------------------------------------------------
# Internal singleton state
# ---------------------------------------------------------------
_model_instance = None
_model_status: str = "NOT_INITIALIZED"
_model_meta: Dict[str, Any] = {}


def _load_model_once() -> None:
    """Load YOLO model once into module-level singleton."""
    global _model_instance, _model_status, _model_meta

    if _model_status in ("READY", "LOAD_ERROR", "MISSING_WEIGHTS",
                         "ULTRALYTICS_MISSING", "NOT_CONFIGURED"):
        return  # already attempted

    weights_path = _WEIGHTS_PATH

    if not weights_path:
        _model_status = "NOT_CONFIGURED"
        _model_meta = {"reason": "No model weights path configured."}
        return

    try:
        from ultralytics import YOLO as _YOLO  # noqa: N811
    except ImportError:
        _model_status = "ULTRALYTICS_MISSING"
        _model_meta = {
            "reason": "ultralytics package not installed. Run: pip install ultralytics"
        }
        return

    if not weights_path.exists():
        _model_status = "MISSING_WEIGHTS"
        _model_meta = {
            "reason": f"Weights file not found: {weights_path}",
            "expected_path": str(weights_path),
            "action": (
                "Train the model and copy weights to the path above, "
                "or set MODEL_WEIGHTS_PATH environment variable."
            ),
        }
        return

    try:
        _model_instance = _YOLO(str(weights_path))
        _model_status = "READY"
        _model_meta = {
            "weights_path": str(weights_path),
            "task": "classification",
            "note": (
                "Task is classification (top-1 label + confidence). "
                "Detection/segmentation tasks require a different model file."
            ),
        }
        print(f"[ModelRegistry] Disease model loaded: {weights_path}")
    except Exception as exc:
        _model_instance = None
        _model_status = "LOAD_ERROR"
        _model_meta = {
            "reason": f"YOLO raised an exception during load: {exc}",
            "weights_path": str(weights_path),
        }
        print(f"[ModelRegistry] LOAD_ERROR: {exc}")


def get_disease_model_status() -> Dict[str, Any]:
    """Return current readiness status of the disease classifier."""
    _load_model_once()
    return {
        "model_name": "disease_classifier",
        "status": _model_status,
        "weights_path": str(_WEIGHTS_PATH),
        "meta": _model_meta,
    }


def predict_disease(image) -> Dict[str, Any]:
    """
    Run disease classification on a single image (numpy array).

    Returns a dict with:
        prediction     — class name string (or status message)
        confidence     — float 0–1 (0.0 when model unavailable)
        model_status   — current registry status
        model_loaded   — bool
    """
    _load_model_once()

    if _model_status != "READY" or _model_instance is None:
        return {
            "prediction": f"model_unavailable ({_model_status})",
            "confidence": 0.0,
            "model_status": _model_status,
            "model_loaded": False,
        }

    try:
        results = _model_instance.predict(image, verbose=False)
        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                top_idx = int(probs.top1)
                top_conf = float(probs.top1conf)
                names = results[0].names
                class_name = names.get(top_idx, f"class_{top_idx}")
                return {
                    "prediction": class_name,
                    "confidence": round(top_conf, 4),
                    "model_status": "READY",
                    "model_loaded": True,
                }
        return {
            "prediction": "no_output",
            "confidence": 0.0,
            "model_status": "READY",
            "model_loaded": True,
        }
    except Exception as exc:
        print(f"[ModelRegistry] Inference error: {exc}")
        return {
            "prediction": "inference_error",
            "confidence": 0.0,
            "model_status": f"INFERENCE_ERROR: {exc}",
            "model_loaded": True,
        }
