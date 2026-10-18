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
import time
import hashlib
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
    try:
        # Compute SHA-256 for provenance tracking
        with open(weights_path, "rb") as f:
            weights_sha256 = hashlib.sha256(f.read()).hexdigest()

        _model_instance = _YOLO(str(weights_path))
        _model_status = "READY"
        model_task = getattr(_model_instance, "task", "detect")
        class_names = list(_model_instance.names.values()) if hasattr(_model_instance, "names") else []
        _model_meta = {
            "model_name": "tomato_disease_yolov8n",
            "model_version": "1.0.0",
            "weights_path": str(weights_path),
            "sha256": weights_sha256,
            "task": model_task,
            "classes": class_names,
            "supported_conditions": [
                "early_blight", "late_blight", "septoria_leaf_spot",
                "bacterial_spot", "yellow_leaf_curl", "mosaic_virus",
                "leaf_mold", "spider_mites", "healthy"
            ],
            "provenance_source": "Hugging Face (NarimanOsman2025/tomato-disease-detector-yolov8n)",
            "note": "Real fine-tuned YOLOv8n tomato disease detection model loaded and active for real-time inference.",
        }
        print(f"[ModelRegistry] Disease model loaded ({model_task}): {weights_path} [SHA-256: {weights_sha256[:10]}...]")
    except Exception as exc:
        _model_instance = None
        _model_status = "LOAD_ERROR"
        _model_meta = {
            "reason": f"YOLO raised an exception during load: {exc}",
            "weights_path": str(weights_path),
        }
        print(f"[ModelRegistry] LOAD_ERROR: {exc}")


def _normalize_disease_name(raw: str) -> str:
    """Normalize raw YOLO class names into standardized taxonomy keys."""
    s = raw.strip().lower()
    if "early" in s and "blight" in s:
        return "early_blight"
    if "late" in s and "blight" in s:
        return "late_blight"
    if "septoria" in s:
        return "septoria_leaf_spot"
    if "bacterial" in s:
        return "bacterial_spot"
    if "yellow" in s:
        return "yellow_leaf_curl"
    if "mosaic" in s:
        return "mosaic_virus"
    if "mold" in s:
        return "leaf_mold"
    if "spider" in s or "mite" in s:
        return "spider_mites"
    if s == "tomato leaf" or "healthy" in s:
        return "healthy"
    return s.replace(" ", "_")


def get_disease_model_status() -> Dict[str, Any]:
    """Return current readiness status and provenance of the disease classifier."""
    _load_model_once()
    return {
        "model_name": _model_meta.get("model_name", "tomato_disease_yolov8n"),
        "model_version": _model_meta.get("model_version", "1.0.0"),
        "status": _model_status,
        "weights_path": str(_WEIGHTS_PATH),
        "sha256": _model_meta.get("sha256"),
        "meta": _model_meta,
    }


def predict_disease(image) -> Dict[str, Any]:
    """
    Run disease inference on a single image (numpy array or image file).
    Supports both YOLO detection models (boxes) and classification models (probs).

    Returns a dict with:
        prediction     — normalized class name string or 'no_supported_condition_detected'
        confidence     — float 0–1
        model_status   — current registry status
        model_loaded   — bool
        boxes          — optional detected bounding boxes
        provenance     — model metadata for traceability
    """
    _load_model_once()

    provenance = {
        "model_name": _model_meta.get("model_name", "tomato_disease_yolov8n"),
        "model_version": _model_meta.get("model_version", "1.0.0"),
        "sha256": _model_meta.get("sha256", "unknown"),
        "timestamp": time.time(),
    }

    if _model_status != "READY" or _model_instance is None:
        return {
            "prediction": f"model_unavailable ({_model_status})",
            "confidence": 0.0,
            "model_status": _model_status,
            "model_loaded": False,
            "provenance": provenance,
        }

    try:
        results = _model_instance.predict(image, verbose=False)
        if results and len(results) > 0:
            res = results[0]
            names = res.names or {}

            # Case A: Object Detection model (boxes)
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                h, w = image.shape[:2] if hasattr(image, "shape") else (640, 640)
                detected_boxes = []
                for b in res.boxes:
                    coords = b.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(b.cls[0].cpu().numpy().item())
                    conf = float(b.conf[0].cpu().numpy().item())
                    lbl = names.get(cls_id, f"class_{cls_id}")
                    nlbl = _normalize_disease_name(lbl)
                    detected_boxes.append({
                        "box": [round(float(c), 1) for c in coords],
                        "norm_box": [
                            round(float(coords[1]) / max(h, 1), 4),
                            round(float(coords[0]) / max(w, 1), 4),
                            round(float(coords[3]) / max(h, 1), 4),
                            round(float(coords[2]) / max(w, 1), 4),
                        ],
                        "class_id": cls_id,
                        "raw_label": lbl,
                        "label": nlbl,
                        "confidence": round(conf, 4)
                    })

                top_box = max(detected_boxes, key=lambda b: b["confidence"])
                return {
                    "prediction": top_box["label"],
                    "confidence": top_box["confidence"],
                    "model_status": "READY",
                    "model_loaded": True,
                    "raw_label": top_box["raw_label"],
                    "box_count": len(detected_boxes),
                    "boxes": detected_boxes,
                    "provenance": provenance,
                }

            # Case B: Classification model (probs)
            if hasattr(res, "probs") and res.probs is not None:
                top_idx = int(res.probs.top1)
                top_conf = float(res.probs.top1conf)
                raw_label = names.get(top_idx, f"class_{top_idx}")
                norm_label = _normalize_disease_name(raw_label)
                return {
                    "prediction": norm_label,
                    "confidence": round(top_conf, 4),
                    "model_status": "READY",
                    "model_loaded": True,
                    "raw_label": raw_label,
                    "boxes": [],
                    "provenance": provenance,
                }

        # Model ran but found no supported condition bounding box on the leaf
        return {
            "prediction": "no_supported_condition_detected",
            "confidence": 0.0,
            "model_status": "READY",
            "model_loaded": True,
            "raw_label": "no_supported_condition_detected",
            "provenance": provenance,
        }
    except Exception as exc:
        print(f"[ModelRegistry] Inference error: {exc}")
        return {
            "prediction": "inference_error",
            "confidence": 0.0,
            "model_status": f"INFERENCE_ERROR: {exc}",
            "model_loaded": True,
            "provenance": provenance,
        }
