"""
YOLO26 Pest Detection Wrapper Module.
Stage 3: Object Detection Interface & Result Parser for Agricultural Health Analysis.
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class DetectionBox:
    """Individual pest bounding-box detection object."""
    box_xyxy: List[float]  # [xmin, ymin, xmax, ymax]
    confidence: float
    ip102_class_id: int
    official_name: str
    display_name: str
    mapped_category: str
    verification_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PestDetectionResult:
    """Structured result returned by YOLO26 Pest Detector."""
    image_path: str
    image_shape: Tuple[int, int, int]
    detections: List[DetectionBox]
    total_pest_count: int
    category_counts: Dict[str, int]
    model_status: str
    inference_time_ms: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["detections"] = [d.to_dict() for d in self.detections]
        return data


class YOLOPestDetector:
    """Ultralytics YOLO26 Pest Detector Wrapper with IP102 Official Mapping."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        classes_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize detector, load configs, and check model/dependency availability."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.config_path = Path(config_path) if config_path else base_dir / "configs" / "yolo_pest_config.yaml"
        self.classes_path = Path(classes_path) if classes_path else base_dir / "configs" / "pest_classes.yaml"

        self.config = self._load_yaml_file(self.config_path)
        self.class_mappings = self._load_class_mappings(self.classes_path)

        # Check ultralytics dependency and weights file
        self.ultralytics_available = self._check_ultralytics()
        weights_setting = self.config.get("model_settings", {}).get("weights_path", "models/pest/yolo26n_ip102_pest.pt")
        self.weights_path = Path(weights_setting)
        if not self.weights_path.is_absolute():
            self.weights_path = base_dir / self.weights_path

        self.model_weights_exist = self.weights_path.exists()
        self.model_instance = None

        if self.ultralytics_available and self.model_weights_exist:
            self.is_model_available = True
            self.model_status = "MODEL_READY"
            self._load_yolo_model()
        else:
            self.is_model_available = False
            reasons = []
            if not self.ultralytics_available:
                reasons.append("ultralytics package not installed")
            if not self.model_weights_exist:
                reasons.append(f"weights file missing at {self.weights_path}")
            self.model_status = f"MODEL_PENDING_DEPENDENCY_OR_WEIGHTS ({', '.join(reasons)})"

    @staticmethod
    def _check_ultralytics() -> bool:
        """Check if ultralytics package is installed without raising errors."""
        try:
            import ultralytics
            return True
        except ImportError:
            return False

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration with PyYAML if available, or lightweight line parser fallback."""
        if not path.exists():
            return {}
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            return self._parse_yaml_fallback(path)

    def _parse_yaml_fallback(self, path: Path) -> Dict[str, Any]:
        """Fallback lightweight text parser for basic YAML structures when pyyaml is missing."""
        result = {"classes": [], "model_settings": {}, "output_settings": {}}
        current_class = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw_line = line.strip()
                if not raw_line or raw_line.startswith("#"):
                    continue

                if raw_line.startswith("- ip102_class_id:"):
                    if current_class:
                        result["classes"].append(current_class)
                    cid = int(raw_line.split(":")[1].strip())
                    current_class = {"ip102_class_id": cid}
                elif current_class and ":" in raw_line:
                    parts = raw_line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip().strip('"').strip("'")
                    current_class[key] = val
                elif ":" in raw_line and not raw_line.startswith("-"):
                    parts = raw_line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip().strip('"').strip("'")
                    if key == "weights_path":
                        result.setdefault("model_settings", {})["weights_path"] = val
                    elif key == "output_directory":
                        result.setdefault("output_settings", {})["output_directory"] = val

        if current_class:
            result["classes"].append(current_class)

        return result

    def _load_class_mappings(self, path: Path) -> Dict[int, Dict[str, Any]]:
        """Load IP102 class mappings indexed by ip102_class_id."""
        raw_cfg = self._load_yaml_file(path)
        mappings = {}
        classes_list = raw_cfg.get("classes", [])
        for c in classes_list:
            if isinstance(c, dict) and "ip102_class_id" in c:
                cid = int(c["ip102_class_id"])
                mappings[cid] = c
        return mappings

    def _load_yolo_model(self):
        """Instantiate Ultralytics YOLO model if package and weights exist."""
        try:
            from ultralytics import YOLO
            self.model_instance = YOLO(str(self.weights_path))
        except Exception as e:
            self.is_model_available = False
            self.model_status = f"MODEL_LOAD_FAILED ({str(e)})"

    def detect(
        self,
        image_source: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> PestDetectionResult:
        """
        Execute pest detection on input image.
        If model is pending weights/dependency, returns clean status report without crashing.
        """
        start_time = time.time()
        model_cfg = self.config.get("model_settings", {})
        conf = conf_threshold if conf_threshold is not None else model_cfg.get("confidence_threshold", 0.25)
        iou = iou_threshold if iou_threshold is not None else model_cfg.get("iou_threshold", 0.45)

        # Load input image
        if isinstance(image_source, np.ndarray):
            img = image_source
            img_path_str = "in_memory_array"
        else:
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"Input image not found: {path}")
            img_path_str = str(path)
            img = cv2.imread(img_path_str)
            if img is None:
                raise ValueError(f"Corrupt or unreadable image: {path}")

        img_shape = img.shape
        detections: List[DetectionBox] = []

        if not self.is_model_available or self.model_instance is None:
            # Model pending - return clear non-fabricated status
            inference_time = (time.time() - start_time) * 1000.0
            return PestDetectionResult(
                image_path=img_path_str,
                image_shape=img_shape,
                detections=[],
                total_pest_count=0,
                category_counts={},
                model_status=self.model_status,
                inference_time_ms=round(inference_time, 2),
                metadata={
                    "confidence_threshold": conf,
                    "iou_threshold": iou,
                    "model_weights_path": str(self.weights_path),
                    "is_real_model_output": False,
                },
            )

        # Execute real YOLO inference
        results = self.model_instance.predict(img, conf=conf, iou=iou, verbose=False)
        category_counts: Dict[str, int] = {}

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Lookup class mapping from IP102 config
                class_info = self.class_mappings.get(class_id, {
                    "official_name": f"IP102_Class_{class_id}",
                    "display_name": f"Pest Class {class_id}",
                    "mapped_category": "unknown",
                    "verification_status": "UNMAPPED",
                })

                det_box = DetectionBox(
                    box_xyxy=[round(coord, 2) for coord in xyxy],
                    confidence=round(confidence, 4),
                    ip102_class_id=class_id,
                    official_name=class_info.get("official_name", f"IP102_Class_{class_id}"),
                    display_name=class_info.get("display_name", f"Pest Class {class_id}"),
                    mapped_category=class_info.get("mapped_category", "unknown"),
                    verification_status=class_info.get("verification_status", "UNMAPPED"),
                )
                detections.append(det_box)
                cat = det_box.mapped_category
                category_counts[cat] = category_counts.get(cat, 0) + 1

        inference_time = (time.time() - start_time) * 1000.0

        return PestDetectionResult(
            image_path=img_path_str,
            image_shape=img_shape,
            detections=detections,
            total_pest_count=len(detections),
            category_counts=category_counts,
            model_status="INFERENCE_SUCCESS",
            inference_time_ms=round(inference_time, 2),
            metadata={
                "confidence_threshold": conf,
                "iou_threshold": iou,
                "model_weights_path": str(self.weights_path),
                "is_real_model_output": True,
            },
        )

    def annotate_image(self, image: np.ndarray, result: PestDetectionResult) -> np.ndarray:
        """Draw bounding boxes and class labels onto image copy."""
        annotated = image.copy()
        for det in result.detections:
            xmin, ymin, xmax, ymax = map(int, det.box_xyxy)
            label = f"{det.display_name} ({det.confidence:.2f})"
            color = (0, 0, 255)  # Red for pest bounding box

            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                annotated,
                label,
                (xmin, max(ymin - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return annotated

    def save_results(
        self,
        result: PestDetectionResult,
        output_dir: Optional[Union[str, Path]] = None,
        image_input: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """Save JSON detection results and annotated image output to disk."""
        out_dir = Path(output_dir) if output_dir else Path(self.config.get("output_settings", {}).get("output_directory", "data/processed/pest_detections"))
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(result.image_path).stem if result.image_path != "in_memory_array" else "pest_detection"
        json_path = out_dir / f"{base_name}_pest_result.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        output_paths = {"json_path": str(json_path)}

        if image_input is not None and self.config.get("output_settings", {}).get("save_annotated_images", True):
            annotated_img = self.annotate_image(image_input, result)
            img_path = out_dir / f"{base_name}_annotated.png"
            cv2.imwrite(str(img_path), annotated_img)
            output_paths["image_path"] = str(img_path)

        return output_paths
