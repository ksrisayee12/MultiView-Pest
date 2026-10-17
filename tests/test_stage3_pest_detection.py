"""
Stage 3 Unit Tests for YOLO26 Pest Detection Module.
Uses Python built-in unittest framework. Works without requiring external weights or datasets.
"""

import json
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.pest_detector import YOLOPestDetector, DetectionBox, PestDetectionResult


class TestStage3PestDetection(unittest.TestCase):
    """Unit test suite for YOLO26 Pest Detector engine and configurations."""

    def setUp(self):
        """Set up test instances and synthetic image array."""
        self.detector = YOLOPestDetector()
        self.test_img = np.zeros((300, 400, 3), dtype=np.uint8)

    def test_configuration_loading(self):
        """Verify detector loads yolo_pest_config.yaml and pest_classes.yaml correctly."""
        self.assertIsNotNone(self.detector.config)
        self.assertIsNotNone(self.detector.class_mappings)
        self.assertIn(25, self.detector.class_mappings)  # Aphids
        self.assertIn(24, self.detector.class_mappings)  # Armyworm
        self.assertIn(38, self.detector.class_mappings)  # Flea Beetle
        self.assertIn(19, self.detector.class_mappings)  # Black Cutworm
        self.assertIn(72, self.detector.class_mappings)  # Greenhouse Whitefly

    def test_class_mapping_preservation(self):
        """Verify IP102 official class names and warnings are preserved."""
        aphid_info = self.detector.class_mappings[25]
        self.assertEqual(aphid_info["official_name"], "aphids")
        self.assertEqual(aphid_info["display_name"], "Aphids")

        whitefly_info = self.detector.class_mappings[72]
        self.assertEqual(whitefly_info["official_name"], "Trialeurodes vaporariorum")
        self.assertEqual(whitefly_info["verification_status"], "UNVERIFIED_BOUNDING_BOX_SPARSITY")

    def test_detector_schema_and_result_structure(self):
        """Test PestDetectionResult and DetectionBox schema consistency."""
        box = DetectionBox(
            box_xyxy=[10.0, 20.0, 100.0, 150.0],
            confidence=0.85,
            ip102_class_id=25,
            official_name="aphids",
            display_name="Aphids",
            mapped_category="aphid",
            verification_status="VERIFIED",
        )
        self.assertEqual(box.box_xyxy, [10.0, 20.0, 100.0, 150.0])
        self.assertEqual(box.confidence, 0.85)

        res = PestDetectionResult(
            image_path="test_sample.png",
            image_shape=(300, 400, 3),
            detections=[box],
            total_pest_count=1,
            category_counts={"aphid": 1},
            model_status="INFERENCE_SUCCESS",
            inference_time_ms=12.5,
            metadata={"is_real_model_output": True},
        )
        self.assertEqual(res.total_pest_count, 1)
        self.assertEqual(res.category_counts["aphid"], 1)

    def test_json_serialization(self):
        """Test JSON dict conversion of PestDetectionResult."""
        box = DetectionBox(
            box_xyxy=[10.0, 20.0, 50.0, 60.0],
            confidence=0.9,
            ip102_class_id=24,
            official_name="army worm",
            display_name="Armyworm",
            mapped_category="armyworm",
            verification_status="VERIFIED",
        )
        res = PestDetectionResult(
            image_path="sample.png",
            image_shape=(300, 400, 3),
            detections=[box],
            total_pest_count=1,
            category_counts={"armyworm": 1},
            model_status="TEST_STATUS",
            inference_time_ms=5.0,
            metadata={},
        )
        d = res.to_dict()
        json_str = json.dumps(d)
        self.assertIn("army worm", json_str)
        self.assertIn("TEST_STATUS", json_str)

    def test_annotated_image_rendering(self):
        """Test drawing bounding boxes on image array."""
        box = DetectionBox(
            box_xyxy=[20.0, 30.0, 80.0, 90.0],
            confidence=0.75,
            ip102_class_id=38,
            official_name="flea beetle",
            display_name="Flea Beetle",
            mapped_category="beetle",
            verification_status="VERIFIED",
        )
        res = PestDetectionResult(
            image_path="in_memory_array",
            image_shape=self.test_img.shape,
            detections=[box],
            total_pest_count=1,
            category_counts={"beetle": 1},
            model_status="TEST_STATUS",
            inference_time_ms=2.0,
            metadata={},
        )
        annotated = self.detector.annotate_image(self.test_img, res)
        self.assertEqual(annotated.shape, self.test_img.shape)

    def test_missing_model_handling(self):
        """Test graceful offline handling when model weights or package are absent."""
        res = self.detector.detect(self.test_img)
        self.assertIsNotNone(res)
        self.assertEqual(res.total_pest_count, 0)
        self.assertIn("MODEL_PENDING", res.model_status)


if __name__ == "__main__":
    unittest.main()
