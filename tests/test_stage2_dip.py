"""
Stage 2 Unit Tests for Digital Image Processing (DIP) Pipeline.
Uses Python built-in unittest framework.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing.dip_pipeline import ImagePreprocessor, DIPResult


class TestDIPPipeline(unittest.TestCase):
    """Unit test suite covering all Stage 2 DIP capabilities."""

    def setUp(self):
        """Set up test environment and synthetic image array."""
        self.preprocessor = ImagePreprocessor()
        # Create a synthetic 100x200 BGR test image
        self.test_img = np.zeros((100, 200, 3), dtype=np.uint8)
        self.test_img[:50, :100] = (255, 0, 0)     # Blue square
        self.test_img[50:, 100:] = (0, 255, 0)     # Green square

    def test_image_loading_and_validation(self):
        """Test array loading and invalid/missing file exception handling."""
        # 1. Valid array loading
        loaded = self.preprocessor.load_image(self.test_img)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (100, 200, 3))

        # 2. Empty array error handling
        with self.assertRaises(ValueError):
            self.preprocessor.load_image(np.array([]))

        # 3. Non-existent file error handling
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_image("non_existent_image_12345.png")

        # 4. Invalid extension error handling
        temp_invalid = BASE_DIR / "tests" / "temp_invalid.txt"
        temp_invalid.write_text("not an image")
        try:
            with self.assertRaises(ValueError):
                self.preprocessor.load_image(temp_invalid)
        finally:
            if temp_invalid.exists():
                temp_invalid.unlink()

    def test_resizing_aspect_ratio(self):
        """Test aspect-ratio preserving resize and non-aspect resize."""
        # Aspect ratio preserved (640x640 target)
        resized_padded = self.preprocessor.resize(
            self.test_img, target_size=(640, 640), preserve_aspect_ratio=True
        )
        self.assertEqual(resized_padded.shape, (640, 640, 3))

        # Non-aspect ratio stretch
        resized_stretched = self.preprocessor.resize(
            self.test_img, target_size=(300, 150), preserve_aspect_ratio=False
        )
        self.assertEqual(resized_stretched.shape, (150, 300, 3))

    def test_noise_reduction(self):
        """Test Gaussian blur and Median blur operations."""
        gaussian = self.preprocessor.reduce_noise(self.test_img, method="gaussian_blur", kernel_size=3)
        self.assertEqual(gaussian.shape, self.test_img.shape)

        median = self.preprocessor.reduce_noise(self.test_img, method="median_blur", kernel_size=3)
        self.assertEqual(median.shape, self.test_img.shape)

        with self.assertRaises(ValueError):
            self.preprocessor.reduce_noise(self.test_img, method="invalid_method")

    def test_contrast_enhancement(self):
        """Test CLAHE and Global Histogram Equalization."""
        clahe_enhanced = self.preprocessor.enhance_contrast(self.test_img, method="clahe")
        self.assertEqual(clahe_enhanced.shape, self.test_img.shape)

        hist_enhanced = self.preprocessor.enhance_contrast(self.test_img, method="histogram_equalization")
        self.assertEqual(hist_enhanced.shape, self.test_img.shape)

        with self.assertRaises(ValueError):
            self.preprocessor.enhance_contrast(self.test_img, method="unknown_contrast")

    def test_color_space_conversion(self):
        """Test BGR to RGB, HSV, and LAB conversions."""
        rgb = self.preprocessor.convert_color_space(self.test_img, source_space="BGR", target_space="RGB")
        self.assertEqual(rgb.shape, self.test_img.shape)

        hsv = self.preprocessor.convert_color_space(self.test_img, source_space="BGR", target_space="HSV")
        self.assertEqual(hsv.shape, self.test_img.shape)

        lab = self.preprocessor.convert_color_space(self.test_img, source_space="BGR", target_space="LAB")
        self.assertEqual(lab.shape, self.test_img.shape)

    def test_pixel_normalization(self):
        """Test normalization of uint8 [0, 255] to float32 [0.0, 1.0]."""
        normalized = self.preprocessor.normalize_pixels(self.test_img, target_range=(0.0, 1.0))
        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)

    def test_optional_sharpening(self):
        """Test unsharp mask sharpening filter."""
        sharpened = self.preprocessor.sharpen(self.test_img, kernel_strength=1.0)
        self.assertEqual(sharpened.shape, self.test_img.shape)

    def test_full_pipeline_execution(self):
        """Test end-to-end DIP pipeline processing and metadata output."""
        result = self.preprocessor.process_pipeline(self.test_img)
        self.assertIsInstance(result, DIPResult)
        self.assertIsNotNone(result.processed_image)
        self.assertIn("original_dimensions", result.metadata)
        self.assertIn("processed_dimensions", result.metadata)
        self.assertTrue(result.metadata["is_normalized"])
        self.assertIn("original", result.intermediate_stages)
        self.assertIn("resize", result.intermediate_stages)


if __name__ == "__main__":
    unittest.main()
