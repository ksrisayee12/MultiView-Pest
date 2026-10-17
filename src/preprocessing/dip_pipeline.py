"""
Digital Image Processing (DIP) Pipeline Module.
Stage 2: Modular Preprocessing Engine for Agricultural Image Assessment.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class DIPResult:
    """Structured result returned by the DIP pipeline."""
    processed_image: np.ndarray
    metadata: Dict[str, Any]
    intermediate_stages: Dict[str, np.ndarray]


class ImagePreprocessor:
    """Modular Digital Image Processing (DIP) engine."""

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessor with configuration dictionary or defaults."""
        self.config = config or self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Return standard fallback DIP configuration."""
        return {
            "image_constraints": {
                "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
                "target_width": 640,
                "target_height": 640,
                "preserve_aspect_ratio": True,
                "pad_color": [0, 0, 0],
            },
            "pipeline_steps": {
                "resize": {"enabled": True, "preserve_aspect_ratio": True},
                "noise_reduction": {
                    "enabled": True,
                    "method": "gaussian_blur",
                    "kernel_size": 3,
                    "sigma": 1.0,
                },
                "contrast_enhancement": {
                    "enabled": True,
                    "method": "clahe",
                    "clip_limit": 2.0,
                    "tile_grid_size": [8, 8],
                },
                "color_processing": {
                    "enabled": True,
                    "target_color_space": "RGB",
                },
                "normalization": {
                    "enabled": True,
                    "normalize_range": [0.0, 1.0],
                },
                "sharpening": {
                    "enabled": False,
                    "kernel_strength": 1.0,
                },
            },
        }

    def load_image(self, image_source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Load and validate an input image.
        Supports file path (str/Path) or numpy array input.
        Returns a fresh copy of the image array in BGR format.
        """
        if isinstance(image_source, np.ndarray):
            if image_source is None or image_source.size == 0:
                raise ValueError("Input image numpy array is empty or None.")
            return image_source.copy()

        path = Path(image_source)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image file extension '{path.suffix}'. Allowed: {self.ALLOWED_EXTENSIONS}"
            )

        # Load image via OpenCV
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError(f"Corrupt or unreadable image file: {path}")

        return img

    def resize(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        preserve_aspect_ratio: bool = True,
        pad_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Resize image to target dimensions.
        If preserve_aspect_ratio is True, applies letterboxing without distortion.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided for resize operation.")

        target_w, target_h = target_size
        h, w = image.shape[:2]

        if not preserve_aspect_ratio:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Compute scaling ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize scaled region
        resized_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create padded target canvas
        canvas = np.full((target_h, target_w, image.shape[2] if image.ndim == 3 else 1), pad_color, dtype=image.dtype)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2

        canvas[top : top + new_h, left : left + new_w] = resized_scaled
        return canvas

    def reduce_noise(
        self,
        image: np.ndarray,
        method: str = "gaussian_blur",
        kernel_size: int = 3,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Apply noise reduction (Gaussian blur or Median blur)."""
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd

        if method == "gaussian_blur":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
        elif method == "median_blur":
            return cv2.medianBlur(image, kernel_size)
        else:
            raise ValueError(f"Unknown noise reduction method: '{method}'. Options: 'gaussian_blur', 'median_blur'.")

    def enhance_contrast(
        self,
        image: np.ndarray,
        method: str = "clahe",
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Enhance image contrast using CLAHE or Global Histogram Equalization."""
        # Process in LAB color space to preserve color balance
        if image.ndim == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            if method == "clahe":
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
                l_enhanced = clahe.apply(l)
            elif method == "histogram_equalization":
                l_enhanced = cv2.equalizeHist(l)
            else:
                raise ValueError(f"Unknown contrast method: '{method}'. Options: 'clahe', 'histogram_equalization'.")

            enhanced_lab = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale fallback
            if method == "clahe":
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
                return clahe.apply(image)
            elif method == "histogram_equalization":
                return cv2.equalizeHist(image)
            else:
                raise ValueError(f"Unknown contrast method: '{method}'")

    def convert_color_space(
        self,
        image: np.ndarray,
        source_space: str = "BGR",
        target_space: str = "RGB",
    ) -> np.ndarray:
        """Convert image between color spaces (RGB, BGR, HSV, LAB)."""
        source_space = source_space.upper()
        target_space = target_space.upper()

        if source_space == target_space:
            return image.copy()

        conversion_code_name = f"COLOR_{source_space}2{target_space}"
        if hasattr(cv2, conversion_code_name):
            code = getattr(cv2, conversion_code_name)
            return cv2.cvtColor(image, code)
        else:
            # Handle multi-hop conversion via BGR if direct code does not exist
            to_bgr_code = getattr(cv2, f"COLOR_{source_space}2BGR", None)
            from_bgr_code = getattr(cv2, f"COLOR_BGR2{target_space}", None)
            if to_bgr_code is not None and from_bgr_code is not None:
                bgr_img = cv2.cvtColor(image, to_bgr_code)
                return cv2.cvtColor(bgr_img, from_bgr_code)

            raise ValueError(f"Unsupported color space conversion from {source_space} to {target_space}")

    def normalize_pixels(
        self,
        image: np.ndarray,
        target_range: Tuple[float, float] = (0.0, 1.0),
    ) -> np.ndarray:
        """Normalize pixel intensities to target range (default [0.0, 1.0] float32)."""
        img_float = image.astype(np.float32)
        min_val, max_val = target_range

        if image.dtype == np.uint8:
            img_normalized = (img_float / 255.0) * (max_val - min_val) + min_val
        else:
            curr_min, curr_max = img_float.min(), img_float.max()
            if curr_max > curr_min:
                img_normalized = ((img_float - curr_min) / (curr_max - curr_min)) * (max_val - min_val) + min_val
            else:
                img_normalized = np.full_like(img_float, min_val)

        return img_normalized.astype(np.float32)

    def sharpen(self, image: np.ndarray, kernel_strength: float = 1.0) -> np.ndarray:
        """Apply image sharpening using a Laplacian unsharp kernel."""
        kernel = np.array([[0, -1, 0], [-1, 4 + kernel_strength, -1], [0, -1, 0]], dtype=np.float32)
        if image.dtype == np.float32:
            return cv2.filter2D(image, -1, kernel)
        else:
            sharpened = cv2.filter2D(image, -1, kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)

    def process_pipeline(
        self,
        image_source: Union[str, Path, np.ndarray],
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> DIPResult:
        """
        Execute full configurable DIP pipeline:
        original -> resize -> noise_reduction -> contrast_enhancement -> color_processing -> normalization -> optional_sharpening
        """
        cfg = custom_config or self.config
        constraints = cfg.get("image_constraints", {})
        steps = cfg.get("pipeline_steps", {})

        # 1. Load Original Image
        original = self.load_image(image_source)
        orig_h, orig_w = original.shape[:2]
        filename = str(image_source) if isinstance(image_source, (str, Path)) else "in_memory_array"

        current_img = original.copy()
        intermediate_stages: Dict[str, np.ndarray] = {"original": original.copy()}
        ops_applied = ["load_image"]

        # 2. Resize
        resize_cfg = steps.get("resize", {})
        if resize_cfg.get("enabled", True):
            target_w = constraints.get("target_width", 640)
            target_h = constraints.get("target_height", 640)
            preserve_aspect = resize_cfg.get("preserve_aspect_ratio", True)
            pad_color = tuple(constraints.get("pad_color", [0, 0, 0]))

            current_img = self.resize(
                current_img,
                target_size=(target_w, target_h),
                preserve_aspect_ratio=preserve_aspect,
                pad_color=pad_color,
            )
            intermediate_stages["resize"] = current_img.copy()
            ops_applied.append("resize")

        # 3. Noise Reduction
        noise_cfg = steps.get("noise_reduction", {})
        if noise_cfg.get("enabled", True):
            method = noise_cfg.get("method", "gaussian_blur")
            k_size = noise_cfg.get("kernel_size", 3)
            sigma = noise_cfg.get("sigma", 1.0)

            current_img = self.reduce_noise(current_img, method=method, kernel_size=k_size, sigma=sigma)
            intermediate_stages["noise_reduction"] = current_img.copy()
            ops_applied.append(f"noise_reduction:{method}")

        # 4. Contrast Enhancement
        contrast_cfg = steps.get("contrast_enhancement", {})
        if contrast_cfg.get("enabled", True):
            method = contrast_cfg.get("method", "clahe")
            clip_limit = contrast_cfg.get("clip_limit", 2.0)
            grid_size = contrast_cfg.get("tile_grid_size", [8, 8])

            current_img = self.enhance_contrast(current_img, method=method, clip_limit=clip_limit, tile_grid_size=grid_size)
            intermediate_stages["contrast_enhancement"] = current_img.copy()
            ops_applied.append(f"contrast_enhancement:{method}")

        # 5. Color Processing
        color_cfg = steps.get("color_processing", {})
        if color_cfg.get("enabled", True):
            target_space = color_cfg.get("target_color_space", "RGB")
            current_img = self.convert_color_space(current_img, source_space="BGR", target_space=target_space)
            intermediate_stages["color_processing"] = current_img.copy()
            ops_applied.append(f"color_processing:{target_space}")

        # 6. Normalization
        norm_cfg = steps.get("normalization", {})
        is_normalized = False
        if norm_cfg.get("enabled", True):
            norm_range = tuple(norm_cfg.get("normalize_range", [0.0, 1.0]))
            current_img = self.normalize_pixels(current_img, target_range=norm_range)
            intermediate_stages["normalization"] = current_img.copy()
            ops_applied.append("normalization")
            is_normalized = True

        # 7. Optional Sharpening
        sharp_cfg = steps.get("sharpening", {})
        if sharp_cfg.get("enabled", False):
            strength = sharp_cfg.get("kernel_strength", 1.0)
            current_img = self.sharpen(current_img, kernel_strength=strength)
            intermediate_stages["sharpening"] = current_img.copy()
            ops_applied.append("sharpening")

        proc_h, proc_w = current_img.shape[:2]

        metadata = {
            "input_filename": os.path.basename(filename),
            "original_dimensions": {"height": orig_h, "width": orig_w},
            "processed_dimensions": {"height": proc_h, "width": proc_w},
            "operations_applied": ops_applied,
            "is_normalized": is_normalized,
            "data_type": str(current_img.dtype),
        }

        return DIPResult(
            processed_image=current_img,
            metadata=metadata,
            intermediate_stages=intermediate_stages,
        )
