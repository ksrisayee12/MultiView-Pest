import cv2
import numpy as np
from typing import Dict, Any

def compute_color_metrics(image: np.ndarray) -> Dict[str, Any]:
    """
    Computes greenness index, average RGB channels, and green pixel percentage.
    Formula: Greenness Index = G / (R + G + B + epsilon)
    """
    if image is None or image.size == 0:
        return {
            "greenness_index": 0.0,
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "green_pixel_percentage": 0.0
        }

    # Convert BGR (OpenCV default) to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image.copy()

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    epsilon = 1e-6
    total_rgb = r + g + b + epsilon
    greenness_map = g / total_rgb
    mean_greenness = float(np.mean(greenness_map))

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    # Green pixel identification (where green dominant over red and blue)
    green_mask = (g > r) & (g > b) & (g > 30)
    green_pixels = np.count_nonzero(green_mask)
    total_pixels = image.shape[0] * image.shape[1]
    green_pixel_pct = float((green_pixels / total_pixels) * 100.0)

    return {
        "greenness_index": round(mean_greenness, 4),
        "mean_r": round(mean_r, 2),
        "mean_g": round(mean_g, 2),
        "mean_b": round(mean_b, 2),
        "green_pixel_percentage": round(green_pixel_pct, 2)
    }
