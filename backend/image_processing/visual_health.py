import cv2
import numpy as np
from typing import Dict, Any
from backend.image_processing.color_analysis import compute_color_metrics
from backend.image_processing.lesion_detection import compute_lesion_percentage

def analyze_visual_health(image: np.ndarray) -> Dict[str, Any]:
    """
    Method B: Classical digital image processing visual health classification.
    Returns visual_status: 'normal' or 'abnormal'.
    Does NOT fabricate disease names.
    """
    color_metrics = compute_color_metrics(image)
    lesion_metrics = compute_lesion_percentage(image)

    greenness_index = color_metrics["greenness_index"]
    lesion_pct = lesion_metrics["lesion_percentage"]

    # Rule-based thresholding for normal vs abnormal
    # Normal: Sufficient greenness index (> 0.32) AND low lesion percentage (< 8.0%)
    if greenness_index >= 0.32 and lesion_pct < 8.0:
        visual_status = "normal"
    else:
        visual_status = "abnormal"

    return {
        "visual_status": visual_status,
        "greenness_index": greenness_index,
        "green_pixel_percentage": color_metrics["green_pixel_percentage"],
        "lesion_percentage": lesion_pct,
        "mean_rgb": [color_metrics["mean_r"], color_metrics["mean_g"], color_metrics["mean_b"]]
    }
