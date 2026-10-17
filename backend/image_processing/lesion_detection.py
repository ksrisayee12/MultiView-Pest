import cv2
import numpy as np
from typing import Dict, Any

def compute_lesion_percentage(image: np.ndarray) -> Dict[str, Any]:
    """
    Estimates leaf area vs abnormal/lesion area using HSV thresholding.
    Calculates: Lesion Percentage = (Abnormal Pixels / Leaf Pixels) * 100
    """
    if image is None or image.size == 0:
        return {
            "leaf_pixel_count": 0,
            "lesion_pixel_count": 0,
            "lesion_percentage": 0.0
        }

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Broad Foliage Mask (Greenish & Yellowish-green hues)
    lower_foliage = np.array([20, 20, 20])
    upper_foliage = np.array([90, 255, 255])
    foliage_mask = cv2.inRange(hsv, lower_foliage, upper_foliage)

    # 2. Abnormal / Lesion Mask (Brown, Dark Yellow, Necrotic spots, Blight patches)
    lower_brown = np.array([0, 30, 20])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    abnormal_mask = cv2.bitwise_or(brown_mask, yellow_mask)

    leaf_pixels = np.count_nonzero(foliage_mask)
    abnormal_pixels = np.count_nonzero(abnormal_mask)

    total_pixels = image.shape[0] * image.shape[1]
    effective_leaf_pixels = max(leaf_pixels, int(total_pixels * 0.1)) # fallback to prevent zero division

    lesion_pct = (abnormal_pixels / effective_leaf_pixels) * 100.0
    lesion_pct = min(100.0, max(0.0, lesion_pct))

    return {
        "leaf_pixel_count": int(effective_leaf_pixels),
        "lesion_pixel_count": int(abnormal_pixels),
        "lesion_percentage": round(float(lesion_pct), 2)
    }
