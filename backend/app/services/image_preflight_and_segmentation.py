"""
backend/app/services/image_preflight_and_segmentation.py

Advanced Computer Vision Diagnostic Preprocessor:
  1. Pre-flight Quality Assessment (blur detection, lighting validation, foliage coverage)
  2. Foliar Isolation & Background Suppression (neutralizes hands, soil, tables, sky)
  3. Saliency-Guided High-Resolution Lesion Patch Extraction (preserves micro-textures)
  4. Test-Time Augmentation (TTA) Generator (4-angle probability smoothing)
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torchvision.transforms as T


def evaluate_image_quality(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Performs pre-flight optical validation before diagnosis.
    Returns blur index, foliar coverage %, exposure status, and actionable recommendations.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {
            "is_acceptable": False,
            "blur_score": 0.0,
            "foliage_coverage_pct": 0.0,
            "exposure": "INVALID",
            "warning": "Input image buffer is empty or corrupted.",
        }

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Blur Detection using Laplacian Variance
    blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
    is_blurry = blur_score < 30.0

    # 2. Plant Foliage Coverage using Excess Green Index (ExG)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    foliage_mask = (exg > 8.0).astype(np.uint8)
    foliage_pixels = int(np.sum(foliage_mask > 0))
    foliage_pct = round((foliage_pixels / max(h * w, 1)) * 100.0, 1)

    # 3. Exposure / Illumination
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 35.0:
        exposure = "UNDEREXPOSED"
        exp_warning = "Photo is too dark. Increase ambient lighting or turn on flash."
    elif mean_brightness > 225.0:
        exposure = "OVEREXPOSED"
        exp_warning = "Photo is washed out by direct glare. Reduce harsh lighting."
    else:
        exposure = "OPTIMAL"
        exp_warning = None

    warnings = []
    if is_blurry:
        warnings.append(f"Blurry focus detected (score: {blur_score}). Hold camera steady 4–8 inches away.")
    if foliage_pct < 4.0:
        warnings.append(f"Low plant foliage detected ({foliage_pct}%). Please frame the leaf clearly in view.")
    if exp_warning:
        warnings.append(exp_warning)

    is_acceptable = (not is_blurry or blur_score >= 20.0) and foliage_pct >= 2.0

    return {
        "is_acceptable": is_acceptable,
        "blur_score": blur_score,
        "is_blurry": is_blurry,
        "foliage_coverage_pct": foliage_pct,
        "exposure": exposure,
        "mean_brightness": round(mean_brightness, 1),
        "warning": " | ".join(warnings) if warnings else None,
    }


def segment_foliage_background(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Isolates plant leaf tissue and suppresses noisy in-the-wild backgrounds
    (soil, fingers, greenhouse plastic, table surfaces).
    
    Returns:
      segmented_bgr: Leaf tissue placed on a neutral gray canvas.
      foliage_mask: Binary mask of plant material.
      meta: Bounding box and segmentation metadata.
    """
    h, w = img_bgr.shape[:2]
    
    # Excess Green + Plant Hue HSV
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_mask = (exg > 8.0).astype(np.uint8) * 255

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Green and yellow chlorotic plant hues
    lower_plant = np.array([15, 30, 25], dtype=np.uint8)
    upper_plant = np.array([95, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_plant, upper_plant)

    # Necrotic lesion recovery (brown/tan spots on leaf)
    lower_brown = np.array([8, 30, 25], dtype=np.uint8)
    upper_brown = np.array([32, 255, 220], dtype=np.uint8)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    combined_mask = cv2.bitwise_or(exg_mask, hsv_mask)
    combined_mask = cv2.bitwise_or(combined_mask, brown_mask)

    # Morphological filtering to close veins and remove speckles
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)

    # Find dominant contours (leaf canopy)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    min_leaf_area = h * w * 0.01  # At least 1% of frame

    bounding_rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_leaf_area:
            cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)
            bounding_rects.append(cv2.boundingRect(c))

    # If mask is empty or degenerate, fallback to original
    if np.sum(final_mask > 0) < (h * w * 0.02):
        final_mask = np.ones((h, w), dtype=np.uint8) * 255
        segmented_bgr = img_bgr.copy()
        crop_box = [0, 0, w, h]
    else:
        # Create neutral studio background (neutral dark gray RGB 45,45,45)
        neutral_bg = np.full_like(img_bgr, 45)
        segmented_bgr = np.where(final_mask[:, :, None] == 255, img_bgr, neutral_bg)
        
        # Bounding box of the largest leaf region
        ys, xs = np.where(final_mask > 0)
        crop_box = [int(np.min(xs)), int(np.min(ys)), int(np.max(xs) - np.min(xs)), int(np.max(ys) - np.min(ys))]

    return segmented_bgr, final_mask, {"crop_box": crop_box, "leaf_pixel_ratio": float(np.mean(final_mask > 0))}


def extract_high_res_lesion_patches(img_bgr: np.ndarray, foliage_mask: np.ndarray, target_size: int = 224) -> List[Dict[str, Any]]:
    """
    Extracts high-resolution focused patches around symptomatic lesion sites.
    Preserves micro-textures (concentric rings, spot margins) that vanish during full-frame downsampling.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Necrotic lesion detector
    lower_brown = np.array([8, 35, 30], dtype=np.uint8)
    upper_brown = np.array([32, 255, 215], dtype=np.uint8)
    lesion_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    lesion_mask = cv2.bitwise_and(lesion_mask, foliage_mask)

    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patches = []

    # Sort contours by area (largest lesion hotspots first)
    valid_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.001)]
    valid_contours.sort(key=cv2.contourArea, reverse=True)

    for i, c in enumerate(valid_contours[:2]):
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw // 2, y + bh // 2
        
        # Determine patch crop window (at least 200px or 1.5x lesion dimension)
        crop_dim = max(int(max(bw, bh) * 1.6), 180)
        crop_dim = min(crop_dim, min(h, w))
        
        x1 = max(0, cx - crop_dim // 2)
        y1 = max(0, cy - crop_dim // 2)
        x2 = min(w, x1 + crop_dim)
        y2 = min(h, y1 + crop_dim)
        
        # Adjust if hitting frame edges
        if x2 - x1 < crop_dim and x1 > 0:
            x1 = max(0, x2 - crop_dim)
        if y2 - y1 < crop_dim and y1 > 0:
            y1 = max(0, y2 - crop_dim)

        patch_bgr = img_bgr[y1:y2, x1:x2]
        if patch_bgr.shape[0] > 20 and patch_bgr.shape[1] > 20:
            resized_patch = cv2.resize(patch_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)
            patches.append({
                "patch_bgr": resized_patch,
                "bbox": [x1, y1, x2, y2],
                "norm_bbox": [y1 / h, x1 / w, y2 / h, x2 / w],
                "type": f"lesion_hotspot_{i+1}",
                "saliency_area": int(cv2.contourArea(c))
            })

    # Always include a central high-res zoom patch
    center_dim = int(min(h, w) * 0.70)
    cx, cy = w // 2, h // 2
    cx1 = max(0, cx - center_dim // 2)
    cy1 = max(0, cy - center_dim // 2)
    cx2 = min(w, cx1 + center_dim)
    cy2 = min(h, cy1 + center_dim)
    
    center_crop = img_bgr[cy1:cy2, cx1:cx2]
    if center_crop.shape[0] > 20 and center_crop.shape[1] > 20:
        patches.append({
            "patch_bgr": cv2.resize(center_crop, (target_size, target_size), interpolation=cv2.INTER_AREA),
            "bbox": [cx1, cy1, cx2, cy2],
            "norm_bbox": [cy1 / h, cx1 / w, cy2 / h, cx2 / w],
            "type": "canopy_center_crop",
            "saliency_area": 0
        })

    return patches


def generate_tta_batch(img_rgb: np.ndarray, transform: T.Compose) -> torch.Tensor:
    """
    Generates a 4-variant Test-Time Augmentation (TTA) batch:
      1. Original
      2. Horizontal Mirror Flip
      3. Tight Zoom (90% Central Crop)
      4. Contrast-Normalized (CLAHE Luminance Enhancement)
    
    Returns:
      torch.Tensor of shape (4, 3, 224, 224) ready for a single batched forward pass.
    """
    tensors = []
    
    # 1. Standard
    tensors.append(transform(img_rgb))
    
    # 2. Horizontal Flip
    flipped = cv2.flip(img_rgb, 1)
    tensors.append(transform(flipped))
    
    # 3. Tight Central Zoom (90%)
    h, w = img_rgb.shape[:2]
    zh, zw = int(h * 0.88), int(w * 0.88)
    y1, x1 = (h - zh) // 2, (w - zw) // 2
    zoomed = cv2.resize(img_rgb[y1:y1+zh, x1:x1+zw], (w, h), interpolation=cv2.INTER_LINEAR)
    tensors.append(transform(zoomed))
    
    # 4. Contrast Balanced (LAB Luminance Equalization)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    balanced_lab = cv2.merge((l_eq, a, b))
    balanced_rgb = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2RGB)
    tensors.append(transform(balanced_rgb))
    
    return torch.stack(tensors, dim=0)


def detect_leaf_perforations_and_pest_damage(
    img_bgr: np.ndarray,
) -> Tuple[bool, List[Dict[str, Any]], float]:
    """
    Detects physical foliar perforations, insect feeding holes, and caterpillar edge defoliation.
    Distinguishes physical missing leaf tissue from fungal chlorosis / necrotic lesions.
    
    Returns:
        (is_pest_chewed, hole_bounding_boxes, defoliation_percentage)
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, [], 0.0

    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    exg = 2.0 * g - r - b

    # Segment healthy green tissue + brown necrotic tissue to identify all plant matter
    green_mask = (exg > 10.0) | (
        (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 35)
    )
    necrotic_mask = (
        (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 32) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 35)
    )
    plant_mask = ((green_mask | necrotic_mask).astype(np.uint8)) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    plant_clean = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)

    # 1. Enclosed interior feeding holes via two-level contour hierarchy (cv2.RETR_CCOMP)
    contours, hierarchy = cv2.findContours(plant_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_boxes: List[Dict[str, Any]] = []
    total_hole_area = 0.0

    # Minimum hole area: real insect feeding holes are at least 120px or 0.12% of image area
    min_hole_px = max(120, int(h * w * 0.0012))
    # Holes larger than 15% of image area are background voids, not insect holes
    max_hole_px = int(h * w * 0.15)

    interior_holes_count = 0
    if hierarchy is not None:
        for i, h_info in enumerate(hierarchy[0]):
            # If h_info[3] >= 0, this contour is an interior hole enclosed completely by plant tissue
            if h_info[3] >= 0:
                area = cv2.contourArea(contours[i])
                if min_hole_px <= area <= max_hole_px:
                    perim = cv2.arcLength(contours[i], True)
                    circularity = (4 * 3.14159 * area) / (perim * perim + 1e-6)
                    # Filter out long thin cracks, leaf veins, or irregular gaps (circularity must be >= 0.16)
                    if circularity >= 0.16:
                        x, y, bw, bh = cv2.boundingRect(contours[i])
                        total_hole_area += area
                        interior_holes_count += 1
                        hole_boxes.append({
                            "box": [float(x), float(y), float(x + bw), float(y + bh)],
                            "norm_box": [
                                round(y / max(h, 1), 4),
                                round(x / max(w, 1), 4),
                                round((y + bh) / max(h, 1), 4),
                                round((x + bw) / max(w, 1), 4),
                            ],
                            "label": "pest_feeding_hole",
                            "raw_label": "Pest Feeding Hole",
                            "confidence": 0.88,
                            "area_px": int(area),
                            "type": "interior",
                        })

    # 2. Defoliation margin bites & notches: ONLY for a single isolated leaf lamina
    # Compound leaves (e.g. tomato/potato) naturally have spaces between leaflets; never treat those as bites!
    ext_contours, _ = cv2.findContours(plant_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    major_leaves = [c for c in ext_contours if cv2.contourArea(c) > (h * w * 0.06)]
    total_blade_area = sum(cv2.contourArea(c) for c in major_leaves) if major_leaves else float(np.sum(plant_clean > 0))

    if len(major_leaves) == 1:
        c = major_leaves[0]
        hull = cv2.convexHull(c)
        leaf_solid = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(leaf_solid, [hull], -1, 255, -1)

        missing = cv2.bitwise_and(leaf_solid, cv2.bitwise_not(plant_clean))
        missing = cv2.morphologyEx(missing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        bite_contours, _ = cv2.findContours(missing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Valid margin bites must be large, deep, and few in number (<= 5)
        valid_bites = [bc for bc in bite_contours if cv2.contourArea(bc) >= max(200, int(h * w * 0.003))]
        if 1 <= len(valid_bites) <= 5:
            for bc in valid_bites:
                area = cv2.contourArea(bc)
                x, y, bw, bh = cv2.boundingRect(bc)
                is_dup = any(
                    abs(x - hb["box"][0]) < 15 and abs(y - hb["box"][1]) < 15 for hb in hole_boxes
                )
                if not is_dup:
                    total_hole_area += area
                    hole_boxes.append({
                        "box": [float(x), float(y), float(x + bw), float(y + bh)],
                        "norm_box": [
                            round(y / max(h, 1), 4),
                            round(x / max(w, 1), 4),
                            round((y + bh) / max(h, 1), 4),
                            round((x + bw) / max(w, 1), 4),
                        ],
                        "label": "pest_feeding_hole",
                        "raw_label": "Pest Chewing Defoliation",
                        "confidence": 0.85,
                        "area_px": int(area),
                        "type": "margin",
                    })

    defoliation_pct = round((total_hole_area / max(total_blade_area, 1)) * 100.0, 2)
    # Genuine pest chewing requires either:
    # A. At least 2 interior holes with defoliation >= 2.5%
    # B. At least 1 significant hole/bite with defoliation >= 7.0%
    is_pest_chewed = (interior_holes_count >= 2 and defoliation_pct >= 2.5) or (
        len(hole_boxes) >= 1 and defoliation_pct >= 7.0
    )

    return is_pest_chewed, hole_boxes, defoliation_pct

    return is_pest_chewed, hole_boxes, defoliation_pct

