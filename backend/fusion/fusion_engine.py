from typing import Dict, Any, List
from backend.app.config.settings import settings

CLASSES = ["healthy", "early_blight", "late_blight", "septoria_leaf_spot"]

def evaluate_tile_fusion(yolo_prediction: str, yolo_confidence: float, opencv_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates dual-evidence fusion logic for a single tile.
    
    YOLO Disease + OpenCV Abnormal -> CONFIRMED DISEASE
    YOLO Healthy + OpenCV Normal   -> HEALTHY
    Otherwise                      -> DISPUTED
    """
    yolo_is_healthy = (yolo_prediction == "healthy")
    yolo_is_disease = (yolo_prediction in ["early_blight", "late_blight", "septoria_leaf_spot"])
    opencv_status = opencv_result.get("visual_status", "abnormal")
    opencv_is_normal = (opencv_status == "normal")

    if yolo_is_healthy and opencv_is_normal:
        fusion_status = "HEALTHY"
        confirmed_disease = None
    elif yolo_is_disease and not opencv_is_normal:
        fusion_status = "CONFIRMED DISEASE"
        confirmed_disease = yolo_prediction
    else:
        fusion_status = "DISPUTED"
        confirmed_disease = yolo_prediction if yolo_is_disease else None

    lesion_pct = opencv_result.get("lesion_percentage", 0.0)
    severity = calculate_severity(lesion_pct)

    return {
        "yolo_prediction": yolo_prediction,
        "yolo_confidence": round(yolo_confidence, 4),
        "opencv_status": opencv_status,
        "greenness_index": opencv_result.get("greenness_index", 0.0),
        "lesion_percentage": lesion_pct,
        "fusion_status": fusion_status,
        "confirmed_disease": confirmed_disease,
        "severity": severity,
        "requires_closeup": (fusion_status == "DISPUTED")
    }

def calculate_severity(lesion_pct: float) -> str:
    """Configurable project severity thresholding."""
    if lesion_pct <= settings.SEVERITY_LOW_MAX:
        return "Low"
    elif lesion_pct <= settings.SEVERITY_MODERATE_MAX:
        return "Moderate"
    elif lesion_pct <= settings.SEVERITY_HIGH_MAX:
        return "High"
    else:
        return "Severe"

def calculate_plant_health_index(tile_results: List[Dict[str, Any]]) -> float:
    """
    Computes Prototype Plant Health Index (0-100).
    Formula:
      + 6.25 points per HEALTHY tile (16 tiles max = 100 points)
      + 3.0 points per DISPUTED tile (penalty for uncertainty)
      - Lesion percentage penalty factor
    """
    if not tile_results:
        return 100.0

    healthy_count = sum(1 for t in tile_results if t["fusion_status"] == "HEALTHY")
    disputed_count = sum(1 for t in tile_results if t["fusion_status"] == "DISPUTED")
    confirmed_count = sum(1 for t in tile_results if t["fusion_status"] == "CONFIRMED DISEASE")

    total_tiles = len(tile_results)
    tile_score = (healthy_count * 6.25) + (disputed_count * 3.0) + (confirmed_count * 0.0)

    avg_lesion_pct = sum(t["lesion_percentage"] for t in tile_results) / total_tiles
    lesion_penalty = (avg_lesion_pct / 100.0) * 20.0

    health_index = max(0.0, min(100.0, tile_score - lesion_penalty))
    return round(float(health_index), 1)

def generate_priority_list(tile_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates priority inspection list ordered by severity and uncertainty."""
    priority_items = []
    for t in tile_results:
        if t["fusion_status"] == "HEALTHY":
            continue

        priority_level = 3
        if t["fusion_status"] == "CONFIRMED DISEASE":
            priority_level = 1 if t["severity"] in ["High", "Severe"] else 2
        elif t["fusion_status"] == "DISPUTED":
            priority_level = 2

        disease_label = t.get("confirmed_disease")
        if not disease_label and t.get("yolo_prediction") != "YOLO model not trained":
            disease_label = t.get("yolo_prediction")

        priority_items.append({
            "tile_id": t["tile_id"],
            "fusion_status": t["fusion_status"],
            "disease": disease_label,
            "severity": t["severity"],
            "lesion_percentage": t["lesion_percentage"],
            "priority_level": priority_level
        })

    priority_items.sort(key=lambda x: (x["priority_level"], -x["lesion_percentage"]))
    return priority_items
