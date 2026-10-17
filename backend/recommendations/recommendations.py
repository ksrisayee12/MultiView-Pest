from typing import Dict, Any, List

RECOMMENDATION_DATABASE: Dict[str, Dict[str, Any]] = {
    "healthy": {
        "title": "Healthy Crop Status",
        "treatment": "No active disease treatment required. Maintain current agronomic schedule.",
        "prevention": [
            "Inspect foliage regularly for early pest or lesion signs.",
            "Maintain proper plant spacing for efficient airflow.",
            "Ensure proper drip irrigation to avoid prolonged foliage wetness.",
            "Monitor soil nutrient and moisture balance."
        ]
    },
    "early_blight": {
        "title": "Tomato Early Blight (Alternaria solani)",
        "treatment": "Prune severely damaged lower leaves. Apply label-approved copper-based or chlorothalonil fungicide if disease progression persists.",
        "prevention": [
            "Remove and destroy infected leaf debris after harvest.",
            "Rotate tomato crops with non-solanaceous crops every 2-3 years.",
            "Apply mulch around plant base to minimize soil splash onto leaves.",
            "Avoid overhead sprinkler irrigation."
        ]
    },
    "late_blight": {
        "title": "Tomato Late Blight (Phytophthora infestans)",
        "treatment": "Isolate and promptly remove severely affected plant tissue. Apply label-registered protective or systemic fungicides (e.g., mancozeb or copper octanoate) as directed by local extension services.",
        "prevention": [
            "Use certified disease-free tomato seeds and transplants.",
            "Maintain wide plant spacing to facilitate rapid drying of leaves.",
            "Monitor field continuously during cool, wet weather conditions.",
            "Destroy volunteer tomato and potato plants nearby."
        ]
    },
    "septoria_leaf_spot": {
        "title": "Septoria Leaf Spot (Septoria lycopersici)",
        "treatment": "Remove heavily spotted lower leaves. Apply recommended broad-spectrum protective fungicides according to local agricultural authority guidelines.",
        "prevention": [
            "Practice strict field sanitation and crop rotation.",
            "Control weeds and wild solanaceous plants surrounding the field.",
            "Ensure adequate sunlight penetration through canopy management.",
            "Disinfect pruning tools between plants."
        ]
    },
    "yolo_model_not_trained": {
        "title": "YOLO Model Not Trained",
        "treatment": "YOLO disease classification model is not yet trained. Train the YOLO26 model using the Colab notebook or place weights in backend/models/yolo26_tomato.pt for specific disease control recommendations.",
        "prevention": [
            "Inspect foliage visually for lesion progression.",
            "Maintain proper plant spacing for efficient airflow.",
            "Avoid overhead irrigation to reduce foliage moisture.",
            "Follow standard agricultural extension advice until model training is complete."
        ]
    }
}

DISCLAIMER = (
    "NOTICE: Treatment and prevention recommendations provided by this prototype system are for "
    "informational and educational guidance only. Always verify recommendations against "
    "registered product labels and consult local agricultural extension officers before applying treatments."
)

def get_recommendation(disease_key: str) -> Dict[str, Any]:
    """Retrieves recommendation entry with safety disclaimer."""
    normalized_key = (disease_key or "healthy").lower().replace("tomato___", "").replace(" ", "_")
    rec = RECOMMENDATION_DATABASE.get(normalized_key, RECOMMENDATION_DATABASE["healthy"])

    return {
        "problem": normalized_key,
        "title": rec["title"],
        "treatment": rec["treatment"],
        "prevention": rec["prevention"],
        "disclaimer": DISCLAIMER
    }
