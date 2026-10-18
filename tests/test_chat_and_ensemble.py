"""
tests/test_chat_and_ensemble.py

Tests for:
  1. General Botanical Chatbot (guardrails against non-plant questions)
  2. Clinical Disease & Treatment Chatbot (locked to diagnosis context)
  3. Multi-Model Ensemble Diagnostic Endpoint (/api/ensemble/assess)
  4. Ensemble Models Metadata & Provenance Endpoint (/api/ensemble/models)
"""

import pytest
import numpy as np
import cv2
from starlette.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_chatbot_status_endpoint():
    res = client.get("/api/chat/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "has_groq_api_key" in data
    assert len(data["supported_bots"]) == 2


def test_general_chatbot_guardrail_rejection():
    # Submit non-plant query (coding/crypto)
    res = client.post(
        "/api/chat/general",
        json={"message": "Write a python script to trade bitcoin on binance"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["guardrail_triggered"] is True
    assert "AgriHealth" in data["reply"]
    assert "cannot answer" in data["reply"].lower() or "guardrail" in data["reply"].lower()


def test_general_chatbot_accepts_plant_query():
    # Submit authentic plant query
    res = client.post(
        "/api/chat/general",
        json={"message": "What causes tomato blossom end rot and how can I prevent it?"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["guardrail_triggered"] is False
    assert len(data["reply"]) > 20


def test_disease_chatbot_clinical_guidance():
    # Submit disease consultation query
    context = {
        "disease_name": "early_blight",
        "crop_type": "tomato",
        "confidence": 0.92,
        "severity": "Moderate",
        "lesion_percentage": 14.5,
        "affected_views": ["front", "underside"]
    }
    res = client.post(
        "/api/chat/disease",
        json={
            "message": "What fungicide should I spray and what is the application interval?",
            "context": context
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert "disease" in data
    assert len(data["reply"]) > 50
    # Should contain clinical pathogen or treatment references
    assert any(term in data["reply"].lower() for term in ["alternaria", "copper", "fungicide", "blight", "cultural"])


def test_ensemble_models_status_endpoint():
    res = client.get("/api/ensemble/models")
    assert res.status_code == 200
    data = res.json()
    assert data["ensemble_status"] == "READY"
    assert "model_1_yolo_detector" in data["models"]
    assert "model_2_leaf_severity_net" in data["models"]
    assert "model_3_pest_vector_net" in data["models"]
    # Verify SHA-256 is present for trained models
    assert len(data["models"]["model_2_leaf_severity_net"]["sha256"]) == 64
    assert len(data["models"]["model_3_pest_vector_net"]["sha256"]) == 64


def test_ensemble_leaf_assessment():
    # Generate dummy leaf image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:] = (35, 160, 45)  # green leaf
    cv2.circle(img, (110, 110), 30, (20, 70, 110), -1)  # brown lesion spot
    _, buf = cv2.imencode(".jpg", img)
    
    files = {"file": ("test_leaf.jpg", buf.tobytes(), "image/jpeg")}
    data = {"crop_type": "tomato"}
    
    res = client.post("/api/ensemble/assess", files=files, data=data)
    assert res.status_code == 200
    report = res.json()
    
    assert "primary_diagnosis" in report
    assert "multi_model_consensus" in report
    assert report["multi_model_consensus"]["models_evaluated"] == 3
    assert "severity_analysis" in report
    assert "vector_surveillance" in report
    assert "pathogen_profile" in report
    assert "treatment_prescription" in report
    assert "model_provenance" in report
    assert "cultural_controls" in report["treatment_prescription"]
    assert "organic_biocontrols" in report["treatment_prescription"]
    assert "chemical_interventions" in report["treatment_prescription"]
    assert "visual_localization" in report
    vis = report["visual_localization"]
    assert "annotated_image" in vis
    assert "heatmap_image" in vis
    assert "original_image" in vis
    assert "detected_regions" in vis
    assert vis["annotated_image"].startswith("data:image/jpeg;base64,")
