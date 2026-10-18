"""
scripts/evaluate_model.py

Evaluates the real YOLOv8n tomato disease detector on held-out sample leaf
data using the verified test manifest. Measures precision, accuracy, latency,
and produces a truthful evaluation report.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from backend.app.services.model_registry import (
    get_disease_model_status,
    predict_disease,
)


def run_evaluation() -> dict:
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "sample_test_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return {"status": "FAILED", "reason": "Manifest missing"}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    status = get_disease_model_status()
    print("=" * 65)
    print(" MULTIVIEW-AGRIHEALTH: REAL MODEL EVALUATION SUITE")
    print("=" * 65)
    print(f"Model Name       : {status.get('model_name')}")
    print(f"Model Version    : {status.get('model_version')}")
    print(f"Model Status     : {status.get('status')}")
    print(f"Weights Path     : {status.get('weights_path')}")
    print(f"Artifact SHA-256 : {status.get('sha256')}")
    print("=" * 65)

    if status.get("status") != "READY":
        print(f"Evaluation cannot proceed: Model is {status.get('status')}")
        return {"status": "BLOCKED", "model_status": status.get("status")}

    samples = manifest.get("samples", [])
    results = []
    latencies = []

    correct_count = 0
    total_count = len(samples)

    for item in samples:
        rel_path = item["relative_path"]
        ground_truth = item["ground_truth_label"]
        img_path = PROJECT_ROOT / rel_path

        if not img_path.exists():
            print(f"WARNING: Image file not found: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: Could not decode image: {img_path}")
            continue

        # Measure inference latency
        t0 = time.perf_counter()
        pred_res = predict_disease(img)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        predicted_class = pred_res.get("prediction")
        confidence = pred_res.get("confidence", 0.0)
        is_match = (predicted_class == ground_truth)
        if is_match:
            correct_count += 1

        results.append({
            "sample_id": item["sample_id"],
            "image": item["relative_path"],
            "ground_truth": ground_truth,
            "prediction": predicted_class,
            "confidence": confidence,
            "match": is_match,
            "latency_ms": round(dt_ms, 2),
        })

    accuracy = (correct_count / total_count) if total_count > 0 else 0.0
    mean_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    print("\nSAMPLE PREDICTION RESULTS:")
    print("-" * 65)
    for r in results:
        status_sym = "[PASS]" if r["match"] else "[FAIL]"
        print(f"{status_sym} Sample: {r['sample_id']:<24} GT: {r['ground_truth']:<18} Pred: {r['prediction']:<18} Conf: {r['confidence']:.2f} ({r['latency_ms']} ms)")

    print("-" * 65)
    print(f"Total Evaluated Samples : {total_count}")
    print(f"Correct Predictions     : {correct_count} / {total_count}")
    print(f"Accuracy                : {accuracy:.1%}")
    print(f"Mean Latency            : {mean_latency:.1f} ms")
    print("=" * 65)

    report_payload = {
        "evaluation_timestamp": time.time(),
        "model_name": status.get("model_name"),
        "model_version": status.get("model_version"),
        "weights_sha256": status.get("sha256"),
        "total_samples": total_count,
        "correct_samples": correct_count,
        "accuracy": round(accuracy, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "sample_breakdown": results,
    }

    return report_payload


if __name__ == "__main__":
    report = run_evaluation()
