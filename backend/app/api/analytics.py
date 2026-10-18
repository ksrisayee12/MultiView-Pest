"""
backend/app/api/analytics.py

Analytics API — aggregates scan history statistics.

Phase 1 fix: Removed hard-coded accuracy strings ("78.4%", "92.1%") that were
presented as real model accuracy. Those were from a synthetic evaluation using
the ground-truth label as the simulated YOLO prediction — not real model metrics.

Real performance metrics will be recorded here once a trained model is evaluated
on a genuine held-out test split. Until then, benchmark fields are omitted.
"""

from fastapi import APIRouter
from backend.database.mongodb import db_manager

router = APIRouter()


@router.get("/analytics", tags=["Analytics"])
def get_analytics():
    """
    Returns aggregated scan history statistics.

    Note: 'disease_breakdown' counts are based on YOLO predictions or fusion-confirmed
    diseases from historical scans stored in the database. If the YOLO model weights
    are unavailable, all predictions will be 'model_unavailable (...)' and disease
    counts will remain zero.

    Accuracy benchmarks are NOT included here — they will only be added after a
    reproducible evaluation run against a real held-out test set.
    """
    scans = db_manager.get_scans(limit=100)

    total_scans = len(scans)
    total_tiles = sum(s.get("total_tiles", 16) for s in scans)
    healthy_tiles = sum(s.get("healthy_tiles", 0) for s in scans)
    disease_tiles = sum(s.get("confirmed_disease_tiles", 0) for s in scans)
    disputed_tiles = sum(s.get("disputed_tiles", 0) for s in scans)

    disease_breakdown: dict = {
        "healthy": healthy_tiles,
        "early_blight": 0,
        "late_blight": 0,
        "septoria_leaf_spot": 0,
        "other": 0,
    }

    known_diseases = {"early_blight", "late_blight", "septoria_leaf_spot", "healthy"}

    for scan in scans:
        for tile in scan.get("tile_results", []):
            d = tile.get("confirmed_disease") or tile.get("yolo_prediction", "")
            if d and d in known_diseases:
                disease_breakdown[d] = disease_breakdown.get(d, 0) + 1
            elif d and d not in ("", "model_unavailable", "YOLO model not trained"):
                disease_breakdown["other"] = disease_breakdown.get("other", 0) + 1

    avg_health_index = (
        round(sum(s.get("plant_health_index", 100.0) for s in scans) / total_scans, 1)
        if total_scans > 0
        else None
    )

    return {
        "total_scans": total_scans,
        "total_tiles": total_tiles,
        "healthy_tiles": healthy_tiles,
        "disease_tiles": disease_tiles,
        "disputed_tiles": disputed_tiles,
        "average_health_index": avg_health_index,
        "disease_breakdown": disease_breakdown,
        # accuracy_benchmarks intentionally omitted — not yet measured against real test data.
        # Will be added after: train model → evaluate on held-out split → record results.
    }
