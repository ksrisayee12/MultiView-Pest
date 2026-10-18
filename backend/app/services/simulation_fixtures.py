"""
backend/app/services/simulation_fixtures.py

Phase 6 — Deterministic Simulation Fixtures

Generates reproducible, synthetic multi-view plant health scenarios for:
  - Pipeline testing without real video files
  - Evaluation framework validation
  - Frontend demonstration / onboarding

Fixture design principles:
  - All fixtures use numpy-generated synthetic frames — no external files needed.
  - Each fixture has a documented 'ground truth' status and expected outputs.
  - Results are deterministic given the same numpy random seed.
  - Fixtures explicitly test edge cases: unobserved views, conflicting views,
    low coverage, maximum coverage, healthy appearing, symptoms only from one angle.
  - No data is synthesized to claim model-level accuracy;
    fixtures validate pipeline logic, not model quality.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Synthetic frame generators
# ---------------------------------------------------------------------------

def _make_synthetic_frame(
    mode: str = "healthy",
    size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic BGR numpy frame for testing.

    mode options:
        'healthy'  — bright green, low noise
        'diseased' — greenish with brown/yellow patches
        'dark'     — underexposed (low brightness)
        'blank'    — near-empty grey frame
    """
    rng = np.random.default_rng(seed)
    h, w = size

    if mode == "healthy":
        # Mostly bright green
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = rng.integers(120, 200, size=(h, w))   # high green
        frame[:, :, 0] = rng.integers(20, 60, size=(h, w))     # low blue
        frame[:, :, 2] = rng.integers(20, 60, size=(h, w))     # low red
        return frame

    elif mode == "diseased":
        # Green base with brown/yellow patches
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = rng.integers(80, 160, size=(h, w))
        frame[:, :, 0] = rng.integers(20, 50, size=(h, w))
        frame[:, :, 2] = rng.integers(20, 50, size=(h, w))

        # Add brown lesion patches (BGR: low G, mid R, low B)
        for _ in range(6):
            cx = rng.integers(30, w - 30)
            cy = rng.integers(30, h - 30)
            r = rng.integers(15, 40)
            y_coords, x_coords = np.ogrid[:h, :w]
            circle = (y_coords - cy) ** 2 + (x_coords - cx) ** 2 <= r ** 2
            frame[:, :, 2][circle] = rng.integers(100, 160)  # brown-red
            frame[:, :, 1][circle] = rng.integers(40, 80)    # low green
            frame[:, :, 0][circle] = rng.integers(10, 40)    # low blue
        return frame

    elif mode == "dark":
        frame = rng.integers(0, 30, size=(h, w, 3), dtype=np.uint8)
        return frame

    else:  # blank / background
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        return frame


# ---------------------------------------------------------------------------
# Observation record builder (mimics video_processor output)
# ---------------------------------------------------------------------------

def _make_obs(
    clip_id: str,
    frame_index: int,
    view_label: str,
    frame_mode: str = "healthy",
    seed: int = 42,
) -> Dict[str, Any]:
    """Build a synthetic frame observation dict."""
    import cv2  # lazy import — may not be available in all test contexts
    frame = _make_synthetic_frame(mode=frame_mode, seed=seed + frame_index)

    # Simulate DIP analysis inline (not calling real pipeline — for determinism)
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([20, 25, 20])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    leaf_pixels = int(np.count_nonzero(green_mask))
    total = frame.shape[0] * frame.shape[1]
    greenness_index = round(leaf_pixels / max(total, 1), 4)

    lower_brown = np.array([0, 30, 20])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    abnormal_pixels = int(np.count_nonzero(brown_mask))
    lesion_pct = round(min(100.0, (abnormal_pixels / max(leaf_pixels, 1)) * 100.0), 2)

    if frame_mode == "healthy":
        visual_status = "normal"
        yolo_pred = "healthy"
        yolo_conf = 0.82
    elif frame_mode == "diseased":
        visual_status = "abnormal"
        yolo_pred = "early_blight"
        yolo_conf = 0.71
    elif frame_mode == "dark":
        visual_status = "abnormal"
        yolo_pred = "model_unavailable (MISSING_WEIGHTS)"
        yolo_conf = 0.0
    else:
        visual_status = "normal"
        yolo_pred = "model_unavailable (MISSING_WEIGHTS)"
        yolo_conf = 0.0

    return {
        "clip_id": clip_id,
        "frame_index": frame_index,
        "timestamp_sec": round(frame_index * 0.5, 3),
        "view_label": view_label,
        "yolo_prediction": yolo_pred,
        "yolo_confidence": yolo_conf,
        "yolo_model_status": "READY" if yolo_conf > 0 else "MISSING_WEIGHTS",
        "model_loaded": yolo_conf > 0,
        "opencv_visual_status": visual_status,
        "greenness_index": greenness_index,
        "green_pixel_percentage": round(greenness_index * 100, 2),
        "lesion_percentage": lesion_pct,
        "frame_shape": list(frame.shape),
        # Leaf analysis fields (Phase 4)
        "leaf_mask_available": frame_mode in ("healthy", "diseased"),
        "leaf_segmentation_method": "HEURISTIC",
        "leaf_pixel_count": leaf_pixels,
        "leaf_coverage_fraction": round(leaf_pixels / max(total, 1), 4),
        "visible_symptom_area_percent": lesion_pct,
        "leaf_bbox": [20, 20, frame.shape[1] - 20, frame.shape[0] - 20],
        "quality_flags": ["leaf_mask_heuristic: mask is approximate"] if frame_mode == "healthy" else
                         ["leaf_mask_heuristic: mask is approximate",
                          "abnormal_pixels_detected: potential lesion sites present"],
    }


# ---------------------------------------------------------------------------
# Fixture definitions
# ---------------------------------------------------------------------------

def build_fixture_healthy_all_views() -> Dict[str, Any]:
    """
    Scenario: Plant observed from 4 views, all views show healthy green leaves.
    Expected plant_status: HEALTHY_APPEARING
    """
    views = ["north", "south", "east", "west"]
    obs: Dict[str, List[Dict[str, Any]]] = {}
    for i, view in enumerate(views):
        obs[view] = [
            _make_obs("clip_h1", fi, view, "healthy", seed=10 * i + fi)
            for fi in range(8)
        ]
    return {
        "scenario_name": "healthy_all_views",
        "description": "Fully covered plant — all 4 views healthy",
        "declared_viewpoints": views,
        "view_observations": obs,
        "ground_truth_status": "HEALTHY_APPEARING",
        "ground_truth_class": "healthy",
    }


def build_fixture_diseased_majority_views() -> Dict[str, Any]:
    """
    Scenario: 3/4 views show disease symptoms, 1 view is healthy-appearing.
    Expected plant_status: SYMPTOMS_DETECTED
    """
    views = ["north", "south", "east", "west"]
    obs: Dict[str, List[Dict[str, Any]]] = {}
    modes = {"north": "diseased", "south": "diseased", "east": "diseased", "west": "healthy"}
    for i, view in enumerate(views):
        obs[view] = [
            _make_obs("clip_d1", fi, view, modes[view], seed=20 * i + fi)
            for fi in range(8)
        ]
    return {
        "scenario_name": "diseased_majority_views",
        "description": "3 of 4 views show early_blight symptoms",
        "declared_viewpoints": views,
        "view_observations": obs,
        "ground_truth_status": "SYMPTOMS_DETECTED",
        "ground_truth_class": "early_blight",
    }


def build_fixture_low_coverage() -> Dict[str, Any]:
    """
    Scenario: Only 1 of 4 declared views has observations (coverage = 0.25).
    Expected plant_status: INSUFFICIENT_COVERAGE
    """
    views = ["north", "south", "east", "west"]
    obs: Dict[str, List[Dict[str, Any]]] = {
        "north": [
            _make_obs("clip_lc", fi, "north", "healthy", seed=30 + fi)
            for fi in range(6)
        ]
        # south, east, west → unobserved
    }
    return {
        "scenario_name": "low_coverage",
        "description": "Only north view captured; other 3 views unobserved",
        "declared_viewpoints": views,
        "view_observations": obs,
        "ground_truth_status": "INSUFFICIENT_COVERAGE",
        "ground_truth_class": None,
    }


def build_fixture_conflicting_views() -> Dict[str, Any]:
    """
    Scenario: North view shows disease; South view shows healthy.
    Expected plant_status: UNCERTAIN (conflict between views)
    """
    views = ["north", "south"]
    obs: Dict[str, List[Dict[str, Any]]] = {
        "north": [
            _make_obs("clip_c1", fi, "north", "diseased", seed=40 + fi)
            for fi in range(8)
        ],
        "south": [
            _make_obs("clip_c1", fi, "south", "healthy", seed=50 + fi)
            for fi in range(8)
        ],
    }
    return {
        "scenario_name": "conflicting_views",
        "description": "North view: symptoms; South view: healthy — conflicting evidence",
        "declared_viewpoints": views,
        "view_observations": obs,
        "ground_truth_status": "UNCERTAIN",
        "ground_truth_class": None,
    }


def build_fixture_underexposed_views() -> Dict[str, Any]:
    """
    Scenario: All frames are underexposed (dark). Should report UNCERTAIN or no confident status.
    Expected plant_status: UNCERTAIN (insufficient visual evidence)
    """
    views = ["north", "east"]
    obs: Dict[str, List[Dict[str, Any]]] = {}
    for i, view in enumerate(views):
        obs[view] = [
            _make_obs("clip_u1", fi, view, "dark", seed=60 * i + fi)
            for fi in range(6)
        ]
    return {
        "scenario_name": "underexposed_frames",
        "description": "All frames underexposed — analysis unreliable",
        "declared_viewpoints": views,
        "view_observations": obs,
        "ground_truth_status": "UNCERTAIN",
        "ground_truth_class": None,
    }


def get_all_fixtures() -> List[Dict[str, Any]]:
    """Return all registered simulation fixtures."""
    return [
        build_fixture_healthy_all_views(),
        build_fixture_diseased_majority_views(),
        build_fixture_low_coverage(),
        build_fixture_conflicting_views(),
        build_fixture_underexposed_views(),
    ]


# ---------------------------------------------------------------------------
# Simulation runner (integrates cross_view_matcher + evaluation)
# ---------------------------------------------------------------------------

def run_all_simulation_scenarios() -> Dict[str, Any]:
    """
    Run all fixtures through the cross-view matcher and score results
    against ground truth. Returns a full evaluation report.
    """
    from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
    from backend.app.services.evaluation import run_evaluation_suite

    fixtures = get_all_fixtures()
    scenario_inputs = []

    scenario_reports = []
    for fx in fixtures:
        evidence = aggregate_cross_view_evidence(
            session_id="simulation",
            plant_id=fx["scenario_name"],
            view_observations=fx["view_observations"],
            declared_viewpoints=fx["declared_viewpoints"],
        )

        # Flatten observations for scoring
        all_obs: List[Dict[str, Any]] = []
        for obs_list in fx["view_observations"].values():
            all_obs.extend(obs_list)

        # Add status to each obs for scoring compatibility
        for obs in all_obs:
            obs["status"] = evidence.get("plant_status", "UNKNOWN")

        scenario_inputs.append({
            "name": fx["scenario_name"],
            "observations": all_obs,
            "expected_status": fx["ground_truth_status"],
            "expected_class": fx.get("ground_truth_class"),
        })

        scenario_reports.append({
            "scenario_name": fx["scenario_name"],
            "description": fx["description"],
            "ground_truth_status": fx["ground_truth_status"],
            "predicted_plant_status": evidence.get("plant_status"),
            "observed_views": evidence.get("observed_views"),
            "unobserved_views": evidence.get("unobserved_views"),
            "has_conflicts": evidence.get("has_conflicts"),
            "coverage_fraction": evidence.get("coverage_fraction"),
            "cross_view_classification": evidence.get("cross_view_classification"),
            "cross_view_confidence": evidence.get("cross_view_confidence"),
        })

    suite_results = run_evaluation_suite(scenario_inputs)

    return {
        "simulation_type": "deterministic_fixtures",
        "total_fixtures": len(fixtures),
        "suite_summary": suite_results,
        "scenario_reports": scenario_reports,
        "caveats": [
            "Fixtures use synthetic numpy frames — results do not reflect real-world model performance.",
            "YOLO predictions in fixtures are hard-coded for determinism, not real model output.",
            "Scenario 'pass' means the pipeline produced the expected status — not that a real plant would be assessed identically.",
        ],
    }
