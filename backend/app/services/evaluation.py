"""
backend/app/services/evaluation.py

Phase 6 — Multi-View Assessment Evaluation Framework

Provides evaluation metrics and scenario-level validation for the multi-view
plant health assessment pipeline. Enables:

  1. Per-session score cards (precision, recall, F1 at the plant level).
  2. Coverage analysis — how observed viewpoints influence assessment confidence.
  3. Calibration reporting — whether stated confidence tracks empirical accuracy.
  4. Simulation scenario scoring — evaluating synthetic test scenario results.

Design principles:
  - Only evaluates when ground truth labels are explicitly provided.
  - Never fabricates performance metrics from unlabelled data.
  - All metrics carry a 'sample_size' — small N → large uncertainty interval.
  - Uncertainty bounds reported via Wilson score interval (appropriate for small N).
  - Calibration uses binned reliability plots, not scalar ECE alone.
  - Results are JSON-serialisable; no external ML libraries required.

Caveats:
  - Ground truth labels are user-declared — downstream results inherit that quality.
  - F1 / precision / recall are plant-level (binary: diseased vs healthy).
  - Multi-class evaluation requires per-class breakdowns not yet implemented.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Binary classification metrics with Wilson interval
# ---------------------------------------------------------------------------

def _wilson_interval(
    n_successes: int, n_total: int, z: float = 1.96
) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion (95% by default)."""
    if n_total == 0:
        return (0.0, 0.0)
    p = n_successes / n_total
    denom = 1 + z * z / n_total
    centre = (p + z * z / (2 * n_total)) / denom
    half = (z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total))) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def compute_plant_level_metrics(
    predictions: List[str],
    ground_truth: List[str],
    positive_label: str = "SYMPTOMS_DETECTED",
) -> Dict[str, Any]:
    """
    Compute binary classification metrics at the plant level.

    Parameters:
        predictions   — list of predicted plant status strings
        ground_truth  — list of true plant status strings (user-declared)
        positive_label — the label treated as 'positive' (diseased)

    Returns precision, recall, F1, accuracy with Wilson intervals and N.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) must have equal length."
        )

    n = len(predictions)
    if n == 0:
        return {"error": "No samples provided.", "n": 0}

    tp = fp = fn = tn = 0
    for pred, true in zip(predictions, ground_truth):
        p_pos = pred == positive_label
        t_pos = true == positive_label
        if p_pos and t_pos:
            tp += 1
        elif p_pos and not t_pos:
            fp += 1
        elif not p_pos and t_pos:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / n
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "n": n,
        "positive_label": positive_label,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "precision_95ci": _wilson_interval(tp, max(tp + fp, 1)),
        "recall_95ci": _wilson_interval(tp, max(tp + fn, 1)),
        "accuracy_95ci": _wilson_interval(tp + tn, n),
        "interpretation_note": (
            f"Metrics computed at plant level (binary: {positive_label} vs other). "
            f"N={n} — {'small sample, wide confidence intervals' if n < 30 else 'moderate sample'}. "
            "Ground truth is user-declared, not expert-validated clinical annotation."
        ),
    }


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def analyze_view_coverage(
    session_view_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyse how viewpoint coverage correlates with assessment quality.

    Parameters:
        session_view_reports — list of cross_view_evidence dicts (one per plant session)

    Returns coverage statistics and correlation between coverage and confidence.
    """
    if not session_view_reports:
        return {"error": "No session reports provided."}

    coverage_fractions = []
    confidence_scores = []
    status_distribution: Dict[str, int] = {}

    for report in session_view_reports:
        cov = report.get("coverage_fraction", 0.0)
        conf = report.get("cross_view_confidence", 0.0)
        status = report.get("plant_status", "UNKNOWN")

        coverage_fractions.append(cov)
        confidence_scores.append(conf)
        status_distribution[status] = status_distribution.get(status, 0) + 1

    n = len(coverage_fractions)
    mean_coverage = sum(coverage_fractions) / n
    mean_confidence = sum(confidence_scores) / n

    # Pearson correlation between coverage and confidence
    cov_corr = _pearson(coverage_fractions, confidence_scores)

    return {
        "n_sessions": n,
        "mean_coverage_fraction": round(mean_coverage, 4),
        "mean_confidence": round(mean_confidence, 4),
        "coverage_confidence_correlation": round(cov_corr, 4),
        "status_distribution": status_distribution,
        "coverage_note": (
            "Coverage fraction = (observed views) / (declared views). "
            "Sessions with coverage < 0.5 are classified INSUFFICIENT_COVERAGE. "
            "Correlation is Pearson r between coverage and cross_view_confidence."
        ),
    }


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    denom = dx * dy
    return num / denom if denom > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def calibration_report(
    confidences: List[float],
    correct_flags: List[bool],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Generate a calibration report: bins predictions by confidence, compares
    mean confidence in bin vs empirical accuracy.

    Parameters:
        confidences   — predicted confidence values [0.0–1.0]
        correct_flags — whether each prediction was correct (True/False)
        n_bins        — number of equal-width bins

    Returns bin reliability data and Expected Calibration Error (ECE).
    """
    if len(confidences) != len(correct_flags):
        raise ValueError("confidences and correct_flags must have equal length.")
    if not confidences:
        return {"error": "No samples provided."}

    bin_width = 1.0 / n_bins
    bins: List[Dict[str, Any]] = []
    ece_numerator = 0.0
    n_total = len(confidences)

    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width
        in_bin = [(c, f) for c, f in zip(confidences, correct_flags) if lo <= c < hi]
        if i == n_bins - 1:
            in_bin = [(c, f) for c, f in zip(confidences, correct_flags) if lo <= c <= hi]

        if not in_bin:
            bins.append({
                "bin_lo": round(lo, 3),
                "bin_hi": round(hi, 3),
                "count": 0,
                "mean_confidence": None,
                "empirical_accuracy": None,
                "gap": None,
            })
            continue

        mean_conf = sum(c for c, _ in in_bin) / len(in_bin)
        accuracy = sum(1 for _, f in in_bin if f) / len(in_bin)
        gap = mean_conf - accuracy
        ece_numerator += len(in_bin) * abs(gap)

        bins.append({
            "bin_lo": round(lo, 3),
            "bin_hi": round(hi, 3),
            "count": len(in_bin),
            "mean_confidence": round(mean_conf, 4),
            "empirical_accuracy": round(accuracy, 4),
            "gap": round(gap, 4),  # positive = overconfident, negative = underconfident
        })

    ece = round(ece_numerator / n_total, 4)

    return {
        "n": n_total,
        "n_bins": n_bins,
        "expected_calibration_error": ece,
        "bins": bins,
        "calibration_note": (
            "ECE near 0 indicates good calibration. "
            "Positive bin gap = model is overconfident. "
            "Negative bin gap = model is underconfident. "
            f"N={n_total} — {'very small sample; interpret with caution' if n_total < 50 else 'moderate sample'}."
        ),
    }


# ---------------------------------------------------------------------------
# Simulation scenario scoring (Phase 6)
# ---------------------------------------------------------------------------

def score_simulation_scenario(
    scenario_name: str,
    scenario_observations: List[Dict[str, Any]],
    expected_status: str,
    expected_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score a deterministic simulation scenario against its expected outcome.
    Returns pass/fail with detailed evidence.

    Parameters:
        scenario_name         — human-readable scenario label
        scenario_observations — list of observation dicts from the scenario
        expected_status       — expected plant status (e.g., "SYMPTOMS_DETECTED")
        expected_class        — optional expected YOLO class label

    Returns:
        scenario score card dict with pass/fail and evidence breakdown.
    """
    if not scenario_observations:
        return {
            "scenario": scenario_name,
            "passed": False,
            "reason": "No observations in scenario.",
        }

    # Count statuses, visual results, class predictions
    detected_statuses: Dict[str, int] = {}
    detected_classes: Dict[str, int] = {}
    visual_abnormal = 0

    for obs in scenario_observations:
        status = obs.get("status", obs.get("opencv_visual_status", "unknown"))
        detected_statuses[status] = detected_statuses.get(status, 0) + 1

        cls = obs.get("yolo_prediction", obs.get("dominant_class", ""))
        if cls:
            detected_classes[cls] = detected_classes.get(cls, 0) + 1

        if obs.get("opencv_visual_status") == "abnormal":
            visual_abnormal += 1

    total = len(scenario_observations)
    dominant_status = max(detected_statuses, key=detected_statuses.get, default="UNKNOWN")
    dominant_class = max(detected_classes, key=detected_classes.get, default=None) if detected_classes else None

    status_match = dominant_status == expected_status
    class_match = (expected_class is None) or (dominant_class == expected_class)

    passed = status_match and class_match

    return {
        "scenario": scenario_name,
        "passed": passed,
        "expected_status": expected_status,
        "dominant_status": dominant_status,
        "status_match": status_match,
        "expected_class": expected_class,
        "dominant_class": dominant_class,
        "class_match": class_match,
        "total_observations": total,
        "status_distribution": detected_statuses,
        "class_distribution": detected_classes,
        "visual_abnormal_count": visual_abnormal,
        "visual_abnormal_fraction": round(visual_abnormal / total, 3) if total > 0 else 0.0,
        "note": (
            "Simulation scoring uses majority-vote dominance across frame observations. "
            "Deterministic fixtures should always produce the same dominant status."
        ),
    }


def run_evaluation_suite(
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run multiple scenario score cards and return a summary report.

    Each element of 'scenarios' must contain:
        name, observations, expected_status, and optionally expected_class.
    """
    results = []
    n_passed = 0

    for scenario in scenarios:
        result = score_simulation_scenario(
            scenario_name=scenario.get("name", "unnamed"),
            scenario_observations=scenario.get("observations", []),
            expected_status=scenario.get("expected_status", "UNKNOWN"),
            expected_class=scenario.get("expected_class"),
        )
        results.append(result)
        if result.get("passed"):
            n_passed += 1

    n_total = len(results)
    return {
        "total_scenarios": n_total,
        "passed": n_passed,
        "failed": n_total - n_passed,
        "pass_rate": round(n_passed / max(n_total, 1), 4),
        "scenario_results": results,
    }
