"""
backend/app/services/cross_view_matcher.py

Phase 5 — Cross-View Plant Association

Associates observations from multiple viewpoints (North, South, East, West, Top, etc.)
as belonging to the same physical plant, and aggregates evidence across views while
explicitly accounting for which views are unobserved.

Design principles:
  - Plants are identified by a user-supplied session ID, not auto-detected across cameras.
  - Within a session, each viewpoint contributes independent observations.
  - Cross-view evidence is FUSED, not averaged blindly:
      * Majority vote for class labels (weighted by confidence).
      * Upper-bound lesion estimate = max visible area across views (most impacted angle).
      * Uncertainty propagates: if views strongly disagree, status = UNCERTAIN.
  - Views explicitly marked UNOBSERVED remain UNKNOWN — no imputation.
  - Conflicting view labels documented, not silently resolved.
  - Produces a canonical PlantEvidence report usable by plant_fusion.py.

Important caveats (disclosed):
  - "Same plant" is based on user-declared session, not geometric camera calibration.
  - If cameras are miscalibrated or labels are wrong, cross-view fusion may be meaningless.
  - Lesion area estimates from different angles may overlap (same lesion, different view).
    The upper-bound aggregation may therefore overestimate total affected area.
  - Cross-view tracking of individual leaves (not whole plants) requires calibrated stereo.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------

ViewLabel = str          # e.g., "north", "south", "top", "side_45"
TrackSummary = Dict[str, Any]
FrameObservation = Dict[str, Any]


# ---------------------------------------------------------------------------
# Cross-view aggregation
# ---------------------------------------------------------------------------

CANONICAL_VIEWS = frozenset([
    "north", "south", "east", "west",
    "top", "bottom", "side_45", "side_135", "custom"
])


def aggregate_cross_view_evidence(
    session_id: str,
    plant_id: str,
    view_observations: Dict[ViewLabel, List[FrameObservation]],
    declared_viewpoints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate multi-view frame observations into a single PlantEvidence record.

    Parameters:
        session_id           — plant session this evidence belongs to
        plant_id             — unique identifier for this plant
        view_observations    — {view_label: [frame_observation_dicts]}
        declared_viewpoints  — list of all viewpoints that were declared in the
                               session setup (even if no data was collected yet).
                               Views in this list without observations → UNOBSERVED.

    Returns a CrossViewEvidence dict.
    """
    declared = set(declared_viewpoints or [])
    observed_views = set(view_observations.keys())
    unobserved_views = sorted(declared - observed_views)
    extra_views = sorted(observed_views - declared)   # views captured but not pre-declared

    view_reports: Dict[str, Any] = {}
    all_class_votes: Dict[str, float] = {}  # class → sum of weighted confidence
    all_lesion_pcts: List[float] = []
    view_status_list: List[str] = []

    for view_label, observations in view_observations.items():
        if not observations:
            view_reports[view_label] = _empty_view_report(view_label, "NO_DATA")
            view_status_list.append("NO_DATA")
            continue

        vr = _summarize_single_view(view_label, observations)
        view_reports[view_label] = vr
        view_status_list.append(vr["view_status"])

        # Accumulate class votes weighted by confidence
        for cls, weight in vr["class_votes_weighted"].items():
            all_class_votes[cls] = all_class_votes.get(cls, 0.0) + weight

        # Accumulate lesion percentages across views (for upper-bound)
        if vr.get("median_lesion_pct") is not None:
            all_lesion_pcts.append(vr["median_lesion_pct"])
        if vr.get("max_lesion_pct") is not None:
            all_lesion_pcts.append(vr["max_lesion_pct"])

    # Mark unobserved views
    for uv in unobserved_views:
        view_reports[uv] = _empty_view_report(uv, "UNOBSERVED")

    # Derive cross-view dominant classification
    cross_view_class = None
    cross_view_confidence = 0.0
    if all_class_votes:
        cross_view_class = max(all_class_votes, key=all_class_votes.get)
        total_weight = sum(all_class_votes.values())
        cross_view_confidence = round(
            all_class_votes[cross_view_class] / max(total_weight, 1e-9), 4
        )

    # Cross-view lesion estimate
    max_lesion_across_views = round(max(all_lesion_pcts), 2) if all_lesion_pcts else None

    # Overall plant status
    plant_status = _derive_plant_status(
        view_status_list=view_status_list,
        cross_view_class=cross_view_class,
        cross_view_confidence=cross_view_confidence,
        observed_count=len(observed_views),
        declared_count=len(declared),
    )

    # Conflict detection: views that disagree significantly
    conflicts = _detect_view_conflicts(view_reports)

    return {
        "session_id": session_id,
        "plant_id": plant_id,
        # Cross-view summary
        "plant_status": plant_status,
        "cross_view_classification": cross_view_class,
        "cross_view_confidence": cross_view_confidence,
        "max_lesion_pct_across_views": max_lesion_across_views,
        "lesion_estimate_note": (
            "max_lesion_pct_across_views is the UPPER BOUND of visible symptom area "
            "across all camera angles. Different views may capture the same lesion, "
            "so this value does NOT represent total unique affected area. "
            "It is NOT a validated severity score."
        ) if max_lesion_across_views is not None else None,
        # Coverage
        "observed_views": sorted(observed_views),
        "unobserved_views": unobserved_views,
        "extra_views": extra_views,
        "total_declared_views": len(declared),
        "coverage_fraction": round(len(observed_views) / max(len(declared), 1), 3),
        # Per-view detail
        "view_reports": view_reports,
        # Conflict summary
        "view_conflicts": conflicts,
        "has_conflicts": len(conflicts) > 0,
        # Caveats
        "cross_view_caveats": [
            "Plant identity is declared by user session — no geometric camera calibration.",
            "Lesion area may overlap between views — upper-bound can over-estimate total affected area.",
            "Cross-view leaf-level tracking requires calibrated stereo and is NOT implemented here.",
            "Unobserved views are explicitly marked UNOBSERVED — they are never imputed.",
        ],
    }


def _summarize_single_view(
    view_label: str,
    observations: List[FrameObservation],
) -> Dict[str, Any]:
    """
    Summarize all frame observations for a single viewpoint.
    """
    if not observations:
        return _empty_view_report(view_label, "NO_DATA")

    classes: List[str] = []
    confidences: List[float] = []
    lesion_pcts: List[float] = []
    visual_statuses: List[str] = []

    class_votes_weighted: Dict[str, float] = {}

    for obs in observations:
        cls = obs.get("yolo_prediction") or obs.get("dominant_class") or "UNKNOWN"
        conf = float(obs.get("yolo_confidence") or obs.get("mean_confidence") or 0.0)
        lesion = obs.get("lesion_percentage") or obs.get("visible_symptom_area_percent")
        vs = obs.get("opencv_visual_status") or obs.get("view_status") or "unknown"

        if cls and cls not in ("UNKNOWN", "model_unavailable") and cls.startswith("model") is False:
            classes.append(cls)
            confidences.append(conf)
            class_votes_weighted[cls] = class_votes_weighted.get(cls, 0.0) + conf

        if lesion is not None:
            lesion_pcts.append(float(lesion))
        if vs:
            visual_statuses.append(vs)

    # Dominant class from weighted votes
    dominant_class = None
    dominant_confidence = 0.0
    if class_votes_weighted:
        dominant_class = max(class_votes_weighted, key=class_votes_weighted.get)
        total_w = sum(class_votes_weighted.values())
        dominant_confidence = round(
            class_votes_weighted[dominant_class] / max(total_w, 1e-9), 4
        )

    median_lesion = None
    max_lesion = None
    if lesion_pcts:
        sorted_l = sorted(lesion_pcts)
        n = len(sorted_l)
        median_lesion = round(sorted_l[n // 2], 2)
        max_lesion = round(max(sorted_l), 2)

    # View status
    abnormal_count = visual_statuses.count("abnormal")
    view_status = _view_status_from_observations(
        dominant_class=dominant_class,
        dominant_confidence=dominant_confidence,
        abnormal_count=abnormal_count,
        total_count=len(observations),
        lesion_max=max_lesion,
    )

    return {
        "view_label": view_label,
        "view_status": view_status,
        "frame_count": len(observations),
        "dominant_class": dominant_class,
        "dominant_confidence": dominant_confidence,
        "class_votes_weighted": class_votes_weighted,
        "median_lesion_pct": median_lesion,
        "max_lesion_pct": max_lesion,
        "abnormal_frame_count": abnormal_count,
    }


def _view_status_from_observations(
    dominant_class: Optional[str],
    dominant_confidence: float,
    abnormal_count: int,
    total_count: int,
    lesion_max: Optional[float],
) -> str:
    if total_count == 0:
        return "NO_DATA"

    abnormal_fraction = abnormal_count / total_count

    # Explicit disease detected with reasonable confidence
    if dominant_class and dominant_confidence >= 0.55 and dominant_class.lower() not in (
        "healthy", "normal", "no_disease", "background"
    ):
        return "SYMPTOMS_DETECTED"

    # Visual analysis flagged majority of frames
    if abnormal_fraction >= 0.6:
        return "SYMPTOMS_DETECTED"
    if abnormal_fraction >= 0.3 or (lesion_max is not None and lesion_max >= 10.0):
        return "UNCERTAIN"

    if dominant_class and dominant_confidence >= 0.60:
        return "HEALTHY_APPEARING"

    return "INSUFFICIENT_DATA"


def _derive_plant_status(
    view_status_list: List[str],
    cross_view_class: Optional[str],
    cross_view_confidence: float,
    observed_count: int,
    declared_count: int,
) -> str:
    """
    Derive overall plant status from aggregated view statuses.
    Follows the honest taxonomy: HEALTHY_APPEARING, SYMPTOMS_DETECTED,
    UNCERTAIN, INSUFFICIENT_COVERAGE.
    """
    if observed_count == 0:
        return "INSUFFICIENT_COVERAGE"

    # If ALL observed views have NO_DATA → insufficient
    all_no_data = all(s in ("NO_DATA", "UNOBSERVED") for s in view_status_list)
    if all_no_data:
        return "INSUFFICIENT_COVERAGE"

    symptoms_detected = sum(1 for s in view_status_list if s == "SYMPTOMS_DETECTED")
    healthy_count = sum(1 for s in view_status_list if s == "HEALTHY_APPEARING")
    uncertain_count = sum(1 for s in view_status_list if s == "UNCERTAIN")

    # Any view detecting symptoms → at minimum UNCERTAIN, likely SYMPTOMS_DETECTED
    if symptoms_detected > 0:
        # Only call SYMPTOMS_DETECTED if corroborated or majority agreement
        if symptoms_detected >= max(2, observed_count // 2):
            return "SYMPTOMS_DETECTED"
        return "UNCERTAIN"

    if uncertain_count > 0:
        return "UNCERTAIN"

    coverage = observed_count / max(declared_count, 1)
    if coverage < 0.5:
        return "INSUFFICIENT_COVERAGE"

    if healthy_count == observed_count:
        return "HEALTHY_APPEARING"

    return "UNCERTAIN"


def _detect_view_conflicts(view_reports: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find pairs of views with conflicting classifications.
    A conflict exists when one view is SYMPTOMS_DETECTED and another is HEALTHY_APPEARING.
    """
    conflicts = []
    view_list = [
        (vl, vr) for vl, vr in view_reports.items()
        if vr.get("view_status") not in ("UNOBSERVED", "NO_DATA")
    ]

    for i in range(len(view_list)):
        for j in range(i + 1, len(view_list)):
            labelA, repA = view_list[i]
            labelB, repB = view_list[j]

            statusA = repA.get("view_status", "")
            statusB = repB.get("view_status", "")

            if (
                (statusA == "SYMPTOMS_DETECTED" and statusB == "HEALTHY_APPEARING") or
                (statusB == "SYMPTOMS_DETECTED" and statusA == "HEALTHY_APPEARING")
            ):
                conflicts.append({
                    "view_a": labelA,
                    "status_a": statusA,
                    "class_a": repA.get("dominant_class"),
                    "view_b": labelB,
                    "status_b": statusB,
                    "class_b": repB.get("dominant_class"),
                    "note": (
                        "Views disagree on health status. This may indicate partial disease "
                        "visible only from certain angles, lighting differences, or a "
                        "false positive / false negative from one view."
                    ),
                })

    return conflicts


def _empty_view_report(view_label: str, status: str) -> Dict[str, Any]:
    return {
        "view_label": view_label,
        "view_status": status,
        "frame_count": 0,
        "dominant_class": None,
        "dominant_confidence": 0.0,
        "class_votes_weighted": {},
        "median_lesion_pct": None,
        "max_lesion_pct": None,
        "abnormal_frame_count": 0,
    }
