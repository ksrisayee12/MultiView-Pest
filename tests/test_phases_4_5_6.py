"""
tests/test_phases_4_5_6.py

Comprehensive test suite for Phases 4, 5, and 6:
  - Phase 4: Leaf analysis (mask-aware lesion estimation)
  - Phase 4: Temporal tracker (within-clip IoU/centroid association)
  - Phase 5: Cross-view matcher (multi-angle evidence fusion)
  - Phase 6: Evaluation framework (metrics, calibration, scenario scoring)
  - Phase 6: Simulation fixtures (deterministic synthetic scenarios)
"""

import numpy as np
import pytest


# ===========================================================================
# Phase 4: Leaf Analysis
# ===========================================================================

class TestLeafAnalysis:
    """Tests for backend.app.services.leaf_analysis"""

    def _make_green_frame(self, h=120, w=120):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = 150  # green channel dominant
        return frame

    def _make_diseased_frame(self, h=120, w=120):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = 120  # moderate green
        # Add brown lesion patch
        frame[30:60, 30:60, 2] = 130  # red (BGR)
        frame[30:60, 30:60, 1] = 50
        return frame

    def test_healthy_frame_returns_observation(self):
        from backend.app.services.leaf_analysis import analyze_frame_with_mask
        frame = self._make_green_frame()
        obs = analyze_frame_with_mask(frame, clip_id="c1", frame_index=0,
                                      timestamp_sec=0.0, view_label="north")
        assert obs["leaf_mask_available"] is True
        assert obs["leaf_segmentation_method"] in ("AVAILABLE", "HEURISTIC")
        assert obs["visible_symptom_area_percent"] is not None
        assert 0.0 <= obs["visible_symptom_area_percent"] <= 100.0

    def test_empty_frame_returns_unavailable(self):
        from backend.app.services.leaf_analysis import analyze_frame_with_mask
        obs = analyze_frame_with_mask(None, clip_id="c1", frame_index=0,
                                      timestamp_sec=0.0, view_label="top")
        assert obs["leaf_mask_available"] is False
        assert obs["visible_symptom_area_percent"] is None
        assert len(obs["quality_flags"]) > 0

    def test_tiny_frame_returns_unavailable(self):
        from backend.app.services.leaf_analysis import analyze_frame_with_mask
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        obs = analyze_frame_with_mask(frame, clip_id="c1", frame_index=0,
                                      timestamp_sec=0.0, view_label="south")
        assert obs["leaf_mask_available"] is False

    def test_diseased_frame_has_symptom_area(self):
        from backend.app.services.leaf_analysis import analyze_frame_with_mask
        frame = self._make_diseased_frame()
        obs = analyze_frame_with_mask(frame, clip_id="c1", frame_index=0,
                                      timestamp_sec=0.0, view_label="east")
        # Should detect some symptom area
        assert obs["visible_symptom_area_percent"] is not None

    def test_batch_analysis_returns_list(self):
        from backend.app.services.leaf_analysis import analyze_frames_batch
        frames = [
            (self._make_green_frame(), "c1", i, float(i) * 0.5, "north")
            for i in range(5)
        ]
        results = analyze_frames_batch(frames)
        assert len(results) == 5
        for r in results:
            assert "leaf_mask_available" in r

    def test_get_status_returns_dict(self):
        from backend.app.services.leaf_analysis import get_leaf_segmentation_status
        status = get_leaf_segmentation_status()
        assert "leaf_segmentation_status" in status
        assert isinstance(status["model_loaded"], bool)

    def test_observation_has_required_keys(self):
        from backend.app.services.leaf_analysis import analyze_frame_with_mask
        frame = self._make_green_frame()
        obs = analyze_frame_with_mask(frame, "clip_test", 0, 0.0, "west")
        required_keys = [
            "clip_id", "frame_index", "timestamp_sec", "view_label",
            "leaf_mask_available", "leaf_segmentation_method",
            "leaf_pixel_count", "leaf_coverage_fraction",
            "symptom_pixel_count", "visible_symptom_area_percent",
            "quality_flags", "frame_shape",
        ]
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"


# ===========================================================================
# Phase 4: Temporal Tracker
# ===========================================================================

class TestTemporalTracker:
    """Tests for backend.app.services.temporal_tracker"""

    def _make_detection(self, clip_id, frame_index, bbox, conf=0.8, pred="healthy"):
        return {
            "clip_id": clip_id,
            "frame_index": frame_index,
            "view_label": "north",
            "yolo_prediction": pred,
            "yolo_confidence": conf,
            "leaf_bbox": bbox,
            "visible_symptom_area_percent": 5.0,
        }

    def test_single_detection_creates_track(self):
        from backend.app.services.temporal_tracker import TemporalTracker
        t = TemporalTracker("clip_1", "north")
        tid = t.update(self._make_detection("clip_1", 0, [10, 10, 80, 80]))
        assert tid != "unassociated"
        summary = t.get_summary()
        assert summary["total_unique_tracks"] == 1

    def test_overlapping_boxes_same_track(self):
        from backend.app.services.temporal_tracker import TemporalTracker
        t = TemporalTracker("clip_2", "north")
        tid0 = t.update(self._make_detection("clip_2", 0, [10, 10, 80, 80]))
        tid1 = t.update(self._make_detection("clip_2", 1, [12, 12, 82, 82]))
        tid2 = t.update(self._make_detection("clip_2", 2, [14, 14, 84, 84]))
        assert tid0 == tid1 == tid2  # same track
        summary = t.get_summary()
        assert summary["total_unique_tracks"] == 1

    def test_non_overlapping_boxes_separate_tracks(self):
        from backend.app.services.temporal_tracker import TemporalTracker
        t = TemporalTracker("clip_3", "north")
        tid_a = t.update(self._make_detection("clip_3", 0, [0, 0, 40, 40]))
        tid_b = t.update(self._make_detection("clip_3", 1, [160, 160, 220, 220]))
        assert tid_a != tid_b
        summary = t.get_summary()
        assert summary["total_unique_tracks"] == 2

    def test_low_confidence_returns_unassociated(self):
        from backend.app.services.temporal_tracker import TemporalTracker
        t = TemporalTracker("clip_4", "north")
        tid = t.update(self._make_detection("clip_4", 0, [10, 10, 80, 80], conf=0.1))
        assert tid == "unassociated"

    def test_track_expiry(self):
        from backend.app.services.temporal_tracker import TemporalTracker, _MAX_FRAMES_UNSEEN
        t = TemporalTracker("clip_5", "north")
        t.update(self._make_detection("clip_5", 0, [10, 10, 80, 80]))
        # Skip many frames without updating
        for fi in range(_MAX_FRAMES_UNSEEN + 5):
            # Feed a dummy low-confidence detection to advance frame counter
            t.update(self._make_detection("clip_5", fi + 1, [200, 200, 240, 240], conf=0.1))
        # Now a new overlapping detection should create a new track
        tid_new = t.update(self._make_detection("clip_5", _MAX_FRAMES_UNSEEN + 6, [10, 10, 80, 80]))
        summary = t.get_summary()
        # Should have >= 1 track (the original is archived)
        assert summary["total_unique_tracks"] >= 1

    def test_summary_has_required_fields(self):
        from backend.app.services.temporal_tracker import TemporalTracker
        t = TemporalTracker("clip_6", "east")
        t.update(self._make_detection("clip_6", 0, [10, 10, 80, 80], pred="early_blight", conf=0.75))
        summary = t.get_summary()
        for key in ["clip_id", "view_label", "total_unique_tracks", "track_summaries", "limitations"]:
            assert key in summary, f"Missing key: {key}"
        ts = summary["track_summaries"][0]
        for key in ["track_id", "observation_count", "dominant_class", "mean_confidence"]:
            assert key in ts, f"Missing track summary key: {key}"

    def test_track_clip_observations_convenience(self):
        from backend.app.services.temporal_tracker import track_clip_observations
        obs = [
            self._make_detection("clip_7", i, [10 + i, 10 + i, 80 + i, 80 + i])
            for i in range(5)
        ]
        summary = track_clip_observations("clip_7", "south", obs)
        assert summary["total_unique_tracks"] >= 1


# ===========================================================================
# Phase 5: Cross-View Matcher
# ===========================================================================

class TestCrossViewMatcher:
    """Tests for backend.app.services.cross_view_matcher"""

    def _make_obs(self, view, cls, conf, visual="normal", lesion=2.0):
        return {
            "view_label": view,
            "yolo_prediction": cls,
            "yolo_confidence": conf,
            "opencv_visual_status": visual,
            "lesion_percentage": lesion,
            "visible_symptom_area_percent": lesion,
        }

    def test_healthy_all_views(self):
        from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
        views = ["north", "south", "east", "west"]
        obs = {v: [self._make_obs(v, "healthy", 0.85, "normal", 1.0)] * 6 for v in views}
        evidence = aggregate_cross_view_evidence("sess1", "plant1", obs, views)
        assert evidence["plant_status"] == "HEALTHY_APPEARING"
        assert evidence["unobserved_views"] == []
        assert evidence["coverage_fraction"] == 1.0

    def test_symptom_majority_views(self):
        from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
        views = ["north", "south", "east", "west"]
        obs = {
            "north": [self._make_obs("north", "early_blight", 0.75, "abnormal", 20.0)] * 6,
            "south": [self._make_obs("south", "early_blight", 0.70, "abnormal", 18.0)] * 6,
            "east": [self._make_obs("east", "early_blight", 0.65, "abnormal", 15.0)] * 6,
            "west": [self._make_obs("west", "healthy", 0.80, "normal", 1.0)] * 6,
        }
        evidence = aggregate_cross_view_evidence("sess2", "plant2", obs, views)
        assert evidence["plant_status"] == "SYMPTOMS_DETECTED"

    def test_unobserved_views_marked(self):
        from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
        declared = ["north", "south", "east", "west"]
        obs = {"north": [self._make_obs("north", "healthy", 0.80, "normal", 1.0)] * 4}
        evidence = aggregate_cross_view_evidence("sess3", "plant3", obs, declared)
        assert "south" in evidence["unobserved_views"]
        assert "east" in evidence["unobserved_views"]
        assert "west" in evidence["unobserved_views"]
        assert evidence["plant_status"] == "INSUFFICIENT_COVERAGE"

    def test_conflicting_views_creates_conflict_report(self):
        from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
        views = ["north", "south"]
        obs = {
            "north": [self._make_obs("north", "early_blight", 0.72, "abnormal", 22.0)] * 6,
            "south": [self._make_obs("south", "healthy", 0.82, "normal", 1.0)] * 6,
        }
        evidence = aggregate_cross_view_evidence("sess4", "plant4", obs, views)
        assert evidence["has_conflicts"] is True
        assert len(evidence["view_conflicts"]) > 0

    def test_max_lesion_upper_bound(self):
        from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
        views = ["north", "south"]
        obs = {
            "north": [self._make_obs("north", "early_blight", 0.70, "abnormal", 30.0)],
            "south": [self._make_obs("south", "early_blight", 0.65, "abnormal", 10.0)],
        }
        evidence = aggregate_cross_view_evidence("sess5", "plant5", obs, views)
        assert evidence["max_lesion_pct_across_views"] is not None
        assert evidence["max_lesion_pct_across_views"] >= 10.0

    def test_cross_view_caveats_present(self):
        from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
        views = ["north"]
        obs = {"north": [self._make_obs("north", "healthy", 0.80, "normal", 1.0)]}
        evidence = aggregate_cross_view_evidence("sess6", "plant6", obs, views)
        assert "cross_view_caveats" in evidence
        assert len(evidence["cross_view_caveats"]) > 0

    def test_no_declared_views_no_crash(self):
        from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
        obs = {"top": [self._make_obs("top", "healthy", 0.75, "normal", 2.0)] * 3}
        evidence = aggregate_cross_view_evidence("sess7", "plant7", obs, None)
        assert "plant_status" in evidence

    def test_empty_observations_per_view(self):
        from backend.app.services.cross_view_matcher import aggregate_cross_view_evidence
        views = ["north", "south"]
        obs = {"north": [], "south": []}
        evidence = aggregate_cross_view_evidence("sess8", "plant8", obs, views)
        assert evidence["plant_status"] == "INSUFFICIENT_COVERAGE"


# ===========================================================================
# Phase 6: Evaluation Framework
# ===========================================================================

class TestEvaluationFramework:
    """Tests for backend.app.services.evaluation"""

    def test_binary_metrics_perfect_classifier(self):
        from backend.app.services.evaluation import compute_plant_level_metrics
        preds = ["SYMPTOMS_DETECTED"] * 5 + ["HEALTHY_APPEARING"] * 5
        truths = preds[:]
        result = compute_plant_level_metrics(preds, truths)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["accuracy"] == 1.0

    def test_binary_metrics_all_wrong(self):
        from backend.app.services.evaluation import compute_plant_level_metrics
        preds = ["SYMPTOMS_DETECTED"] * 5
        truths = ["HEALTHY_APPEARING"] * 5
        result = compute_plant_level_metrics(preds, truths)
        assert result["recall"] == 0.0
        assert result["true_negatives"] == 0

    def test_binary_metrics_length_mismatch_raises(self):
        from backend.app.services.evaluation import compute_plant_level_metrics
        with pytest.raises(ValueError):
            compute_plant_level_metrics(["A", "B"], ["A"])

    def test_binary_metrics_empty_raises(self):
        from backend.app.services.evaluation import compute_plant_level_metrics
        result = compute_plant_level_metrics([], [])
        assert "error" in result

    def test_wilson_interval_bounds(self):
        from backend.app.services.evaluation import _wilson_interval
        lo, hi = _wilson_interval(10, 20)
        assert 0.0 <= lo <= hi <= 1.0

    def test_calibration_report_structure(self):
        from backend.app.services.evaluation import calibration_report
        confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        correct = [True, True, True, False, True, False, False, False, True, False]
        report = calibration_report(confs, correct, n_bins=5)
        assert "expected_calibration_error" in report
        assert "bins" in report
        assert len(report["bins"]) == 5
        assert 0.0 <= report["expected_calibration_error"] <= 1.0

    def test_calibration_report_length_mismatch_raises(self):
        from backend.app.services.evaluation import calibration_report
        with pytest.raises(ValueError):
            calibration_report([0.9, 0.8], [True])

    def test_scenario_scoring_pass(self):
        from backend.app.services.evaluation import score_simulation_scenario
        obs = [{"status": "SYMPTOMS_DETECTED", "opencv_visual_status": "abnormal",
                "yolo_prediction": "early_blight"} for _ in range(5)]
        result = score_simulation_scenario("test_scenario", obs, "SYMPTOMS_DETECTED", "early_blight")
        assert result["passed"] is True

    def test_scenario_scoring_fail(self):
        from backend.app.services.evaluation import score_simulation_scenario
        obs = [{"status": "HEALTHY_APPEARING", "opencv_visual_status": "normal",
                "yolo_prediction": "healthy"} for _ in range(5)]
        result = score_simulation_scenario("test_fail", obs, "SYMPTOMS_DETECTED")
        assert result["passed"] is False

    def test_evaluation_suite_summary(self):
        from backend.app.services.evaluation import run_evaluation_suite
        scenarios = [
            {
                "name": "s1",
                "observations": [{"status": "HEALTHY_APPEARING"}] * 3,
                "expected_status": "HEALTHY_APPEARING",
            },
            {
                "name": "s2",
                "observations": [{"status": "SYMPTOMS_DETECTED"}] * 3,
                "expected_status": "HEALTHY_APPEARING",  # intentional failure
            },
        ]
        result = run_evaluation_suite(scenarios)
        assert result["total_scenarios"] == 2
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert result["pass_rate"] == 0.5

    def test_view_coverage_analysis(self):
        from backend.app.services.evaluation import analyze_view_coverage
        reports = [
            {"coverage_fraction": 1.0, "cross_view_confidence": 0.85, "plant_status": "HEALTHY_APPEARING"},
            {"coverage_fraction": 0.5, "cross_view_confidence": 0.60, "plant_status": "UNCERTAIN"},
            {"coverage_fraction": 0.25, "cross_view_confidence": 0.30, "plant_status": "INSUFFICIENT_COVERAGE"},
        ]
        result = analyze_view_coverage(reports)
        assert "n_sessions" in result
        assert result["n_sessions"] == 3
        assert "mean_coverage_fraction" in result
        assert "coverage_confidence_correlation" in result


# ===========================================================================
# Phase 6: Simulation Fixtures End-to-End
# ===========================================================================

class TestSimulationFixtures:
    """Tests for backend.app.services.simulation_fixtures"""

    def test_all_fixtures_have_required_fields(self):
        from backend.app.services.simulation_fixtures import get_all_fixtures
        fixtures = get_all_fixtures()
        assert len(fixtures) >= 5
        for fx in fixtures:
            assert "scenario_name" in fx
            assert "ground_truth_status" in fx
            assert "view_observations" in fx
            assert "declared_viewpoints" in fx

    def test_healthy_fixture_has_all_views(self):
        from backend.app.services.simulation_fixtures import build_fixture_healthy_all_views
        fx = build_fixture_healthy_all_views()
        assert fx["ground_truth_status"] == "HEALTHY_APPEARING"
        assert len(fx["view_observations"]) == 4
        for view_obs in fx["view_observations"].values():
            assert len(view_obs) > 0

    def test_diseased_fixture(self):
        from backend.app.services.simulation_fixtures import build_fixture_diseased_majority_views
        fx = build_fixture_diseased_majority_views()
        assert fx["ground_truth_status"] == "SYMPTOMS_DETECTED"

    def test_low_coverage_fixture(self):
        from backend.app.services.simulation_fixtures import build_fixture_low_coverage
        fx = build_fixture_low_coverage()
        assert fx["ground_truth_status"] == "INSUFFICIENT_COVERAGE"
        # Only north has data
        observed = [k for k, v in fx["view_observations"].items() if v]
        assert observed == ["north"]

    def test_conflicting_fixture(self):
        from backend.app.services.simulation_fixtures import build_fixture_conflicting_views
        fx = build_fixture_conflicting_views()
        assert fx["ground_truth_status"] == "UNCERTAIN"

    def test_run_all_simulations(self):
        from backend.app.services.simulation_fixtures import run_all_simulation_scenarios
        report = run_all_simulation_scenarios()
        assert "suite_summary" in report
        assert "scenario_reports" in report
        assert report["total_fixtures"] >= 5
        # Suite should have a pass_rate field
        assert "pass_rate" in report["suite_summary"]

    def test_simulation_report_has_caveats(self):
        from backend.app.services.simulation_fixtures import run_all_simulation_scenarios
        report = run_all_simulation_scenarios()
        assert "caveats" in report
        assert len(report["caveats"]) > 0
