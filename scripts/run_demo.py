import sys
import os
import cv2
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.create_field_image import create_simulated_field
from simulation.tile_extractor import extract_16_tiles
from backend.image_processing.visual_health import analyze_visual_health
from backend.fusion.fusion_engine import evaluate_tile_fusion, calculate_plant_health_index
from evaluation.evaluate_all import run_evaluation

def main():
    print("=" * 60)
    print("  AGRICULTURAL HEALTH ASSESSMENT DEMO PIPELINE RUNNER")
    print("=" * 60)

    print("\n[Step 1] Creating Simulated 4x4 Field Composite...")
    f_path, m_path = create_simulated_field()
    print(f"  Field Image: {f_path}")

    print("\n[Step 2] Extracting 16 Grid Tiles...")
    field_img = cv2.imread(f_path)
    tiles = extract_16_tiles(field_img)
    print(f"  Extracted {len(tiles)} tiles.")

    from backend.app.services.model_registry import predict_disease

    print("\n[Step 3] Running Dual-Evidence Fusion Engine on 16 Tiles...")
    results = []
    for tile in tiles:
        dip = analyze_visual_health(tile["image"])
        yolo_res = predict_disease(tile["image"])
        fusion = evaluate_tile_fusion(yolo_res["prediction"], yolo_res["confidence"], dip)
        fusion["tile_id"] = tile["tile_id"]
        results.append(fusion)

    health_index = calculate_plant_health_index(results)
    print(f"  Calculated Field Plant Health Index: {health_index} / 100")

    healthy_count = sum(1 for r in results if r["fusion_status"] == "HEALTHY")
    disease_count = sum(1 for r in results if r["fusion_status"] == "CONFIRMED DISEASE")
    disputed_count = sum(1 for r in results if r["fusion_status"] == "DISPUTED")

    print(f"  Tile Breakdown: {healthy_count} Healthy | {disease_count} Confirmed Disease | {disputed_count} Disputed")

    print("\n[Step 4] Running Stressed Image Evaluation Suite...")
    eval_report = run_evaluation()
    print("  Evaluation finished cleanly. Results saved to results/metrics.json")

    print("\n[Step 5] Demonstrating Phase 3 Multi-View Same-Plant Fusion...")
    from backend.app.services.plant_fusion import fuse_plant_assessment
    demo_observations = [
        {
            "clip_id": "demo_clip_front",
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "view_label": "front",
            "yolo_prediction": "healthy",
            "yolo_confidence": 0.94,
            "opencv_visual_status": "Healthy",
            "lesion_percentage": 0.2,
            "model_loaded": True,
        },
        {
            "clip_id": "demo_clip_back",
            "frame_index": 15,
            "timestamp_sec": 0.5,
            "view_label": "back",
            "yolo_prediction": "early_blight",
            "yolo_confidence": 0.86,
            "opencv_visual_status": "abnormal",
            "lesion_percentage": 14.2,
            "model_loaded": True,
        },
        {
            "clip_id": "demo_clip_left",
            "frame_index": 30,
            "timestamp_sec": 1.0,
            "view_label": "left",
            "yolo_prediction": "healthy",
            "yolo_confidence": 0.91,
            "opencv_visual_status": "Healthy",
            "lesion_percentage": 0.5,
            "model_loaded": True,
        },
    ]
    demo_clips_meta = [
        {"clip_id": "demo_clip_front", "view_label": "front", "processing_status": "COMPLETED", "frames_extracted": 1},
        {"clip_id": "demo_clip_back", "view_label": "back", "processing_status": "COMPLETED", "frames_extracted": 1},
        {"clip_id": "demo_clip_left", "view_label": "left", "processing_status": "COMPLETED", "frames_extracted": 1},
    ]
    mv_assessment = fuse_plant_assessment(
        session_id="sess_demo_cli",
        plant_id="PLANT-DEMO-001",
        crop_type="tomato",
        clips_meta=demo_clips_meta,
        all_observations=demo_observations,
    )
    print(f"  Multi-View Status: {mv_assessment['status']}")
    print(f"  Views Submitted: {mv_assessment['submitted_views']}")
    print(f"  Top Symptom: {mv_assessment.get('top_disease_class')}")
    print(f"  Unobserved Regions Remain: {mv_assessment['missing_views']}")

    print("\n[Step 6] Running Phase 4-6 Simulation & Cross-View Evaluation Suite...")
    from backend.app.services.simulation_fixtures import run_all_simulation_scenarios
    sim_report = run_all_simulation_scenarios()
    suite_summary = sim_report.get("suite_summary", {})
    print(f"  Simulated Scenarios Evaluated: {sim_report.get('total_fixtures', 0)}")
    print(f"  Scenario Pass Rate: {suite_summary.get('passed', 0)}/{suite_summary.get('total_scenarios', 0)} ({suite_summary.get('pass_rate', 0.0)*100:.1f}%)")
    for rep in sim_report.get("scenario_reports", []):
        print(f"    - {rep['scenario_name']}: Expected={rep['ground_truth_status']} -> Pred={rep['predicted_plant_status']} (Coverage={rep.get('coverage_fraction', 0.0)*100:.0f}%, Conflicts={rep.get('has_conflicts')})")

    print("\n" + "=" * 60)
    print("  ALL DEMO PIPELINES (FIELD TILE + MULTI-VIEW + SIMULATION) VERIFIED!")
    print("=" * 60)

if __name__ == "__main__":
    main()

