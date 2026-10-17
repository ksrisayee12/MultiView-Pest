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

    from backend.app.api.analyze import predict_yolo_tile

    print("\n[Step 3] Running Dual-Evidence Fusion Engine on 16 Tiles...")
    results = []
    for tile in tiles:
        dip = analyze_visual_health(tile["image"])
        yolo_res = predict_yolo_tile(tile["image"])
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

    print("\n" + "=" * 60)
    print("  DEMO PIPELINE VERIFICATION SUCCESSFUL!")
    print("=" * 60)

if __name__ == "__main__":
    main()
