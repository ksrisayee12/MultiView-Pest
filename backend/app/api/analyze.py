import os
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, Dict, Any, List

from backend.image_processing.visual_health import analyze_visual_health
from backend.fusion.fusion_engine import (
    evaluate_tile_fusion,
    calculate_plant_health_index,
    generate_priority_list
)
from backend.recommendations.recommendations import get_recommendation
from backend.database.mongodb import db_manager
from simulation.tile_extractor import extract_16_tiles
from simulation.create_field_image import create_simulated_field

router = APIRouter()

MODEL_WEIGHTS_PATH = os.path.join("backend", "models", "yolo26_tomato.pt")

def predict_yolo_tile(tile_image: np.ndarray) -> Dict[str, Any]:
    """
    Runs YOLO26 model if weights exist; otherwise cleanly reports model pending status.
    No fabricated AI predictions.
    """
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_WEIGHTS_PATH)
            results = model.predict(tile_image, verbose=False)
            if results and len(results) > 0:
                probs = results[0].probs
                if probs is not None:
                    top_idx = int(probs.top1)
                    top_conf = float(probs.top1conf)
                    names = results[0].names
                    class_name = names.get(top_idx, "healthy")
                    return {"prediction": class_name, "confidence": top_conf, "model_loaded": True}
        except Exception as e:
            print(f"[YOLO26] Error during inference: {e}")

    return {
        "prediction": "YOLO model not trained",
        "confidence": 0.0,
        "model_loaded": False
    }

@router.post("/analyze/field")
async def analyze_field(file: Optional[UploadFile] = File(None)):
    """
    Processes a 4x4 composite field image (16 tiles), runs YOLO + OpenCV on each tile,
    executes Fusion engine, computes Plant Health Index, and stores results.
    """
    if file:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        field_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filename = file.filename
    else:
        # Fallback to simulated field
        f_path, _ = create_simulated_field()
        field_img = cv2.imread(f_path)
        filename = "simulated_field_image.jpg"

    if field_img is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    # Split composite into 16 tiles
    tiles = extract_16_tiles(field_img)
    tile_results = []

    for tile in tiles:
        tile_id = tile["tile_id"]
        tile_crop = tile["image"]

        # Method A: YOLO
        yolo_res = predict_yolo_tile(tile_crop)

        # Method B: OpenCV DIP
        opencv_res = analyze_visual_health(tile_crop)

        # Fusion Engine
        fusion_res = evaluate_tile_fusion(
            yolo_prediction=yolo_res["prediction"],
            yolo_confidence=yolo_res["confidence"],
            opencv_result=opencv_res
        )

        recommendation = get_recommendation(
            fusion_res["confirmed_disease"] or yolo_res["prediction"]
        )

        tile_results.append({
            "tile_id": tile_id,
            "grid_row": tile["grid_row"],
            "grid_col": tile["grid_col"],
            "bbox": tile["bbox"],
            "yolo_prediction": yolo_res["prediction"],
            "yolo_confidence": yolo_res["confidence"],
            "opencv_status": opencv_res["visual_status"],
            "greenness_index": opencv_res["greenness_index"],
            "green_pixel_percentage": opencv_res["green_pixel_percentage"],
            "lesion_percentage": opencv_res["lesion_percentage"],
            "fusion_status": fusion_res["fusion_status"],
            "confirmed_disease": fusion_res["confirmed_disease"],
            "severity": fusion_res["severity"],
            "requires_closeup": fusion_res["requires_closeup"],
            "recommendation": recommendation
        })

    health_index = calculate_plant_health_index(tile_results)
    priority_list = generate_priority_list(tile_results)

    scan_summary = {
        "crop": "Tomato",
        "image_name": filename,
        "total_tiles": len(tile_results),
        "healthy_tiles": sum(1 for t in tile_results if t["fusion_status"] == "HEALTHY"),
        "confirmed_disease_tiles": sum(1 for t in tile_results if t["fusion_status"] == "CONFIRMED DISEASE"),
        "disputed_tiles": sum(1 for t in tile_results if t["fusion_status"] == "DISPUTED"),
        "plant_health_index": health_index,
        "yolo_model_loaded": os.path.exists(MODEL_WEIGHTS_PATH),
        "tile_results": tile_results,
        "priority_list": priority_list
    }

    scan_id = db_manager.save_scan(scan_summary)
    scan_summary["scan_id"] = scan_id

    return scan_summary

@router.post("/analyze/tile")
async def analyze_tile(file: UploadFile = File(...)):
    """Analyzes a single leaf tile image with YOLO + OpenCV + Fusion."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    tile_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if tile_img is None:
        raise HTTPException(status_code=400, detail="Invalid leaf tile image.")

    yolo_res = predict_yolo_tile(tile_img)
    opencv_res = analyze_visual_health(tile_img)

    fusion_res = evaluate_tile_fusion(
        yolo_prediction=yolo_res["prediction"],
        yolo_confidence=yolo_res["confidence"],
        opencv_result=opencv_res
    )

    recommendation = get_recommendation(
        fusion_res["confirmed_disease"] or yolo_res["prediction"]
    )

    return {
        "yolo_prediction": yolo_res["prediction"],
        "yolo_confidence": yolo_res["confidence"],
        "yolo_model_loaded": yolo_res["model_loaded"],
        "opencv_status": opencv_res["visual_status"],
        "greenness_index": opencv_res["greenness_index"],
        "green_pixel_percentage": opencv_res["green_pixel_percentage"],
        "lesion_percentage": opencv_res["lesion_percentage"],
        "fusion_status": fusion_res["fusion_status"],
        "confirmed_disease": fusion_res["confirmed_disease"],
        "severity": fusion_res["severity"],
        "requires_closeup": fusion_res["requires_closeup"],
        "recommendation": recommendation
    }

@router.post("/analyze/closeup")
async def analyze_closeup(
    file: UploadFile = File(...),
    tile_id: int = Form(...),
    scan_id: Optional[str] = Form(None)
):
    """
    Analyzes a close-up leaf image for a disputed tile and updates the decision.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    closeup_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if closeup_img is None:
        raise HTTPException(status_code=400, detail="Invalid close-up leaf image.")

    yolo_res = predict_yolo_tile(closeup_img)
    opencv_res = analyze_visual_health(closeup_img)

    fusion_res = evaluate_tile_fusion(
        yolo_prediction=yolo_res["prediction"],
        yolo_confidence=yolo_res["confidence"],
        opencv_result=opencv_res
    )

    final_decision = fusion_res["fusion_status"]
    if final_decision == "DISPUTED":
        final_decision = "REQUIRES MANUAL REVIEW"

    recommendation = get_recommendation(
        fusion_res["confirmed_disease"] or yolo_res["prediction"]
    )

    resolution_result = {
        "tile_id": tile_id,
        "scan_id": scan_id,
        "closeup_yolo_prediction": yolo_res["prediction"],
        "closeup_yolo_confidence": yolo_res["confidence"],
        "closeup_opencv_status": opencv_res["visual_status"],
        "closeup_lesion_percentage": opencv_res["lesion_percentage"],
        "final_decision": final_decision,
        "confirmed_disease": fusion_res["confirmed_disease"],
        "severity": fusion_res["severity"],
        "recommendation": recommendation
    }

    return resolution_result
