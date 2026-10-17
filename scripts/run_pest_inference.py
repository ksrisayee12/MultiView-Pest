"""
Pest Detection Inference Runner Script.
Stage 3: Loads image, executes detector wrapper, logs JSON results, saves annotated image.
"""

import sys
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.pest_detector import YOLOPestDetector

SAMPLES_DIR = BASE_DIR / "data" / "samples"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "pest_detections"


def run_sample_pest_inference(sample_image_path: Path):
    """Execute pest detector on sample image and log results."""
    detector = YOLOPestDetector()
    img = cv2.imread(str(sample_image_path))

    result = detector.detect(img)
    saved_paths = detector.save_results(result, output_dir=PROCESSED_DIR, image_input=img)

    print("=" * 60)
    print(" PEST INFERENCE EXECUTION RESULT")
    print("=" * 60)
    print(f"Image Path        : {result.image_path}")
    print(f"Model Status      : {result.model_status}")
    print(f"Total Pests Found : {result.total_pest_count}")
    print(f"Category Counts   : {result.category_counts}")
    print(f"Inference Time    : {result.inference_time_ms} ms")
    print(f"JSON Log Path     : {saved_paths.get('json_path')}")
    if "image_path" in saved_paths:
        print(f"Annotated Image   : {saved_paths.get('image_path')}")
    print("=" * 60)


if __name__ == "__main__":
    sample_file = SAMPLES_DIR / "sample_leaf.png"
    if sample_file.exists():
        run_sample_pest_inference(sample_file)
    else:
        print(f"Sample image not found at {sample_file}")
