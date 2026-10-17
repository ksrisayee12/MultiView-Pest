"""
DIP Before/After Visualization Script.
Stage 2: Generates visual comparisons and metadata logs saved to data/processed/.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing.dip_pipeline import ImagePreprocessor

PROCESSED_DIR = BASE_DIR / "data" / "processed"
SAMPLES_DIR = BASE_DIR / "data" / "samples"


def create_sample_leaf_image(output_path: Path) -> Path:
    """Generate a sample synthetic agricultural leaf image for DIP demonstration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Background soil color (brownish)
    img[:, :] = (35, 65, 90)

    # Draw synthetic leaf shape (green blob with texture and spots)
    cv2.ellipse(img, (200, 200), (140, 80), 30, 0, 360, (40, 160, 50), -1)
    cv2.ellipse(img, (200, 200), (130, 70), 30, 0, 360, (50, 190, 60), -1)

    # Draw leaf veins
    cv2.line(img, (100, 250), (300, 150), (80, 220, 90), 3)
    cv2.line(img, (160, 220), (180, 250), (70, 200, 80), 2)
    cv2.line(img, (220, 190), (250, 220), (70, 200, 80), 2)

    # Add artificial spot / noise
    cv2.circle(img, (210, 180), 12, (20, 40, 140), -1)  # Simulated disease spot
    cv2.circle(img, (170, 210), 8, (15, 30, 110), -1)

    cv2.imwrite(str(output_path), img)
    return output_path


def run_dip_visualization(sample_image_path: Path) -> Path:
    """Execute DIP pipeline on sample image and save before/after comparison plot + metadata."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor = ImagePreprocessor()

    # Execute pipeline
    result = preprocessor.process_pipeline(sample_image_path)

    # Save metadata JSON
    metadata = result.metadata
    metadata["output_image_path"] = str(PROCESSED_DIR / "processed_output.png")
    json_path = PROCESSED_DIR / f"{sample_image_path.stem}_dip_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Generate Visualization Grid
    stages = result.intermediate_stages
    num_stages = len(stages)

    fig, axes = plt.subplots(1, num_stages, figsize=(4 * num_stages, 4))
    if num_stages == 1:
        axes = [axes]

    for idx, (stage_name, img_data) in enumerate(stages.items()):
        ax = axes[idx]
        disp_img = img_data.copy()

        # Handle float normalized image [0.0, 1.0] for plt display
        if disp_img.dtype == np.float32 or disp_img.dtype == np.float64:
            disp_img = np.clip(disp_img, 0.0, 1.0)
            if disp_img.ndim == 3:
                ax.imshow(disp_img)
            else:
                ax.imshow(disp_img, cmap="gray")
        else:
            if disp_img.ndim == 3:
                # Convert BGR to RGB for matplotlib
                disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
                ax.imshow(disp_img)
            else:
                ax.imshow(disp_img, cmap="gray")

        ax.set_title(f"{stage_name.capitalize()}", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plot_output_path = PROCESSED_DIR / f"{sample_image_path.stem}_dip_comparison.png"
    plt.savefig(plot_output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[DIP] Comparison plot saved to: {plot_output_path}")
    print(f"[DIP] Metadata JSON saved to: {json_path}")
    return plot_output_path


if __name__ == "__main__":
    sample_file = SAMPLES_DIR / "sample_leaf.png"
    if not sample_file.exists():
        create_sample_leaf_image(sample_file)

    run_dip_visualization(sample_file)
