import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple

CLASSES = ["healthy", "early_blight", "late_blight", "septoria_leaf_spot"]

def generate_synthetic_leaf(label: str, width: int = 256, height: int = 256) -> np.ndarray:
    """Generates a realistic synthetic leaf tile for simulation fallback."""
    # Base green leaf canvas
    img = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8) # Forest green

    # Add leaf vein structures
    cv2.line(img, (width // 2, height), (width // 2, 20), (20, 100, 20), 3)
    cv2.line(img, (width // 2, height // 2), (width // 4, height // 3), (20, 100, 20), 2)
    cv2.line(img, (width // 2, height // 2), (3 * width // 4, height // 3), (20, 100, 20), 2)

    # Add disease spots if not healthy
    if label == "early_blight":
        # Concentric dark brown spots with yellow halos
        cv2.circle(img, (width // 3, height // 3), 22, (20, 215, 255), -1) # Yellow halo
        cv2.circle(img, (width // 3, height // 3), 16, (19, 69, 139), -1) # Brown spot
        cv2.circle(img, (2 * width // 3, 2 * height // 3), 18, (20, 215, 255), -1)
        cv2.circle(img, (2 * width // 3, 2 * height // 3), 12, (19, 69, 139), -1)
    elif label == "late_blight":
        # Irregular water-soaked dark gray/brown patches
        cv2.ellipse(img, (width // 2, height // 3), (35, 20), 30, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(img, (width // 3, 2 * height // 3), (25, 40), 15, 0, 360, (40, 40, 80), -1)
    elif label == "septoria_leaf_spot":
        # Numerous small circular spots with dark borders
        for pos in [(60, 60), (180, 70), (120, 150), (80, 200), (190, 190)]:
            cv2.circle(img, pos, 10, (200, 200, 200), -1) # Center gray
            cv2.circle(img, pos, 10, (20, 20, 20), 2) # Dark border

    # Add subtle Gaussian noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

def create_simulated_field(output_dir: str = "data/processed") -> Tuple[str, str]:
    """
    Creates a 4x4 composite simulated field image (16 tiles) and saves metadata.
    Does NOT pretend real drone data was used.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tile_size = 256
    grid_size = 4
    canvas = np.zeros((tile_size * grid_size, tile_size * grid_size, 3), dtype=np.uint8)

    # Preset deterministic labels for reproducible 4x4 simulation
    tile_labels = [
        "healthy", "early_blight", "healthy", "septoria_leaf_spot",
        "healthy", "late_blight", "early_blight", "healthy",
        "late_blight", "healthy", "early_blight", "healthy",
        "septoria_leaf_spot", "healthy", "healthy", "late_blight"
    ]

    metadata = {
        "description": "Simulated 4x4 Field Composite Image created from leaf tile samples.",
        "disclaimer": "Drone view is simulated using a multi-tile composite image because real drone imagery is not available for this prototype.",
        "tiles": {}
    }

    tile_id = 1
    for r in range(grid_size):
        for c in range(grid_size):
            label = tile_labels[tile_id - 1]
            tile_img = generate_synthetic_leaf(label, tile_size, tile_size)

            y_start, y_end = r * tile_size, (r + 1) * tile_size
            x_start, x_end = c * tile_size, (c + 1) * tile_size

            canvas[y_start:y_end, x_start:x_end] = tile_img

            metadata["tiles"][f"tile_{tile_id:02d}"] = {
                "tile_id": tile_id,
                "grid_row": r,
                "grid_col": c,
                "true_label": label,
                "bbox": [x_start, y_start, x_end, y_end]
            }
            tile_id += 1

    field_path = os.path.join(output_dir, "field_image.jpg")
    meta_path = os.path.join(output_dir, "field_metadata.json")

    cv2.imwrite(field_path, canvas)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return field_path, meta_path

if __name__ == "__main__":
    f_path, m_path = create_simulated_field()
    print(f"Simulated field created: {f_path}")
