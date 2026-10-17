import cv2
import numpy as np
from typing import List, Dict, Any

def extract_16_tiles(field_image: np.ndarray) -> List[Dict[str, Any]]:
    """Splits a composite 4x4 field image into 16 individual tile images."""
    if field_image is None or field_image.size == 0:
        return []

    h, w = field_image.shape[:2]
    tile_h = h // 4
    tile_w = w // 4

    tiles = []
    tile_id = 1
    for r in range(4):
        for c in range(4):
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w

            crop = field_image[y1:y2, x1:x2].copy()
            tiles.append({
                "tile_id": tile_id,
                "grid_row": r,
                "grid_col": c,
                "bbox": [x1, y1, x2, y2],
                "image": crop
            })
            tile_id += 1

    return tiles
