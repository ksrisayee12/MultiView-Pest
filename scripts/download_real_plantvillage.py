"""
scripts/download_real_plantvillage.py

Downloads a real, multi-class agricultural dataset from the official PlantVillage repository
(spMohanty/PlantVillage-Dataset) to train accurate production diagnostic models.

Classes downloaded:
  1. Tomato___healthy
  2. Tomato___Early_blight
  3. Tomato___Late_blight
  4. Tomato___Septoria_leaf_spot
  5. Tomato___Bacterial_spot
  6. Tomato___Leaf_Mold
  7. Tomato___Spider_mites Two-spotted_spider_mite
  8. Tomato___Target_Spot
  9. Tomato___Tomato_Yellow_Leaf_Curl_Virus
 10. Tomato___Tomato_mosaic_virus
 11. Potato___Early_blight
 12. Potato___Late_blight
 13. Pepper,_bell___Bacterial_spot
 14. Pepper,_bell___healthy
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "disease" / "PlantVillage"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
]

SAMPLES_PER_CLASS = 120  # ~1,680 real verified images total


def get_image_list_for_class(cls_name: str):
    url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{urllib.parse.quote(cls_name)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AgriHealth/1.0)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            images = [item for item in data if item["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
            return cls_name, images
    except Exception as e:
        print(f"Error listing {cls_name}: {e}")
        return cls_name, []


def download_single_image(dl_url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0 (AgriHealth/1.0)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
            if len(data) > 1000:
                with open(dest_path, "wb") as f:
                    f.write(data)
                return True
    except Exception:
        pass
    return False


def main():
    print("=" * 65)
    print("  PlantVillage Real Agronomic Dataset Downloader")
    print(f"  Target: {len(TARGET_CLASSES)} classes, {SAMPLES_PER_CLASS} samples each")
    print(f"  Destination: {RAW_DIR}")
    print("=" * 65)

    t0 = time.time()
    download_tasks = []

    print("\n[1/2] Querying class manifests from spMohanty/PlantVillage-Dataset...", flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(get_image_list_for_class, cls): cls for cls in TARGET_CLASSES}
        for fut in as_completed(futures):
            cls_name, images = fut.result()
            target_folder = RAW_DIR / cls_name
            target_folder.mkdir(parents=True, exist_ok=True)
            selected = images[:SAMPLES_PER_CLASS]
            print(f"  -> {cls_name}: found {len(images)} images (selected {len(selected)})", flush=True)
            for item in selected:
                dest = target_folder / item["name"]
                download_tasks.append((item["download_url"], dest))

    print(f"\n[2/2] Downloading {len(download_tasks)} real plant pathology images concurrently...", flush=True)
    success_count = 0
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(download_single_image, url, path) for url, path in download_tasks]
        for fut in as_completed(futures):
            if fut.result():
                success_count += 1
                if success_count % 50 == 0 or success_count == len(download_tasks):
                    print(f"  Downloaded: {success_count}/{len(download_tasks)} images...", flush=True)

    elapsed = round(time.time() - t0, 1)
    print("=" * 65, flush=True)
    print(f"  SUCCESS: Downloaded {success_count} real pathology images in {elapsed}s", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
