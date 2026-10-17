"""
Dataset Validator Script for Multi-View Agricultural Analysis System.
Stage 1 Baseline Integrity Checker.

Checks directory structures, configuration file parsing, image file integrity,
and reports dataset readiness without auto-downloading or fabricating data.
"""

import os
import sys
from pathlib import Path

# OpenCV is verified installed in the agrivision environment
import cv2

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

REQUIRED_DIRS = [
    DATA_DIR / "raw" / "pest",
    DATA_DIR / "raw" / "disease",
    DATA_DIR / "raw" / "multiview",
    DATA_DIR / "raw" / "drone",
    DATA_DIR / "processed",
    DATA_DIR / "samples",
]

REQUIRED_CONFIGS = [
    CONFIGS_DIR / "pest_classes.yaml",
    CONFIGS_DIR / "disease_classes.yaml",
    CONFIGS_DIR / "dataset_sources.yaml",
    CONFIGS_DIR / "preprocessing.yaml",
    CONFIGS_DIR / "severity.yaml",
    CONFIGS_DIR / "recommendations.yaml",
]


def load_yaml_simple(file_path: Path) -> dict:
    """Read configuration file using PyYAML if available, or simple parser fallback."""
    try:
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback text check if pyyaml is not yet installed in environment
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return {"_raw_lines_count": len(lines), "_file_name": file_path.name}


def check_directories() -> dict:
    """Verify presence of required project data directories."""
    results = {}
    for d in REQUIRED_DIRS:
        exists = d.exists() and d.is_dir()
        results[str(d.relative_to(BASE_DIR))] = exists
    return results


def check_configurations() -> dict:
    """Check availability and readability of configuration files."""
    results = {}
    for c in REQUIRED_CONFIGS:
        if not c.exists():
            results[c.name] = {"status": "MISSING", "data": None}
            continue
        try:
            parsed = load_yaml_simple(c)
            results[c.name] = {"status": "READABLE", "data": parsed}
        except Exception as e:
            results[c.name] = {"status": f"ERROR: {str(e)}", "data": None}
    return results


def scan_images_in_dir(target_dir: Path) -> dict:
    """Scan directory for images, count valid vs corrupt files."""
    if not target_dir.exists():
        return {"total": 0, "valid": 0, "corrupt": 0}

    valid_count = 0
    corrupt_count = 0
    total_count = 0

    for root, _, files in os.walk(target_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in ALLOWED_EXTENSIONS:
                total_count += 1
                img_path = os.path.join(root, file)
                # Attempt reading image header/pixels using OpenCV
                try:
                    img = cv2.imread(img_path)
                    if img is not None and img.size > 0:
                        valid_count += 1
                    else:
                        corrupt_count += 1
                except Exception:
                    corrupt_count += 1

    return {"total": total_count, "valid": valid_count, "corrupt": corrupt_count}


def run_validation_report() -> bool:
    """Run full dataset validation sweep and print summary report."""
    print("=" * 60)
    print(" STAGE 1: DATASET & CONFIGURATION VALIDATION REPORT")
    print("=" * 60)

    # 1. Directories Check
    dir_status = check_directories()
    all_dirs_ok = all(dir_status.values())
    print("\n[1] Directory Structure Check:")
    for path_str, ok in dir_status.items():
        status_str = "EXISTS" if ok else "MISSING"
        print(f"  - {path_str:<25}: [{status_str}]")

    # 2. Config Files Check
    config_status = check_configurations()
    all_configs_ok = all(v["status"] in ("READABLE", "OK") for v in config_status.values())
    print("\n[2] Configuration Files Check:")
    for cfg_name, res in config_status.items():
        print(f"  - {cfg_name:<25}: [{res['status']}]")

    # 3. Image Datasets Scan
    print("\n[3] Image Dataset File Scan:")
    for d in REQUIRED_DIRS:
        scan = scan_images_in_dir(d)
        rel_path = str(d.relative_to(BASE_DIR))
        print(f"  - {rel_path:<25}: Total: {scan['total']} | Valid: {scan['valid']} | Corrupt: {scan['corrupt']}")

    print("\n" + "=" * 60)
    passed = all_dirs_ok and all_configs_ok
    if passed:
        print(" RESULT: PASS - Workspace structure and configurations are valid.")
    else:
        print(" RESULT: FAIL - Some required directories or configs are missing.")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    success = run_validation_report()
    sys.exit(0 if success else 1)
