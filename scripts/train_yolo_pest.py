"""
YOLO26 Pest Training Preparation Script.
Stage 3: Configures training workflow for IP102 pest dataset without downloading data or running auto-training.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = BASE_DIR / "configs" / "ip102_yolo_dataset.yaml"
RAW_PEST_DIR = BASE_DIR / "data" / "raw" / "pest"


def check_training_readiness() -> dict:
    """Inspect dataset availability and report training readiness status."""
    train_images = RAW_PEST_DIR / "images" / "train"
    val_images = RAW_PEST_DIR / "images" / "val"

    dataset_yaml_exists = DATASET_YAML.exists()
    has_train_data = train_images.exists() and any(train_images.iterdir())
    has_val_data = val_images.exists() and any(val_images.iterdir())

    ready = dataset_yaml_exists and has_train_data and has_val_data

    status_message = (
        "READY_FOR_TRAINING"
        if ready
        else "Training pending IP102 dataset acquisition. Run dataset acquisition when ready."
    )

    return {
        "is_ready": ready,
        "status_message": status_message,
        "dataset_config": str(DATASET_YAML),
        "train_images_exist": has_train_data,
        "val_images_exist": has_val_data,
    }


def main():
    print("=" * 60)
    print(" YOLO26 PEST DETECTION: TRAINING PREPARATION CHECK")
    print("=" * 60)

    report = check_training_readiness()
    print(f"Status              : {report['status_message']}")
    print(f"Dataset Config YAML : {report['dataset_config']}")
    print(f"Train Images Found  : {report['train_images_exist']}")
    print(f"Val Images Found    : {report['val_images_exist']}")
    print("=" * 60)

    if not report["is_ready"]:
        print("\n[NOTE] Training was NOT started. No dataset download or model training was executed.")
    else:
        print("\n[READY] Run 'yolo detect train data=configs/ip102_yolo_dataset.yaml model=yolo26n.pt' when ready.")


if __name__ == "__main__":
    main()
