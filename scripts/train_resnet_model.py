"""
scripts/train_resnet_model.py

Trains an orthogonal deep architecture (ResNet18 with transfer learning)
on the real PlantVillage pathology dataset (data/raw/disease/PlantVillage/).

Fuses with AgriHealthNet (MobileNetV3) to create a dual-architecture
deep voting ensemble with high field-grade accuracy.
"""

import os
import sys
import time
import json
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "raw" / "disease" / "PlantVillage"
MODELS_DIR = BASE_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
]

CLASS_TO_TAXONOMY_KEY = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites",
    "Tomato___Target_Spot": "target_spot",
    "Tomato___Tomato_mosaic_virus": "mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Pepper,_bell___Bacterial_spot": "pepper_bacterial_spot",
    "Pepper,_bell___healthy": "pepper_healthy",
}


class PlantVillageDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        if is_train:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        bgr = cv2.imread(path)
        if bgr is None:
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb), label


def train_resnet():
    print("=" * 65)
    print("  Training AgriHealthNet-ResNet18 Dual Ensemble Architecture")
    print("=" * 65)

    all_samples = []
    for cls_idx, cls_name in enumerate(DATASET_CLASSES):
        cls_dir = DATA_DIR / cls_name
        if not cls_dir.exists():
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.png"))
        unique_paths = list({str(p.resolve()): p for p in imgs}.values())
        for p in unique_paths:
            all_samples.append((str(p), cls_idx))

    print(f"Loaded {len(all_samples)} unique verified pathology samples across {len(DATASET_CLASSES)} classes.")
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    train_ds = PlantVillageDataset(train_samples, is_train=True)
    val_ds = PlantVillageDataset(val_samples, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Initialize ResNet18 pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, len(DATASET_CLASSES))
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_val_acc = 0.0
    best_weights = None

    # =================================================================
    # Phase 1: Train Classifier Head & Layer 4 (Base Layers Frozen)
    # =================================================================
    print("\n>> Phase 1: Training ResNet18 Head & Layer 4 (Base Layers Frozen)...")
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_p1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    epochs_p1 = 3

    for epoch in range(1, epochs_p1 + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_p1.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer_p1.step()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        train_acc = (train_correct / train_total) * 100.0
        val_acc = (val_correct / val_total) * 100.0
        print(f"  [Head Align] Epoch [{epoch:02d}/{epochs_p1:02d}] Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.1f}%")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # =================================================================
    # Phase 2: Fine-Tuning Layer 3, Layer 4 & Classifier Head
    # =================================================================
    print("\n>> Phase 2: Fine-Tuning Deep Convolutional Blocks (Layer 3 & 4)...")
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    epochs_p2 = 4
    optimizer_p2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=epochs_p2)

    for epoch in range(1, epochs_p2 + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_p2.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer_p2.step()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(out, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        scheduler_p2.step()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        train_acc = (train_correct / train_total) * 100.0
        val_acc = (val_correct / val_total) * 100.0
        print(f"  [Fine Tune]  Epoch [{epoch:02d}/{epochs_p2:02d}] Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.1f}%")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out_path = MODELS_DIR / "plant_disease_resnet18.pth"
    torch.save(best_weights, str(out_path))

    with open(out_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "model_name": "AgriHealthNet_ResNet18",
        "architecture": "resnet18",
        "classes": DATASET_CLASSES,
        "taxonomy_mapping": CLASS_TO_TAXONOMY_KEY,
        "val_accuracy": round(best_val_acc, 2),
        "sha256": sha256_hash,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(MODELS_DIR / "plant_disease_resnet18_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  -> Best ResNet18 Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  -> Model saved: {out_path} [SHA-256: {sha256_hash[:10]}...]")


if __name__ == "__main__":
    train_resnet()
