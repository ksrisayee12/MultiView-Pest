# MultiView-AgriHealth: Agricultural Plant Health Assessment System

A modular, dual-evidence AI & Digital Image Processing (DIP) agricultural decision-support application
for plant health assessment, disease detection, damage area estimation, and multi-view consensus monitoring.

> **Production Decision-Support System** — Fully operational multi-view video/image pipeline, temporal leaf tracking, cross-view consensus fusion, multi-plant batch mode with sample isolation, report export (Markdown & JSON), and a real fine-tuned YOLOv8n tomato disease detection model with verified provenance. Missing conditions return explicit unobserved/not-detected states without fabricating predictions.

---

## Capabilities & Workflows

### 1. Flexible Media Ingestion Workflow (Primary Mode)
Accepts **single images, multiple images, single videos, or multiple videos** for individual plants or across a batch:
- **Streamed Video & Image Ingestion:** Decodes uploaded video clips on the fly with OpenCV `VideoCapture` (never loads full video into memory) and processes still images (`.jpg`, `.png`, `.webp`) as 1-frame observations.
- **Bulk Media Queue:** Drag-and-drop or multi-select file ingestion with per-file viewpoint angle tagging (`front`, `back`, `left`, `right`, `top`, `underside`, `unknown`).
- **Multi-Plant Batch Mode (Strict Isolation):** Accepts bulk media mapped to distinct Plant IDs (`PLANT-001`, `PLANT-002`). Evidence from Plant A is strictly isolated from Plant B.
- **Content-Hash Deduplication:** SHA-256 content hashing automatically detects and suppresses duplicate uploads.
- **Leaf-Region Lesion Estimation:** Restricts visible symptom measurement strictly to detected foliage masks (via YOLO segmentation or HSV fallback), preventing background/soil false positives.
- **Within-Clip Temporal Tracking:** Associates leaf bounding boxes across adjacent frames using IoU/centroid matching, suppressing duplicate leaf counts.
- **Cross-View Evidence Association:** Merges multi-angle visual evidence, detects conflicting observations, bounds maximum visible lesion area, and records unobserved viewpoints as `UNKNOWN`.
- **Honest Status Taxonomy:**
  - `HEALTHY_APPEARING` — Sufficient viewpoints observed with no symptoms detected above threshold.
  - `SYMPTOMS_DETECTED` — Credible disease/symptom evidence detected in one or more viewpoints.
  - `UNCERTAIN` — Conflicting evidence between views or models.
  - `INSUFFICIENT_COVERAGE` — Fewer than required viewpoints observed.
  - `MODEL_UNAVAILABLE` — Model weights missing or offline.
- **Decision-Support Report Export:** One-click download of full reports in Markdown (`.md`) and JSON (`.json`) with complete model provenance (SHA-256 hash), observation summaries, and agronomic disclaimers.

### 2. Legacy Field-Tile Composite Workflow (Preserved Mode)
Retained for spatial drone composite scans:
- Splits a field composite into a 4×4 grid of 16 spatial tiles.
- Runs dual-evidence fusion combining YOLO predictions with OpenCV color/lesion metrics.
- Computes a prototype Field Plant Health Index (0–100).

---

## Active Real ML Model

- **Model Architecture:** Fine-Tuned YOLOv8n Object Detector
- **Artifact Path:** `backend/models/yolo26_tomato.pt` (6.25 MB)
- **Artifact SHA-256:** `ccc05ab88bb550567fb7a7a07c564d37e344e7e69e094da54481ea47c0f134b3`
- **Supported Conditions (9 Classes):**
  1. Tomato Early Blight (`early_blight`)
  2. Tomato Septoria Leaf Spot (`septoria_leaf_spot`)
  3. Tomato Healthy Foliage (`healthy`)
  4. Tomato Bacterial Spot (`bacterial_spot`)
  5. Tomato Late Blight (`late_blight`)
  6. Tomato Mosaic Virus (`mosaic_virus`)
  7. Tomato Yellow Leaf Curl Virus (`yellow_leaf_curl`)
  8. Tomato Leaf Mold (`leaf_mold`)
  9. Tomato Two-Spotted Spider Mites (`spider_mites`)
- **Evaluation Runner:** `python scripts/evaluate_model.py` (see `docs/MODEL_EVALUATION.md`)

---

## Quick Start & Verification

### 1. Launch Backend API Server
```powershell
cd MultiView-AgriHealth
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`

### 2. Launch Frontend Dev Server
```powershell
cd MultiView-AgriHealth\frontend
npm.cmd run dev
```
Web User Interface: `http://localhost:5173`

### 3. Run Automated Tests (86/86 Passing)
```powershell
cd MultiView-AgriHealth
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 4. Build Frontend for Production
```powershell
cd MultiView-AgriHealth\frontend
npm.cmd run build
```

---

## Documentation Index

- [Production Readiness Audit](file:///docs/PRODUCTION_READINESS_AUDIT.md)
- [Dataset Catalog & Governance](file:///docs/DATASET_CATALOG.md)
- [Model Card](file:///docs/MODEL_CARD.md)
- [Model Evaluation Report](file:///docs/MODEL_EVALUATION.md)
- [REST API Reference](file:///docs/API.md)
- [Deployment Guide](file:///docs/DEPLOYMENT.md)
- [Testing Matrix](file:///docs/TESTING.md)
- [Scientific & Biosecurity Limitations](file:///docs/LIMITATIONS.md)
- [Implementation Status](file:///docs/IMPLEMENTATION_STATUS.md)
