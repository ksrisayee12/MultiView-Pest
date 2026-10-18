# START HERE — MultiView-AgriHealth Setup Guide

> Comprehensive setup and verification guide for the MultiView-AgriHealth precision agriculture system.

---

## Step 1: Verify Prerequisites

- [ ] Python 3.10, 3.11, or 3.12 installed → `python --version`
- [ ] Node.js 18+ installed → `node --version`
- [ ] Git installed → `git --version`

---

## Step 2: Create the Python Virtual Environment

**Windows PowerShell (from project root):**
```powershell
.\scripts\setup_venv.ps1
```

This creates `.venv/` in the project root and installs dependencies (`fastapi`, `uvicorn`, `opencv-python-headless`, `numpy`, `pillow`, `pyyaml`, `python-multipart`, `pymongo`, `requests`, `ultralytics`, `pytest`).

---

## Step 3: Start the Backend

```powershell
.\scripts\run_backend.ps1
```

Or without script activation:
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

✅ Expected: `Uvicorn running on http://0.0.0.0:8000`

Open in browser:
- **API Docs (Swagger):** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health
- **Model Readiness Status:** http://localhost:8000/api/models/status
- **Leaf Segmentation Model Status:** http://localhost:8000/api/plant-sessions/leaf-segmentation-status

---

## Step 4: Start the Frontend

In a separate terminal:
```powershell
cd frontend
npm install
npm run dev
```

✅ Expected: Vite dev server running at http://localhost:5173

Key Navigation Tabs:
- **Dashboard:** System metrics and high-level health overview.
- **Multi-View Plant:** Dedicated single-plant analysis page supporting multi-video uploads tagged with viewpoints (front, back, left, right, top, underside), video frame extraction, within-clip tracking, cross-view association, and one-click quick demo simulation.
- **Field Scan / Tile Analysis:** Legacy 4×4 spatial image composite analysis.
- **Scan History:** Historical records stored in MongoDB (or in-memory fallback).

---

## Step 5: Run the Full Test Suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

✅ Expected: **70/70 tests pass** across all stages:
- `tests/test_stage1.py`: Baseline configuration, dataset validator, mappings
- `tests/test_stage2_dip.py`: Digital Image Processing (DIP) pipeline, HSV thresholds, noise filtering
- `tests/test_stage3_pest_detection.py`: Pest detector schemas, class maps, missing-model handling
- `tests/test_video_processor.py`: Video upload validation, streamed frame sampling, duplicate suppression
- `tests/test_plant_sessions.py`: Plant sessions, multi-clip ingestion, coverage tracking, fusion rules
- `tests/test_phases_4_5_6.py`: Leaf segmentation (YOLO/HSV fallback), temporal tracker, cross-view matcher, evaluation framework, simulation suite

---

## Step 6: Run the Comprehensive Demo Pipeline

```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py
```

This verifies:
1. 4×4 Field composite simulation & 16-tile dual-evidence fusion
2. Stressed evaluation pipeline
3. Multi-view same-plant evidence fusion
4. Phase 4–6 deterministic simulation & cross-view evaluation suite (scenario scoring, pass rates, coverage metrics)

---

## Step 7: Frontend Production Build

```powershell
cd frontend
npm.cmd run build
```

✅ Expected: TypeScript compilation and Vite production bundle generated in `frontend/dist/` with 0 errors.

---

## Current Architecture & Design Philosophy

| Component | Status | Behavior When Model Weights Absent |
|---|---|---|
| Disease Classifier | Plug-in slot | Honest `MODEL_UNAVAILABLE` status; uses DIP heuristic visual analysis without falsifying disease labels |
| Leaf Segmentation | Modular YOLO / HSV | Uses conservative HSV-based foliage mask (`HEURISTIC`), marks availability explicitly |
| Lesion Estimation | Mask-aware | Restricted strictly to detected leaf pixels — never estimated across entire video frame |
| Temporal Tracker | Active | Within-clip IoU/centroid tracker; suppresses counting repeated frames as unique leaves |
| Cross-View Matcher | Active | Multi-angle consensus, upper-bound lesion propagation, explicit unobserved viewpoint tracking |
| Simulation Suite | Active | 5 deterministic multi-view test fixtures with ground truth scoring and Wilson 95% CIs |
| Persistence | Active | MongoDB with automatic in-memory fallback for zero-dependency development |

---

## To Train Custom YOLO Disease Weights

1. Place [PlantVillage tomato dataset](https://github.com/spMohanty/PlantVillage-Dataset) inside `data/raw/disease/`.
2. Open `notebooks/YOLO26_PlantVillage_Training.ipynb`.
3. Train YOLO classification model (e.g. `yolov8n-cls.pt` or compatible checkpoint).
4. Save the trained checkpoint to `backend/models/yolo26_tomato.pt`.
5. Restart backend — model readiness endpoint `GET /api/models/status` will report `READY`.
