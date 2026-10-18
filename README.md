# MultiView-AgriHealth: Agricultural Plant Health Assessment System

A modular, dual-evidence AI & Digital Image Processing (DIP) agricultural monitoring application
for single-plant multi-view health assessment, disease detection, damage area estimation, and management recommendations.

> **Academic & Precision Agriculture Prototype** — Fully operational multi-view video pipeline, temporal leaf tracking, cross-view consensus fusion, and deterministic simulation suite. Deep learning model slots operate cleanly: when weights are not present, the system runs with explicit `MODEL_UNAVAILABLE` or `HEURISTIC` statuses without inventing predictions.

---

## Capabilities & Workflows

### 1. Multi-View Plant Analysis Workflow (Primary Mode)
Analyzes one individual plant across multiple camera viewpoints (`front`, `back`, `left`, `right`, `top`, `underside`, `arbitrary`):
- **Streamed Video Ingestion:** Decodes uploaded video clips on the fly with OpenCV `VideoCapture` (never loads full video into memory).
- **Configurable Frame Sampling:** Extracts representative frames with near-duplicate suppression.
- **Leaf-Region Lesion Estimation:** Restricts visible symptom measurement strictly to detected foliage masks (via YOLO segmentation or HSV fallback), preventing background/soil false positives.
- **Within-Clip Temporal Tracking:** Associates leaf bounding boxes across adjacent frames using IoU/centroid matching, suppressing duplicate leaf counts.
- **Cross-View Evidence Association:** Merges multi-angle visual evidence, detects conflicting observations, bounds maximum visible lesion area, and records unobserved viewpoints as `UNKNOWN`.
- **Honest Status Taxonomy:**
  - `HEALTHY_APPEARING` — Sufficient viewpoints with no symptoms detected above threshold.
  - `SYMPTOMS_DETECTED` — Credible disease/symptom evidence detected in one or more viewpoints.
  - `UNCERTAIN` — Conflicting evidence between views or models.
  - `INSUFFICIENT_COVERAGE` — Fewer than required viewpoints observed.
  - `PROCESSING_ERROR` / `MODEL_UNAVAILABLE` — Explicit system states.

### 2. Legacy Field-Tile Composite Workflow (Preserved Mode)
Retained for spatial drone composite scans:
- Splits a field composite into a 4×4 grid of 16 spatial tiles.
- Runs dual-evidence fusion combining YOLO predictions with OpenCV color/lesion metrics.
- Computes a prototype Field Plant Health Index (0–100).

---

## Project Structure

```
MultiView-AgriHealth/
├── backend/
│   ├── app/
│   │   ├── api/              health.py, analyze.py, scans.py, analytics.py, simulation.py,
│   │   │                     models_status.py, plant_sessions.py (Phase 2-6 endpoints)
│   │   ├── config/           settings.py (env-configurable thresholds)
│   │   ├── schemas/          plant_session.py (Pydantic request/response models)
│   │   ├── services/         model_registry.py (singleton model loader)
│   │   │                     video_processor.py (streamed video decoder & sampler)
│   │   │                     leaf_analysis.py (YOLO segmentation + HSV heuristic fallback)
│   │   │                     temporal_tracker.py (within-clip IoU/centroid leaf tracking)
│   │   │                     cross_view_matcher.py (multi-angle evidence fusion & consensus)
│   │   │                     plant_fusion.py (plant-level assessment & coverage engine)
│   │   │                     simulation_fixtures.py (deterministic multi-view fixtures)
│   │   │                     evaluation.py (metrics, Wilson 95% CIs, calibration reports)
│   │   └── main.py           FastAPI entrypoint with lifespan startup & CORS
│   ├── database/             mongodb.py (MongoDB + in-memory fallback)
│   ├── fusion/               fusion_engine.py (legacy 16-tile fusion rules)
│   ├── image_processing/     color_analysis.py, lesion_detection.py, visual_health.py
│   ├── models/               Place trained .pt weights here (gitignored)
│   └── recommendations/      recommendations.py
├── configs/                  YAML configuration files for classes, thresholds, datasets
├── data/
│   ├── raw/                  Dataset directories (disease, pest, multiview, drone)
│   ├── processed/            Generated output images
│   └── samples/              Sample leaf images for testing
├── docs/
│   └── IMPLEMENTATION_STATUS.md   Phase-by-phase implementation report
├── evaluation/               evaluate_all.py (stressed evaluation runner)
├── frontend/                 React 18 + TypeScript + Vite + Tailwind CSS
│   ├── src/
│   │   ├── components/       Header.tsx
│   │   ├── pages/            MultiViewPage.tsx (Phase 3 UI), DashboardPage.tsx,
│   │   │                     FieldScanPage.tsx, TileAnalysisPage.tsx, ScanHistoryPage.tsx
│   │   ├── types.ts          Full TypeScript contracts for multi-view & legacy APIs
│   │   └── main.tsx          Application routing & navigation
├── notebooks/                YOLO26_PlantVillage_Training.ipynb (reproducible training)
├── scripts/                  setup_venv.ps1, run_backend.ps1, run_tests.ps1, run_demo.py
├── tests/                    test_stage1.py, test_stage2_dip.py, test_stage3_pest_detection.py,
│                             test_video_processor.py, test_plant_sessions.py, test_phases_4_5_6.py
├── .env.example              Configuration keys
├── requirements.txt          Runtime dependencies
└── START_HERE.md             Step-by-step setup guide
```

---

## Quick Start

### 1. Backend Setup

**Windows PowerShell:**
```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

**macOS/Linux:**
```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
python -m uvicorn backend.app.main:app --reload --port 8000
```

- **Backend:** http://localhost:8000
- **Interactive OpenAPI Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health
- **Model Readiness:** http://localhost:8000/api/models/status

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

- **Frontend App:** http://localhost:5173

### 3. Run All Tests (70/70 Passing)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 4. Run End-to-End Demo Script

```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py
```

### 5. Frontend Production Build

```powershell
cd frontend
npm.cmd run build
```

---

## API Endpoints Reference

### Plant-Session & Multi-View APIs (Phase 2–6)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/plant-sessions` | Create a plant analysis session with stable `plant_id` |
| `GET` | `/api/plant-sessions` | List recent plant sessions |
| `GET` | `/api/plant-sessions/{session_id}` | Retrieve session metadata and clips |
| `DELETE` | `/api/plant-sessions/{session_id}` | Delete session and clean up temporary assets |
| `POST` | `/api/plant-sessions/{session_id}/videos` | Upload video clip with viewpoint metadata |
| `POST` | `/api/plant-sessions/{session_id}/analyze` | Start asynchronous multi-view video analysis |
| `GET` | `/api/plant-sessions/{session_id}/status` | Check background analysis progress and stage |
| `GET` | `/api/plant-sessions/{session_id}/assessment` | Retrieve finalized fused plant assessment |
| `GET` | `/api/plant-sessions/{session_id}/observations` | Inspect all frame observations across clips |
| `POST` | `/api/plant-sessions/simulation` | Generate deterministic synthetic 4-view session |
| `GET` | `/api/plant-sessions/leaf-segmentation-status` | Check leaf segmentation model availability |
| `POST` | `/api/plant-sessions/{session_id}/leaf-analysis` | Run mask-aware leaf quality & lesion analysis |
| `GET` | `/api/plant-sessions/{session_id}/temporal-tracks` | Get within-clip temporal leaf tracks |
| `GET` | `/api/plant-sessions/{session_id}/cross-view-evidence` | Get multi-angle cross-view evidence report |
| `POST` | `/api/plant-sessions/run-simulation-suite` | Run all 5 deterministic simulation scenarios |
| `POST` | `/api/plant-sessions/evaluate` | Evaluate predictions against ground truth (Wilson 95% CI) |

### System & Legacy APIs
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status |
| `GET` | `/api/models/status` | Model readiness and loaded version information |
| `POST` | `/api/analyze/field` | Legacy 16-tile composite field analysis |
| `POST` | `/api/analyze/tile` | Single-tile YOLO + DIP analysis |
| `POST` | `/api/analyze/closeup` | High-resolution close-up tile resolution |
| `GET` | `/api/scans` | Historical scan records |
| `GET` | `/api/analytics` | Aggregate scan statistics |

---

## Model Readiness & Training Instructions

No model weights are committed to Git. Check status at `GET /api/models/status`:

| Model Slot | Task | When Weights Missing |
|---|---|---|
| Disease Classifier | Classification (e.g. YOLOv8-cls) | Reports `MISSING_WEIGHTS`, skips ML prediction, relies on heuristic DIP |
| Leaf Segmenter | Instance Segmentation (e.g. YOLOv8-seg) | Reports `MODEL_MISSING`, falls back to HSV foliage mask |
| Pest Detector | Object Detection (e.g. YOLOv8) | Reports `MISSING_WEIGHTS`, handles missing model cleanly |

### Training Disease Classification Weights:
1. Place the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) into `data/raw/disease/`.
2. Open and run `notebooks/YOLO26_PlantVillage_Training.ipynb`.
3. Save the trained weights to `backend/models/yolo26_tomato.pt`.
4. Restart the backend — model status will switch to `READY`.

---

## Verification & Acceptance

- **Backend Pytest Suite:** 70 passed in 2.26s.
- **Frontend Build:** `tsc && vite build` bundled successfully in 11.48s.
- **Demo Runner:** `python scripts/run_demo.py` verified field tile fusion, stressed evaluation, multi-view session fusion, and deterministic simulation fixtures.
- **Detailed Audit & Implementation Logs:** See [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).
