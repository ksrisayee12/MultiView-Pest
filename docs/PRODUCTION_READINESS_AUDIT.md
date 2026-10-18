# MultiView-AgriHealth — Production Readiness Audit Report

**Date:** 2026-10-18  
**Auditor:** Antigravity Production Lead  
**Scope:** Evaluation of MultiView-AgriHealth baseline post Phase 7 against the Production Completion Master Specification.  
**Baseline Test Status:** **80 passed / 80 tests in 20.08s (100% pass rate)**  

---

## 1. Executive Summary

An exhaustive audit of the MultiView-AgriHealth codebase was performed covering backend services, real model artifacts, dataset governance, frontend architecture, and test coverage.

The application possesses a robust core foundation:
- Real ML model weights (`backend/models/yolo26_tomato.pt`, 6.2 MB) are loaded and active.
- Temporal leaf tracking and mask-aware lesion estimation prevent counting duplicate video frames as distinct leaves.
- Multi-angle viewpoint consensus and unobserved view handling are implemented.
- End-to-end simulated and real media workflows pass 80 automated unit and integration tests.

However, several critical gaps must be resolved to meet the production standard:
1. **Media Ingestion Flexibility:** Batch multi-image/multi-video upload and multi-plant sample isolation need explicit endpoints and queue UX.
2. **Inference Grounding Integrity:** Line 188 of `model_registry.py` returns hardcoded `prediction: "healthy", confidence: 0.85` when no disease bounding boxes are detected, violating the strict "no fabricated predictions" and "distinguish not-detected from healthy" rules.
3. **Model Provenance & Dataset Governance:** Missing SHA-256 hash tracking in inference metadata, dataset catalogs (`docs/DATASET_CATALOG.md`), model cards (`docs/MODEL_CARD.md`), and held-out validation reports (`docs/MODEL_EVALUATION.md`).
4. **Export & Reporting:** PDF/Markdown analysis export with explicit provenance, disclaimers, and view coverage is missing.

---

## 2. Feature-by-Feature Status Matrix

| ID | Feature / Component | Status | File Path(s) | Evidence & Observations |
|---|---|---|---|---|
| **F-01** | **Single Image Upload & Analysis** | `VERIFIED` | `backend/app/services/video_processor.py`, `backend/app/api/plant_sessions.py` | Image decoded via `cv2.imread()`, wrapped as 1-frame observation, processed through DIP & YOLO. Passes `test_app_simulation_e2e.py`. |
| **F-02** | **Multiple Images Upload** | `PARTIAL` | `backend/app/api/plant_sessions.py` | Can upload multiple images sequentially to a session via `/videos`. Lacks a bulk multi-file upload endpoint (`/media/batch`). |
| **F-03** | **Single Video Upload & Frame Sampling** | `VERIFIED` | `backend/app/services/video_processor.py` | Streams video via OpenCV `VideoCapture`, samples at configurable FPS (default 2.0), suppresses duplicate frames. Passes 6 tests. |
| **F-04** | **Multiple Videos Upload** | `VERIFIED` | `backend/app/api/plant_sessions.py` | Multiple video clips can be uploaded per session with distinct view labels (`front`, `back`, `left`, `right`, `top`, `underside`). |
| **F-05** | **Mixed Media Submissions (Images + Videos)** | `PARTIAL` | `backend/app/api/plant_sessions.py` | Backend processes both types in same session, but UI does not provide a unified drag-and-drop mixed queue with file-type indicators. |
| **F-06** | **Multi-Plant Batch Mode with Isolation** | `MISSING` | `backend/app/api/plant_sessions.py`, `frontend/src/pages/MultiViewPage.tsx` | Sessions are strictly 1:1 with `plant_id`. No multi-plant batch submission endpoint or sample-boundary isolation UI. |
| **F-07** | **Real Model Loading & Inference** | `VERIFIED` | `backend/app/services/model_registry.py`, `backend/models/yolo26_tomato.pt` | Singleton `_YOLO` loader, fine-tuned YOLOv8n tomato disease detector (9 classes) active on CPU/GPU. Passes inference tests. |
| **F-08** | **Strict Grounded Inference (No Fabricated Output)** | `BROKEN` | `backend/app/services/model_registry.py` (L188-194) | When no disease bounding boxes are detected, code returns hardcoded `healthy` with `0.85` confidence instead of `no_supported_condition_detected`. |
| **F-09** | **Model Provenance Tracking** | `PARTIAL` | `backend/app/services/model_registry.py` | Model status reported (`READY`), but artifact SHA-256 hash, model version string, and inference timestamp are not attached to responses. |
| **F-10** | **Temporal Within-Clip Tracking** | `VERIFIED` | `backend/app/services/temporal_tracker.py` | Centroid/IoU tracking with expiry ensures duplicate consecutive video frames do not inflate leaf count. Passes 8 tests. |
| **F-11** | **Mask-Aware Lesion Estimation** | `VERIFIED` | `backend/app/services/leaf_analysis.py` | Restricts lesion calculation strictly to segmented leaf foliage mask. Reports as heuristic visible estimate. Passes 7 tests. |
| **F-12** | **Cross-View Evidence Association** | `VERIFIED` | `backend/app/services/cross_view_matcher.py` | Evaluates multi-angle consensus, surfaces view conflicts, tracks unobserved angles as `UNKNOWN`. Passes 8 tests. |
| **F-13** | **Plant-Level Evidence Fusion** | `VERIFIED` | `backend/app/services/plant_fusion.py` | Evaluates view coverage, disease frame fraction, conservative thresholds. Leaves unobserved views as unknown. Passes 5 tests. |
| **F-14** | **Simulation Suite & Statistical Evaluation** | `VERIFIED` | `backend/app/services/simulation_fixtures.py`, `backend/app/services/evaluation.py` | 5 deterministic fixtures, Wilson 95% CI, calibration curves. Passes 18 tests. |
| **F-15** | **Session Persistence & Cleanup** | `VERIFIED` | `backend/database/mongodb.py`, `backend/app/api/plant_sessions.py` | MongoDB persistence with automatic in-memory fallback. Deletion endpoint removes files and session records safely. |
| **F-16** | **Frontend MultiView UI** | `VERIFIED` | `frontend/src/pages/MultiViewPage.tsx` | Full interactive UI with sample pickers, media preview, status badges, view coverage matrix, and assessment cards. |
| **F-17** | **Report Export & Provenance Summary** | `MISSING` | `backend/app/services/report_generator.py`, frontend export | No JSON/Markdown/PDF downloadable report summarizing provenance, evidence, and limitations. |
| **F-18** | **Dataset Catalog & Manifests** | `MISSING` | `data/manifests/`, `docs/DATASET_CATALOG.md` | Candidate datasets not formally cataloged with licenses, split files, and SHA-256 checksums. |
| **F-19** | **Model Card & Evaluation Documentation** | `MISSING` | `docs/MODEL_CARD.md`, `docs/MODEL_EVALUATION.md` | Formal model card describing scope, limitations, license, and evaluated benchmark metrics on held-out data. |

---

## 3. Real Model and Dataset Status

### 3.1 Loaded Model Artifact
- **Artifact Path:** `backend/models/yolo26_tomato.pt`
- **File Size:** 6,248,345 bytes (~6.2 MB)
- **Base Architecture:** YOLOv8n (detection)
- **Provenance:** `NarimanOsman2025/tomato-disease-detector-yolov8n` (Hugging Face)
- **Supported Conditions (9 classes):**
  1. Tomato Early Blight (`early_blight`)
  2. Tomato Late Blight (`late_blight`)
  3. Tomato Septoria Leaf Spot (`septoria_leaf_spot`)
  4. Tomato Bacterial Spot (`bacterial_spot`)
  5. Tomato Yellow Leaf Curl Virus (`yellow_leaf_curl`)
  6. Tomato Mosaic Virus (`mosaic_virus`)
  7. Tomato Leaf Mold (`leaf_mold`)
  8. Tomato Two-Spotted Spider Mites (`spider_mites`)
  9. Tomato Healthy (`healthy`)
- **Execution Mode:** CPU verified (GPU automatic fallback via PyTorch/CUDA).
- **Inference Verification:** Confirmed active and tested on real sample leaf images (`tomato_healthy.jpg`, `tomato_early_blight.jpg`, etc.).

### 3.2 Dataset Status
- Candidate datasets (PlantVillage, PlantDoc, IP102) documented in configs (`configs/dataset_sources.yaml`).
- Formal dataset catalog (`docs/DATASET_CATALOG.md`) and immutable manifests (`data/manifests/`) are not yet constructed.
- Sample dataset for local smoke testing exists in `data/samples/`.

---

## 4. Security, Privacy, and Operational Risks

1. **File Upload Security:**
   - Extensions and MIME types are validated against allowed sets.
   - Max file size is capped at 500 MB per clip.
   - Path traversal risk mitigated: Files are stored in `data/uploads/{session_id}/{clip_id}_{safe_filename}` with sanitized names.
   - Content hash deduplication at upload time is recommended to prevent repeated storage of identical media.
2. **Persistence & Data Retention:**
   - Single-user local development defaults to in-memory fallback if MongoDB is not reachable.
   - Session deletion cleans up physical files in `data/uploads/{session_id}`.
3. **Error Leakage:**
   - HTTP 400/404/413/415 errors return clean structured details without raw Python stack traces.
4. **Performance & Memory Management:**
   - Video decoding streams frame-by-frame; full videos are not loaded into memory.
   - Duplicate frame diffing suppresses redundant frames before deep learning inference.
   - Inference runs on a bounded `ThreadPoolExecutor(max_workers=2)` to prevent resource exhaustion.

---

## 5. Prioritized Implementation Plan

To bring the application to genuine production readiness, the work is organized into the following phases:

### Phase 1: Grounded Inference & Provenance Hardening
- Fix `model_registry.py` to return `no_supported_condition_detected` (with 0.0 confidence) instead of hardcoding `healthy (0.85)` when no boxes are detected.
- Add artifact SHA-256 checksum computation and include model version, hash, and inference timestamp in all predictions and assessments.
- Verify that `plant_fusion.py` handles `no_supported_condition_detected` appropriately without assuming health.

### Phase 2: Dynamic Media Ingestion & Multi-Plant Batch Mode
- Implement `POST /api/plant-sessions/{session_id}/media/batch` accepting multiple images and videos simultaneously with optional view labels.
- Implement `POST /api/plant-sessions/batch` for multi-plant submissions with explicit sample boundaries, ensuring evidence from plant A is never conflated with plant B.
- Add content-hash (SHA-256) upload deduplication.
- Enhance frontend `MultiViewPage.tsx` with a multi-file media queue, drag-and-drop support, per-file view tagger, and multi-plant batch selector.

### Phase 3: Dataset Catalog, Manifests & Model Card
- Create `docs/DATASET_CATALOG.md` detailing PlantVillage, PlantDoc, IP102, licenses, sample counts, and inclusion/exclusion rationale.
- Create `data/manifests/` with validated manifest templates.
- Create `docs/MODEL_CARD.md` with model scope, supported classes, known failure modes, and hardware benchmarks.
- Create reproducible evaluation script `scripts/evaluate_model.py` and write `docs/MODEL_EVALUATION.md`.

### Phase 4: Report Export & Enterprise API Extensions
- Implement report generation service (`backend/app/services/report_generator.py`) supporting structured JSON and Markdown/PDF export with full provenance and limitations.
- Add endpoint `GET /api/plant-sessions/{session_id}/report` for direct download.
- Add report download button in frontend Results view.

### Phase 5: Comprehensive Verification & Release Package
- Add test coverage for batch upload, multi-plant isolation, grounded inference, report export, and provenance.
- Ensure all 80+ tests pass with zero failures.
- Update `README.md`, `docs/DEPLOYMENT.md`, `docs/API.md`, `docs/TESTING.md`, `docs/LIMITATIONS.md`, and `docs/IMPLEMENTATION_STATUS.md`.
- Verify frontend production build (`npm.cmd run build`).

---

## 6. External Blockers

None. The target environment possesses active Python 3.12 with `.venv`, PyTorch, TorchVision, Ultralytics, OpenCV, Node.js, and running local servers.
