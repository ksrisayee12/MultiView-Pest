# MultiView-AgriHealth — Test Matrix & Verification

**Latest Verification Status:** **86/86 Tests Passing (100% Pass Rate)**  
**Test Runner:** `pytest 9.1.1` in isolated `.venv`  

---

## 1. Test Suite Architecture

| Test Module | Test Count | Scope | Key Validations |
|---|---|---|---|
| `tests/test_production_batch_and_provenance.py` | 6 | Production Endpoints | Multi-file batch upload, content hash deduplication, multi-plant strict isolation, report export (Markdown & JSON), SHA-256 provenance |
| `tests/test_app_simulation_e2e.py` | 10 | E2E Integration | Full video upload, streamed decoding, temporal tracking, plant fusion, evaluation suite |
| `tests/test_phases_4_5_6.py` | 40 | Advanced CV Services | Leaf segmentation, within-clip temporal IoU tracking, cross-view consensus, simulation suite, Wilson 95% CI |
| `tests/test_plant_sessions.py` | 5 | Session Lifecycle | Creation, retrieval, simulated multi-view sessions, insufficient coverage threshold |
| `tests/test_video_processor.py` | 6 | Ingestion & Sampling | Upload validation, duplicate frame suppression, streamed OpenCV frame decoding |
| `tests/test_stage1.py` | 5 | Baseline Integrity | Directory structures, config validation, disease mappings |
| `tests/test_stage2_dip.py` | 8 | Image Processing | HSV/LAB color conversion, CLAHE contrast enhancement, noise reduction, aspect ratio resize |
| `tests/test_stage3_pest_detection.py` | 6 | Pest Detector Core | Missing model handling, schema preservation, JSON serialization |

---

## 2. Executing Automated Tests

### Run Full Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected output: `86 passed, 2 warnings in ~18s`

### Run Production Batch & Provenance Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_production_batch_and_provenance.py -v
```

### Run Real Model Evaluation on Held-Out Samples
```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py
```

### Run End-to-End Pipeline Demo
```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py
```

### Validate Frontend Production Build
```powershell
cd frontend
npm.cmd run build
```
Expected output: `✓ built in ~20s with 0 errors`.
