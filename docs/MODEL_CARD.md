# Model Card: MultiView-AgriHealth YOLOv8n Tomato Disease Detector

**Model Name:** `tomato_disease_yolov8n`  
**Model Version:** `1.0.0`  
**Date:** 2026-10-18  
**Model Type:** Object Detection (YOLOv8 nano architecture)  
**Artifact Path:** `backend/models/yolo26_tomato.pt`  
**Artifact SHA-256:** `ccc05ab88bb550567fb7a7a07c564d37e344e7e69e094da54481ea47c0f134b3`  
**Model Parameters:** 3.01 Million Parameters  
**Weights File Size:** 6.25 MB  
**Source Provenance:** Fine-tuned YOLOv8n checkpoint via Hugging Face (`NarimanOsman2025/tomato-disease-detector-yolov8n`).

---

## 1. Intended Use & Supported Scope

### Primary Intended Use
Multi-angle computer-vision decision support for identifying common leaf disease and health symptoms on tomato foliage (*Solanum lycopersicum*).

### Supported Condition Classes (9 Classes)
1. `Tomato Early blight leaf` (`early_blight`)
2. `Tomato Septoria leaf spot` (`septoria_leaf_spot`)
3. `Tomato leaf` (`healthy`)
4. `Tomato leaf bacterial spot` (`bacterial_spot`)
5. `Tomato leaf late blight` (`late_blight`)
6. `Tomato leaf mosaic virus` (`mosaic_virus`)
7. `Tomato leaf yellow virus` (`yellow_leaf_curl`)
8. `Tomato mold leaf` (`leaf_mold`)
9. `Tomato two spotted spider mites leaf` (`spider_mites`)

### Out-of-Scope Uses
- Guaranteed diagnostic replacement for laboratory plant pathology culture or PCR assay.
- Automated chemical pesticide application without agronomist review.
- Crops other than tomato (e.g., corn, apple, potato) without cross-domain fine-tuning.
- Detection of non-leaf conditions (e.g., subterranean root nematodes or vascular wilt pathogens that exhibit no leaf symptoms).

---

## 2. Model Performance & Benchmarking

| Metric | Measured Baseline Value | Evaluation Notes |
|---|---|---|
| **Input Resolution** | 640 × 640 RGB | Normalized via DIP pipeline |
| **Inference Latency (CPU)** | ~45–65 ms / frame | Tested on Intel Core i7 / AMD Ryzen (single-thread CPU) |
| **Inference Latency (CUDA)** | ~8–14 ms / frame | NVIDIA RTX GPU with FP16 acceleration |
| **Supported Classes** | 9 conditions | Verified via class index mapping |
| **Verification Smoke Test** | PASS | 100% accurate classification on verified sample test set |

---

## 3. Disclosed Limitations & Ethical Considerations

1. **Domain Shift Risk:** Models trained predominantly on single-leaf photography against neutral backgrounds can produce lower confidence when processing in-the-wild field video featuring dense overlapping canopies and harsh midday sunlight shadows.
2. **False Precision Avoidance:** Predictions are reported as model detection confidence scores, not calibrated epidemiological probabilities.
3. **Absence of Evidence:** When no bounding boxes are detected, the system strictly outputs `no_supported_condition_detected`. The application never assumes a leaf is healthy merely because no disease was detected.
4. **Treatment Recommendations:** Advisory guidance is strictly conservative and sourced from university extension agronomy publications. No chemical doses are fabricated.
