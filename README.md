# Multi-View Drone-Based Agricultural Pest and Plant Disease Detection System

A modular, dual-evidence AI & Digital Image Processing (DIP) agricultural monitoring application designed for plant health assessment, disease detection, damage area estimation, severity classification, multi-view verification, and management recommendations.

---

## 📌 Core Research Architecture

### Dual-Evidence Fusion Engine
- **Method A (YOLO26 DL Classifier)**: Classifies leaf tiles into 4 target tomato classes (`healthy`, `early_blight`, `late_blight`, `septoria_leaf_spot`).
- **Method B (OpenCV DIP Engine)**: Computes Greenness Index ($G / (R+G+B+\epsilon)$), mean channel values, green pixel percentage, and estimated lesion area percentage. Classifies visual status strictly as `NORMAL` or `ABNORMAL`.
- **Fusion Decision Matrix**:
  - `YOLO Disease` + `OpenCV Abnormal` $\rightarrow$ **CONFIRMED DISEASE**
  - `YOLO Healthy` + `OpenCV Normal` $\rightarrow$ **HEALTHY**
  - Disagreement $\rightarrow$ **DISPUTED** (Requests Close-up Image Upload)

### Prototype Plant Health Index ($0 - 100$)
$$H_{\text{index}} = \max\left(0, \min\left(100, (N_{\text{healthy}} \times 6.25) + (N_{\text{disputed}} \times 3.0) - P_{\text{lesion}}\right)\right)$$

---

## 🏗️ Project Structure

```text
MultiView_AgriHealth/
├── backend/
│   ├── app/
│   │   ├── api/                # FastAPI Routers (health, analyze, scans, analytics, simulation)
│   │   ├── config/             # Configuration & severity thresholds
│   │   └── main.py             # FastAPI entrypoint
│   ├── database/               # MongoDB Atlas manager with in-memory fallback
│   ├── fusion/                 # Dual-evidence fusion engine
│   ├── image_processing/       # OpenCV Greenness Index & Lesion area segmentation
│   ├── recommendations/        # Controlled tomato treatment lookup table
│   └── models/                 # Model weights directory (yolo26_tomato.pt)
├── frontend/                   # Vite + React + TypeScript + Tailwind CSS (Government Blue Theme)
├── simulation/                 # 4x4 composite field simulator & 16-tile extractor
├── training/                   # YOLO26 training pipeline configurations
├── notebooks/                  # Google Colab GPU training notebook (YOLO26_PlantVillage_Training.ipynb)
├── evaluation/                 # Stressed image evaluation runner (Clean, Blurred, Darkened, Noisy)
├── data/                       # Processed images & field composites
├── results/                    # Metric evaluation logs
├── scripts/                    # Beginner runner scripts (run_demo.py, create_simulated_field.py)
├── START_HERE.md               # Beginner step-by-step instructions
└── README.md
```

---

## 📚 Dataset References

1. **PlantVillage Dataset**:
   - Source: https://github.com/spMohanty/PlantVillage-Dataset
   - Selected Tomato Subset: `Tomato___healthy`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___Septoria_leaf_spot`.
2. **PlantDoc Dataset**:
   - Source: https://github.com/pratikkayal/PlantDoc-Dataset
3. **IP102 Benchmark**:
   - Source: https://github.com/xpwu95/IP102

---

## 💻 Local Running Instructions

### Backend Startup:
```powershell
& "C:\Users\Senthil kumar\anaconda3\envs\agrivision\python.exe" -m uvicorn backend.app.main:app --reload --port 8000
```

### Run Demo Verification Script:
```powershell
& "C:\Users\Senthil kumar\anaconda3\envs\agrivision\python.exe" scripts/run_demo.py
```

---

## ⚠️ Academic Disclaimers

1. **Simulated Drone Imagery**: Wide-field drone view is simulated using a 4x4 composite of 16 leaf images because real drone field imagery is not available for this prototype.
2. **Recommendation Safety**: Treatment recommendations are for educational guidance only. Always follow product label instructions and local agricultural extension guidance.
