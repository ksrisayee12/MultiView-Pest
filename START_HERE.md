# START HERE: Beginner Project Guide

Welcome to your academic capstone project: **Multi-View Drone-Based Agricultural Pest and Plant Disease Detection System**.

---

## 🚀 Quick Start Instructions (Windows PowerShell)

### STEP 1: Activate Python Environment
Open PowerShell in VS Code and verify the environment:
```powershell
& "C:\Users\Senthil kumar\anaconda3\envs\agrivision\python.exe" --version
```

### STEP 2: Start the FastAPI Backend Server
Run the backend application:
```powershell
& "C:\Users\Senthil kumar\anaconda3\envs\agrivision\python.exe" -m uvicorn backend.app.main:app --reload --port 8000
```
- Interactive API Documentation: Open `http://localhost:8000/docs` in your browser.
- Health Check: `http://localhost:8000/api/health`

### STEP 3: Run the Pipeline Verification Script
Test simulation, DIP processing, fusion engine, and evaluation in one command:
```powershell
& "C:\Users\Senthil kumar\anaconda3\envs\agrivision\python.exe" scripts/run_demo.py
```

---

## 🔬 Google Colab Model Training (YOLO26)

Because deep-learning training requires a GPU, the YOLO26 training pipeline is prepared in:
`notebooks/YOLO26_PlantVillage_Training.ipynb`

1. Upload `notebooks/YOLO26_PlantVillage_Training.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select Runtime $\rightarrow$ Change runtime type $\rightarrow$ **T4 GPU**.
3. Run all cells.
4. Download the generated `yolo26_tomato.pt` model weights.
5. Place `yolo26_tomato.pt` inside `backend/models/`.

---

## 📊 Dual-Evidence Fusion Architecture

- **Method A (YOLO26 DL Classifier)**: Predicts disease class (`healthy`, `early_blight`, `late_blight`, `septoria_leaf_spot`).
- **Method B (OpenCV DIP Engine)**: Calculates Greenness Index ($G / (R+G+B+\epsilon)$) and Lesion Area Percentage ($\text{Normal} / \text{Abnormal}$).
- **Fusion Decision**:
  - `YOLO Disease` + `OpenCV Abnormal` $\rightarrow$ **CONFIRMED DISEASE**
  - `YOLO Healthy` + `OpenCV Normal` $\rightarrow$ **HEALTHY**
  - Disagreement $\rightarrow$ **DISPUTED** (Prompts Close-up Leaf Image Verification)

---

## 📌 Simulated Multi-View Disclaimer
Drone field view is simulated using a 4x4 composite of 16 leaf images because real drone field imagery is not available for this prototype.
