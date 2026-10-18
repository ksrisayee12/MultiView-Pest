# MultiView-AgriHealth — Deployment Guide

**Target Runtime:** Local Workstation, Edge Server, or Cloud VM (Windows PowerShell / Linux Bash)

---

## 1. System Requirements

- **Python:** 3.10 – 3.12 (Isolated `.venv` recommended)
- **Node.js:** v18+ with `npm`
- **Memory (RAM):** 8 GB minimum, 16 GB recommended for multi-video streaming
- **Storage:** 2 GB free disk space (weights file: 6.25 MB)
- **Compute:** CPU-ready (AVX2 supported); optional CUDA-enabled NVIDIA GPU for accelerated inference.

---

## 2. Windows PowerShell Setup (Step-by-Step)

### 2.1 Backend Environment Setup
```powershell
# 1. Navigate to project root
cd MultiView-AgriHealth

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install backend dependencies
pip install -r requirements.txt
pip install ultralytics torch torchvision opencv-python-headless
```

### 2.2 Verify Model Weights Placement
Ensure model artifact exists at:
```text
backend/models/yolo26_tomato.pt
```
Verify the SHA-256 checksum:
```powershell
.\.venv\Scripts\python.exe -c "import hashlib; print(hashlib.sha256(open('backend/models/yolo26_tomato.pt','rb').read()).hexdigest())"
# Expected: ccc05ab88bb550567fb7a7a07c564d37e344e7e69e094da54481ea47c0f134b3
```

### 2.3 Frontend Environment Setup
```powershell
cd frontend
npm.cmd install
npm.cmd run build
```

---

## 3. Launching the Application

### Start Backend API Server (Port 8000)
```powershell
cd MultiView-AgriHealth
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Swagger UI available at: `http://127.0.0.1:8000/docs`

### Start Frontend Dev Server (Port 5173)
```powershell
cd MultiView-AgriHealth\frontend
npm.cmd run dev
```
Open browser to: `http://localhost:5173`

---

## 4. Storage & Persistence Configuration

- **Default (Zero-Config):** In-memory storage with automatic persistence fallback. No external database required for testing or demonstration.
- **MongoDB Atlas / Local MongoDB:** Set the environment variable `MONGODB_URI` in `.env`:
  ```bash
  MONGODB_URI=mongodb://localhost:27017
  DB_NAME=agrihealth_db
  ```
- **Temporary Upload Deletion:** When a plant session is deleted via `DELETE /api/plant-sessions/{id}`, all associated uploaded clips in `data/uploads/{id}/` are automatically wiped.
