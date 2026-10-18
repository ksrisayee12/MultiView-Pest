# MultiView-AgriHealth — REST API Specification

**Base URL:** `http://127.0.0.1:8000/api`  
**Protocol:** HTTP/1.1 REST  
**Authentication:** Local single-user (Enterprise token auth ready via reverse proxy / API gateway)

---

## 1. Plant Sessions & Media Ingestion

### `POST /plant-sessions`
Create a new plant health analysis session.
- **Request Body (JSON):**
  ```json
  {
    "plant_id": "PLANT-001",
    "crop_type": "tomato",
    "notes": "Field Plot 4, row 12"
  }
  ```
- **Response (`201 Created`):**
  ```json
  {
    "session_id": "sess_89f31ab2c01d",
    "plant_id": "PLANT-001",
    "crop_type": "tomato",
    "status": "CREATED",
    "clips": [],
    "created_at": 1729253400.0
  }
  ```

---

### `POST /plant-sessions/{session_id}/videos`
Upload a single video clip or leaf photograph with viewpoint metadata.
- **Form Data:**
  - `file`: Media binary (`.mp4`, `.avi`, `.mov`, `.jpg`, `.png`, `.webp`)
  - `view_label`: View angle (`front`, `back`, `left`, `right`, `top`, `underside`, `unknown`, `arbitrary`)
- **Response (`201 Created`):**
  ```json
  {
    "message": "Video clip uploaded successfully for view 'front'.",
    "clip": {
      "clip_id": "clip_4a9b1c2d",
      "session_id": "sess_89f31ab2c01d",
      "original_filename": "leaf_front.jpg",
      "view_label": "front",
      "upload_status": "UPLOADED",
      "processing_status": "PENDING"
    }
  }
  ```

---

### `POST /plant-sessions/{session_id}/media/batch`
Upload multiple images and/or videos in a single batch request to an existing session.
- **Form Data:**
  - `files`: Multiple media binaries (`multipart/form-data`)
  - `metadata`: Optional JSON mapping filenames to view labels, e.g.:
    ```json
    {"leaf_front.jpg": "front", "leaf_back.jpg": "back"}
    ```
- **Features:**
  - Content SHA-256 deduplication: duplicate files already uploaded to this session are skipped automatically.
- **Response (`201 Created`):**
  ```json
  {
    "message": "Processed 2 files: 2 added, 0 duplicates skipped, 0 rejected.",
    "added_count": 2,
    "added_clips": [...],
    "skipped_duplicates": []
  }
  ```

---

### `POST /plant-sessions/batch`
Multi-Plant Batch Ingestion Mode with Strict Boundary Isolation.
- **Form Data:**
  - `files`: Multiple media binaries
  - `batch_manifest`: JSON mapping filenames to plant targets:
    ```json
    {
      "photo1.jpg": {"plant_id": "PLANT-001", "view_label": "front", "crop_type": "tomato"},
      "photo2.jpg": {"plant_id": "PLANT-002", "view_label": "top", "crop_type": "tomato"}
    }
    ```
- **Strict Isolation Guarantee:**
  - Evidence from `PLANT-001` is strictly partitioned from `PLANT-002`.
  - Separate sessions are created and queued for asynchronous analysis.
- **Response (`201 Created`):**
  ```json
  {
    "message": "Successfully created 2 isolated plant sessions.",
    "batch_count": 2,
    "sessions": [
      {"session_id": "sess_001...", "plant_id": "PLANT-001", "status": "QUEUED"},
      {"session_id": "sess_002...", "plant_id": "PLANT-002", "status": "QUEUED"}
    ]
  }
  ```

---

## 2. Analysis Execution & Reporting

### `POST /plant-sessions/{session_id}/analyze`
Trigger asynchronous multi-view evidence fusion on uploaded media.
- **Response (`200 OK`):**
  ```json
  {
    "message": "Analysis started in background.",
    "session_id": "sess_89f31ab2c01d",
    "clips_count": 4
  }
  ```

---

### `GET /plant-sessions/{session_id}/status`
Query truthful job processing progress.
- **Response (`200 OK`):**
  ```json
  {
    "session_id": "sess_89f31ab2c01d",
    "status": "COMPLETED",
    "stage": "Finished",
    "progress_percent": 100,
    "has_assessment": true,
    "assessment": { ... }
  }
  ```

---

### `GET /plant-sessions/{session_id}/export`
Export decision-support report with model provenance, view coverage, and disclaimers.
- **Query Parameter:** `format` (`markdown` or `json`, default `json`)
- **Responses:**
  - `format=json`: Structured JSON containing session, observation counts, and assessment.
  - `format=markdown`: GitHub-Flavored Markdown report (`text/markdown; charset=utf-8`) with attachment headers.

---

### `DELETE /plant-sessions/{session_id}`
Delete session records and permanently remove associated uploaded temporary files.
- **Response (`200 OK`):**
  ```json
  {
    "message": "Session 'sess_89f31ab2c01d' deleted successfully."
  }
  ```

---

## 3. System & Model Health

### `GET /models/status`
Inspect model readiness, weights path, and artifact SHA-256 hash.
- **Response (`200 OK`):**
  ```json
  {
    "overall_status": "READY",
    "models": {
      "disease_classifier": {
        "status": "READY",
        "weights_path": "backend/models/yolo26_tomato.pt",
        "sha256": "ccc05ab88bb550567fb7a7a07c564d37e344e7e69e094da54481ea47c0f134b3",
        "meta": {
          "model_name": "tomato_disease_yolov8n",
          "classes": ["early_blight", "late_blight", "septoria_leaf_spot", "healthy", ...]
        }
      }
    }
  }
  ```
