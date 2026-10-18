"""
backend/app/schemas/plant_session.py

Pydantic schemas for multi-view plant analysis sessions, clip uploads,
observation evidence, and plant assessments.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class CreateSessionRequest(BaseModel):
    plant_id: str = Field(..., min_length=1, max_length=64, description="Plant identifier, e.g. PLANT-001")
    crop_type: str = Field(default="tomato", min_length=1, max_length=64, description="Crop type")
    notes: Optional[str] = Field(default=None, max_length=500, description="Optional field or plot notes")


class ClipMetaResponse(BaseModel):
    clip_id: str
    session_id: str
    original_filename: str
    view_label: str
    upload_status: str
    processing_status: str
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    total_video_frames: Optional[int] = None
    frames_extracted: int = 0
    frames_skipped_duplicate: int = 0
    sample_fps: float = 2.0
    error_message: Optional[str] = None
    created_at: float
    processed_at: Optional[float] = None


class PlantAssessmentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    plant_id: str
    crop_type: str
    status: str
    status_description: str
    overall_health_assessment: str
    top_predicted_disease: Optional[str] = None
    max_disease_confidence: float = 0.0
    mean_disease_confidence: float = 0.0
    frames_with_symptoms: int = 0
    total_usable_frames: int = 0
    symptom_frame_fraction: float = 0.0
    mean_lesion_percent: float = 0.0
    views_submitted: List[str] = []
    missing_views: List[str] = []
    coverage_assessment: str = "INSUFFICIENT"
    clip_summaries: List[Dict[str, Any]] = []
    limitations: List[str] = []
    model_readiness: Dict[str, Any] = {}
    recommendations: List[str] = []
    created_at: float


class PlantSessionResponse(BaseModel):
    session_id: str
    plant_id: str
    crop_type: str
    status: str
    created_at: float
    updated_at: float
    notes: Optional[str] = None
    clips: List[ClipMetaResponse] = []
    assessment: Optional[Dict[str, Any]] = None
    observation_count: int = 0


class SessionStatusResponse(BaseModel):
    session_id: str
    plant_id: str
    status: str
    stage: str
    progress_percent: int
    message: str
    clips_total: int
    clips_processed: int
    has_assessment: bool
    assessment: Optional[Dict[str, Any]] = None
