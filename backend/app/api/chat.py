"""
backend/app/api/chat.py

API router providing the two specialized AgriHealth Chatbots:
  1. POST /api/chat/general — General Plant & Agronomy Assistant (Strictly Guardrailed)
  2. POST /api/chat/disease — Clinical Disease & Treatment Consultant (Context-Locked)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status

from backend.app.services.chatbot_service import (
    query_general_plant_bot,
    query_disease_treatment_bot,
    _get_groq_api_key,
)

router = APIRouter()


class GeneralChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt to the general plant advisor")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Prior chat history turns")


class DiseaseChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query about the diagnosed disease or treatment")
    context: Dict[str, Any] = Field(..., description="Active clinical diagnosis context (disease_name, confidence, severity, etc.)")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Prior clinical consultation history")


@router.get("/chat/status")
async def get_chatbot_status():
    """
    Returns the operational readiness of the Groq LLM integration.
    """
    api_key = _get_groq_api_key()
    has_key = bool(api_key)
    return {
        "status": "ONLINE" if has_key else "OFFLINE_FALLBACK_READY",
        "has_groq_api_key": has_key,
        "active_llm_model": "llama-3.3-70b-versatile" if has_key else "local_clinical_knowledgebase",
        "supported_bots": [
            {
                "id": "general_plant_bot",
                "name": "AgriHealth Botanical Assistant",
                "guardrail": "Strictly locked to plant science, botany, agronomy, soil, fertilizers, and pests."
            },
            {
                "id": "disease_treatment_bot",
                "name": "Clinical Pathologist & Treatment Specialist",
                "guardrail": "Context-locked to diagnosed disease etiology, causal pathogen, organic/chemical cure, and prevention."
            }
        ]
    }


@router.post("/chat/general")
async def chat_general(req: GeneralChatRequest):
    """
    Submits a query to the General Plant & Agronomy Assistant.
    Enforces strict guardrails rejecting non-plant inquiries.
    """
    try:
        response = await query_general_plant_bot(req.message, req.history)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(exc)}"
        )


@router.post("/chat/disease")
async def chat_disease(req: DiseaseChatRequest):
    """
    Submits a query to the Clinical Disease & Treatment Consultant.
    Enforces context-locking to the active diagnosis.
    """
    try:
        response = await query_disease_treatment_bot(req.message, req.context, req.history)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process clinical consultation: {str(exc)}"
        )
