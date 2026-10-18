import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config.settings import settings
from backend.app.api import health, analyze, scans, analytics, simulation
from backend.app.api import models_status, plant_sessions, chat, ensemble



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: runs once at startup to pre-initialize the model registry and ensemble models.
    """
    print("[Startup] Pre-initializing model registry...")
    from backend.app.services.model_registry import get_disease_model_status
    status = get_disease_model_status()
    print(f"[Startup] Disease model status: {status['status']}")
    
    print("[Startup] Initializing multi-model ensemble...")
    from backend.app.services.ensemble_engine import _init_ensemble_models
    _init_ensemble_models()
    yield
    # Cleanup on shutdown (if needed in future)


app = FastAPI(
    title="Multi-View Agricultural Health Assessment API",
    description=(
        "Dual-evidence AI & DIP crop health monitoring engine. "
        "Supports multi-view plant sessions, multi-model ensemble diagnosis, and clinical chatbots."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware
# Allow all origins for local development. For production, restrict to known
# frontend origins via the ALLOWED_ORIGINS environment variable.
# ---------------------------------------------------------------------------
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register Routers
# ---------------------------------------------------------------------------
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(models_status.router, prefix="/api", tags=["Models"])
app.include_router(ensemble.router, prefix="/api", tags=["Ensemble"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(plant_sessions.router, prefix="/api", tags=["PlantSessions"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(scans.router, prefix="/api", tags=["Scans"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])
app.include_router(simulation.router, prefix="/api", tags=["Simulation"])

# ---------------------------------------------------------------------------
# Static files for processed images and real leaf samples
# ---------------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/samples", exist_ok=True)
app.mount("/static/processed", StaticFiles(directory="data/processed"), name="processed")
app.mount("/static/samples", StaticFiles(directory="data/samples"), name="samples")


@app.get("/api/samples", tags=["Samples"])
def list_samples():
    """Returns sample leaf images available for instant real-time testing."""
    samples_dir = Path("data/samples")
    items = []
    for f in sorted(samples_dir.glob("*.jpg")):
        items.append({
            "filename": f.name,
            "url": f"/static/samples/{f.name}",
            "label": f.stem.replace("_", " ").title(),
        })
    return {"samples": items}


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Multi-View Agricultural Health Assessment API is running.",
        "documentation": "/docs",
        "health_check": "/api/health",
        "models_status": "/api/models/status",
        "version": "1.1.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
