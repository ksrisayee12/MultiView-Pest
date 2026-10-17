import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config.settings import settings
from backend.app.api import health, analyze, scans, analytics, simulation

app = FastAPI(
    title="Multi-View Agricultural Health Assessment API",
    description="Dual-evidence AI & DIP crop health monitoring engine using YOLO26 and OpenCV.",
    version="1.0.0"
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(scans.router, prefix="/api", tags=["Scans"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])
app.include_router(simulation.router, prefix="/api", tags=["Simulation"])

# Ensure data processed directory exists and mount static files if available
os.makedirs("data/processed", exist_ok=True)
app.mount("/static/processed", StaticFiles(directory="data/processed"), name="processed")

@app.get("/")
def root():
    return {
        "message": "Multi-View Drone-Based Agricultural Health Assessment API is running.",
        "documentation": "/docs",
        "health_check": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
