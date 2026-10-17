import os
from dataclasses import dataclass

@dataclass
class Settings:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "agrihealth_db")
    
    # Severity Thresholds (percentage)
    SEVERITY_LOW_MAX: float = float(os.getenv("SEVERITY_LOW_MAX", "10.0"))
    SEVERITY_MODERATE_MAX: float = float(os.getenv("SEVERITY_MODERATE_MAX", "25.0"))
    SEVERITY_HIGH_MAX: float = float(os.getenv("SEVERITY_HIGH_MAX", "50.0"))
    
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

settings = Settings()
