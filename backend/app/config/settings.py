import os
from pathlib import Path
from dataclasses import dataclass

# Automatically parse .env if present
_env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v

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

    # Groq AI Chatbots
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

settings = Settings()

