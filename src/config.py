"""
Centralized configuration for paths and environment variables.
"""
import os
from pathlib import Path

# Project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_PROJECT_ROOT / "data")))
CUSTOM_ZONES_PATH = DATA_ROOT / "custom_zones.json"

# ML service
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(_PROJECT_ROOT / "models")))
ML_API_URL = os.environ.get("ML_API_URL", "http://localhost:8000")
