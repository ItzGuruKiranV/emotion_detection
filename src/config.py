"""
Global Configuration and Taxonomy for Multimodal Emotion Analysis.
"""

from pathlib import Path
from typing import Dict, List
import os

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_MODEL_PATH = MODELS_DIR / "best_emotion_model.h5"
FACE_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
CLASSES_JSON_PATH = PROJECT_ROOT / "classes.json"
RESULTS_DIR = PROJECT_ROOT / "results"
TEXT_CHECKPOINT_DIR = RESULTS_DIR / "checkpoint-1600"

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SAMPLE_DIR = DATA_DIR / "sample"

# Database
DATABASE_PATH = PROJECT_ROOT / "emotion_platform.db"

# Canonical 7 Ekman Emotion Taxonomy
CANONICAL_EMOTIONS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Color mapping for visualizations
EMOTION_COLORS: Dict[str, str] = {
    "angry": "#EF4444",      # Red
    "disgust": "#10B981",    # Emerald
    "fear": "#8B5CF6",       # Purple
    "happy": "#F59E0B",      # Amber/Yellow
    "neutral": "#6B7280",    # Gray
    "sad": "#3B82F6",        # Blue
    "surprise": "#EC4899"    # Pink
}

EMOTION_EMOJIS: Dict[str, str] = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

# Mapping from text dataset / BERT classes to canonical emotions
TEXT_TO_CANONICAL: Dict[str, str] = {
    "anger": "angry",
    "fear": "fear",
    "joy": "happy",
    "love": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "happy": "happy",
    "neutral": "neutral"
}

# Image input specifications
IMAGE_INPUT_SIZE = (96, 96)
IMAGE_MAX_SIZE_MB = 10
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# Audio specifications
AUDIO_TARGET_SR = 16000
AUDIO_MAX_DURATION_SEC = 60.0
AUDIO_MIN_DURATION_SEC = 0.5
AUDIO_MAX_SIZE_MB = 20
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".ogg", ".flac"]

# Text specifications
TEXT_MAX_LENGTH = 1000
TEXT_MIN_LENGTH = 2

# Multimodal Fusion Configuration
DEFAULT_MODALITY_WEIGHTS: Dict[str, float] = {
    "vision": 0.40,
    "audio": 0.35,
    "text": 0.25
}

# Conflict and Uncertainty Thresholds
CONFLICT_ENTROPY_THRESHOLD = 1.60
ABSTAIN_CONFIDENCE_THRESHOLD = 0.30

# Environment Flags for Memory Safety on Windows
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
