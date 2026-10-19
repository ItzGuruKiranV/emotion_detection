"""
Facial Emotion Classifier using Pretrained MobileNetV2 Architecture.
Performs face detection, per-face emotion classification, and probability calibration.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
import numpy as np
from PIL import Image

from src.config import (
    IMAGE_MODEL_PATH,
    CLASSES_JSON_PATH,
    CANONICAL_EMOTIONS,
    IMAGE_INPUT_SIZE
)
from src.vision.face_detector import FaceDetector
from src.vision.preprocessor import preprocess_face_crop
from src.utils.logger import get_logger

logger = get_logger("vision.classifier")


class VisionEmotionClassifier:
    """Singleton Classifier for facial emotion recognition."""

    _instance: Optional["VisionEmotionClassifier"] = None

    def __init__(self, model_path: Optional[Path] = None, classes_path: Optional[Path] = None):
        self.model_path = model_path or IMAGE_MODEL_PATH
        self.classes_path = classes_path or CLASSES_JSON_PATH
        self.model = None
        self.class_indices: Dict[str, int] = {}
        self.inv_classes: Dict[int, str] = {}
        self.detector = FaceDetector.get_instance()
        self._load_classes()

    @classmethod
    def get_instance(cls) -> "VisionEmotionClassifier":
        if cls._instance is None:
            cls._instance = VisionEmotionClassifier()
        return cls._instance

    def _load_classes(self) -> None:
        """Load class index mapping from classes.json."""
        if not self.classes_path.exists():
            # Standard 7 Ekman classes default
            default_classes = {emo: i for i, emo in enumerate(CANONICAL_EMOTIONS)}
            self.class_indices = default_classes
        else:
            with open(self.classes_path, "r") as f:
                self.class_indices = json.load(f)

        self.inv_classes = {int(v): k for k, v in self.class_indices.items()}
        logger.info(f"Loaded {len(self.inv_classes)} emotion classes: {list(self.inv_classes.values())}")

    def _load_model(self) -> None:
        """Lazy loader for TensorFlow MobileNetV2 model to preserve system RAM."""
        if self.model is None:
            import tensorflow as tf
            if not self.model_path.exists():
                raise FileNotFoundError(f"Vision model weights not found at {self.model_path}")
            logger.info(f"Loading TensorFlow vision model from {self.model_path}...")
            # compile=False avoids compiling optimizer weights which saves memory and avoids warnings
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            logger.info("Vision model loaded successfully.")

    def predict_tensor(self, tensor: np.ndarray) -> Dict[str, float]:
        """
        Run forward pass on preprocessed tensor shape (1, 96, 96, 3).
        Returns normalized probability dictionary across canonical emotions.
        """
        self._load_model()
        raw_preds = self.model.predict(tensor, verbose=0)[0]

        probs: Dict[str, float] = {}
        for idx, val in enumerate(raw_preds):
            label = self.inv_classes.get(idx, f"unknown_{idx}").lower()
            probs[label] = float(val)

        # Normalize across canonical emotions
        canonical_probs: Dict[str, float] = {e: 0.0 for e in CANONICAL_EMOTIONS}
        for lbl, score in probs.items():
            if lbl in canonical_probs:
                canonical_probs[lbl] += score
            else:
                canonical_probs["neutral"] += score

        total = sum(canonical_probs.values())
        if total > 0:
            canonical_probs = {k: round(v / total, 4) for k, v in canonical_probs.items()}

        return canonical_probs

    def analyze_image(
        self,
        image: Image.Image,
        annotate: bool = True
    ) -> Dict[str, Any]:
        """
        End-to-end image analysis:
        1. Face detection
        2. Per-face crop and classification
        3. Multi-face aggregation
        4. Optional annotation
        """
        t0 = time.perf_counter()
        boxes = self.detector.detect_faces(image)

        face_results: List[Dict[str, Any]] = []

        if len(boxes) == 0:
            # Fallback: classify entire image (in case cropped portrait without detected bounding box)
            logger.info("No face bounding box detected. Evaluating full image frame.")
            tensor = preprocess_face_crop(image, bbox=None, target_size=IMAGE_INPUT_SIZE)
            probs = self.predict_tensor(tensor)
            dominant = max(probs, key=probs.get)
            conf = probs[dominant]

            face_results.append({
                "box": (0, 0, image.width, image.height),
                "dominant_emotion": dominant,
                "confidence": conf,
                "probabilities": probs,
                "is_fallback": True
            })
            dominant_emotion = dominant
            overall_confidence = conf
            overall_probs = probs
        else:
            for bbox in boxes:
                tensor = preprocess_face_crop(image, bbox=bbox, target_size=IMAGE_INPUT_SIZE)
                probs = self.predict_tensor(tensor)
                dom = max(probs, key=probs.get)
                face_results.append({
                    "box": bbox,
                    "dominant_emotion": dom,
                    "confidence": probs[dom],
                    "probabilities": probs,
                    "is_fallback": False
                })

            # Primary face is the largest bounding box (w * h)
            primary_face = max(face_results, key=lambda f: f["box"][2] * f["box"][3])
            dominant_emotion = primary_face["dominant_emotion"]
            overall_confidence = primary_face["confidence"]
            overall_probs = primary_face["probabilities"]

        annotated_img = None
        if annotate:
            annotated_img = self.detector.draw_annotations(image, face_results)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "modality": "vision",
            "faces_detected": len(boxes),
            "faces": face_results,
            "dominant_emotion": dominant_emotion,
            "confidence": round(float(overall_confidence), 4),
            "probabilities": overall_probs,
            "latency_ms": latency_ms,
            "annotated_image": annotated_img
        }
