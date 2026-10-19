"""
Face Detection Module using OpenCV Haar Cascade.
Detects human faces, handles multiple faces, and returns bounded crops.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

from src.config import FACE_CASCADE_PATH
from src.utils.logger import get_logger

logger = get_logger("vision.face_detector")


class FaceDetector:
    """Haar Cascade face detector with fallback and multi-face support."""

    _instance: Optional["FaceDetector"] = None

    def __init__(self, cascade_path: Optional[Path] = None):
        path_to_use = cascade_path or FACE_CASCADE_PATH
        if not path_to_use.exists():
            # Fallback to OpenCV built-in cascade if local file missing
            cv2_data_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            logger.warning(f"Local cascade {path_to_use} not found. Falling back to {cv2_data_path}")
            self.cascade = cv2.CascadeClassifier(cv2_data_path)
        else:
            self.cascade = cv2.CascadeClassifier(str(path_to_use))

        if self.cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV Haar Cascade from {path_to_use}")

    @classmethod
    def get_instance(cls) -> "FaceDetector":
        """Singleton pattern to avoid re-initializing Haar cascade."""
        if cls._instance is None:
            cls._instance = FaceDetector()
        return cls._instance

    def detect_faces(
        self,
        image: Image.Image,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in an image.
        Returns list of (x, y, w, h) bounding boxes.
        """
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Equalize histogram for illumination invariance
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        boxes: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in faces:
            boxes.append((int(x), int(y), int(w), int(h)))

        logger.info(f"Detected {len(boxes)} face(s) in image of size {image.size}")
        return boxes

    @staticmethod
    def draw_annotations(
        image: Image.Image,
        faces_results: List[dict]
    ) -> Image.Image:
        """
        Draw bounding boxes and emotion predictions on the image.
        """
        annotated = np.array(image.copy())

        for res in faces_results:
            bbox = res.get("box", (0, 0, 0, 0))
            x, y, w, h = bbox
            emotion = res.get("dominant_emotion", "unknown")
            conf = res.get("confidence", 0.0)

            label = f"{emotion.capitalize()} ({conf * 100:.1f}%)"

            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 230, 115), 2)

            # Draw label banner
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            y_banner = max(y - 8, th + 8)
            cv2.rectangle(
                annotated,
                (x, y_banner - th - 6),
                (x + tw + 6, y_banner + baseline),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x + 3, y_banner - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 128),
                1,
                cv2.LINE_AA
            )

        return Image.fromarray(annotated)
