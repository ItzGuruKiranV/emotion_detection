"""
Text Emotion Classifier using Fine-Tuned BERT Architecture.
Loads checkpoint-1600 with memory-conscious initialization and canonical label mapping.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import os
import torch
import numpy as np

from src.config import (
    TEXT_CHECKPOINT_DIR,
    CANONICAL_EMOTIONS,
    TEXT_TO_CANONICAL
)
from src.text.preprocessor import preprocess_text
from src.utils.logger import get_logger

logger = get_logger("text.classifier")

# Checkpoint label mapping
CHECKPOINT_ID2LABEL = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "happy"
}


class TextEmotionClassifier:
    """Singleton Text Emotion Classifier with lazy model loading."""

    _instance: Optional["TextEmotionClassifier"] = None

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or TEXT_CHECKPOINT_DIR
        self.tokenizer = None
        self.model = None

    @classmethod
    def get_instance(cls) -> "TextEmotionClassifier":
        if cls._instance is None:
            cls._instance = TextEmotionClassifier()
        return cls._instance

    def _load_model(self) -> None:
        """Lazy load tokenizer and BERT sequence classification model."""
        if self.model is None or self.tokenizer is None:
            # Set memory-safety flags
            os.environ["TRANSFORMERS_NO_TF"] = "1"
            os.environ["USE_TF"] = "0"

            from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
            from safetensors.torch import load_file

            logger.info("Loading BERT tokenizer ('bert-base-uncased')...")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            safetensors_path = self.checkpoint_dir / "model.safetensors"
            config_path = self.checkpoint_dir / "config.json"

            if not safetensors_path.exists():
                raise FileNotFoundError(f"Checkpoint model weights not found at {safetensors_path}")

            logger.info(f"Loading BERT model weights from {safetensors_path}...")
            config = BertConfig.from_pretrained(str(config_path))
            self.model = BertForSequenceClassification(config)

            # Direct state_dict loading avoids Windows memory-mapping issues with large safetensors files
            state_dict = load_file(str(safetensors_path), device="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("BERT text classifier loaded and ready for evaluation.")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Tokenize and classify emotion in input text.
        Returns:
            Structured dictionary with dominant emotion, confidence, and canonical probabilities.
        """
        t0 = time.perf_counter()
        cleaned_text = preprocess_text(text)

        self._load_model()

        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()

        # Map model outputs to canonical emotions
        canonical_probs: Dict[str, float] = {e: 0.02 for e in CANONICAL_EMOTIONS}

        for idx, prob in enumerate(raw_probs):
            raw_label = CHECKPOINT_ID2LABEL.get(idx, "neutral")
            canonical_target = TEXT_TO_CANONICAL.get(raw_label, "neutral")
            canonical_probs[canonical_target] += float(prob)

        # Baseline adjustment for neutral / surprise / disgust
        total = sum(canonical_probs.values())
        canonical_probs = {k: round(v / total, 4) for k, v in canonical_probs.items()}

        dominant_emotion = max(canonical_probs, key=canonical_probs.get)
        confidence = canonical_probs[dominant_emotion]

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "modality": "text",
            "dominant_emotion": dominant_emotion,
            "confidence": round(float(confidence), 4),
            "probabilities": canonical_probs,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "latency_ms": latency_ms
        }
