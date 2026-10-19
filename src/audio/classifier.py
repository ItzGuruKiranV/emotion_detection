"""
Speech Emotion Recognition (SER) Classifier.
Evaluates prosodic, spectral, and cepstral features to classify acoustic emotional tone.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import io
import time
import numpy as np

from src.config import CANONICAL_EMOTIONS
from src.audio.preprocessor import load_and_preprocess_audio
from src.audio.feature_extractor import extract_acoustic_features
from src.utils.logger import get_logger

logger = get_logger("audio.classifier")


class AudioEmotionClassifier:
    """
    Acoustic Speech Emotion Recognition (SER) Classifier based on
    psychoacoustic features (pitch, spectral contrast, MFCCs, and RMS dynamics).
    """

    _instance: Optional["AudioEmotionClassifier"] = None

    def __init__(self):
        logger.info("Initialized Acoustic AudioEmotionClassifier.")

    @classmethod
    def get_instance(cls) -> "AudioEmotionClassifier":
        if cls._instance is None:
            cls._instance = AudioEmotionClassifier()
        return cls._instance

    def predict_from_features(
        self,
        features: np.ndarray,
        summary: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calibrate emotional probabilities using acoustic evidence:
        pitch, energy, dynamic range, and spectral balance.
        """
        pitch = summary.get("mean_pitch_hz", 120.0)
        pitch_var = summary.get("pitch_variability_hz", 20.0)
        energy = summary.get("mean_rms_energy", 0.05)
        zcr = summary.get("zero_crossing_rate", 0.05)

        # Baseline logits for the 7 canonical emotions
        logits = {e: 0.0 for e in CANONICAL_EMOTIONS}

        # 1. Energy correlations
        if energy > 0.12:
            logits["angry"] += 2.2
            logits["happy"] += 1.2
            logits["surprise"] += 0.8
            logits["sad"] -= 1.8
            logits["neutral"] -= 1.5
        elif energy < 0.04:
            logits["sad"] += 2.0
            logits["neutral"] += 1.2
            logits["disgust"] += 0.5
            logits["angry"] -= 2.0
            logits["happy"] -= 1.5

        # 2. Pitch & Pitch Variability correlations
        if pitch > 200:
            logits["fear"] += 1.8
            logits["surprise"] += 1.6
            logits["happy"] += 1.4
            logits["sad"] -= 1.2
        elif pitch < 120:
            logits["sad"] += 1.5
            logits["neutral"] += 1.3
            logits["angry"] += 0.8
            logits["disgust"] += 1.0

        if pitch_var > 45:
            logits["happy"] += 1.5
            logits["surprise"] += 1.4
            logits["fear"] += 1.2
            logits["neutral"] -= 2.0
        elif pitch_var < 18:
            logits["neutral"] += 1.8
            logits["sad"] += 1.0
            logits["happy"] -= 1.5

        # 3. High Frequency / Zero-Crossing Rate
        if zcr > 0.10:
            logits["angry"] += 1.0
            logits["fear"] += 1.0
            logits["disgust"] += 0.6
        else:
            logits["neutral"] += 0.5
            logits["sad"] += 0.5

        # 4. Mel / MFCC shape contribution
        # Use first 3 MFCC means as spectral envelope anchors
        mfcc0 = float(features[0])  # overall energy level
        mfcc1 = float(features[1])  # spectral tilt
        if mfcc1 > 0:
            logits["happy"] += 0.4
            logits["surprise"] += 0.4
        else:
            logits["sad"] += 0.4
            logits["disgust"] += 0.4

        # Softmax normalization
        exp_logits = {k: np.exp(np.clip(v, -10.0, 10.0)) for k, v in logits.items()}
        sum_exp = sum(exp_logits.values())
        probabilities = {k: round(float(v / sum_exp), 4) for k, v in exp_logits.items()}

        return probabilities

    def analyze_audio(
        self,
        file_or_bytes: Union[str, Path, bytes, io.BytesIO]
    ) -> Dict[str, Any]:
        """
        Analyze audio input and return acoustic emotion predictions,
        spectrogram data, and summary metrics.
        """
        t0 = time.perf_counter()

        # Preprocess waveform
        audio, sr, duration = load_and_preprocess_audio(file_or_bytes)

        # Extract features and spectrogram
        features, mel_spec_db, summary = extract_acoustic_features(audio, sr=sr)

        # Classify
        probabilities = self.predict_from_features(features, summary)
        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "modality": "audio",
            "dominant_emotion": dominant_emotion,
            "confidence": round(float(confidence), 4),
            "probabilities": probabilities,
            "acoustic_summary": summary,
            "waveform": audio,
            "mel_spectrogram_db": mel_spec_db,
            "sample_rate": sr,
            "duration_sec": duration,
            "latency_ms": latency_ms
        }
