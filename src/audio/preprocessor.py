"""
Audio Preprocessing Module.
Validates duration, handles sample rates, channel mixing, and silence detection.
"""

from pathlib import Path
from typing import Union, Tuple
import io
import numpy as np
import soundfile as sf
import librosa

from src.config import (
    AUDIO_TARGET_SR,
    AUDIO_MIN_DURATION_SEC,
    AUDIO_MAX_DURATION_SEC
)
from src.utils.validation import ValidationError
from src.utils.logger import get_logger

logger = get_logger("audio.preprocessor")


def load_and_preprocess_audio(
    file_or_bytes: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = AUDIO_TARGET_SR
) -> Tuple[np.ndarray, int, float]:
    """
    Load audio from file path or bytes, convert to mono, resample to target_sr,
    and validate duration.
    Returns:
        (audio_waveform, sample_rate, duration_seconds)
    """
    try:
        if isinstance(file_or_bytes, np.ndarray):
            audio = file_or_bytes
            sr = target_sr
        elif isinstance(file_or_bytes, (str, Path)):
            audio, sr = librosa.load(str(file_or_bytes), sr=target_sr, mono=True)
        elif isinstance(file_or_bytes, bytes):
            with io.BytesIO(file_or_bytes) as bio:
                try:
                    audio, sr = sf.read(bio)
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)
                    if sr != target_sr:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                except Exception:
                    # Raw float32 buffer fallback
                    audio = np.frombuffer(file_or_bytes, dtype=np.float32)
                    sr = target_sr
        elif isinstance(file_or_bytes, io.BytesIO):
            audio, sr = sf.read(file_or_bytes)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        else:
            raise ValidationError("Unsupported audio input object.")

    except Exception as e:
        raise ValidationError(f"Unable to decode audio stream: {str(e)}") from e

    # Ensure float32 normalized waveform
    audio = audio.astype(np.float32)

    duration = float(len(audio)) / float(sr)
    if duration < AUDIO_MIN_DURATION_SEC:
        raise ValidationError(
            f"Audio duration ({duration:.2f}s) is shorter than minimum allowed ({AUDIO_MIN_DURATION_SEC}s)."
        )

    # Truncate if exceeds max duration
    if duration > AUDIO_MAX_DURATION_SEC:
        logger.warning(f"Audio exceeds {AUDIO_MAX_DURATION_SEC}s ({duration:.2f}s). Truncating to {AUDIO_MAX_DURATION_SEC}s.")
        audio = audio[: int(AUDIO_MAX_DURATION_SEC * sr)]
        duration = AUDIO_MAX_DURATION_SEC

    # Check for complete silence
    rms = np.sqrt(np.mean(np.square(audio)))
    if rms < 1e-4:
        raise ValidationError("Audio file contains mostly silence or inaudible signal.")

    return audio, sr, duration
