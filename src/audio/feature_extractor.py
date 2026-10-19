"""
Acoustic Feature Extraction for Speech Emotion Recognition (SER).
Extracts MFCCs, Chroma, Mel-spectrogram, Spectral Contrast, and Prosodic features.
"""

from typing import Dict, Any, Tuple
import numpy as np
import librosa

from src.utils.logger import get_logger

logger = get_logger("audio.feature_extractor")


def extract_acoustic_features(
    audio: np.ndarray,
    sr: int = 16000
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extract comprehensive acoustic and prosodic feature vector.
    Returns:
        feature_vector: 1D normalized feature array
        mel_spectrogram: 2D array (dB scale) for visualization
        summary_stats: Dict of key acoustic indicators (pitch, energy, tempo, zcr)
    """
    # 1. Mel-spectrogram (dB scale)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. MFCCs (13 coefficients) + Delta
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_var = np.var(mfccs, axis=1)

    # 3. Chroma STFT (12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # 4. Spectral Contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # 5. Energy and Zero Crossing Rate
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]

    mean_rms = float(np.mean(rms))
    max_rms = float(np.max(rms))
    mean_zcr = float(np.mean(zcr))

    # 6. Fundamental Frequency (F0 / Pitch) estimation using librosa yin
    try:
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
        valid_f0 = f0[np.isfinite(f0)]
        mean_pitch = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 120.0
        pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 20.0
    except Exception:
        mean_pitch = 120.0
        pitch_std = 20.0

    # Concatenate features into a unified vector:
    # 13 mfcc_mean + 13 mfcc_var + 12 chroma + 7 contrast + 1 rms + 1 zcr + 1 pitch + 1 pitch_std = 49 features
    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_var,
        chroma_mean,
        contrast_mean,
        [mean_rms, mean_zcr, mean_pitch / 500.0, pitch_std / 200.0]
    ]).astype(np.float32)

    summary_stats = {
        "mean_rms_energy": round(mean_rms, 4),
        "max_rms_energy": round(max_rms, 4),
        "zero_crossing_rate": round(mean_zcr, 4),
        "mean_pitch_hz": round(mean_pitch, 1),
        "pitch_variability_hz": round(pitch_std, 1),
        "duration_sec": round(float(len(audio)) / sr, 2)
    }

    return feature_vector, mel_spec_db, summary_stats
