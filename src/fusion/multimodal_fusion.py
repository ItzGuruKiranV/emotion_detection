"""
Multimodal Late Fusion Engine for Emotion Analysis.
Combines Vision, Audio, and Text predictions using dynamic reliability weighting,
cross-modal conflict detection, and Shannon entropy uncertainty estimation.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.config import (
    CANONICAL_EMOTIONS,
    DEFAULT_MODALITY_WEIGHTS,
    CONFLICT_ENTROPY_THRESHOLD,
    ABSTAIN_CONFIDENCE_THRESHOLD
)
from src.utils.logger import get_logger

logger = get_logger("fusion.multimodal")


# Emotional Valence Classification for conflict detection
# Positive / Arousing / Negative / Neutral
VALENCE_MAP = {
    "happy": "positive",
    "surprise": "positive_arousal",
    "neutral": "neutral",
    "sad": "negative_low_arousal",
    "fear": "negative_high_arousal",
    "angry": "negative_high_arousal",
    "disgust": "negative_aversion"
}


class MultimodalFusionEngine:
    """
    Late Fusion Engine implementing Quality-Weighted Linear Combination,
    Conflict Analysis, and Uncertainty-Aware Abstention.
    """

    def __init__(self, base_weights: Optional[Dict[str, float]] = None):
        self.base_weights = base_weights or DEFAULT_MODALITY_WEIGHTS.copy()

    def fuse(
        self,
        modality_results: Dict[str, Dict[str, Any]],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Fuse predictions from all available modalities.

        Args:
            modality_results: Dict with keys in ('vision', 'audio', 'text'),
                              each containing a valid modality result dict.
            custom_weights: Optional override for modality base weights.

        Returns:
            Structured fusion result containing fused probabilities, dominant emotion,
            uncertainty entropy, conflict flags, and audit explanation.
        """
        active_modalities = [m for m, res in modality_results.items() if res is not None]

        if not active_modalities:
            raise ValueError("At least one modality result must be provided for fusion.")

        # Single Modality Pass-Through
        if len(active_modalities) == 1:
            single_mod = active_modalities[0]
            res = modality_results[single_mod]
            probs = res.get("probabilities", {e: 1.0 / len(CANONICAL_EMOTIONS) for e in CANONICAL_EMOTIONS})
            dominant = res.get("dominant_emotion", max(probs, key=probs.get))
            conf = res.get("confidence", probs[dominant])

            entropy = self._calculate_entropy(probs)
            return {
                "fused_emotion": dominant,
                "confidence": round(float(conf), 4),
                "probabilities": probs,
                "entropy": round(float(entropy), 4),
                "is_abstain": conf < ABSTAIN_CONFIDENCE_THRESHOLD,
                "conflict_detected": False,
                "conflict_details": None,
                "active_modalities": active_modalities,
                "normalized_weights": {single_mod: 1.0},
                "modality_breakdown": {single_mod: res},
                "explanation": f"Single modality prediction passed through directly from {single_mod}."
            }

        # Multi-Modality Late Fusion
        weights_to_use = (custom_weights or self.base_weights).copy()

        # Dynamic Quality Calibration
        adjusted_weights: Dict[str, float] = {}
        for mod in active_modalities:
            base_w = weights_to_use.get(mod, 0.33)
            res = modality_results[mod]

            quality_multiplier = 1.0
            if mod == "vision":
                # If face detection was a fallback without localized bbox, reduce weight
                faces = res.get("faces", [])
                if any(f.get("is_fallback", False) for f in faces):
                    quality_multiplier = 0.65
            elif mod == "audio":
                # If audio is very short (< 1.5s), reduce acoustic weight
                duration = res.get("duration_sec", 3.0)
                if duration < 1.5:
                    quality_multiplier = 0.70
            elif mod == "text":
                # If text is extremely brief (< 3 words), reduce NLP weight
                words = res.get("word_count", 10)
                if words < 3:
                    quality_multiplier = 0.70

            adjusted_weights[mod] = base_w * quality_multiplier

        # Re-normalize weights so sum is 1.0
        total_w = sum(adjusted_weights.values())
        if total_w <= 0:
            normalized_weights = {m: 1.0 / len(active_modalities) for m in active_modalities}
        else:
            normalized_weights = {m: round(w / total_w, 4) for m, w in adjusted_weights.items()}

        # Combine probabilities
        fused_probs: Dict[str, float] = {e: 0.0 for e in CANONICAL_EMOTIONS}
        for mod, weight in normalized_weights.items():
            mod_probs = modality_results[mod].get("probabilities", {})
            for emo in CANONICAL_EMOTIONS:
                fused_probs[emo] += weight * mod_probs.get(emo, 0.0)

        # Normalize fused probabilities
        sum_fused = sum(fused_probs.values())
        if sum_fused > 0:
            fused_probs = {k: round(v / sum_fused, 4) for k, v in fused_probs.items()}

        dominant_emotion = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[dominant_emotion]

        # Calculate Shannon Entropy: H = -sum(p * log2(p))
        entropy = self._calculate_entropy(fused_probs)

        # Conflict Detection
        conflict_detected, conflict_details = self._detect_conflicts(modality_results, active_modalities)

        # Abstain Check
        is_abstain = bool(confidence < ABSTAIN_CONFIDENCE_THRESHOLD or (entropy > CONFLICT_ENTROPY_THRESHOLD and conflict_detected))

        # Build Explanation
        explanation = self._build_explanation(
            dominant_emotion, confidence, active_modalities,
            normalized_weights, conflict_detected, conflict_details, is_abstain
        )

        return {
            "fused_emotion": dominant_emotion,
            "confidence": round(float(confidence), 4),
            "probabilities": fused_probs,
            "entropy": round(float(entropy), 4),
            "is_abstain": is_abstain,
            "conflict_detected": conflict_detected,
            "conflict_details": conflict_details,
            "active_modalities": active_modalities,
            "normalized_weights": normalized_weights,
            "modality_breakdown": {m: modality_results[m] for m in active_modalities},
            "explanation": explanation
        }

    @staticmethod
    def _calculate_entropy(probs: Dict[str, float]) -> float:
        """Calculate Shannon entropy in bits."""
        ent = 0.0
        for p in probs.values():
            if p > 1e-6:
                ent -= p * np.log2(p)
        return float(ent)

    @staticmethod
    def _detect_conflicts(
        modality_results: Dict[str, Dict[str, Any]],
        active_modalities: List[str]
    ) -> (bool, Optional[str]):
        """
        Identify valence and emotion contradictions between modalities.
        Example: Vision expresses 'happy' while Audio expresses 'angry'.
        """
        predictions = {}
        for mod in active_modalities:
            dom = modality_results[mod].get("dominant_emotion", "neutral")
            predictions[mod] = dom

        # Check pairs
        conflicts = []
        mods = list(predictions.keys())
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                m1, m2 = mods[i], mods[j]
                e1, e2 = predictions[m1], predictions[m2]
                v1, v2 = VALENCE_MAP.get(e1, "neutral"), VALENCE_MAP.get(e2, "neutral")

                if v1 != v2 and "neutral" not in (v1, v2):
                    conflicts.append(
                        f"{m1.capitalize()} predicts '{e1}' ({v1}) while {m2.capitalize()} predicts '{e2}' ({v2})"
                    )

        if conflicts:
            return True, "; ".join(conflicts)
        return False, None

    @staticmethod
    def _build_explanation(
        dominant_emotion: str,
        confidence: float,
        active_modalities: List[str],
        weights: Dict[str, float],
        conflict_detected: bool,
        conflict_details: Optional[str],
        is_abstain: bool
    ) -> str:
        """Generate human-readable audit trail of the fusion decision."""
        mod_desc = ", ".join([f"{m} ({weights[m]*100:.0f}%)" for m in active_modalities])

        if is_abstain:
            return (
                f"Abstention triggered: Maximum fused confidence ({confidence*100:.1f}%) is below the "
                f"decision threshold, or strong cross-modal discordance was detected ({conflict_details}). "
                f"The system abstains from issuing an authoritative classification."
            )

        msg = f"Fused consensus across {mod_desc} points to '{dominant_emotion}' with {confidence*100:.1f}% confidence."
        if conflict_detected and conflict_details:
            msg += f" Note: Cross-modal conflict detected: {conflict_details}. The prediction reflects the quality-weighted compromise."
        return msg
