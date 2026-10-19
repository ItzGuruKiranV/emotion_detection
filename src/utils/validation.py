"""
Input Validation and Integrity Checks for Multimodal Emotion Analysis.
"""

from pathlib import Path
from typing import Union, Tuple
import io
from PIL import Image

from src.config import (
    IMAGE_MAX_SIZE_MB,
    SUPPORTED_IMAGE_FORMATS,
    AUDIO_MAX_SIZE_MB,
    SUPPORTED_AUDIO_FORMATS,
    AUDIO_MIN_DURATION_SEC,
    AUDIO_MAX_DURATION_SEC,
    TEXT_MIN_LENGTH,
    TEXT_MAX_LENGTH,
)


class ValidationError(Exception):
    """Raised when input fails domain or format validation."""
    pass


def validate_image_file(file_or_bytes: Union[str, Path, bytes, io.BytesIO]) -> Image.Image:
    """
    Validate that the provided input is a valid, readable image within limits.
    Returns a verified PIL Image in RGB format.
    """
    try:
        if isinstance(file_or_bytes, (str, Path)):
            p = Path(file_or_bytes)
            if not p.exists():
                raise ValidationError(f"Image file does not exist: {p}")
            suffix = p.suffix.lower()
            if suffix not in SUPPORTED_IMAGE_FORMATS:
                raise ValidationError(
                    f"Unsupported image format '{suffix}'. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
                )
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > IMAGE_MAX_SIZE_MB:
                raise ValidationError(
                    f"Image size ({size_mb:.2f} MB) exceeds maximum allowed limit ({IMAGE_MAX_SIZE_MB} MB)"
                )
            img = Image.open(p)
        elif isinstance(file_or_bytes, bytes):
            size_mb = len(file_or_bytes) / (1024 * 1024)
            if size_mb > IMAGE_MAX_SIZE_MB:
                raise ValidationError(
                    f"Image size ({size_mb:.2f} MB) exceeds maximum limit ({IMAGE_MAX_SIZE_MB} MB)"
                )
            img = Image.open(io.BytesIO(file_or_bytes))
        elif isinstance(file_or_bytes, io.BytesIO):
            file_or_bytes.seek(0, 2)
            size_mb = file_or_bytes.tell() / (1024 * 1024)
            file_or_bytes.seek(0)
            if size_mb > IMAGE_MAX_SIZE_MB:
                raise ValidationError(
                    f"Image size ({size_mb:.2f} MB) exceeds maximum limit ({IMAGE_MAX_SIZE_MB} MB)"
                )
            img = Image.open(file_or_bytes)
        else:
            raise ValidationError("Invalid image input type provided.")

        img.verify()
        # Re-open after verify() because verify() invalidates the image stream in PIL
        if isinstance(file_or_bytes, (str, Path)):
            img = Image.open(file_or_bytes).convert("RGB")
        elif isinstance(file_or_bytes, bytes):
            img = Image.open(io.BytesIO(file_or_bytes)).convert("RGB")
        else:
            file_or_bytes.seek(0)
            img = Image.open(file_or_bytes).convert("RGB")

        if img.width < 10 or img.height < 10:
            raise ValidationError(f"Image dimensions ({img.width}x{img.height}) are too small for facial analysis.")

        return img

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Failed to decode image file: {str(e)}") from e


def validate_text_input(text: str) -> str:
    """
    Validate and clean input text.
    """
    if not isinstance(text, str):
        raise ValidationError("Text input must be a string.")

    cleaned = text.strip()
    if len(cleaned) < TEXT_MIN_LENGTH:
        raise ValidationError(f"Text too short. Minimum length is {TEXT_MIN_LENGTH} characters.")
    if len(cleaned) > TEXT_MAX_LENGTH:
        raise ValidationError(
            f"Text too long ({len(cleaned)} chars). Maximum allowed is {TEXT_MAX_LENGTH} characters."
        )

    return cleaned


def validate_audio_file(file_path: Union[str, Path]) -> Path:
    """
    Validate that an audio file exists, has a supported format, and does not exceed size limits.
    """
    p = Path(file_path)
    if not p.exists():
        raise ValidationError(f"Audio file does not exist: {p}")

    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise ValidationError(
            f"Unsupported audio format '{suffix}'. Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > AUDIO_MAX_SIZE_MB:
        raise ValidationError(
            f"Audio file size ({size_mb:.2f} MB) exceeds maximum limit ({AUDIO_MAX_SIZE_MB} MB)"
        )

    return p
