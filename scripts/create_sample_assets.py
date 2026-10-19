"""
Generate Lightweight Sample Assets for Immediate Offline Testing and Demonstration.
Creates sample facial images, acoustic wav tones, and labeled emotion sentences.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from PIL import Image, ImageDraw
import soundfile as sf

from src.config import DATA_SAMPLE_DIR, AUDIO_TARGET_SR

SAMPLE_IMAGES_DIR = DATA_SAMPLE_DIR / "images"
SAMPLE_AUDIO_DIR = DATA_SAMPLE_DIR / "audio"
SAMPLE_TEXT_DIR = DATA_SAMPLE_DIR / "texts"


def generate_sample_images():
    """Create lightweight synthetic face-like portrait images."""
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Image 1: Smiling face with yellow background
    img1 = Image.new("RGB", (200, 200), color=(245, 245, 245))
    draw = ImageDraw.Draw(img1)
    # Head
    draw.ellipse((40, 30, 160, 150), fill=(255, 220, 185), outline=(100, 100, 100), width=2)
    # Eyes
    draw.ellipse((65, 65, 80, 80), fill=(40, 40, 40))
    draw.ellipse((120, 65, 135, 80), fill=(40, 40, 40))
    # Smile
    draw.arc((70, 85, 130, 125), start=0, end=180, fill=(180, 40, 40), width=3)
    img1.save(SAMPLE_IMAGES_DIR / "sample_happy_face.jpg", quality=90)

    # Image 2: Sad face
    img2 = Image.new("RGB", (200, 200), color=(240, 240, 245))
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse((40, 30, 160, 150), fill=(240, 215, 190), outline=(100, 100, 100), width=2)
    draw2.ellipse((65, 70, 80, 85), fill=(40, 40, 40))
    draw2.ellipse((120, 70, 135, 85), fill=(40, 40, 40))
    # Frown
    draw2.arc((70, 105, 130, 140), start=180, end=360, fill=(80, 80, 120), width=3)
    img2.save(SAMPLE_IMAGES_DIR / "sample_sad_face.jpg", quality=90)

    print(f"Generated sample images in {SAMPLE_IMAGES_DIR}")


def generate_sample_audio():
    """Create short WAV audio samples with distinct acoustic frequencies and envelopes."""
    SAMPLE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    sr = AUDIO_TARGET_SR
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Happy Audio: High pitch (350Hz) with rapid vibrato and high energy
    vibrato = np.sin(2 * np.pi * 6 * t) * 30
    waveform_happy = 0.35 * np.sin(2 * np.pi * (350 + vibrato) * t)
    # Add harmonics
    waveform_happy += 0.15 * np.sin(2 * np.pi * 2 * (350 + vibrato) * t)
    sf.write(SAMPLE_AUDIO_DIR / "sample_happy_tone.wav", waveform_happy.astype(np.float32), sr)

    # 2. Sad Audio: Low pitch (95Hz), declining amplitude envelope, low energy
    envelope = np.linspace(0.20, 0.04, len(t))
    waveform_sad = envelope * np.sin(2 * np.pi * 95 * t)
    sf.write(SAMPLE_AUDIO_DIR / "sample_sad_tone.wav", waveform_sad.astype(np.float32), sr)

    # 3. Angry Audio: Loud, energetic with harsh clipping distortion and high ZCR
    waveform_angry = 0.50 * np.sin(2 * np.pi * 220 * t) + 0.30 * np.sin(2 * np.pi * 440 * t)
    # Add noise
    noise = np.random.normal(0, 0.05, len(t))
    waveform_angry = np.clip(waveform_angry + noise, -0.6, 0.6)
    sf.write(SAMPLE_AUDIO_DIR / "sample_angry_tone.wav", waveform_angry.astype(np.float32), sr)

    # 4. Neutral Audio: Calm, steady 130Hz tone with gentle onset
    window = np.hanning(len(t))
    waveform_neutral = 0.20 * np.sin(2 * np.pi * 130 * t) * window
    sf.write(SAMPLE_AUDIO_DIR / "sample_neutral_tone.wav", waveform_neutral.astype(np.float32), sr)

    print(f"Generated sample audio clips in {SAMPLE_AUDIO_DIR}")


def generate_sample_texts():
    """Create representative text inputs across canonical emotion categories."""
    SAMPLE_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    samples = [
        {"text": "I am absolutely thrilled and overjoyed with this incredible milestone!", "expected_emotion": "happy"},
        {"text": "I feel so devastated, alone, and completely heartbroken today.", "expected_emotion": "sad"},
        {"text": "I am furious that our work was destroyed without any explanation!", "expected_emotion": "angry"},
        {"text": "I am terrified and panicking about what will happen next.", "expected_emotion": "fear"},
        {"text": "Wow, I was totally astonished and stunned by the sudden turn of events!", "expected_emotion": "surprise"},
        {"text": "That spoiled food smells completely repulsive, gross, and nauseating.", "expected_emotion": "disgust"},
        {"text": "The meeting is scheduled tomorrow morning at 10 AM in room four.", "expected_emotion": "neutral"}
    ]

    with open(SAMPLE_TEXT_DIR / "sample_benchmark_texts.json", "w") as f:
        json.dump(samples, f, indent=4)

    print(f"Generated sample texts in {SAMPLE_TEXT_DIR}")


if __name__ == "__main__":
    generate_sample_images()
    generate_sample_audio()
    generate_sample_texts()
    print("All sample assets created successfully!")
