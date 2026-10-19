"""
Reproducible Dataset Acquisition and Ingestion Script.
Supports FER-2013 (vision), RAVDESS (audio), and GoEmotions (text).
Includes provenance metadata, license documentation, and local folder ingestion.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import urllib.request
import zipfile
import json
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SAMPLE_DIR

# Dataset Provenance and License Registry
DATASET_METADATA = {
    "fer2013": {
        "name": "Facial Expression Recognition 2013 (FER-2013)",
        "task": "Facial Expression Classification (7 Ekman emotions)",
        "source": "ICML 2013 Workshop on Challenges in Representation Learning (Goodfellow et al.)",
        "license": "Open Database License (ODbL) / Public Research Use",
        "labels": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        "size_mb": 300,
        "notes": "Requires Kaggle account authentication for direct tarball download (kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge)."
    },
    "ravdess": {
        "name": "Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)",
        "task": "Speech Emotion Recognition (Acoustic audio)",
        "source": "Zenodo (Livingstone & Russo, 2018)",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "url": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
        "labels": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "size_mb": 1050,
        "notes": "Direct open-access research repository download hosted on Zenodo."
    },
    "goemotions": {
        "name": "Google GoEmotions",
        "task": "Fine-Grained Text Emotion Classification",
        "source": "Google Research / GitHub (Demszky et al., ACL 2020)",
        "license": "Apache License 2.0",
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv",
        "labels": ["28 emotion categories mapped to positive, negative, and ambiguous valences"],
        "size_mb": 15,
        "notes": "Direct open download from official Google Research GitHub repository."
    }
}


def print_status():
    """Print current dataset setup status."""
    print("=" * 70)
    print("DATASET ACQUISITION AND LOCAL STORAGE STATUS")
    print("=" * 70)
    print(f"Data Root Directory: {PROJECT_ROOT / 'data'}")
    print(f"  - Sample Assets (Instant Offline Inference): {DATA_SAMPLE_DIR} "
          f"({'EXISTS' if DATA_SAMPLE_DIR.exists() else 'MISSING'})")
    print(f"  - Raw Downloads: {DATA_RAW_DIR} "
          f"({'EXISTS' if DATA_RAW_DIR.exists() else 'MISSING'})")
    print(f"  - Processed Datasets: {DATA_PROCESSED_DIR} "
          f"({'EXISTS' if DATA_PROCESSED_DIR.exists() else 'MISSING'})")

    print("\nCandidate Datasets Provenance & Licensing:")
    for key, meta in DATASET_METADATA.items():
        print(f"\n[{key.upper()}] - {meta['name']}")
        print(f"  Task:     {meta['task']}")
        print(f"  Source:   {meta['source']}")
        print(f"  License:  {meta['license']}")
        print(f"  Approx:   ~{meta['size_mb']} MB")
        print(f"  Notes:    {meta['notes']}")
    print("=" * 70)
    print("\nNOTE: The platform operates fully in inference and evaluation mode using")
    print("the pretrained model checkpoints and sample assets without requiring large downloads.")
    print("=" * 70)


def download_progress_hook(block_num, block_size, total_size):
    """Console progress bar for downloads."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()


def download_goemotions():
    """Download Google GoEmotions open dataset subset (Apache 2.0)."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_file = DATA_RAW_DIR / "goemotions_train.tsv"

    print("\nDownloading GoEmotions dataset (Apache 2.0)...")
    url = DATASET_METADATA["goemotions"]["url"]

    try:
        urllib.request.urlretrieve(url, str(target_file), reporthook=download_progress_hook)
        print(f"\nSuccessfully downloaded GoEmotions to {target_file} ({target_file.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"\nFailed to download GoEmotions: {e}")


def import_local_dataset(source_path: str, dataset_type: str):
    """
    Import manually downloaded dataset archives or folders.
    Useful when user has downloaded FER-2013 or RAVDESS from Kaggle/Zenodo manually.
    """
    src = Path(source_path)
    if not src.exists():
        print(f"Error: Source file or folder '{src}' does not exist.")
        return

    dest_dir = DATA_RAW_DIR / dataset_type
    dest_dir.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".zip":
        print(f"Extracting zip archive {src} into {dest_dir}...")
        with zipfile.ZipFile(src, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete!")
    else:
        print(f"Dataset files registered in {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Dataset Acquisition Manager")
    parser.add_argument(
        "--action",
        choices=["status", "download-goemotions", "import-local"],
        default="status",
        help="Action to perform"
    )
    parser.add_argument("--path", type=str, help="Local file path for import-local action")
    parser.add_argument("--type", type=str, choices=["fer2013", "ravdess", "goemotions"], help="Dataset type for import-local")

    args = parser.parse_args()

    if args.action == "status":
        print_status()
    elif args.action == "download-goemotions":
        download_goemotions()
    elif args.action == "import-local":
        if not args.path or not args.type:
            print("Error: --path and --type are required for import-local.")
            sys.exit(1)
        import_local_dataset(args.path, args.type)


if __name__ == "__main__":
    main()
