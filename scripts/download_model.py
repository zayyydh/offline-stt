"""
scripts/download_model.py
--------------------------
Downloads a Whisper model from HuggingFace into the local ./models/ directory.

Usage
-----
  python scripts/download_model.py                     # downloads base.en (default)
  python scripts/download_model.py --model small.en
  python scripts/download_model.py --model large-v3
  python scripts/download_model.py --list

Model sizes (approximate)
--------------------------
  tiny.en   ~39 MB    CPU: ~3s/min audio
  base.en   ~74 MB    CPU: ~8s/min audio   ← RECOMMENDED
  small.en  ~244 MB   CPU: ~20s/min audio
  medium.en ~769 MB   CPU: ~50s/min audio
  large-v3  ~1.5 GB   CPU: ~90s/min audio

All models are downloaded from:
  https://huggingface.co/Systran/faster-whisper-<model>
They are saved to <project_root>/models/<model_name>/
After downloading, the app works fully offline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Project root is one level above scripts/
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

AVAILABLE_MODELS = {
    "tiny.en":   "~39 MB  — very fast, lower accuracy",
    "base.en":   "~74 MB  — RECOMMENDED: good balance of speed and accuracy",
    "small.en":  "~244 MB — better accuracy, slower on CPU",
    "medium.en": "~769 MB — high accuracy, significantly slower on CPU",
    "large-v3":  "~1.5 GB — best accuracy, slow on CPU, needs 4GB+ RAM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Whisper model for offline use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/download_model.py
  python scripts/download_model.py --model small.en
  python scripts/download_model.py --list
        """,
    )
    parser.add_argument(
        "--model", "-m",
        default="base.en",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to download (default: base.en)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help=f"Directory to save models (default: {MODELS_DIR})",
    )
    return parser.parse_args()


def list_models():
    print("\nAvailable Whisper models for offline-stt")
    print("─" * 55)
    for name, desc in AVAILABLE_MODELS.items():
        local_path = MODELS_DIR / name
        status = "✓ already downloaded" if local_path.exists() else "  not downloaded"
        print(f"  {name:<14} {desc}")
        print(f"  {'':14} {status}")
    print(f"\nModels directory: {MODELS_DIR}\n")


def check_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        print("ERROR: faster-whisper is not installed.")
        print("Run: pip install faster-whisper")
        return False


def download_model(model_name: str, models_dir: Path) -> bool:
    """
    Download a Whisper model using faster-whisper's built-in downloader.

    faster-whisper downloads from HuggingFace Hub into the specified directory.
    On subsequent runs with the same model, it verifies checksums and skips
    the download if files are already present.

    Returns True on success, False on failure.
    """
    from faster_whisper import WhisperModel

    target_path = models_dir / model_name
    models_dir.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and any(target_path.iterdir()):
        print(f"\nModel '{model_name}' already exists at: {target_path}")
        print("Verifying files...\n")

    print(f"Downloading model: {model_name}")
    print(f"Destination      : {target_path}")
    print(f"Size             : {AVAILABLE_MODELS.get(model_name, 'unknown')}")
    print("\nThis may take a few minutes on first run...\n")

    t0 = time.perf_counter()
    try:
        # Loading the model triggers the download if not cached
        # We immediately unload after to free RAM
        print("Connecting to HuggingFace Hub...")
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(models_dir),
        )
        del model  # release RAM immediately

        elapsed = time.perf_counter() - t0
        print(f"\n✓ Model '{model_name}' downloaded successfully in {elapsed:.0f}s")
        print(f"  Saved to: {target_path}\n")
        return True

    except Exception as exc:
        print(f"\n✗ Download failed: {exc}", file=sys.stderr)
        print("Check your internet connection and try again.", file=sys.stderr)
        return False


def verify_model(model_name: str, models_dir: Path) -> bool:
    """Check that the expected model files are present."""
    model_path = models_dir / model_name
    if not model_path.exists():
        return False

    # faster-whisper models always contain model.bin and config.json
    required = ["model.bin", "config.json"]
    present = [f.name for f in model_path.iterdir()]
    missing = [r for r in required if r not in present]

    if missing:
        print(f"WARNING: Model directory exists but is incomplete.")
        print(f"  Missing files: {', '.join(missing)}")
        print(f"  Re-run this script to re-download.\n")
        return False

    print(f"✓ Verification passed — all required files present.")
    return True


def main():
    args = parse_args()

    if args.list:
        list_models()
        return

    if not check_faster_whisper():
        sys.exit(1)

    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR

    success = download_model(args.model, models_dir)
    if not success:
        sys.exit(1)

    verify_model(args.model, models_dir)

    print("─" * 50)
    print("Setup complete. You can now run:")
    print(f"  python main.py --input your_audio.wav --model {args.model}")
    print()


if __name__ == "__main__":
    main()