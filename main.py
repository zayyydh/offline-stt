"""
main.py — Offline English Speech-to-Text CLI
=============================================

SynoraAI Labs internship assignment prototype.
Transcribes audio files (or live mic) completely offline using
faster-whisper (CTranslate2 backend, INT8 quantised).

Usage examples
--------------
  # Basic file transcription
  python main.py --input audio/sample.wav

  # With timestamps and noise reduction
  python main.py --input audio/interview.mp3 --timestamps --denoise

  # Save as JSON with all metadata
  python main.py --input audio/sample.wav --format json --output outputs/result.json

  # List available models and audio devices
  python main.py --list-models
  python main.py --list-devices

  # Live microphone recording (5 seconds)
  python main.py --mic --duration 5

  # Use a specific model and save as SRT subtitles
  python main.py --input lecture.mp3 --model small.en --format srt

Full help
---------
  python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Configure logging BEFORE importing src modules ────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("offline-stt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offline-stt",
        description=(
            "Offline English Speech-to-Text — powered by faster-whisper.\n"
            "Runs completely without internet. No cloud APIs.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --input sample.wav
  python main.py --input interview.mp3 --timestamps --denoise
  python main.py --input lecture.wav --model small.en --format srt
  python main.py --mic --duration 10
  python main.py --list-models
  python main.py --list-devices
        """,
    )

    # ── Input source ──────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Path to audio file (.wav .mp3 .flac .ogg .m4a)",
    )
    input_group.add_argument(
        "--mic",
        action="store_true",
        help="Record from system microphone instead of a file",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", "-m",
        default="base.en",
        metavar="NAME",
        help=(
            "Whisper model name (default: base.en).\n"
            "Options: tiny.en, base.en, small.en, medium.en, large-v3\n"
            "Larger = more accurate but slower."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Directory containing downloaded model weights (default: ./models/)",
    )

    # ── Preprocessing ─────────────────────────────────────────────────────
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply noise reduction before transcription (adds ~1s latency)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip amplitude normalisation (not recommended for quiet recordings)",
    )
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Trim leading/trailing silence before transcription",
    )

    # ── Transcription options ──────────────────────────────────────────────
    parser.add_argument(
        "--language",
        default="en",
        metavar="LANG",
        help="Language code (default: en). Use 'auto' to auto-detect.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        metavar="N",
        help="Beam search width (default: 5). Use 1 for greedy/fastest.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filter (slower; useful if VAD cuts real speech)",
    )

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Print segment-level timestamps in output",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "json", "srt"],
        default=None,
        metavar="FMT",
        help="Save output to file. Format: txt, json, or srt",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Output file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Print transcript only — suppress metadata and segment table",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    # ── Microphone options ─────────────────────────────────────────────────
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Microphone recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        metavar="N",
        help="Microphone device index (use --list-devices to find yours)",
    )

    # ── Info commands ──────────────────────────────────────────────────────
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List downloaded models in ./models/ and exit",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--info",
        metavar="FILE",
        help="Print audio file info without transcribing",
    )

    return parser


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_list_models(args):
    from src.transcribe import DEFAULT_MODEL_DIR
    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR

    known_models = {
        "tiny.en":   ("39 MB",  "~3s/min audio",  "~80% accuracy"),
        "base.en":   ("74 MB",  "~8s/min audio",  "~93% accuracy — RECOMMENDED"),
        "small.en":  ("244 MB", "~20s/min audio", "~96% accuracy"),
        "medium.en": ("769 MB", "~50s/min audio", "~98% accuracy"),
        "large-v3":  ("1.5 GB", "~90s/min audio", "~99% accuracy"),
    }

    print("\nAvailable Whisper models")
    print("─" * 56)
    print(f"{'Model':<14} {'Size':<8} {'CPU Speed':<18} {'Accuracy'}")
    print("─" * 56)
    for name, (size, speed, acc) in known_models.items():
        local_path = model_dir / name
        status = "✓ downloaded" if local_path.exists() else "  not downloaded"
        print(f"  {name:<12} {size:<8} {speed:<18} {acc}")
        print(f"  {'':12} {status}")
    print(f"\nModels directory: {model_dir}")
    print("To download: python scripts/download_model.py --model base.en\n")


def cmd_list_devices(args):
    from src.mic_capture import MicCapture
    devices = MicCapture.list_devices()

    if not devices:
        print("\nNo input devices found (or PyAudio not installed).")
        print("Install PyAudio: pip install pyaudio\n")
        return

    print("\nAvailable audio input devices")
    print("─" * 50)
    for d in devices:
        print(f"  {d}")
    default = MicCapture.get_default_device()
    if default:
        print(f"\nDefault: {default}")
    print()


def cmd_info(args):
    from src.audio_loader import get_info
    try:
        info = get_info(args.info)
        print(f"\nFile   : {info['file']}")
        print(f"Format : {info['format']}")
        print(f"Size   : {info['size_mb']} MB")
        if "sample_rate" in info:
            print(f"Rate   : {info['sample_rate']} Hz")
        if "channels" in info:
            print(f"Channels: {info['channels']}")
        if "duration_sec" in info:
            print(f"Duration: {info['duration_sec']} s")
        print()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_transcribe_file(args):
    from src.audio_loader import load as load_audio, AudioLoadError
    from src.preprocessor import Preprocessor, PreprocessConfig
    from src.transcribe import TranscribeEngine
    from src.output_handler import save_transcript, print_result

    input_path = Path(args.input)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"Loading  : {input_path.name}")
    try:
        audio_data = load_audio(input_path)
    except FileNotFoundError:
        print(f"Error: File not found — '{input_path}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AudioLoadError as exc:
        print(f"Error loading audio: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Duration : {audio_data.duration:.1f} s  |  {audio_data.sample_rate} Hz  |  {audio_data.channels}ch")

    # ── Preprocess ────────────────────────────────────────────────────────
    cfg = PreprocessConfig(
        denoise=args.denoise,
        normalize=not args.no_normalize,
        trim_silence=args.trim_silence,
    )
    preprocessor = Preprocessor(cfg)

    print(f"Preprocessing: {', '.join([k for k, v in [('denoise', args.denoise), ('normalize', not args.no_normalize), ('trim_silence', args.trim_silence)] if v])  or 'normalise only'}")
    prep_result = preprocessor.process(audio_data.samples, audio_data.sample_rate)

    # ── Transcribe ────────────────────────────────────────────────────────
    engine = TranscribeEngine(
        model_name=args.model,
        model_dir=args.model_dir,
    )
    lang = None if args.language == "auto" else args.language

    print(f"Model    : {args.model}  (loading...)")
    t0 = time.perf_counter()
    result = engine.transcribe(
        prep_result.samples,
        sample_rate=prep_result.sample_rate,
        language=lang,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
    )
    elapsed = time.perf_counter() - t0
    print(f"Done in  : {elapsed:.1f} s  (RTF={result.rtf:.2f}x)\n")

    # ── Output ────────────────────────────────────────────────────────────
    if args.quiet:
        print(result.transcript)
    else:
        print_result(
            result,
            show_segments=args.timestamps,
            show_metadata=True,
        )

    if args.format:
        saved_path = save_transcript(
            result,
            output_path=args.output,
            fmt=args.format,
            source_file=str(input_path),
        )
        print(f"Saved → {saved_path}")


def cmd_transcribe_mic(args):
    from src.mic_capture import MicCapture
    from src.preprocessor import Preprocessor, PreprocessConfig
    from src.transcribe import TranscribeEngine
    from src.output_handler import save_transcript, print_result

    print(f"\nRecording {args.duration}s from microphone (device={args.device_index or 'default'})...")
    print("Speak now.\n")

    mic = MicCapture(
        duration=args.duration,
        device_index=args.device_index,
    )

    try:
        samples, sr = mic.record()
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Microphone error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Recording complete. Transcribing...\n")

    cfg = PreprocessConfig(
        denoise=args.denoise,
        normalize=not args.no_normalize,
    )
    prep_result = Preprocessor(cfg).process(samples, sr)

    engine = TranscribeEngine(model_name=args.model, model_dir=args.model_dir)
    lang = None if args.language == "auto" else args.language

    result = engine.transcribe(
        prep_result.samples,
        sample_rate=prep_result.sample_rate,
        language=lang,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
    )

    if args.quiet:
        print(result.transcript)
    else:
        print_result(result, show_segments=args.timestamps, show_metadata=True)

    if args.format:
        saved_path = save_transcript(
            result,
            output_path=args.output,
            fmt=args.format,
            source_file="mic_recording",
        )
        print(f"Saved → {saved_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Info commands (no transcription)
    if args.list_models:
        cmd_list_models(args)
        return

    if args.list_devices:
        cmd_list_devices(args)
        return

    if args.info:
        cmd_info(args)
        return

    # Transcription commands
    if args.mic:
        cmd_transcribe_mic(args)
    elif args.input:
        cmd_transcribe_file(args)
    else:
        parser.print_help()
        print("\nProvide --input FILE or --mic to transcribe audio.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()