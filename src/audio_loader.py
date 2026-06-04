"""
audio_loader.py
---------------
Loads audio files (WAV, MP3, FLAC, OGG, M4A) into a normalised
numpy float32 array suitable for Whisper (expects 16-bit PCM-range floats).

Supported formats
-----------------
  .wav  — soundfile (fastest, no extra deps)
  .mp3  — audioread (ffmpeg / libav backend)
  .flac — soundfile
  .ogg  — soundfile
  .m4a  — audioread

Returns
-------
  AudioData(samples: np.ndarray[float32], sample_rate: int, duration: float, channels: int)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import audioread


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}


@dataclass
class AudioData:
    """Container for a loaded audio clip."""
    samples: np.ndarray     # float32, shape (num_samples,) — always mono after load
    sample_rate: int        # original sample rate in Hz
    duration: float         # seconds
    channels: int           # number of channels in original file
    file_path: str          # source file path


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded."""


def load(file_path: str | Path) -> AudioData:
    """
    Load an audio file and return an AudioData object.

    The returned samples are:
      - float32 in the range [-1.0, 1.0]
      - mono (averaged across channels if stereo/multi-channel)
      - at the ORIGINAL sample rate (resampling is Preprocessor's job)

    Parameters
    ----------
    file_path : str or Path
        Path to the audio file.

    Returns
    -------
    AudioData

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    AudioLoadError
        If the file exists but cannot be decoded.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Route to the appropriate loader
    if ext in {".wav", ".flac", ".ogg"}:
        return _load_soundfile(path)
    else:
        # .mp3, .m4a, .aac — use audioread which wraps ffmpeg/libav
        return _load_audioread(path)


def _load_soundfile(path: Path) -> AudioData:
    """Load formats natively handled by libsndfile (wav, flac, ogg)."""
    try:
        samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioLoadError(f"soundfile could not read '{path}': {exc}") from exc

    channels = samples.shape[1]
    # Collapse to mono by averaging channels
    mono = samples.mean(axis=1).astype(np.float32)
    duration = len(mono) / sr

    return AudioData(
        samples=mono,
        sample_rate=sr,
        duration=duration,
        channels=channels,
        file_path=str(path),
    )


def _load_audioread(path: Path) -> AudioData:
    """Load MP3/M4A using audioread (wraps ffmpeg/libav)."""
    try:
        with audioread.audio_open(str(path)) as f:
            sr = f.samplerate
            channels = f.channels
            # audioread yields raw 16-bit little-endian PCM blocks
            raw_blocks = []
            for block in f:
                raw_blocks.append(block)
    except Exception as exc:
        raise AudioLoadError(f"audioread could not decode '{path}': {exc}") from exc

    if not raw_blocks:
        raise AudioLoadError(f"'{path}' decoded to zero audio bytes — file may be corrupt.")

    # Concatenate raw bytes → int16 array
    raw_bytes = b"".join(raw_blocks)
    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

    # Normalise int16 range to float32 [-1.0, 1.0]
    pcm /= 32768.0

    # Reshape to (num_samples, num_channels) and mix to mono
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32)

    duration = len(pcm) / sr

    return AudioData(
        samples=pcm,
        sample_rate=sr,
        duration=duration,
        channels=channels,
        file_path=str(path),
    )


def get_info(file_path: str | Path) -> dict:
    """
    Return a lightweight info dict without fully loading the file.
    Useful for validation before committing to a full load.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    size_bytes = os.path.getsize(path)
    info = {
        "file": path.name,
        "format": ext.lstrip(".").upper(),
        "size_mb": round(size_bytes / 1_048_576, 2),
        "supported": ext in SUPPORTED_EXTENSIONS,
    }

    # Try to get sample rate without full decode (soundfile only)
    if ext in {".wav", ".flac", ".ogg"}:
        try:
            sf_info = sf.info(str(path))
            info["sample_rate"] = sf_info.samplerate
            info["channels"] = sf_info.channels
            info["duration_sec"] = round(sf_info.duration, 2)
        except Exception:
            pass

    return info