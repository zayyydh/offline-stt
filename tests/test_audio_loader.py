"""
tests/test_audio_loader.py
--------------------------
Unit tests for src/audio_loader.py

Tests cover:
  - WAV loading (soundfile path)
  - Stereo → mono downmix
  - Short clip edge case (< 0.5s)
  - File not found raises FileNotFoundError
  - Unsupported extension raises ValueError
  - get_info() returns correct metadata
  - Output dtype is always float32
  - Output values are in [-1.0, 1.0]
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Ensure src/ is importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_loader import load, get_info, AudioData, AudioLoadError, SUPPORTED_EXTENSIONS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_wav(path: str, samples: np.ndarray, sr: int = 16_000):
    """Helper: write a numpy array to a WAV file."""
    sf.write(path, samples, sr)


def _sine(freq: float = 440.0, duration: float = 1.0, sr: int = 16_000) -> np.ndarray:
    """Generate a mono sine wave as float32."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)


@pytest.fixture
def mono_wav(tmp_path):
    """Temporary 1-second mono 16kHz WAV file."""
    samples = _sine(440, 1.0, 16_000)
    p = tmp_path / "mono.wav"
    _write_wav(str(p), samples, 16_000)
    return p, samples


@pytest.fixture
def stereo_wav(tmp_path):
    """Temporary 1-second stereo 44.1kHz WAV file."""
    left  = _sine(440, 1.0, 44_100)
    right = _sine(880, 1.0, 44_100)
    stereo = np.stack([left, right], axis=1)
    p = tmp_path / "stereo.wav"
    sf.write(str(p), stereo, 44_100)
    return p


@pytest.fixture
def short_wav(tmp_path):
    """Temporary very short WAV file (0.05s — edge case)."""
    samples = _sine(440, 0.05, 16_000)
    p = tmp_path / "short.wav"
    _write_wav(str(p), samples, 16_000)
    return p


@pytest.fixture
def silent_wav(tmp_path):
    """All-zeros WAV file — tests near-silent handling."""
    samples = np.zeros(16_000, dtype=np.float32)
    p = tmp_path / "silent.wav"
    _write_wav(str(p), samples, 16_000)
    return p


# ── Tests: load() ─────────────────────────────────────────────────────────────

class TestLoad:

    def test_returns_audio_data(self, mono_wav):
        path, _ = mono_wav
        result = load(path)
        assert isinstance(result, AudioData)

    def test_dtype_is_float32(self, mono_wav):
        path, _ = mono_wav
        result = load(path)
        assert result.samples.dtype == np.float32

    def test_values_in_range(self, mono_wav):
        path, _ = mono_wav
        result = load(path)
        assert np.all(result.samples >= -1.0)
        assert np.all(result.samples <= 1.0)

    def test_mono_shape(self, mono_wav):
        path, _ = mono_wav
        result = load(path)
        assert result.samples.ndim == 1

    def test_stereo_downmixed_to_mono(self, stereo_wav):
        result = load(stereo_wav)
        assert result.samples.ndim == 1
        assert result.channels == 2

    def test_sample_rate_preserved(self, stereo_wav):
        result = load(stereo_wav)
        assert result.sample_rate == 44_100

    def test_duration_accuracy(self, mono_wav):
        path, _ = mono_wav
        result = load(path)
        assert abs(result.duration - 1.0) < 0.01  # within 10ms

    def test_short_clip(self, short_wav):
        """Edge case: very short clip should not raise."""
        result = load(short_wav)
        assert len(result.samples) > 0
        assert result.duration < 0.1

    def test_silent_clip(self, silent_wav):
        """All-zeros clip should load without error."""
        result = load(silent_wav)
        assert result.samples.dtype == np.float32
        assert np.allclose(result.samples, 0.0)

    def test_file_path_recorded(self, mono_wav):
        path, _ = mono_wav
        result = load(path)
        assert Path(result.file_path) == path

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.wav")

    def test_unsupported_extension(self, tmp_path):
        fake = tmp_path / "audio.xyz"
        fake.write_text("not audio")
        with pytest.raises(ValueError, match="Unsupported format"):
            load(fake)

    def test_accepts_string_path(self, mono_wav):
        path, _ = mono_wav
        result = load(str(path))  # str not Path
        assert isinstance(result, AudioData)

    def test_accepts_path_object(self, mono_wav):
        path, _ = mono_wav
        result = load(path)       # Path object
        assert isinstance(result, AudioData)

    def test_flac_loads(self, tmp_path):
        """FLAC loading via soundfile."""
        samples = _sine(440, 0.5, 16_000)
        p = tmp_path / "test.flac"
        sf.write(str(p), samples, 16_000)
        result = load(p)
        assert isinstance(result, AudioData)
        assert result.sample_rate == 16_000

    def test_ogg_loads(self, tmp_path):
        """OGG/Vorbis loading via soundfile."""
        samples = _sine(440, 0.5, 16_000)
        p = tmp_path / "test.ogg"
        sf.write(str(p), samples, 16_000, format="OGG", subtype="VORBIS")
        result = load(p)
        assert isinstance(result, AudioData)


# ── Tests: get_info() ─────────────────────────────────────────────────────────

class TestGetInfo:

    def test_returns_dict(self, mono_wav):
        path, _ = mono_wav
        info = get_info(path)
        assert isinstance(info, dict)

    def test_required_keys(self, mono_wav):
        path, _ = mono_wav
        info = get_info(path)
        assert "file" in info
        assert "format" in info
        assert "size_mb" in info
        assert "supported" in info

    def test_wav_metadata(self, mono_wav):
        path, _ = mono_wav
        info = get_info(path)
        assert info["format"] == "WAV"
        assert info["sample_rate"] == 16_000
        assert info["channels"] == 1
        assert abs(info["duration_sec"] - 1.0) < 0.05
        assert info["supported"] is True

    def test_unsupported_extension_flagged(self, tmp_path):
        fake = tmp_path / "audio.xyz"
        fake.write_bytes(b"garbage")
        info = get_info(fake)
        assert info["supported"] is False

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_info(tmp_path / "missing.wav")


# ── Tests: SUPPORTED_EXTENSIONS ───────────────────────────────────────────────

class TestSupportedExtensions:

    def test_includes_common_formats(self):
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            assert ext in SUPPORTED_EXTENSIONS

    def test_all_lowercase(self):
        for ext in SUPPORTED_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext!r} should be lowercase"

    def test_all_start_with_dot(self):
        for ext in SUPPORTED_EXTENSIONS:
            assert ext.startswith("."), f"Extension {ext!r} should start with '.'"