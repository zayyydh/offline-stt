"""
conftest.py
-----------
Pytest configuration and shared fixtures for offline-stt test suite.

Fixtures defined here are available to ALL test files without import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Ensure src/ is importable from any working directory
sys.path.insert(0, str(Path(__file__).parent))


# ── Shared audio fixtures ─────────────────────────────────────────────────────

def _sine_wave(
    freq: float = 440.0,
    duration: float = 1.0,
    sr: int = 16_000,
    amplitude: float = 0.5,
) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


@pytest.fixture(scope="session")
def sample_rate() -> int:
    return 16_000


@pytest.fixture(scope="session")
def sine_1s_16k() -> np.ndarray:
    """1-second 440 Hz sine wave at 16 kHz — used across multiple test modules."""
    return _sine_wave(440, 1.0, 16_000, 0.5)


@pytest.fixture(scope="session")
def sine_3s_44k() -> np.ndarray:
    """3-second 440 Hz sine wave at 44.1 kHz — tests resampling."""
    return _sine_wave(440, 3.0, 44_100, 0.5)


@pytest.fixture
def wav_file_16k(tmp_path) -> Path:
    """Write a temporary mono 16kHz WAV file and return its path."""
    samples = _sine_wave(440, 2.0, 16_000, 0.6)
    p = tmp_path / "test_16k.wav"
    sf.write(str(p), samples, 16_000)
    return p


@pytest.fixture
def wav_file_44k(tmp_path) -> Path:
    """Write a temporary mono 44.1kHz WAV file and return its path."""
    samples = _sine_wave(440, 2.0, 44_100, 0.6)
    p = tmp_path / "test_44k.wav"
    sf.write(str(p), samples, 44_100)
    return p


@pytest.fixture
def wav_file_stereo(tmp_path) -> Path:
    """Write a temporary stereo 44.1kHz WAV file and return its path."""
    left  = _sine_wave(440, 1.0, 44_100, 0.5)
    right = _sine_wave(880, 1.0, 44_100, 0.5)
    stereo = np.stack([left, right], axis=1)
    p = tmp_path / "test_stereo.wav"
    sf.write(str(p), stereo, 44_100)
    return p


@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """All-zeros WAV — tests near-silent handling."""
    samples = np.zeros(16_000, dtype=np.float32)
    p = tmp_path / "silent.wav"
    sf.write(str(p), samples, 16_000)
    return p


@pytest.fixture
def noisy_audio_16k() -> np.ndarray:
    """Gaussian noise at 16 kHz — tests denoise pipeline."""
    rng = np.random.default_rng(seed=0)
    signal  = _sine_wave(440, 2.0, 16_000, 0.5)
    noise   = rng.standard_normal(len(signal)).astype(np.float32) * 0.1
    return signal + noise