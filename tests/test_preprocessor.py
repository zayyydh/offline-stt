"""
tests/test_preprocessor.py
--------------------------
Unit tests for src/preprocessor.py

Tests cover:
  - Resampling correctness (44.1kHz → 16kHz)
  - Output always at target sample rate
  - Output dtype always float32
  - Normalisation brings peak to 1.0
  - Normalisation skips near-silent clips
  - Silence trimming removes leading/trailing zeros
  - PreprocessConfig defaults
  - steps_applied list is populated correctly
  - get_rms_level() and get_snr_estimate() utilities
  - Edge cases: already 16kHz, very short clips
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import (
    Preprocessor,
    PreprocessConfig,
    PreprocessResult,
    TARGET_SAMPLE_RATE,
    get_rms_level,
    get_snr_estimate,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sine(freq: float = 440.0, duration: float = 1.0, sr: int = 44_100, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _noisy(duration: float = 1.0, sr: int = 16_000) -> np.ndarray:
    """Pure Gaussian noise."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.1


def _padded_silence(inner_dur: float = 0.5, pad_dur: float = 0.2, sr: int = 16_000) -> np.ndarray:
    """A signal padded with silence on both ends."""
    silence = np.zeros(int(sr * pad_dur), dtype=np.float32)
    signal  = _sine(440, inner_dur, sr, amp=0.6)
    return np.concatenate([silence, signal, silence])


# ── Tests: PreprocessConfig defaults ─────────────────────────────────────────

class TestPreprocessConfig:

    def test_default_target_sr(self):
        cfg = PreprocessConfig()
        assert cfg.target_sr == 16_000

    def test_default_denoise_off(self):
        assert PreprocessConfig().denoise is False

    def test_default_normalize_on(self):
        assert PreprocessConfig().normalize is True

    def test_default_trim_silence_off(self):
        assert PreprocessConfig().trim_silence is False

    def test_custom_values(self):
        cfg = PreprocessConfig(denoise=True, normalize=False, trim_silence=True)
        assert cfg.denoise is True
        assert cfg.normalize is False
        assert cfg.trim_silence is True


# ── Tests: resample ───────────────────────────────────────────────────────────

class TestResample:

    def test_44100_to_16000(self):
        audio = _sine(440, 1.0, 44_100)
        result = Preprocessor(PreprocessConfig(normalize=False)).process(audio, 44_100)
        assert result.sample_rate == 16_000

    def test_output_length_approx(self):
        """Output length should be close to duration × target_sr."""
        audio = _sine(440, 2.0, 44_100)
        result = Preprocessor(PreprocessConfig(normalize=False)).process(audio, 44_100)
        expected = 2.0 * 16_000
        assert abs(len(result.samples) - expected) < 50  # within 50 samples

    def test_already_16000_no_resample_step(self):
        audio = _sine(440, 1.0, 16_000)
        result = Preprocessor(PreprocessConfig(normalize=False)).process(audio, 16_000)
        assert result.sample_rate == 16_000
        # No 'resampled' step in steps_applied
        assert not any("resampled" in s for s in result.steps_applied)

    def test_8000_to_16000(self):
        """Upsampling from 8kHz (telephone quality) should work."""
        audio = _sine(440, 1.0, 8_000)
        result = Preprocessor(PreprocessConfig(normalize=False)).process(audio, 8_000)
        assert result.sample_rate == 16_000

    def test_22050_to_16000(self):
        audio = _sine(440, 1.0, 22_050)
        result = Preprocessor(PreprocessConfig(normalize=False)).process(audio, 22_050)
        assert result.sample_rate == 16_000

    def test_original_sr_preserved_in_result(self):
        audio = _sine(440, 1.0, 44_100)
        result = Preprocessor().process(audio, 44_100)
        assert result.original_sr == 44_100


# ── Tests: output properties ──────────────────────────────────────────────────

class TestOutputProperties:

    def test_dtype_always_float32(self):
        audio = _sine(440, 1.0, 44_100)
        result = Preprocessor().process(audio, 44_100)
        assert result.samples.dtype == np.float32

    def test_samples_1d(self):
        audio = _sine(440, 1.0, 16_000)
        result = Preprocessor().process(audio, 16_000)
        assert result.samples.ndim == 1

    def test_returns_preprocessresult(self):
        audio = _sine(440, 1.0, 16_000)
        result = Preprocessor().process(audio, 16_000)
        assert isinstance(result, PreprocessResult)

    def test_processed_duration_close_to_input(self):
        audio = _sine(440, 2.0, 44_100)
        result = Preprocessor(PreprocessConfig(normalize=False)).process(audio, 44_100)
        assert abs(result.processed_duration - 2.0) < 0.1

    def test_original_duration_correct(self):
        audio = _sine(440, 3.0, 44_100)
        result = Preprocessor().process(audio, 44_100)
        assert abs(result.original_duration - 3.0) < 0.01


# ── Tests: normalisation ──────────────────────────────────────────────────────

class TestNormalization:

    def test_peak_is_1_after_normalize(self):
        audio = _sine(440, 1.0, 16_000, amp=0.3)
        result = Preprocessor(PreprocessConfig(normalize=True)).process(audio, 16_000)
        assert abs(np.max(np.abs(result.samples)) - 1.0) < 1e-5

    def test_normalize_off_does_not_change_peak(self):
        audio = _sine(440, 1.0, 16_000, amp=0.3)
        result = Preprocessor(PreprocessConfig(normalize=False)).process(audio, 16_000)
        assert abs(np.max(np.abs(result.samples)) - 0.3) < 0.01

    def test_near_silent_not_normalized(self):
        """Near-silent clips should not be boosted (to avoid amplifying noise floor)."""
        audio = np.zeros(16_000, dtype=np.float32) + 1e-8
        result = Preprocessor(PreprocessConfig(normalize=True)).process(audio, 16_000)
        # Peak should remain near-zero, not boosted to 1.0
        assert np.max(np.abs(result.samples)) < 0.01

    def test_normalize_step_recorded(self):
        audio = _sine(440, 1.0, 16_000, amp=0.5)
        result = Preprocessor(PreprocessConfig(normalize=True)).process(audio, 16_000)
        assert any("normalised" in s for s in result.steps_applied)


# ── Tests: silence trimming ───────────────────────────────────────────────────

class TestSilenceTrimming:

    def test_trim_reduces_length(self):
        padded = _padded_silence(inner_dur=0.5, pad_dur=0.3, sr=16_000)
        cfg = PreprocessConfig(normalize=False, trim_silence=True, silence_threshold=0.01)
        result = Preprocessor(cfg).process(padded, 16_000)
        # After trimming, output should be shorter than input
        assert len(result.samples) < len(padded)

    def test_trim_preserves_speech(self):
        """After trimming silence, the signal RMS should be higher (speech kept)."""
        padded = _padded_silence(inner_dur=0.5, pad_dur=0.3, sr=16_000)
        cfg = PreprocessConfig(normalize=False, trim_silence=True)
        result = Preprocessor(cfg).process(padded, 16_000)
        assert get_rms_level(result.samples) > get_rms_level(padded)

    def test_all_silence_returns_original(self):
        """If entire clip is below threshold, return without trimming."""
        silent = np.zeros(16_000, dtype=np.float32)
        cfg = PreprocessConfig(normalize=False, trim_silence=True)
        result = Preprocessor(cfg).process(silent, 16_000)
        # Should not crash, should return something
        assert isinstance(result.samples, np.ndarray)


# ── Tests: steps_applied ─────────────────────────────────────────────────────

class TestStepsApplied:

    def test_steps_is_list(self):
        audio = _sine(440, 1.0, 16_000)
        result = Preprocessor().process(audio, 16_000)
        assert isinstance(result.steps_applied, list)

    def test_steps_not_empty(self):
        audio = _sine(440, 1.0, 44_100)
        result = Preprocessor().process(audio, 44_100)
        assert len(result.steps_applied) > 0


# ── Tests: utility functions ──────────────────────────────────────────────────

class TestUtilities:

    def test_get_rms_level_sine(self):
        audio = _sine(440, 1.0, 16_000, amp=1.0)
        rms = get_rms_level(audio)
        # Sine wave RMS = amplitude / sqrt(2) ≈ 0.707
        assert abs(rms - (1.0 / np.sqrt(2))) < 0.01

    def test_get_rms_level_silence(self):
        assert get_rms_level(np.zeros(1000, dtype=np.float32)) == 0.0

    def test_get_snr_estimate_returns_float(self):
        audio = _sine(440, 2.0, 16_000)
        snr = get_snr_estimate(audio, 16_000)
        assert isinstance(snr, float)

    def test_get_snr_estimate_short_clip(self):
        """Short clips should return 0.0 without crashing."""
        audio = _sine(440, 0.1, 16_000)
        snr = get_snr_estimate(audio, 16_000)
        assert isinstance(snr, float)