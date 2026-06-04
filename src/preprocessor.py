"""
preprocessor.py
---------------
Audio preprocessing pipeline for Whisper input.

Whisper requires audio at 16 kHz, mono, float32.
This module handles:
  1. Resampling to 16 kHz (via scipy)
  2. Optional noise reduction (via noisereduce)
  3. Optional amplitude normalisation (peak or RMS)
  4. Silence trimming (optional)

All operations are non-destructive — they return a new numpy array.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import resample_poly
from math import gcd

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000  # Hz — Whisper's required input rate


@dataclass
class PreprocessConfig:
    """
    Configuration for the preprocessing pipeline.

    Attributes
    ----------
    target_sr : int
        Target sample rate. Always 16000 for Whisper.
    denoise : bool
        Apply spectral subtraction noise reduction via noisereduce.
        Adds ~0.3–1.5 s latency. Recommended for recordings with
        consistent background noise (fans, AC, street).
        NOT recommended for already-clean studio recordings.
    normalize : bool
        Peak-normalise the waveform to [-1.0, 1.0].
        Helps with quiet recordings but can amplify noise if denoise=False.
    trim_silence : bool
        Remove leading/trailing silence below silence_threshold.
    silence_threshold : float
        RMS threshold below which a 10ms window is considered silence.
        Default 0.01 works for most recordings; lower for very quiet audio.
    """
    target_sr: int = TARGET_SAMPLE_RATE
    denoise: bool = False
    normalize: bool = True
    trim_silence: bool = False
    silence_threshold: float = 0.01


@dataclass
class PreprocessResult:
    """Output from the preprocessing pipeline."""
    samples: np.ndarray         # float32 at target_sr
    sample_rate: int            # always target_sr
    original_sr: int            # input sample rate
    original_duration: float    # input duration (seconds)
    processed_duration: float   # output duration after trimming
    steps_applied: list[str] = field(default_factory=list)


class Preprocessor:
    """
    Stateless preprocessing pipeline.
    Instantiate once; call process() for each audio clip.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def process(self, samples: np.ndarray, sample_rate: int) -> PreprocessResult:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        samples : np.ndarray
            Float32 mono audio array.
        sample_rate : int
            Original sample rate of the input.

        Returns
        -------
        PreprocessResult
        """
        cfg = self.config
        original_duration = len(samples) / sample_rate
        steps: list[str] = []

        audio = samples.copy().astype(np.float32)

        # ── Step 1: Resample ──────────────────────────────────────────────
        if sample_rate != cfg.target_sr:
            audio = self._resample(audio, sample_rate, cfg.target_sr)
            steps.append(f"resampled {sample_rate}Hz → {cfg.target_sr}Hz")
            logger.debug("Resampled from %d Hz to %d Hz", sample_rate, cfg.target_sr)
        else:
            steps.append(f"no resample needed ({sample_rate}Hz)")

        # ── Step 2: Noise reduction ───────────────────────────────────────
        if cfg.denoise:
            audio = self._denoise(audio, cfg.target_sr)
            steps.append("noise reduction applied")

        # ── Step 3: Normalise ─────────────────────────────────────────────
        if cfg.normalize:
            audio = self._normalize(audio)
            steps.append("peak normalised")

        # ── Step 4: Silence trimming ──────────────────────────────────────
        if cfg.trim_silence:
            audio = self._trim_silence(audio, cfg.target_sr, cfg.silence_threshold)
            steps.append(f"silence trimmed (threshold={cfg.silence_threshold})")

        processed_duration = len(audio) / cfg.target_sr

        return PreprocessResult(
            samples=audio,
            sample_rate=cfg.target_sr,
            original_sr=sample_rate,
            original_duration=original_duration,
            processed_duration=processed_duration,
            steps_applied=steps,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """
        Resample using scipy.signal.resample_poly.

        resample_poly uses integer up/down ratios, which is more accurate
        than scipy.signal.resample (FFT-based) for non-power-of-two ratios
        and avoids spectral aliasing at high frequencies.
        """
        common = gcd(src_sr, dst_sr)
        up = dst_sr // common
        down = src_sr // common
        resampled = resample_poly(audio, up, down)
        return resampled.astype(np.float32)

    @staticmethod
    def _denoise(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply spectral noise reduction via the noisereduce library.

        noisereduce uses Non-Local Means and spectral gating to estimate
        the noise profile from the first ~0.5s and subtract it.

        Falls back gracefully if noisereduce is not installed.
        """
        try:
            import noisereduce as nr
            reduced = nr.reduce_noise(y=audio, sr=sr, stationary=False)
            return reduced.astype(np.float32)
        except ImportError:
            logger.warning(
                "noisereduce not installed — skipping denoise. "
                "Install with: pip install noisereduce"
            )
            return audio
        except Exception as exc:
            logger.warning("Denoise failed (%s) — returning original audio.", exc)
            return audio

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """Peak-normalise to [-1.0, 1.0]."""
        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            # Near-silent clip — don't amplify noise floor
            logger.warning("Audio is near-silent (peak=%.6f) — skipping normalisation.", peak)
            return audio
        return (audio / peak).astype(np.float32)

    @staticmethod
    def _trim_silence(
        audio: np.ndarray,
        sr: int,
        threshold: float = 0.01,
        frame_ms: int = 10,
    ) -> np.ndarray:
        """
        Remove leading and trailing silence.

        Uses a simple RMS-based detector with 10ms frames.
        More robust than librosa.effects.trim (no extra dependency).
        """
        frame_size = int(sr * frame_ms / 1000)
        if frame_size == 0:
            return audio

        num_frames = len(audio) // frame_size
        if num_frames == 0:
            return audio

        # RMS per frame
        frames = audio[: num_frames * frame_size].reshape(num_frames, frame_size)
        rms = np.sqrt(np.mean(frames**2, axis=1))

        active = rms > threshold
        if not active.any():
            logger.warning("All frames below silence threshold — returning full clip.")
            return audio

        first = int(np.argmax(active)) * frame_size
        last = (len(active) - int(np.argmax(active[::-1]))) * frame_size
        trimmed = audio[first:last]
        logger.debug(
            "Silence trimmed: %.2fs → %.2fs",
            len(audio) / sr,
            len(trimmed) / sr,
        )
        return trimmed.astype(np.float32)


def get_rms_level(samples: np.ndarray) -> float:
    """Utility: return RMS level of an audio array."""
    return float(np.sqrt(np.mean(samples**2)))


def get_snr_estimate(samples: np.ndarray, sr: int, noise_ms: int = 300) -> float:
    """
    Rough SNR estimate using the first `noise_ms` milliseconds as noise floor.
    Returns dB. Useful for deciding whether to enable denoise.
    """
    noise_samples = int(sr * noise_ms / 1000)
    if len(samples) < noise_samples * 2:
        return 0.0

    noise_rms = np.sqrt(np.mean(samples[:noise_samples] ** 2))
    signal_rms = np.sqrt(np.mean(samples[noise_samples:] ** 2))

    if noise_rms < 1e-9:
        return 60.0  # effectively infinite SNR

    snr = 20 * np.log10(signal_rms / (noise_rms + 1e-9))
    return round(float(snr), 1)