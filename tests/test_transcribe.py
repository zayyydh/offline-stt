"""
tests/test_transcribe.py
------------------------
Unit tests for src/transcribe.py

Strategy: mock faster_whisper.WhisperModel so these tests run without
downloading any model weights and without a GPU. This makes the suite
fast and runnable in CI with no special setup.

Tests cover:
  - TranscriptSegment properties (confidence labels, duration, to_dict)
  - TranscriptResult properties (rtf, to_dict)
  - TranscribeEngine lazy loading (model not loaded until first call)
  - TranscribeEngine.transcribe() returns TranscriptResult
  - TranscribeEngine reuses loaded model on second call
  - Error: wrong sample rate raises ValueError
  - Error: missing faster-whisper raises ImportError
  - TranscriptResult.rtf computed correctly
  - Confidence thresholds (high / medium / low)
  - Empty segments (no speech) handled gracefully
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcribe import (
    TranscribeEngine,
    TranscriptResult,
    TranscriptSegment,
    DEFAULT_MODEL_NAME,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_segment(start=0.0, end=2.0, text=" Hello world", avg_logprob=-0.2, no_speech_prob=0.01):
    """Create a mock faster_whisper segment (SimpleNamespace mimics the real object)."""
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


def _make_info(language="en", language_probability=0.98):
    return SimpleNamespace(language=language, language_probability=language_probability)


def _dummy_audio(duration: float = 3.0, sr: int = 16_000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


def _mock_whisper_model(segments=None, info=None):
    """Return a MagicMock that quacks like WhisperModel."""
    if segments is None:
        segments = [_make_segment()]
    if info is None:
        info = _make_info()

    mock = MagicMock()
    mock.transcribe.return_value = (iter(segments), info)
    return mock


# ── Tests: TranscriptSegment ──────────────────────────────────────────────────

class TestTranscriptSegment:

    def test_confidence_high(self):
        seg = TranscriptSegment(0, 2, "hello", avg_logprob=-0.1, no_speech_prob=0.01)
        assert seg.confidence == "high"

    def test_confidence_medium(self):
        seg = TranscriptSegment(0, 2, "hello", avg_logprob=-0.45, no_speech_prob=0.01)
        assert seg.confidence == "medium"

    def test_confidence_low(self):
        seg = TranscriptSegment(0, 2, "hello", avg_logprob=-0.9, no_speech_prob=0.01)
        assert seg.confidence == "low"

    def test_confidence_boundary_high(self):
        seg = TranscriptSegment(0, 2, "hello", avg_logprob=-0.3, no_speech_prob=0.01)
        assert seg.confidence == "high"  # boundary: > -0.3 is high, -0.3 is also high

    def test_confidence_boundary_medium(self):
        seg = TranscriptSegment(0, 2, "hello", avg_logprob=-0.6, no_speech_prob=0.01)
        assert seg.confidence == "medium"  # -0.6 boundary

    def test_duration(self):
        seg = TranscriptSegment(1.0, 3.5, "test", avg_logprob=-0.2, no_speech_prob=0.0)
        assert abs(seg.duration - 2.5) < 0.001

    def test_to_dict_keys(self):
        seg = TranscriptSegment(0, 2, " hello ", avg_logprob=-0.2, no_speech_prob=0.05)
        d = seg.to_dict()
        assert "start" in d
        assert "end" in d
        assert "text" in d
        assert "avg_logprob" in d
        assert "no_speech_prob" in d
        assert "confidence" in d

    def test_to_dict_text_stripped(self):
        seg = TranscriptSegment(0, 2, "  hello world  ", avg_logprob=-0.2, no_speech_prob=0.0)
        assert seg.to_dict()["text"] == "hello world"

    def test_to_dict_rounded_values(self):
        seg = TranscriptSegment(0.0001, 2.9999, "test", avg_logprob=-0.23456, no_speech_prob=0.12345)
        d = seg.to_dict()
        # Values should be rounded, not raw floats
        assert d["start"] == round(0.0001, 3)
        assert d["avg_logprob"] == round(-0.23456, 4)


# ── Tests: TranscriptResult ───────────────────────────────────────────────────

class TestTranscriptResult:

    def _make_result(self, inference_time=5.0, audio_duration=10.0):
        seg = TranscriptSegment(0, 10, "hello world", avg_logprob=-0.2, no_speech_prob=0.01)
        return TranscriptResult(
            transcript="hello world",
            segments=[seg],
            language="en",
            language_probability=0.98,
            audio_duration=audio_duration,
            inference_time=inference_time,
            model_name="base.en",
            avg_confidence=-0.2,
        )

    def test_rtf_faster_than_real_time(self):
        result = self._make_result(inference_time=5.0, audio_duration=10.0)
        assert result.rtf == 0.5  # 5/10

    def test_rtf_slower_than_real_time(self):
        result = self._make_result(inference_time=20.0, audio_duration=10.0)
        assert result.rtf == 2.0

    def test_rtf_zero_duration(self):
        result = self._make_result(inference_time=1.0, audio_duration=0.0)
        assert result.rtf == 0.0  # no division by zero

    def test_to_dict_required_keys(self):
        result = self._make_result()
        d = result.to_dict()
        for key in ["transcript", "segments", "language", "audio_duration_sec",
                    "inference_time_sec", "rtf", "model", "avg_confidence", "num_segments"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_num_segments(self):
        result = self._make_result()
        assert result.to_dict()["num_segments"] == 1

    def test_to_dict_segments_are_dicts(self):
        result = self._make_result()
        for s in result.to_dict()["segments"]:
            assert isinstance(s, dict)


# ── Tests: TranscribeEngine ───────────────────────────────────────────────────

class TestTranscribeEngine:

    def test_model_not_loaded_at_init(self):
        engine = TranscribeEngine()
        assert engine.is_loaded is False

    def test_model_loads_on_first_transcribe(self, tmp_path):
        mock_model = _mock_whisper_model()
        audio = _dummy_audio()

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_dir=tmp_path)
            result = engine.transcribe(audio, sample_rate=16_000)

        assert engine.is_loaded is True
        assert isinstance(result, TranscriptResult)

    def test_model_reused_on_second_call(self, tmp_path):
        mock_model = _mock_whisper_model()
        audio = _dummy_audio()

        with patch("faster_whisper.WhisperModel", return_value=mock_model) as MockClass:
            engine = TranscribeEngine(model_dir=tmp_path)
            engine.transcribe(audio, sample_rate=16_000)
            engine.transcribe(audio, sample_rate=16_000)

        # WhisperModel constructor should only be called ONCE
        assert MockClass.call_count == 1

    def test_wrong_sample_rate_raises(self, tmp_path):
        engine = TranscribeEngine(model_dir=tmp_path)
        audio = _dummy_audio()
        with pytest.raises(ValueError, match="16000"):
            engine.transcribe(audio, sample_rate=44_100)

    def test_transcript_text_assembled(self, tmp_path):
        segments = [
            _make_segment(0, 2, " Hello"),
            _make_segment(2, 4, " world"),
        ]
        mock_model = _mock_whisper_model(segments=segments)
        audio = _dummy_audio(4.0)

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_dir=tmp_path)
            result = engine.transcribe(audio, sample_rate=16_000)

        assert "Hello" in result.transcript
        assert "world" in result.transcript

    def test_segments_count(self, tmp_path):
        segments = [_make_segment(i, i + 1, f" word{i}") for i in range(5)]
        mock_model = _mock_whisper_model(segments=segments)
        audio = _dummy_audio(5.0)

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_dir=tmp_path)
            result = engine.transcribe(audio, sample_rate=16_000)

        assert len(result.segments) == 5

    def test_empty_segments_no_crash(self, tmp_path):
        """No speech detected — empty segment list should not raise."""
        mock_model = _mock_whisper_model(segments=[])
        audio = _dummy_audio(2.0)

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_dir=tmp_path)
            result = engine.transcribe(audio, sample_rate=16_000)

        assert result.transcript == ""
        assert result.segments == []
        assert result.avg_confidence == 0.0

    def test_language_detection_in_result(self, tmp_path):
        info = _make_info(language="en", language_probability=0.97)
        mock_model = _mock_whisper_model(info=info)
        audio = _dummy_audio()

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_dir=tmp_path)
            result = engine.transcribe(audio, sample_rate=16_000)

        assert result.language == "en"
        assert abs(result.language_probability - 0.97) < 0.001

    def test_unload_clears_model(self, tmp_path):
        mock_model = _mock_whisper_model()
        audio = _dummy_audio()

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_dir=tmp_path)
            engine.transcribe(audio, sample_rate=16_000)
            assert engine.is_loaded is True
            engine.unload()
            assert engine.is_loaded is False

    def test_missing_faster_whisper_raises_import_error(self, tmp_path):
        engine = TranscribeEngine(model_dir=tmp_path)
        audio = _dummy_audio()

        with patch.dict("sys.modules", {"faster_whisper": None}):
            # Bust cached _model
            engine._model = None
            with pytest.raises((ImportError, TypeError)):
                engine.transcribe(audio, sample_rate=16_000)

    def test_model_name_in_result(self, tmp_path):
        mock_model = _mock_whisper_model()
        audio = _dummy_audio()

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_name="small.en", model_dir=tmp_path)
            result = engine.transcribe(audio, sample_rate=16_000)

        assert result.model_name == "small.en"

    def test_audio_duration_in_result(self, tmp_path):
        mock_model = _mock_whisper_model()
        duration = 4.0
        audio = _dummy_audio(duration)

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_dir=tmp_path)
            result = engine.transcribe(audio, sample_rate=16_000)

        assert abs(result.audio_duration - duration) < 0.05

    def test_rtf_is_positive(self, tmp_path):
        mock_model = _mock_whisper_model()
        audio = _dummy_audio(3.0)

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            engine = TranscribeEngine(model_dir=tmp_path)
            result = engine.transcribe(audio, sample_rate=16_000)

        assert result.rtf >= 0.0