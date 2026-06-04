"""
transcribe.py
-------------
Whisper transcription engine using faster-whisper.

faster-whisper is a CTranslate2-based reimplementation of OpenAI Whisper.
It is 4x faster than openai-whisper on CPU and uses ~70% less memory
due to INT8 quantisation — critical for offline, CPU-only environments.

Model selection guidance
------------------------
  tiny.en   — 39 MB  — very fast, lower accuracy (~WER 10–15%)
  base.en   — 74 MB  — RECOMMENDED — good accuracy, fast CPU inference
  small.en  — 244 MB — better accuracy, ~3x slower than base
  medium.en — 769 MB — high accuracy, ~8x slower than base
  large-v3  — 1.5 GB — best accuracy, slow on CPU, requires RAM

For this prototype, base.en is the default:
  - Word Error Rate ~7% on clean English speech
  - Processes 60s audio in ~8s on a modern CPU
  - Fits comfortably in 2 GB RAM

References
----------
  https://github.com/SYSTRAN/faster-whisper
  https://github.com/openai/whisper
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model directory relative to project root
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_NAME = "base.en"


@dataclass
class TranscriptSegment:
    """A single time-aligned segment from Whisper."""
    start: float        # seconds
    end: float          # seconds
    text: str           # transcribed text for this segment
    avg_logprob: float  # average log-probability (proxy for confidence)
    no_speech_prob: float  # probability this segment is silence

    @property
    def confidence(self):
        if self.avg_logprob >= -0.3:
            return "high"
        elif self.avg_logprob >= -0.6:
            return "medium"
        else:
            return "low"

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text.strip(),
            "avg_logprob": round(self.avg_logprob, 4),
            "no_speech_prob": round(self.no_speech_prob, 4),
            "confidence": self.confidence,
        }


@dataclass
class TranscriptResult:
    """Full result returned by TranscribeEngine.transcribe()."""
    transcript: str                         # full concatenated text
    segments: list[TranscriptSegment]       # time-aligned segments
    language: str                           # detected language code (e.g. 'en')
    language_probability: float             # confidence in language detection
    audio_duration: float                   # input audio duration (seconds)
    inference_time: float                   # wall-clock inference time (seconds)
    model_name: str
    avg_confidence: float = 0.0             # mean avg_logprob across segments

    @property
    def rtf(self) -> float:
        """Real-time factor: inference_time / audio_duration. <1.0 = faster than real time."""
        if self.audio_duration == 0:
            return 0.0
        return round(self.inference_time / self.audio_duration, 3)

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "audio_duration_sec": round(self.audio_duration, 2),
            "inference_time_sec": round(self.inference_time, 2),
            "rtf": self.rtf,
            "model": self.model_name,
            "avg_confidence": round(self.avg_confidence, 4),
            "num_segments": len(self.segments),
        }


class ModelNotFoundError(Exception):
    """Raised when a model directory doesn't exist and auto-download is disabled."""


class TranscribeEngine:
    """
    Wrapper around faster-whisper's WhisperModel.

    Usage
    -----
    engine = TranscribeEngine()          # loads base.en from ./models/
    result = engine.transcribe(samples, sr=16000)
    print(result.transcript)

    The model is loaded lazily on the first transcribe() call to keep
    startup time fast. Subsequent calls reuse the loaded model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_dir: Optional[str | Path] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Parameters
        ----------
        model_name : str
            Whisper model name: tiny.en, base.en, small.en, medium.en, large-v3
        model_dir : Path or str, optional
            Directory where model weights are stored.
            Defaults to <project_root>/models/
        device : str
            'cpu' or 'cuda'. Auto-detected if not specified.
        compute_type : str
            Quantisation: 'int8' (fastest on CPU), 'float16' (GPU), 'float32'
        """
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.device = device
        self.compute_type = compute_type
        self._model = None  # lazy load

        logger.info(
            "TranscribeEngine initialised — model=%s device=%s compute=%s",
            model_name, device, compute_type,
        )

    def _load_model(self):
        """Lazy-load the Whisper model on first use."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            ) from exc

        model_path = self.model_dir / self.model_name

        if model_path.exists():
            # Load from local path
            logger.info("Loading model from local path: %s", model_path)
            load_target = str(model_path)
        else:
            # Download from HuggingFace and cache to model_dir
            logger.info(
                "Model not found locally — downloading '%s' to %s",
                self.model_name, self.model_dir,
            )
            self.model_dir.mkdir(parents=True, exist_ok=True)
            load_target = self.model_name  # faster-whisper will download

        t0 = time.perf_counter()
        self._model = WhisperModel(
            load_target,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.model_dir),
        )
        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.2f s", elapsed)

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int = 16_000,
        language: Optional[str] = "en",
        beam_size: int = 5,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptResult:
        """
        Transcribe a numpy audio array.

        Parameters
        ----------
        samples : np.ndarray
            Float32 audio at 16 kHz (preprocessed by Preprocessor).
        sample_rate : int
            Must be 16000 — Whisper's required input rate.
        language : str or None
            Force language (e.g. 'en'). None = auto-detect.
        beam_size : int
            Beam search width. 5 is the default; 1 (greedy) is faster.
        word_timestamps : bool
            Return word-level timestamps (slower — uses dtw alignment).
        vad_filter : bool
            Use Silero VAD to skip silent segments before inference.
            Speeds up transcription of audio with long silences.
        initial_prompt : str or None
            Optional text to prime the decoder (e.g. domain vocabulary).

        Returns
        -------
        TranscriptResult
        """
        if self._model is None:
            self._load_model()

        if sample_rate != 16_000:
            raise ValueError(
                f"Whisper requires 16000 Hz input, got {sample_rate} Hz. "
                "Run through Preprocessor first."
            )

        audio_duration = len(samples) / sample_rate
        logger.info(
            "Transcribing %.1f s of audio (beam=%d, vad=%s)",
            audio_duration, beam_size, vad_filter,
        )

        t0 = time.perf_counter()
        segments_gen, info = self._model.transcribe(
            samples,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
            condition_on_previous_text=True,
        )

        # Materialise the generator — faster-whisper is lazy
        segments: list[TranscriptSegment] = []
        for seg in segments_gen:
            segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                )
            )

        inference_time = time.perf_counter() - t0

        # Build full transcript
        transcript = " ".join(s.text.strip() for s in segments).strip()

        # Average confidence across segments (weighted by duration)
        avg_confidence = 0.0
        if segments:
            total_dur = sum(s.duration for s in segments) or 1.0
            avg_confidence = sum(
                s.avg_logprob * s.duration for s in segments
            ) / total_dur

        result = TranscriptResult(
            transcript=transcript,
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            audio_duration=audio_duration,
            inference_time=inference_time,
            model_name=self.model_name,
            avg_confidence=avg_confidence,
        )

        logger.info(
            "Done — %d segments, RTF=%.2f, lang=%s (p=%.2f)",
            len(segments),
            result.rtf,
            info.language,
            info.language_probability,
        )

        return result

    def transcribe_file(
        self,
        file_path: str | Path,
        **kwargs,
    ) -> TranscriptResult:
        """
        Convenience method: load + preprocess + transcribe in one call.
        Uses default PreprocessConfig (normalize=True, denoise=False).
        """
        from .audio_loader import load as load_audio
        from .preprocessor import Preprocessor, PreprocessConfig

        audio_data = load_audio(file_path)
        preprocessor = Preprocessor(PreprocessConfig(
            normalize=True,
            denoise=kwargs.pop("denoise", False),
        ))
        result = preprocessor.process(audio_data.samples, audio_data.sample_rate)
        return self.transcribe(result.samples, sample_rate=result.sample_rate, **kwargs)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        """Release model from memory."""
        self._model = None
        logger.info("Model unloaded from memory.")