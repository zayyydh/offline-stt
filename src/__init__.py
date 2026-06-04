"""
offline-stt — src package
Offline English Speech-to-Text prototype using faster-whisper.
"""

from .audio_loader import load as load_audio, AudioData, AudioLoadError
from .preprocessor import Preprocessor, PreprocessConfig, PreprocessResult
from .transcribe import TranscribeEngine, TranscriptResult, TranscriptSegment
from .output_handler import save_transcript, print_result

__all__ = [
    "load_audio",
    "AudioData",
    "AudioLoadError",
    "Preprocessor",
    "PreprocessConfig",
    "PreprocessResult",
    "TranscribeEngine",
    "TranscriptResult",
    "TranscriptSegment",
    "save_transcript",
    "print_result",
]