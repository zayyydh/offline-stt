"""
mic_capture.py
--------------
Live microphone capture using PyAudio.

Records audio from a system microphone for a fixed duration (or until
stopped) and returns a float32 numpy array at the original sample rate.
The caller is expected to pass the result through Preprocessor before
transcription.

Notes on PyAudio
----------------
PyAudio wraps PortAudio, which is the cross-platform audio I/O library
used by most DAWs and audio tools. On Windows, PortAudio targets
WASAPI (Windows Audio Session API) by default.

Installation
  Windows:  pip install pyaudio
            If it fails: pip install pipwin && pipwin install pyaudio
  Linux:    sudo apt-get install portaudio19-dev && pip install pyaudio
  macOS:    brew install portaudio && pip install pyaudio

If PyAudio is unavailable, MicCapture falls back gracefully and logs a
clear error — the rest of the system (file transcription) still works.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16_000   # Record directly at Whisper's required rate
DEFAULT_CHUNK_SIZE  = 1_024    # Frames per buffer (~64ms at 16kHz)
DEFAULT_DURATION    = 10       # seconds


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({int(self.default_sample_rate)} Hz, {self.max_input_channels}ch)"


class MicCapture:
    """
    Records audio from the system microphone.

    Usage
    -----
    mic = MicCapture(duration=5)
    samples, sr = mic.record()
    # samples is float32 numpy array at sr Hz
    """

    def __init__(
        self,
        duration: float = DEFAULT_DURATION,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        device_index: Optional[int] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.duration = duration
        self.sample_rate = sample_rate
        self.device_index = device_index  # None = system default
        self.chunk_size = chunk_size
        self._stop_event = threading.Event()

    def record(self) -> tuple[np.ndarray, int]:
        """
        Record audio for self.duration seconds.

        Returns
        -------
        (samples: np.ndarray[float32], sample_rate: int)

        Raises
        ------
        ImportError  — if PyAudio is not installed
        RuntimeError — if no microphone is available
        """
        try:
            import pyaudio
        except ImportError as exc:
            raise ImportError(
                "PyAudio is not installed. Install with: pip install pyaudio\n"
                "On Windows if pip fails: pip install pipwin && pipwin install pyaudio"
            ) from exc

        pa = pyaudio.PyAudio()

        # Validate device index
        if self.device_index is not None:
            try:
                info = pa.get_device_info_by_index(self.device_index)
                if info["maxInputChannels"] < 1:
                    raise RuntimeError(
                        f"Device {self.device_index} ({info['name']}) has no input channels."
                    )
            except OSError as exc:
                raise RuntimeError(
                    f"Invalid device index {self.device_index}: {exc}"
                ) from exc

        num_frames = int(self.sample_rate / self.chunk_size * self.duration)
        raw_frames: list[bytes] = []
        self._stop_event.clear()

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )

            logger.info(
                "Recording for %.1f s at %d Hz (device=%s)...",
                self.duration,
                self.sample_rate,
                self.device_index if self.device_index is not None else "default",
            )

            for _ in range(num_frames):
                if self._stop_event.is_set():
                    logger.info("Recording stopped early by user.")
                    break
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                raw_frames.append(data)

            stream.stop_stream()
            stream.close()

        finally:
            pa.terminate()

        if not raw_frames:
            raise RuntimeError("No audio frames captured — check microphone connection.")

        # Convert int16 PCM bytes → float32 [-1.0, 1.0]
        raw = np.frombuffer(b"".join(raw_frames), dtype=np.int16)
        samples = raw.astype(np.float32) / 32768.0

        logger.info("Captured %.2f s of audio (%d samples)", len(samples) / self.sample_rate, len(samples))
        return samples, self.sample_rate

    def stop(self):
        """Signal the recording loop to stop early (call from another thread)."""
        self._stop_event.set()

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        """
        Return all available audio INPUT devices.

        Returns an empty list (with a warning) if PyAudio is not installed.
        """
        try:
            import pyaudio
        except ImportError:
            logger.warning("PyAudio not installed — cannot list devices.")
            return []

        pa = pyaudio.PyAudio()
        devices: list[DeviceInfo] = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append(
                    DeviceInfo(
                        index=i,
                        name=info["name"],
                        max_input_channels=int(info["maxInputChannels"]),
                        default_sample_rate=float(info["defaultSampleRate"]),
                    )
                )

        pa.terminate()
        return devices

    @staticmethod
    def get_default_device() -> Optional[DeviceInfo]:
        """Return the default input device, or None if unavailable."""
        try:
            import pyaudio
        except ImportError:
            return None

        pa = pyaudio.PyAudio()
        try:
            info = pa.get_default_input_device_info()
            device = DeviceInfo(
                index=int(info["index"]),
                name=info["name"],
                max_input_channels=int(info["maxInputChannels"]),
                default_sample_rate=float(info["defaultSampleRate"]),
            )
        except OSError:
            device = None
        finally:
            pa.terminate()

        return device