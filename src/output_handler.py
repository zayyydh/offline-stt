"""
output_handler.py
-----------------
Formats and saves transcript results to disk.

Supported output formats
------------------------
  .txt  — plain text transcript only
  .json — full structured result including segments, metadata, timing
  .srt  — SubRip subtitle format (with timestamps)

All write operations are atomic: the file is written to a temp path
first and then renamed to avoid partial-write corruption.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"txt", "json", "srt"}


def save_transcript(
    result,                        # TranscriptResult from transcribe.py
    output_path: Optional[str | Path] = None,
    fmt: str = "txt",
    source_file: Optional[str] = None,
) -> Path:
    """
    Save a TranscriptResult to disk.

    Parameters
    ----------
    result : TranscriptResult
        Output from TranscribeEngine.transcribe().
    output_path : str or Path, optional
        Full path including filename. If None, a timestamped name is
        generated in <project_root>/outputs/.
    fmt : str
        Output format: 'txt', 'json', or 'srt'.
    source_file : str, optional
        Original audio filename — used in the JSON header.

    Returns
    -------
    Path
        The path where the file was saved.
    """
    fmt = fmt.lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{fmt}'. Choose from: {', '.join(SUPPORTED_FORMATS)}"
        )

    if output_path is None:
        output_path = _auto_path(result, fmt, source_file)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "txt":
        content = _format_txt(result)
    elif fmt == "json":
        content = _format_json(result, source_file)
    else:
        content = _format_srt(result)

    _atomic_write(output_path, content)
    logger.info("Saved %s transcript → %s", fmt.upper(), output_path)
    return output_path


def print_result(result, show_segments: bool = True, show_metadata: bool = True):
    """
    Pretty-print a TranscriptResult to stdout.

    Parameters
    ----------
    result : TranscriptResult
    show_segments : bool
        Print each time-stamped segment.
    show_metadata : bool
        Print duration, RTF, language, confidence.
    """
    print("\n" + "─" * 60)
    print("  TRANSCRIPT")
    print("─" * 60)
    print(result.transcript or "(no speech detected)")

    if show_segments and result.segments:
        print("\n" + "─" * 60)
        print("  SEGMENTS")
        print("─" * 60)
        for seg in result.segments:
            start = _fmt_time(seg.start)
            end   = _fmt_time(seg.end)
            conf  = seg.confidence
            print(f"  [{start} → {end}]  ({conf:6s})  {seg.text.strip()}")

    if show_metadata:
        print("\n" + "─" * 60)
        print("  METADATA")
        print("─" * 60)
        print(f"  Model         : {result.model_name}")
        print(f"  Language      : {result.language} (p={result.language_probability:.2f})")
        print(f"  Audio duration: {result.audio_duration:.1f} s")
        print(f"  Inference time: {result.inference_time:.1f} s")
        print(f"  RTF           : {result.rtf:.3f}x  ({'faster' if result.rtf < 1 else 'slower'} than real-time)")
        print(f"  Avg confidence: {result.avg_confidence:.3f}")
        print(f"  Segments      : {len(result.segments)}")
        print("─" * 60 + "\n")


# ── Format builders ───────────────────────────────────────────────────────────

def _format_txt(result) -> str:
    """Plain transcript text."""
    return result.transcript.strip() + "\n"


def _format_json(result, source_file: Optional[str] = None) -> str:
    """Full structured JSON including metadata and segments."""
    data = result.to_dict()
    data["generated_at"] = datetime.now().isoformat()
    if source_file:
        data["source_file"] = source_file
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _format_srt(result) -> str:
    """
    SubRip (.srt) format:

    1
    00:00:00,000 --> 00:00:02,340
    Hello, world.

    2
    00:00:02,500 --> 00:00:05,100
    This is an offline transcript.
    """
    lines: list[str] = []
    for i, seg in enumerate(result.segments, 1):
        start_srt = _fmt_srt_time(seg.start)
        end_srt   = _fmt_srt_time(seg.end)
        lines.append(str(i))
        lines.append(f"{start_srt} --> {end_srt}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _auto_path(result, fmt: str, source_file: Optional[str]) -> Path:
    """Generate a default output path in outputs/ with a timestamp."""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    stem = Path(source_file).stem if source_file else "transcript"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return outputs_dir / f"{stem}_{timestamp}.{fmt}"


def _atomic_write(path: Path, content: str):
    """Write content to a temp file, then rename (atomic on POSIX)."""
    dir_ = path.parent
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, path)  # atomic rename
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to '{path}'. Check folder permissions."
        ) from exc


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm for display."""
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT files."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"