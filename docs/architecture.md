# Phase 2 — Architecture & Design
## Offline STT System: Processing Pipeline & Design Decisions

---

## 1. System Overview

```
Audio Input (file or mic)
        │
        ▼
┌───────────────────┐
│   AudioLoader     │  Reads WAV/MP3/FLAC/OGG/M4A → numpy float32
│  audio_loader.py  │  Handles stereo→mono downmix
└────────┬──────────┘
         │  AudioData(samples, sample_rate, duration, channels)
         ▼
┌───────────────────┐
│   Preprocessor    │  1. Resample to 16 kHz (resample_poly)
│  preprocessor.py  │  2. Denoise (optional, spectral gating)
└────────┬──────────┘  3. Peak normalise (default on)
         │             4. Trim silence (optional)
         │  PreprocessResult(samples@16kHz, steps_applied)
         ▼
┌───────────────────┐
│ TranscribeEngine  │  faster-whisper WhisperModel
│   transcribe.py   │  INT8 quantised, beam_size=5, VAD filter
└────────┬──────────┘
         │  TranscriptResult(transcript, segments, rtf, confidence)
         ▼
┌───────────────────┐
│  OutputHandler    │  .txt / .json / .srt — atomic write
│ output_handler.py │  Pretty-print to terminal
└───────────────────┘
```

---

## 2. Module Design Decisions

### 2.1 Separation of Concerns

Each module has a single responsibility and a clean interface:

- `AudioLoader` — knows about file formats, nothing else
- `Preprocessor` — knows about signal processing, nothing about models
- `TranscribeEngine` — knows about Whisper, nothing about file I/O
- `OutputHandler` — knows about formatting, nothing about audio

This means any module can be replaced independently. If we later switch from
faster-whisper to whisper.cpp, only `transcribe.py` changes.

### 2.2 Dataclasses as Return Types

All modules return typed dataclasses (`AudioData`, `PreprocessResult`,
`TranscriptResult`) rather than raw tuples or dicts. This provides:
- IDE autocomplete on field names
- Self-documenting interfaces
- Easy serialisation via `.to_dict()`

A plain tuple `(samples, sr)` was considered and rejected because it breaks
at the call site when the signature changes (positional coupling).

### 2.3 Lazy Model Loading

`TranscribeEngine` does **not** load the Whisper model at `__init__()` time.
The model is loaded on the **first call to `.transcribe()`**.

Reason: startup time. Loading `base.en` takes ~2–4 seconds. If the engine is
instantiated at application startup (e.g., in a CLI that then waits for user
input), lazy loading avoids paying this cost until it's actually needed.

In a long-running server context, the model is loaded once and reused for all
subsequent requests — this is handled naturally by the lazy pattern since
`_model` is set to `None` only at `__init__()` and `unload()`.

### 2.4 Stateless Preprocessor

`Preprocessor` is stateless — `process()` takes `(samples, sample_rate)` and
returns a new array. It does not modify the input array in-place.

This makes it safe to call from multiple threads (future streaming use case)
and makes each call independently testable without shared state side effects.

### 2.5 INT8 Quantisation Strategy

faster-whisper supports three compute types:
- `float32` — full precision, slowest
- `float16` — half precision (GPU only — errors on CPU-only machines)
- `int8`    — 8-bit integer quantisation, fastest on CPU — **selected**

INT8 quantisation introduces a small accuracy degradation (<0.3% WER increase
in benchmarks) in exchange for ~2x speed improvement over float32 on CPU.
This is the correct tradeoff for a CPU-first offline prototype.

---

## 3. CLI Design (`main.py`)

### 3.1 argparse over click/typer

`argparse` is Python stdlib — zero additional dependency. For a prototype of
this scope, adding `click` or `typer` would be over-engineering.

### 3.2 Mutually Exclusive Input Group

```
--input FILE  (file transcription)
--mic         (microphone capture)
```

These are defined as a `mutually_exclusive_group` in argparse. This enforces
at the parser level that the user cannot specify both simultaneously, producing
a clear error message automatically.

### 3.3 `--quiet` Flag

The `--quiet` flag suppresses all metadata output (RTF, language, segments)
and prints **only the transcript text**. This makes the CLI composable with
Unix pipes:

```bash
python main.py --input audio.wav --quiet > transcript.txt
python main.py --input audio.wav --quiet | wc -w   # word count
```

### 3.4 Format and Output Independence

The `--format` flag saves a file; without it, output is only printed.
The `--output` flag specifies where to save — without it, an auto-timestamped
filename is generated in `outputs/`. This allows:

```bash
# Quick transcription to screen
python main.py --input audio.wav

# Save as JSON with full metadata
python main.py --input audio.wav --format json

# Save to specific location
python main.py --input audio.wav --format txt --output results/my_transcript.txt
```

---

## 4. Testing Strategy

### 4.1 Unit Tests Without Model Weights

`test_transcribe.py` uses `unittest.mock.patch` to replace
`faster_whisper.WhisperModel` with a `MagicMock`. This means:
- Tests run instantly (no model download required)
- Tests work in CI environments with no GPU
- Model behaviour is tested by mocking the contract, not the implementation

### 4.2 Real I/O for Audio Loader & Preprocessor

`test_audio_loader.py` and `test_preprocessor.py` write real temporary WAV files
using `soundfile` and load them through the actual code path. This tests the
full I/O pipeline without requiring real speech content.

Synthetic sine waves are used as test audio because:
- Deterministic and reproducible
- Known exact properties (duration, sample rate, amplitude)
- No copyright or distribution concerns

### 4.3 Session vs Function Scope Fixtures

Fixtures used across multiple test files (sine arrays) use `scope="session"` to
compute them once per test run. Fixtures involving file writes use `scope="function"`
(the default) to ensure each test gets a clean temporary directory via `tmp_path`.

---

## 5. Output Format Design

### 5.1 Three Output Formats

| Format | Use Case |
|--------|----------|
| `.txt` | Quick reading, copy-paste into documents |
| `.json` | Downstream processing, structured data |
| `.srt` | Video subtitles, timeline review |

### 5.2 Atomic Write

All file writes go through `_atomic_write()` which:
1. Writes to a temp file in the same directory
2. Calls `os.replace()` (atomic rename on POSIX; near-atomic on Windows)

This prevents a partial file being visible if the process is interrupted during
write. Important for longer transcripts that take seconds to save.

### 5.3 Confidence Scoring

Whisper's `avg_logprob` per segment is a log-probability value:
- Near 0: high confidence (model assigns high probability to its output)
- Very negative: low confidence (uncertain transcription)

Thresholds used: `> -0.3` → high, `> -0.6` → medium, else → low.
These were determined empirically from the faster-whisper documentation and
community usage patterns.

---

## 6. Known Design Limitations

1. **No streaming transcription** — the current implementation buffers the full
   audio before inference. Streaming requires chunked inference with overlap
   logic to handle words split across chunk boundaries — out of scope for Phase 3.

2. **No speaker diarisation** — single-speaker transcription only. Multi-speaker
   separation requires pyannote.audio (a separate heavy dependency).

3. **Memory limit** — very long audio files (>30 min) load entirely into RAM.
   Production use would require chunked loading with overlap and merge.

4. **Windows-only testing** — pyaudio on macOS/Linux may require additional
   system packages (`portaudio19-dev` on Linux, `brew install portaudio` on macOS).