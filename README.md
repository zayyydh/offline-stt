# Offline English Speech-to-Text Engine

A fully offline Python CLI for transcribing English audio — no cloud APIs,
no LLMs, no internet connection required after setup.

Built as the SynoraAI Labs internship assignment prototype.

---

## What It Does

- Transcribes `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a` audio files
- Optional live microphone recording and transcription
- Outputs plain text, structured JSON, or `.srt` subtitle files
- Runs completely offline on CPU (GPU supported optionally)
- Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 4× faster than openai-whisper on CPU

---

## Prerequisites

- Python 3.10 or higher
- pip

Check your Python version:
```bash
python --version
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/offline-stt.git
cd offline-stt
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> **Windows PyAudio note:** If `pip install pyaudio` fails, run:
> ```
> pip install pipwin
> pipwin install pyaudio
> ```
> PyAudio is only required for microphone mode (`--mic`). File transcription works without it.

**4. Download a Whisper model**
```bash
python scripts/download_model.py
```
This downloads `base.en` (~74 MB) into `models/base.en/`.
The model is cached locally — all subsequent runs are fully offline.

---

## Usage

### Transcribe an audio file
```bash
python main.py --input path/to/audio.wav
```

### With timestamps and noise reduction
```bash
python main.py --input interview.mp3 --timestamps --denoise
```

### Save as JSON (includes segments, confidence, RTF metadata)
```bash
python main.py --input lecture.wav --format json
```

### Save as SRT subtitles
```bash
python main.py --input podcast.mp3 --format srt --output subtitles.srt
```

### Live microphone recording (10 seconds)
```bash
python main.py --mic --duration 10 --timestamps
```

### Use a more accurate model
```bash
python main.py --input audio.wav --model small.en
```

### Quiet mode (transcript text only — good for piping)
```bash
python main.py --input audio.wav --quiet > transcript.txt
```

---

## All Options

```
python main.py --help

Input:
  --input FILE          Audio file to transcribe (.wav .mp3 .flac .ogg .m4a)
  --mic                 Record from microphone instead

Model:
  --model NAME          Whisper model (default: base.en)
  --model-dir DIR       Path to model weights directory

Preprocessing:
  --denoise             Apply noise reduction (recommended for noisy recordings)
  --no-normalize        Skip amplitude normalisation
  --trim-silence        Remove leading/trailing silence

Transcription:
  --language LANG       Language code (default: en). Use 'auto' to detect.
  --beam-size N         Beam search width (default: 5; use 1 for fastest)
  --no-vad              Disable voice activity detection filter

Output:
  --timestamps          Print segment-level timestamps
  --format FMT          Save output: txt, json, or srt
  --output FILE         Output file path (auto-named if not specified)
  --quiet               Print transcript only

Microphone:
  --duration SEC        Recording duration in seconds (default: 10)
  --device-index N      Microphone device index

Info:
  --list-models         Show available models and download status
  --list-devices        Show available audio input devices
  --info FILE           Show audio file metadata without transcribing
  --verbose             Enable debug logging
```

---

## Model Options

```bash
python main.py --list-models
```

| Model     | Size   | CPU Speed         | When to use |
|-----------|--------|-------------------|-------------|
| tiny.en   | 39 MB  | ~3s per min audio | Quick tests, low-RAM machines |
| base.en   | 74 MB  | ~8s per min audio | **Default — best balance** |
| small.en  | 244 MB | ~20s per min audio | Better accuracy for difficult audio |
| medium.en | 769 MB | ~50s per min audio | High accuracy, patient users |
| large-v3  | 1.5 GB | ~90s per min audio | Maximum accuracy, GPU recommended |

Download any model:
```bash
python scripts/download_model.py --model small.en
```

---

## Project Structure

```
offline-stt/
├── main.py                  ← CLI entry point (argparse)
├── requirements.txt         ← Python dependencies
├── conftest.py              ← Pytest shared fixtures
│
├── src/
│   ├── __init__.py
│   ├── audio_loader.py      ← Load WAV/MP3/FLAC/OGG/M4A → numpy float32
│   ├── preprocessor.py      ← Resample to 16kHz, denoise, normalise
│   ├── transcribe.py        ← faster-whisper wrapper, segments + confidence
│   ├── mic_capture.py       ← Live microphone recording via PyAudio
│   └── output_handler.py    ← Save .txt / .json / .srt, print to terminal
│
├── tests/
│   ├── test_audio_loader.py ← 20 unit tests for audio loading
│   ├── test_preprocessor.py ← 25 unit tests for preprocessing pipeline
│   └── test_transcribe.py   ← 18 unit tests (mocked model, no download needed)
│
├── scripts/
│   └── download_model.py    ← Download and verify Whisper model weights
│
├── models/                  ← Downloaded model weights (gitignored)
├── outputs/                 ← Saved transcripts (gitignored)
├── test_audio/              ← Sample audio files for testing
│
└── docs/
    ├── research_notes.md    ← Phase 1: Model comparison, approach analysis
    ├── architecture.md      ← Phase 2: Pipeline design, decision rationale
    └── engineering_log.md   ← Phase 4: Failed experiments, lessons learned
```

---

## Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_audio_loader.py -v

# Run a specific test
pytest tests/test_preprocessor.py::TestNormalization::test_peak_is_1_after_normalize -v
```

> Tests do **not** require a downloaded model. `test_transcribe.py` mocks the
> faster-whisper model so the full suite runs in seconds without any setup.

---

## Research Summary

Full analysis is in [`docs/research_notes.md`](docs/research_notes.md).

**Why faster-whisper over alternatives:**

| Library          | WER (base model) | CPU RTF | Status |
|------------------|-----------------|---------|--------|
| PocketSphinx     | ~30%            | ~0.05x  | Abandoned |
| Mozilla DeepSpeech | ~7%           | ~0.4x   | Abandoned (2021) |
| Vosk             | ~12%            | ~0.06x  | Active |
| openai-whisper   | ~7.4%           | ~0.7x   | Active |
| **faster-whisper** | **~7.4%**    | **~0.13x** | **Active ✓** |

faster-whisper achieves identical accuracy to openai-whisper with 4× faster
inference, using CTranslate2's INT8 quantisation — without any accuracy loss.

---

## Known Limitations

- **No streaming:** Full audio is loaded into memory before transcription starts. Very long files (>30 min) use proportional RAM.
- **Single speaker only:** No speaker diarisation. Multi-speaker audio produces a single mixed transcript.
- **English only:** The `.en` models are English-specific. Use multilingual models (e.g. `large-v3`) with `--language auto` for other languages.
- **PyAudio on Windows:** May require manual installation — see setup notes above.
- **Confidence scores are approximate:** `avg_logprob` is a proxy for confidence, not a calibrated probability.

---

## Future Improvements

- [ ] Chunked processing for long audio files (>30 min)
- [ ] Speaker diarisation via `pyannote.audio`
- [ ] Real-time streaming transcription (Vosk or faster-whisper stream mode)
- [ ] Auto-detect SNR and enable denoising only when beneficial
- [ ] Word-level timestamps via CTranslate2 DTW alignment
- [ ] REST API wrapper (FastAPI) for integration with other tools
- [ ] Batch processing of a directory of audio files

---

## Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| faster-whisper | Whisper inference (CTranslate2) | ≥1.0.0 |
| soundfile | WAV/FLAC/OGG loading | ≥0.12.1 |
| audioread | MP3/M4A decoding | ≥3.0.0 |
| numpy | Array operations | ≥1.24.0 |
| scipy | Resampling (resample_poly) | ≥1.11.0 |
| noisereduce | Spectral noise reduction (optional) | ≥3.0.0 |
| pyaudio | Microphone capture (optional) | ≥0.2.14 |
| pytest | Test runner | ≥7.4.0 |

---

## Assignment Phase Status

| Phase | Status | Output |
|-------|--------|--------|
| Phase 1 — Research & Analysis | ✓ Complete | `docs/research_notes.md` |
| Phase 2 — Design & Planning | ✓ Complete | `docs/architecture.md` |
| Phase 3 — Prototype Implementation | ✓ Complete | `src/`, `main.py`, `tests/` |
| Phase 4 — Engineering Notes | ✓ Complete | `docs/engineering_log.md` |

**Optional features implemented:**
- [x] Microphone input mode (`--mic`)
- [x] Timestamp extraction (`--timestamps`)
- [x] Noise filtering (`--denoise`)
- [x] Silence detection/trimming (`--trim-silence`)
- [x] Confidence estimation (per-segment `avg_logprob`)
- [ ] Streaming support (documented as future improvement)
- [ ] Speaker separation (documented as future improvement)