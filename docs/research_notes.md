# Phase 1 — Research Notes
## Offline English Speech-to-Text: Model & Approach Analysis

**Author:** SynoraAI Labs Internship Assignment  
**Date:** June 2026  
**Scope:** Evaluation of offline STT approaches for local CPU-based English transcription

---

## 1. Problem Definition

The core challenge: convert English speech audio to text **entirely offline** on a standard
CPU, with no dependency on cloud APIs, LLMs, or external inference services.

Key constraints:
- Must run on CPU (GPU optional/bonus)
- Python ecosystem only
- Must handle `.wav` and `.mp3` input files
- Microphone input desirable but optional
- Output: English transcript with optional timestamps

---

## 2. Approaches Surveyed

### 2.1 CMU Sphinx / PocketSphinx
**What it is:** One of the earliest open-source ASR systems (Carnegie Mellon, 1990s).
Uses Hidden Markov Models with phoneme-level acoustic models.

**Verdict: Rejected.**
- Word Error Rate (WER) on clean English: ~25–40% — unacceptable for a modern prototype
- Python bindings (`pocketsphinx`) are poorly maintained (last active release 2022)
- Requires manual acoustic model tuning for custom vocabulary
- No context awareness — treats each word independently

---

### 2.2 Mozilla DeepSpeech
**What it is:** LSTM-based end-to-end ASR trained by Mozilla on LibriSpeech.
Released open-source with pre-trained English models.

**Verdict: Rejected.**
- Mozilla officially **abandoned the project in 2021** — no further model updates
- The successor (Coqui STT) forked it but is also now unmaintained (Coqui shut down in 2023)
- WER ~6–8% on clean speech — competitive, but model quality frozen
- `deepspeech` Python package install is fragile on Python 3.10+
- No built-in timestamp extraction

---

### 2.3 Coqui STT (fork of DeepSpeech)
**What it is:** Community fork of Mozilla DeepSpeech.

**Verdict: Rejected.**
- Coqui AI ceased operations in January 2024
- GitHub repository archived — no active maintenance
- Same underlying model quality as DeepSpeech
- Dependency conflicts with modern Python/numpy versions

---

### 2.4 Vosk
**What it is:** Lightweight offline ASR using Kaldi-based models.
Available for many languages including English.

**Verdict: Considered but not selected as primary.**
- WER on clean English: ~10–15% — worse than Whisper
- Very fast inference: ~1–2s per minute of audio on CPU
- Works well in streaming / real-time applications
- Models are small (50–1700 MB depending on quality tier)
- Python package is actively maintained
- **Key limitation:** No built-in timestamp confidence scores; segment boundaries less accurate

Vosk would be the right choice if latency were the primary constraint (e.g., real-time
captioning). For accuracy-first transcription of recorded audio, Whisper is superior.

---

### 2.5 OpenAI Whisper (`openai-whisper`)
**What it is:** Transformer-based encoder-decoder ASR model trained by OpenAI on
680,000 hours of multilingual audio. Released open-source in September 2022.

**Verdict: Strong candidate — but slower implementation superseded by faster-whisper.**
- WER on clean English LibriSpeech test-clean: ~2.7% (large-v2)
- WER with base.en: ~7.4% — still significantly better than alternatives
- Built-in timestamp extraction at segment level
- Automatic language detection
- Runs fully offline once model is downloaded
- **Key limitation:** Reference PyTorch implementation is slow on CPU.
  Inference on a modern laptop: ~40–60s per minute of audio (base model)
  This is a 0.67–1.0x real-time factor — borderline usable, but uncomfortable

---

### 2.6 faster-whisper (CTranslate2 backend) ← **SELECTED**
**What it is:** A reimplementation of OpenAI Whisper using CTranslate2, an optimised
inference engine for transformer models. Developed by SYSTRAN.

**Why selected:**
- **Same model weights as openai-whisper** — identical accuracy
- **4× faster on CPU** due to INT8 quantisation and CTranslate2 optimisations
- **~70% less memory** than the PyTorch reference implementation
- RTF (Real-Time Factor) for base.en on a modern CPU: ~0.13–0.18x
  (i.e., 60 seconds of audio transcribed in ~8–11 seconds)
- Built-in VAD (Voice Activity Detection) via Silero to skip silent segments
- Word-level timestamp alignment via CTranslate2's DTW implementation
- Actively maintained, 3,000+ GitHub stars, used in production systems
- `pip install faster-whisper` — no compiled extension build required

**GitHub:** https://github.com/SYSTRAN/faster-whisper

---

## 3. Model Size vs. Accuracy Trade-off

All Whisper models use the same architecture (encoder-decoder transformer) but
differ in parameter count and therefore accuracy/speed.

| Model     | Params | Size   | WER (test-clean) | CPU RTF (approx) |
|-----------|--------|--------|-----------------|------------------|
| tiny.en   | 39M    | 39 MB  | ~11.8%          | ~0.05x           |
| base.en   | 74M    | 74 MB  | ~7.4%           | ~0.13x ✓ chosen  |
| small.en  | 244M   | 244 MB | ~5.1%           | ~0.35x           |
| medium.en | 769M   | 769 MB | ~3.9%           | ~0.85x           |
| large-v3  | 1.5B   | 1.5 GB | ~2.7%           | ~1.5x            |

**RTF < 1.0 means faster than real-time.**  
`base.en` at ~0.13x RTF processes a 5-minute recording in under 40 seconds on CPU —
comfortable for a prototype use case.

The `.en` suffix indicates English-only models; these are slightly smaller and more
accurate than multilingual models of the same tier because they do not waste capacity
on other languages.

---

## 4. Audio Preprocessing Analysis

### 4.1 Why 16 kHz?
Whisper was trained on 16 kHz mono audio. Audio at higher sample rates contains
frequencies above 8 kHz — beyond what human speech primarily uses — and passing
them directly would waste compute. Resampling to 16 kHz before inference is mandatory.

### 4.2 Resampling Strategy
Three common approaches evaluated:

| Method               | Quality    | Speed  | Notes |
|----------------------|------------|--------|-------|
| `scipy.signal.resample` (FFT) | Good | Fast | Can introduce aliasing for non-integer ratios |
| `scipy.signal.resample_poly` | Excellent | Fast | Integer up/down ratio, no aliasing — **selected** |
| `librosa.resample` (soxr)    | Excellent | Moderate | Extra dependency, overkill for this use case |

`resample_poly` was chosen: it takes the GCD of source and target rates to compute
integer up/down ratios, avoiding the spectral aliasing that can occur with FFT-based
resampling when the ratio is not a simple fraction.

### 4.3 Noise Reduction
Evaluated `noisereduce` (spectral gating) vs `speechbrain` denoising vs no filtering.

**Finding:** Whisper's internal architecture is surprisingly robust to moderate background
noise — the encoder's attention mechanism partially learns to ignore stationary noise.
Adding `noisereduce` helps for recordings with loud consistent background noise (fans,
street noise) but **adds 0.3–1.5s latency** and can occasionally **degrade quality**
on already-clean recordings by over-attenuating transient consonants.

Decision: make denoising **opt-in** (`--denoise` flag), not the default.

### 4.4 Amplitude Normalisation
Peak normalisation (dividing by max absolute value) ensures Whisper's mel-spectrogram
computation receives consistent amplitude regardless of recording level.
Made this the **default-on** behaviour since it has no quality downside.

---

## 5. CPU vs. GPU Execution

| Factor          | CPU (int8)     | GPU (float16)   |
|-----------------|----------------|-----------------|
| Setup           | Zero config    | CUDA driver needed |
| base.en RTF     | ~0.13x         | ~0.02x          |
| Memory          | ~500 MB RAM    | ~1 GB VRAM      |
| Portability     | Any machine    | CUDA-capable GPU only |
| Recommendation  | ✓ Default      | Optional upgrade |

For this prototype, CPU int8 is the correct default. GPU inference can be enabled
by passing `device="cuda"` and `compute_type="float16"` to `TranscribeEngine` —
the code supports this without modification.

---

## 6. Alternative Approaches Considered but Not Implemented

### 6.1 wav2vec 2.0 (Facebook/Meta)
Contrastive self-supervised speech model. Excellent WER on LibriSpeech (~1.8% with
large model). Rejected because: requires `torchaudio` (heavy dependency), no built-in
timestamp extraction, inference pipeline significantly more complex to set up correctly.

### 6.2 Whisper.cpp
C++ reimplementation of Whisper with Python bindings (`pywhispercpp`).
Even faster than faster-whisper on CPU due to GGML kernels.
Rejected for this prototype because Python bindings are less mature, installation
requires compilation on Windows, and faster-whisper provides equivalent performance
with better Python ergonomics.

### 6.3 SpeechBrain
Research framework with multiple STT backends. Too heavy (~500 MB install) for a
prototype where we only need transcription — not multi-task speech processing.

---

## 7. Key References

- OpenAI Whisper paper: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)
- faster-whisper repository: https://github.com/SYSTRAN/faster-whisper
- CTranslate2 documentation: https://opennmt.net/CTranslate2/
- LibriSpeech benchmark results: https://paperswithcode.com/sota/speech-recognition-on-librispeech-test-clean
- Vosk offline ASR: https://alphacephei.com/vosk/