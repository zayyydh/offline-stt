# Phase 4 — Engineering Log
## Decisions, Failed Experiments & Production Considerations

---

## 1. Failed Experiments & What They Taught

### Experiment 1: Using `openai-whisper` directly
**What I tried:** Started with the official `openai-whisper` package as the
transcription backend since it's the reference implementation.

**What happened:** On a test file of 45 seconds of speech, `openai-whisper`
with `base.en` took ~32 seconds on CPU. RTF ≈ 0.71x — borderline real-time.
More importantly, model load time on first call was ~8 seconds due to PyTorch
initialisation overhead.

**What I learned:** The PyTorch reference implementation is designed for
research, not deployment. `faster-whisper` uses CTranslate2 which is
specifically optimised for inference (not training) — different memory layout,
INT8 kernel paths, no gradient tracking overhead. Same model weights, 4x
faster inference. This was a clear win.

**Decision:** Switched to `faster-whisper`. The switch took ~30 minutes since
the API surface is nearly identical.

---

### Experiment 2: Librosa for resampling
**What I tried:** Used `librosa.resample()` for the resampling step since
librosa is the standard audio ML library in Python.

**What happened:** `librosa` imports take ~1.5–2 seconds on first import
due to its transitive dependencies (numba JIT compilation warm-up). For a CLI
tool that the user runs repeatedly, this constant startup penalty is annoying.

**What I learned:** `scipy.signal.resample_poly` does exactly the same job
(high-quality polyphase resampling) with zero import overhead — it's part of
scipy which is already a required dependency for numpy operations.

**Decision:** Replaced librosa with scipy. Removed librosa from requirements.
Startup time improved by ~1.5s.

---

### Experiment 3: Applying denoising by default
**What I tried:** Made `noisereduce` run on all audio as a default preprocessing
step, reasoning that it could only help.

**What happened:** On a test recording of clean speech (recorded in a quiet room),
enabling denoising reduced the WER slightly. But on a recording with fast speech
and hard consonants ('p', 't', 'k' sounds), `noisereduce`'s spectral gating
occasionally attenuated transients as if they were noise bursts — leading to
dropped syllables in the transcript.

Example: "specifically" became "ific-ly" in one test case.

**What I learned:** Denoising is not a free lunch. It works well for steady
background noise (fans, air conditioning) but can hurt quality on clean
recordings or speech with natural plosive consonants. Whisper's own architecture
is already somewhat robust to moderate noise.

**Decision:** Made denoising opt-in via `--denoise` flag with a clear note in
the CLI help. Added SNR estimation utility (`get_snr_estimate()`) so future
versions could auto-decide whether to denoise.

---

### Experiment 4: ffmpeg-based MP3 loading via subprocess
**What I tried:** Before finding `audioread`, attempted to decode MP3 by
shelling out to `ffmpeg` via `subprocess.run()` and piping raw PCM back.

**What happened:** This works but is fragile — requires `ffmpeg` to be in
`PATH`, different systems have different ffmpeg builds, and error handling
for codec failures is complex.

**What I learned:** `audioread` already wraps ffmpeg/libav in a clean Python
interface with proper error handling. It handles the same edge cases (corrupt
files, unusual sample rates) that I would have had to implement manually.

**Decision:** Used `audioread`. Deleted ~60 lines of subprocess glue code.

---

### Experiment 5: Returning samples + sr as a plain tuple
**What I tried:** `audio_loader.load()` initially returned `(samples, sr)` as
a plain Python tuple — common pattern in audio ML codebases.

**What happened:** A few calls downstream, I wrote `sr, samples = load(path)`
instead of `samples, sr = load(path)` — Python happily unpacked it in the
wrong order. The bug was silent: `sr` had 240000 values in it, `samples` was
an integer. This caused a cryptic error much later in the Preprocessor.

**What I learned:** Named return types (dataclasses) eliminate an entire
class of positional argument bugs with zero runtime cost. The IDE can
also autocomplete `result.samples` but not `result[0]`.

**Decision:** Refactored all return types to `@dataclass`. Added type hints
throughout. The bug became impossible.

---

### Experiment 6: Loading model at engine __init__ time
**What I tried:** `TranscribeEngine.__init__()` loaded the Whisper model
immediately during construction.

**What happened:** Any code that instantiated `TranscribeEngine` (including
test code) triggered a 2–4 second model load. Test suite took 45 seconds
before any actual test logic ran.

**What I learned:** Lazy loading is the right pattern for expensive resources.
Load the model exactly when it is first needed, not when the object is created.

**Decision:** Moved model loading to `_load_model()` called on first `transcribe()`.
Test suite now instantiates `TranscribeEngine` for free and mocks the model.

---

## 2. Technical Challenges

### Challenge 1: PyAudio installation on Windows
`pip install pyaudio` frequently fails on Windows because it requires PortAudio
headers at compile time. The standard workaround is:
```
pip install pipwin
pipwin install pyaudio
```
Or use a pre-compiled wheel from Christoph Gohlke's repository.

**Mitigation in code:** `mic_capture.py` wraps the `import pyaudio` in a
try/except with a clear error message and install instructions. The rest of the
application (file transcription) works without PyAudio installed.

### Challenge 2: Very long audio files
`faster-whisper` internally chunks audio into 30-second windows. Files longer
than ~30 minutes will take proportionally long and use significant RAM (the full
numpy array must be loaded).

**Current state:** Acceptable for prototype. Production fix: stream audio in
chunks from disk rather than loading the full file into memory.

### Challenge 3: Corrupt or truncated audio files
`soundfile` raises `sf.SoundFileRuntimeError` for corrupt WAV/FLAC.
`audioread` raises `audioread.exceptions.NoBackendError` if no decoder is available.

**Mitigation:** `AudioLoadError` exception wraps both, with clear messages
pointing the user to the specific file and likely cause.

---

## 3. Production Considerations

If this prototype were to become a production system, the following would need to
be addressed:

### 3.1 Chunked / Streaming Transcription
The current approach loads the full audio array into memory. For long recordings
(podcasts, meetings, lectures), this is wasteful. Production would use:
- Chunked reading from disk in 30-second windows
- Overlap of ~1 second between chunks to avoid cutting words at boundaries
- Merging logic to deduplicate the overlap region in transcripts

### 3.2 Speaker Diarisation
The prototype produces a single speaker transcript. Real meeting transcription
needs speaker labels ("Speaker 1: ...", "Speaker 2: ..."). This requires
`pyannote.audio` (diarisation pipeline) + merging its output with Whisper segments.
Adds ~2 GB of model weights but is the correct approach for multi-speaker content.

### 3.3 Confidence Thresholds as Configuration
The confidence thresholds (`-0.3`, `-0.6`) are currently hardcoded. Production
would expose these as configuration, potentially learning them from user feedback
on a per-domain basis (medical transcription vs casual speech have different profiles).

### 3.4 Model Versioning and Updates
Whisper model weights are downloaded from HuggingFace. In a production offline
system, model updates would be distributed as signed packages with checksum
verification, not pulled from the internet on demand.

### 3.5 Error Telemetry (local only)
No remote telemetry — consistent with the offline-only constraint. But local
logging to a rotating file would allow debugging user-reported issues without
any network calls.

### 3.6 Windows Long Path Support
Windows by default limits file paths to 260 characters. Audio files with long
names in deeply nested directories can trigger `OSError: [WinError 206]`.
Production fix: enable long paths in the Windows registry or use `\\?\` path prefixes.

---

## 4. What I Would Do Differently

1. **Start with `faster-whisper` directly** — I spent time on `openai-whisper`
   before discovering the performance gap. Benchmarking competing implementations
   first would have saved 3–4 hours.

2. **Write tests before implementing modules** — I wrote `audio_loader.py` before
   its test file. Two bugs (stereo shape handling, byte-level MP3 layout) only
   surfaced when writing tests. TDD order would have found them faster.

3. **Pin exact versions in requirements.txt from day one** — Initially used
   unpinned versions (`faster-whisper`, `scipy`) and hit a minor API change
   in `noisereduce` between `2.x` and `3.x`. Added version constraints after
   debugging for 20 minutes.

4. **Generate a synthetic test audio file with known transcription** — Using a
   real audio file for end-to-end testing creates a dependency on that file's
   content. A synthesised audio clip with a known transcript ("hello world")
   would make the integration test fully self-contained.