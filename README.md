# Atonal Music Studio

A cross-platform desktop application for non-traditional music composition — atonal music, mathematical tone sequences, and ambient soundscapes.

---

## Features

| Tab | What it does |
|-----|--------------|
| **Synthesiser** | Sine, sawtooth, and square wave oscillator with ADSR envelope. Dial in any frequency (or pick a standard note name) and hear it instantly. |
| **Equation Synth** | Choose a mathematical concept — Fibonacci, π digits, e digits, golden ratio powers, square roots, primes, harmonic series, the logistic (chaos) map, or write your own expression — and map it to a melody. Control the base frequency, scale quantisation, mapping mode, and wave type. |
| **Sequencer** | A classic 16-step × 8-track step sequencer. Each track has its own note and wave type. Set the BPM, toggle steps on/off, and loop the pattern. |
| **Samples** | Load audio files in any supported format, preview the waveform, adjust volume and playback speed, and export to any other format. |

### Equation mapping modes

| Mode | Description |
|------|-------------|
| `ratio` | Multiply the base frequency by each sequence value, folded into the audible range. Works especially well with multiplicative sequences like Fibonacci or powers of φ. |
| `semitone` | Treat each value as a semitone offset from the base frequency. Good for digit sequences (π, e). |
| `modular` | Map each value to a scale degree (the value wraps within the chosen scale). Produces melodic patterns from any numeric sequence. |

### Audio formats

| Format | Import | Export | Notes |
|--------|--------|--------|-------|
| WAV    | ✅ | ✅ | Lossless, always available |
| FLAC   | ✅ | ✅ | Lossless compressed |
| OGG    | ✅ | ✅ | Open lossy |
| AIFF   | ✅ | ✅ | Apple lossless |
| MP3    | ✅ | ✅ | Requires FFmpeg |
| M4A    | ✅ | ✅ | Requires FFmpeg |
| AAC    | ✅ | ✅ | Requires FFmpeg |
| DSF/DFF | ❌ | ❌ | DSD format — convert to WAV first (see below) |

---

## Installation

### Step 1 — Install Python 3.10 or later

- **Windows**: Download from https://python.org and check "Add Python to PATH" during install.
- **macOS**: `brew install python` or download from python.org.
- **Linux**: Usually pre-installed. `sudo apt install python3 python3-pip` on Debian/Ubuntu.

Verify: open a terminal and run `python --version` (or `python3 --version`).

---

### Step 2 — Download the application

Place all five files from this folder in the same directory on your computer:

```
atonal_music_studio/
├── main.py
├── audio_engine.py
├── equation_engine.py
├── project_io.py
└── requirements.txt
```

---

### Step 3 — Install Python packages

Open a terminal (or Command Prompt on Windows), navigate to the folder, and run:

```bash
pip install -r requirements.txt
```

If `pip` is not found, try `pip3` instead.  On some systems you may also need:

```bash
python -m pip install -r requirements.txt
```

---

### Step 4 (Optional) — Install FFmpeg for MP3 / M4A / AAC support

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html (choose a pre-built Windows binary).
2. Extract the archive and locate the `bin` folder containing `ffmpeg.exe`.
3. Add that `bin` folder to your system PATH, or copy `ffmpeg.exe` next to `main.py`.

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

You can skip this step entirely if you only need WAV / FLAC / OGG / AIFF.

---

### Step 5 — Run the application

```bash
python main.py
```

On macOS/Linux you may need:
```bash
python3 main.py
```

---

## Project files

Projects are saved as `.ams` files (JSON format).  They store all synthesiser settings, sequencer patterns, and the paths to any sample files you loaded.  Sample audio itself is not embedded — the paths are stored, so keep your sample files accessible.

---

## DSD audio (DSF / DFF)

The DSD format requires specialised hardware drivers and cannot be decoded in pure Python.  To use DSD files in the app, convert them to 24-bit WAV first using one of these free tools:

- **dsd2pcm** — command-line, available on GitHub
- **fre:ac** — free audio converter with DSD support (https://www.freac.org)
- **Audirvāna / JRiver** — commercial players with DSD-to-PCM export

---

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New project |
| `Ctrl+O` | Open project |
| `Ctrl+S` | Save project |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+E` | Export audio (current tab) |
| `Space` | Stop all audio |
| `Ctrl+Q` | Quit |

---

## Troubleshooting

**"No module named PyQt6"** — Run `pip install PyQt6` again and make sure you are using the correct Python installation.

**"PortAudio not found" / sounddevice error** — Install the PortAudio library for your OS:
- Windows: usually bundled with the sounddevice wheel, no extra step needed.
- macOS: `brew install portaudio`
- Linux: `sudo apt install libportaudio2`

**No audio output** — Check that your system audio is not muted and that the correct output device is selected in your OS sound settings.

**MP3 / M4A export fails** — Make sure FFmpeg is installed and accessible from the command line (`ffmpeg -version` should work in a terminal).

**App opens but looks odd on a high-DPI screen** — Set the environment variable `QT_SCALE_FACTOR=1` before running, or let your OS handle scaling.

---

## Tips for non-traditional music

- Use **Free (Atonal)** scale mode in the Equation Synth for pure mathematical output with no note quantisation.
- Try **Logistic Map** with r values between 3.57 and 4.0 for chaotic, non-repeating sequences.
- Layer the Sequencer with ambient samples loaded in the Samples tab and play both simultaneously.
- The **ratio** mapping mode with the **Harmonic Series (1/n)** sequence produces descending overtone clouds — good for tonal drones.
- Use very long attack and release times (2–4 s) in the Synthesiser tab for evolving, ambient pads.
- Export individual layers as WAV and mix them in any DAW for a complete production workflow.
