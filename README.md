# Atonal Music Studio

A cross-platform desktop application for non-traditional music composition — atonal music, mathematical tone sequences, iterated function systems, and ambient soundscapes.

---

## Download & Install

> No Python installation required. Just download and run.

### Windows

1. Download **AtonalMusicStudio-Setup.exe** from the [Releases page](https://github.com/seanrugg/atonal-music-studio/releases)
2. Double-click the installer
3. If Windows shows a **"Windows protected your PC"** SmartScreen warning, click **"More info"** → **"Run anyway"** — this appears because the app is currently unsigned
4. Follow the installer steps — a Start Menu shortcut is created automatically
5. Launch **Atonal Music Studio** from the Start Menu

### macOS

1. Download **AtonalMusicStudio-1.0.0-mac.dmg** from the [Releases page](https://github.com/seanrugg/atonal-music-studio/releases)
2. Double-click the DMG to mount it
3. Drag **Atonal Music Studio** into your **Applications** folder
4. On first launch, **right-click the app → Open** (instead of double-clicking) — macOS Gatekeeper requires this one-time step for unsigned apps
5. Click **Open** in the confirmation dialog

### Linux

1. Download **AtonalMusicStudio-1.0.0-linux-x86_64.tar.gz** from the [Releases page](https://github.com/seanrugg/atonal-music-studio/releases)
2. Extract it:
   ```bash
   tar -xzf AtonalMusicStudio-1.0.0-linux-x86_64.tar.gz
   ```
3. Run the app:
   ```bash
   cd AtonalMusicStudio
   ./AtonalMusicStudio
   ```
4. Optional — add to your application menu:
   ```bash
   # Edit Exec= and Icon= paths first, then:
   cp atonal-music-studio.desktop ~/.local/share/applications/
   ```

**Linux system dependencies** (if the app fails to start):
```bash
sudo apt install libportaudio2 libxcb-cursor0   # Debian/Ubuntu
sudo dnf install portaudio libxcb               # Fedora
```

---

## Optional: FFmpeg (for MP3, M4A, AAC export)

The app exports WAV, FLAC, OGG, and AIFF natively. For MP3, M4A, and AAC export you need FFmpeg installed separately:

| Platform | Install command |
|----------|----------------|
| Windows  | Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to PATH |
| macOS    | `brew install ffmpeg` |
| Linux    | `sudo apt install ffmpeg` |

---

## Features

| Tab | What it does |
|-----|--------------|
| **Synthesiser** | Sine, sawtooth, square, and triangle wave oscillator with ADSR envelope and Schroeder reverb |
| **Equation Synth** | Choose a mathematical concept — Fibonacci, π, e, φ, √n, primes, Collatz, Triangular, Catalan, Van der Corput, logistic map, or write your own — and map it to a melody |
| **Sequencer** | 16-step × 8-track step sequencer with per-step velocity (right-click), swing, per-track pattern length (polyrhythm), and batch track export |
| **Samples** | Load audio files in any supported format, preview the waveform, adjust volume and playback speed |
| **Iterated f(x)** | Enter any function f(x), seed it with a starting frequency, and watch the orbit unfold — visualised as a cobweb diagram |

### Equation mapping modes

| Mode | Description |
|------|-------------|
| `ratio` | Multiply the base frequency by each sequence value, folded into the audible range |
| `semitone` | Treat each value as a semitone offset from the base frequency |
| `modular` | Map each value to a scale degree (wraps within the chosen scale) |

### Audio formats

| Format | Import | Export | Notes |
|--------|--------|--------|-------|
| WAV    | ✅ | ✅ | Always available |
| FLAC   | ✅ | ✅ | Lossless compressed |
| OGG    | ✅ | ✅ | Open lossy |
| AIFF   | ✅ | ✅ | Apple lossless |
| MP3    | ✅ | ✅ | Requires FFmpeg |
| M4A    | ✅ | ✅ | Requires FFmpeg |
| AAC    | ✅ | ✅ | Requires FFmpeg |
| DSF/DFF | ❌ | ❌ | DSD — convert to WAV first |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New project |
| `Ctrl+O` | Open project |
| `Ctrl+S` | Save project |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+E` | Export audio (current tab) |
| `Space` | Stop all audio |
| `Ctrl+Q` | Quit |
| Right-click step button | Set step velocity (Sequencer) |

---

## Project Files

Projects are saved as `.ams` files (JSON). They store all synthesiser settings, sequencer patterns, iterated function state, and the paths to any loaded sample files. Sample audio itself is not embedded — keep your sample files accessible.

---

## For Developers — Building from Source

### Requirements

- Python 3.10 or later
- Git

### Setup

```bash
git clone https://github.com/seanrugg/atonal-music-studio.git
cd atonal-music-studio

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # adds PyInstaller + Pillow
```

### Run from source

```bash
python main.py
```

### Build distributable

**Windows** (run in Command Prompt with venv active):
```bat
build_win.bat
```
Produces `dist\AtonalMusicStudio\AtonalMusicStudio.exe` and, if Inno Setup 6 is installed, `installer\AtonalMusicStudio-Setup.exe`.

Download Inno Setup 6: https://jrsoftware.org/isinfo.php

**macOS**:
```bash
chmod +x build_mac.sh
./build_mac.sh
```
Produces `dist/AtonalMusicStudio.app` and, if `create-dmg` is installed (`brew install create-dmg`), `dist/AtonalMusicStudio-1.0.0-mac.dmg`.

**Linux**:
```bash
chmod +x build_linux.sh
./build_linux.sh
```
Produces `dist/AtonalMusicStudio/` and `dist/AtonalMusicStudio-1.0.0-linux-x86_64.tar.gz`.

### Release checklist

- [ ] Update `AppVersion` in `installer/atonal_setup.iss`
- [ ] Update `APP_VERSION` in `build_mac.sh` and `build_linux.sh`
- [ ] Update `CFBundleShortVersionString` in `atonal.spec`
- [ ] Build on each target platform (Windows, macOS, Linux)
- [ ] Test each build on a clean machine before uploading
- [ ] Upload all three distributables to the GitHub Releases page

---

## Troubleshooting

**Windows SmartScreen warning** — Click "More info" → "Run anyway". This is expected for unsigned apps.

**macOS "app is damaged"** — Run in Terminal: `xattr -cr /Applications/AtonalMusicStudio.app`

**Linux: app won't start** — Install `libportaudio2` and `libxcb-cursor0` (see Linux install section above).

**No audio output** — Check your system audio is not muted and the correct output device is selected.

**MP3/M4A export fails** — Install FFmpeg and verify `ffmpeg -version` works in a terminal.

**App opens but looks odd on high-DPI screen** — Set `QT_SCALE_FACTOR=1` before launching, or use your OS display scaling settings.
