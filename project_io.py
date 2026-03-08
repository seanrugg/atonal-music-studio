"""
project_io.py — Atonal Music Studio
=====================================
Project persistence and audio export/import utilities.

Project files are stored as plain JSON (extension ``.ams``).  They contain
all synthesiser settings, sequencer state, and references to any external
sample files used in the project.

Audio I/O
---------
Native (via soundfile):
    WAV, FLAC, OGG, AIFF / AIF

Via pydub + FFmpeg (FFmpeg must be installed separately):
    MP3, M4A, AAC

DSD formats (DSF, DFF):
    Not natively supported.  Use an external DSD converter (e.g. dsd2pcm,
    Saracon, or JRiver Media Center) to convert to 24-bit WAV first.
"""

from __future__ import annotations

import json
import os
import copy
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_EXTENSION = ".ams"      # Atonal Music Studio project file
PROJECT_VERSION   = "1.0"

# soundfile can write these directly
_SF_FORMATS: dict[str, str] = {
    ".wav":  "WAV",
    ".flac": "FLAC",
    ".ogg":  "OGG",
    ".aiff": "AIFF",
    ".aif":  "AIFF",
}

# pydub handles these (requires FFmpeg)
_PYDUB_FORMATS: dict[str, str] = {
    ".mp3": "mp3",
    ".m4a": "ipod",   # pydub codec name for M4A/AAC
    ".aac": "adts",
}

ALL_SUPPORTED_EXTENSIONS = sorted(
    list(_SF_FORMATS.keys()) + list(_PYDUB_FORMATS.keys())
    + [".dsf", ".dff"]          # listed but unsupported — friendly error given
)

EXPORT_FILTER = (
    "Audio Files ("
    + " ".join(f"*{e}" for e in ALL_SUPPORTED_EXTENSIONS)
    + ");;"
    "WAV (*.wav);;FLAC (*.flac);;OGG (*.ogg);;AIFF (*.aiff *.aif);;"
    "MP3 (*.mp3);;M4A (*.m4a);;AAC (*.aac)"
)

IMPORT_FILTER = (
    "Audio Files ("
    + " ".join(f"*{e}" for e in ALL_SUPPORTED_EXTENSIONS)
    + ");;"
    "All Files (*)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Audio export
# ─────────────────────────────────────────────────────────────────────────────

def export_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: str,
) -> None:
    """
    Export *audio_data* to *output_path*.

    *audio_data* should be a float32 array, either (N,) mono or (N, 2) stereo.
    The format is inferred from the file extension.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    RuntimeError
        If a pydub/FFmpeg export fails.
    """
    path = Path(output_path)
    ext = path.suffix.lower()

    # Ensure stereo float32
    if audio_data.ndim == 1:
        audio_data = np.column_stack([audio_data, audio_data])
    audio_data = np.clip(audio_data.astype(np.float32), -1.0, 1.0)

    if ext in (".dsf", ".dff"):
        raise ValueError(
            "DSD formats (DSF / DFF) are not supported for direct export.\n"
            "Export as WAV (24-bit or 32-bit) and convert with a DSD tool."
        )

    if ext in _SF_FORMATS:
        sf.write(str(output_path), audio_data, sample_rate, format=_SF_FORMATS[ext])
        return

    if ext in _PYDUB_FORMATS:
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise RuntimeError(
                f"pydub is required to export {ext} files.\n"
                "Install it with:  pip install pydub\n"
                "You also need FFmpeg installed on your system."
            ) from exc

        # Write a temporary WAV, then convert
        tmp_wav = str(path.with_suffix(".~tmp.wav"))
        try:
            sf.write(tmp_wav, audio_data, sample_rate, format="WAV")
            seg = AudioSegment.from_wav(tmp_wav)
            codec_name = _PYDUB_FORMATS[ext]
            if ext == ".mp3":
                seg.export(str(output_path), format="mp3", bitrate="320k")
            elif ext == ".m4a":
                seg.export(str(output_path), format="mp4", codec="aac")
            elif ext == ".aac":
                seg.export(str(output_path), format="adts")
            else:
                seg.export(str(output_path), format=codec_name)
        except Exception as exc:
            raise RuntimeError(
                f"FFmpeg export to {ext} failed: {exc}\n"
                "Make sure FFmpeg is installed and on your PATH."
            ) from exc
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        return

    raise ValueError(
        f"Unsupported export format: '{ext}'.\n"
        f"Supported: {', '.join(sorted(_SF_FORMATS) + sorted(_PYDUB_FORMATS))}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Audio import
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_file(path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return ``(audio_data, sample_rate)``.

    *audio_data* is always returned as a float32 (N, 2) stereo array.

    Raises
    ------
    ValueError
        For unsupported or DSD formats.
    RuntimeError
        If a pydub/FFmpeg load fails.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".dsf", ".dff"):
        raise ValueError(
            "DSD formats (DSF / DFF) cannot be loaded directly.\n"
            "Convert to WAV first using a DSD tool."
        )

    if ext in _SF_FORMATS or ext in (".wav", ".flac", ".ogg", ".aiff", ".aif"):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # Ensure stereo
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        return data, sr

    if ext in _PYDUB_FORMATS or ext in (".mp3", ".m4a", ".aac"):
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise RuntimeError(
                f"pydub is required to load {ext} files.\n"
                "Install it with:  pip install pydub\n"
                "You also need FFmpeg installed on your system."
            ) from exc
        try:
            seg = AudioSegment.from_file(str(path))
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            # Normalise int samples to float32 [-1, 1]
            max_val = float(2 ** (seg.sample_width * 8 - 1))
            samples /= max_val
            if seg.channels == 1:
                data = np.column_stack([samples, samples])
            elif seg.channels == 2:
                data = samples.reshape(-1, 2)
            else:
                data = samples.reshape(-1, seg.channels)[:, :2]
            return data.astype(np.float32), sr
        except Exception as exc:
            raise RuntimeError(
                f"Could not load '{p.name}': {exc}\n"
                "Make sure FFmpeg is installed and on your PATH."
            ) from exc

    raise ValueError(
        f"Unsupported audio format: '{ext}'.\n"
        f"Supported: {', '.join(ALL_SUPPORTED_EXTENSIONS)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Project data structure
# ─────────────────────────────────────────────────────────────────────────────

def new_project() -> dict[str, Any]:
    """Return a default empty project dictionary."""
    return {
        "version": PROJECT_VERSION,
        "name": "Untitled Project",
        # Transport
        "bpm": 120,
        "steps": 16,
        # Synthesiser tab state
        "synth": {
            "wave_type": "sine",
            "frequency": 440.0,
            "duration": 1.0,
            "amplitude": 0.7,
            "attack":   0.01,
            "decay":    0.05,
            "sustain":  0.8,
            "release":  0.1,
            "duty":     0.5,
        },
        # Equation synthesiser tab state
        "equation": {
            "sequence_type": "Fibonacci",
            "custom_expr":   "sin(i * pi / 4) + 1",
            "n_notes":       16,
            "base_freq":     220.0,
            "scale":         "Chromatic",
            "mode":          "ratio",
            "octave_range":  3,
            "wave_type":     "sine",
            "note_duration": 0.3,
            "gap":           0.02,
            "attack":        0.01,
            "release":       0.1,
        },
        # Sequencer — list of tracks, each track has a step pattern
        "sequencer": {
            "tracks": [
                {
                    "name": f"Track {i + 1}",
                    "wave_type": "sine",
                    "note": "A4",
                    "volume": 0.7,
                    "muted": False,
                    "steps": [False] * 16,   # 16 steps, all off
                }
                for i in range(8)
            ],
        },
        # Loaded sample paths (list of absolute path strings)
        "samples": [],
    }


def _sanitise_project(data: dict[str, Any]) -> dict[str, Any]:
    """
    Merge *data* into a fresh default project, so that old or partial
    project files load safely even if fields are missing.
    """
    base = new_project()
    base.update({k: v for k, v in data.items() if k in base})
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_project(path: str, project_data: dict[str, Any]) -> None:
    """
    Serialise *project_data* to *path* as pretty-printed JSON.

    The file will have a ``.ams`` extension by convention, but any path
    may be used.
    """
    data = copy.deepcopy(project_data)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str, ensure_ascii=False)


def load_project(path: str) -> dict[str, Any]:
    """
    Load a project from *path* and return the sanitised project dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is not valid JSON or has an incompatible version.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Project file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    version = raw.get("version", "0.0")
    if version != PROJECT_VERSION:
        # Attempt to load anyway; _sanitise_project fills in defaults
        pass

    return _sanitise_project(raw)
