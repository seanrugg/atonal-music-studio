"""
project_io.py — Atonal Music Studio
=====================================
Project persistence, audio export/import, recent-projects tracking,
and batch track export utilities.

Project files are stored as plain JSON (extension ``.ams``).  They contain
all synthesiser settings, sequencer state, and references to any external
sample files used in the project.

Recent Projects
---------------
A sidecar file ``<project_name>.ams.recent`` is written alongside each
``.ams`` file.  It stores a JSON list of up to MAX_RECENT_PROJECTS
absolute paths, most-recently-used first.  The sidecar lives in the same
directory as the project file so that project folders remain self-contained.

Batch Export
------------
``batch_export_tracks()`` renders each active sequencer track to a separate
WAV file in a nominated output directory.  Track files are named
``track_01_<note>.wav`` … ``track_08_<note>.wav``.

Audio I/O
---------
Native (via soundfile):
    WAV, FLAC, OGG, AIFF / AIF

Via pydub + FFmpeg (FFmpeg must be installed separately):
    MP3, M4A, AAC

DSD formats (DSF, DFF):
    Not natively supported.  Use an external DSD converter first.
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

PROJECT_EXTENSION   = ".ams"
PROJECT_VERSION     = "1.0"
RECENT_EXTENSION    = ".ams.recent"
MAX_RECENT_PROJECTS = 10

_SF_FORMATS: dict[str, str] = {
    ".wav":  "WAV",
    ".flac": "FLAC",
    ".ogg":  "OGG",
    ".aiff": "AIFF",
    ".aif":  "AIFF",
}

_PYDUB_FORMATS: dict[str, str] = {
    ".mp3": "mp3",
    ".m4a": "ipod",
    ".aac": "adts",
}

ALL_SUPPORTED_EXTENSIONS = sorted(
    list(_SF_FORMATS.keys()) + list(_PYDUB_FORMATS.keys())
    + [".dsf", ".dff"]
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
        "bpm": 120,
        "steps": 16,
        "synth": {
            "wave_type":  "sine",
            "frequency":  440.0,
            "duration":   1.0,
            "amplitude":  0.7,
            "attack":     0.01,
            "decay":      0.05,
            "sustain":    0.8,
            "release":    0.1,
            "duty":       0.5,
            "reverb_enabled": False,
            "reverb_room":    0.5,
            "reverb_wet":     0.3,
        },
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
            "reverb_enabled": False,
            "reverb_room":    0.5,
            "reverb_wet":     0.3,
        },
        "iterated_function": {
            "expr":           "x * 1.5",
            "seed_freq":      220.0,
            "n_steps":        16,
            "scale":          "Free (Atonal)",
            "octave_range":   3,
            "wave_type":      "sine",
            "note_duration":  0.3,
            "gap":            0.05,
            "attack":         0.01,
            "release":        0.1,
            "reverb_enabled": False,
            "reverb_room":    0.5,
            "reverb_wet":     0.3,
        },
        "sequencer": {
            "tracks": [
                {
                    "name":      f"Track {i + 1}",
                    "wave_type": "sine",
                    "note":      "A4",
                    "volume":    0.7,
                    "muted":     False,
                    "steps":     [False] * 16,
                    "velocities": [1.0] * 16,
                    "pattern_length": 16,
                }
                for i in range(8)
            ],
            "swing": 0.0,
        },
        "samples": [],
    }


def _sanitise_project(data: dict[str, Any]) -> dict[str, Any]:
    """
    Merge *data* into a fresh default project so that old or partial
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
    Serialise *project_data* to *path* as pretty-printed JSON and record
    it in the recent-projects sidecar.
    """
    data = copy.deepcopy(project_data)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
    _record_recent_project(path)


def load_project(path: str) -> dict[str, Any]:
    """
    Load a project from *path* and return the sanitised project dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is not valid JSON.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Project file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    _record_recent_project(path)
    return _sanitise_project(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Recent projects sidecar
# ─────────────────────────────────────────────────────────────────────────────

def _sidecar_path(project_path: str) -> str:
    """Return the path of the recent-projects sidecar for *project_path*."""
    p = Path(project_path)
    return str(p.parent / (p.stem + RECENT_EXTENSION))


def _record_recent_project(project_path: str) -> None:
    """
    Add *project_path* to the recent-projects sidecar (MRU first).

    Creates the sidecar if it does not exist.  Silently ignores I/O errors
    so that a missing or unwritable sidecar never breaks a save/load.
    """
    abs_path = str(Path(project_path).resolve())
    sidecar = _sidecar_path(project_path)
    try:
        recents = _load_recent_projects_raw(sidecar)
        # Remove duplicates, prepend new entry, trim to max
        recents = [p for p in recents if p != abs_path]
        recents.insert(0, abs_path)
        recents = recents[:MAX_RECENT_PROJECTS]
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(recents, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _load_recent_projects_raw(sidecar: str) -> list[str]:
    """Return the raw list from a sidecar file, or [] on any error."""
    try:
        with open(sidecar, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [str(p) for p in data]
    except Exception:
        pass
    return []


def get_recent_projects(project_path: str) -> list[str]:
    """
    Return a list of recently opened/saved project paths for the project
    at *project_path*, filtered to paths that still exist on disk.

    The sidecar is stored alongside the project file.
    """
    sidecar = _sidecar_path(project_path)
    raw = _load_recent_projects_raw(sidecar)
    return [p for p in raw if os.path.isfile(p)]


def get_recent_projects_from_dir(directory: str) -> list[str]:
    """
    Scan *directory* for any ``.ams.recent`` sidecar files and aggregate
    all recent paths.  Useful for populating the File → Recent Projects
    menu when no project is currently open.
    """
    aggregated: list[str] = []
    seen: set[str] = set()
    try:
        for entry in Path(directory).iterdir():
            if entry.name.endswith(RECENT_EXTENSION):
                for p in _load_recent_projects_raw(str(entry)):
                    if p not in seen and os.path.isfile(p):
                        aggregated.append(p)
                        seen.add(p)
    except Exception:
        pass
    return aggregated[:MAX_RECENT_PROJECTS]


# ─────────────────────────────────────────────────────────────────────────────
# Batch track export
# ─────────────────────────────────────────────────────────────────────────────

def batch_export_tracks(
    sequencer_state: dict[str, Any],
    output_dir: str,
    sample_rate: int,
    file_format: str = ".wav",
) -> list[str]:
    """
    Render each active sequencer track to a separate audio file.

    Parameters
    ----------
    sequencer_state : the dict returned by ``SequencerTab.get_state()``
    output_dir      : directory to write files into (created if absent)
    sample_rate     : audio sample rate (typically SAMPLE_RATE = 44 100)
    file_format     : extension including dot, e.g. ``".wav"`` or ``".flac"``

    Returns
    -------
    List of paths written (one per active, non-muted track).

    Raises
    ------
    ValueError  if *file_format* is not supported
    RuntimeError on FFmpeg/pydub failure
    """
    from audio_engine import generate_note, mix_buffers, midi_to_freq, note_name_to_midi

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bpm        = int(sequencer_state.get("bpm", 120))
    swing      = float(sequencer_state.get("swing", 0.0))
    tracks     = sequencer_state.get("tracks", [])
    step_dur   = 60.0 / bpm / 4.0   # 16th-note duration at this BPM

    written: list[str] = []

    for t_idx, track in enumerate(tracks):
        if track.get("muted", False):
            continue

        steps      = track.get("steps", [False] * 16)
        velocities = track.get("velocities", [1.0] * len(steps))
        pat_len    = int(track.get("pattern_length", len(steps)))
        note_name  = track.get("note", "A4")
        wave       = track.get("wave_type", "sine")
        volume     = float(track.get("volume", 0.7))

        active_steps = [s for s in steps[:pat_len] if s]
        if not any(steps[:pat_len]):
            continue   # skip silent tracks

        try:
            freq = midi_to_freq(note_name_to_midi(note_name))
        except Exception:
            freq = 440.0

        note_dur   = step_dur * 0.9
        total_len  = int(step_dur * pat_len * sample_rate)
        track_buf  = np.zeros(total_len, dtype=np.float32)

        for s_idx in range(pat_len):
            if s_idx >= len(steps) or not steps[s_idx]:
                continue

            # Swing: delay even-numbered steps (0-indexed)
            swing_offset = 0.0
            if s_idx % 2 == 1 and swing > 0.0:
                swing_offset = step_dur * swing * 0.5

            vel = float(velocities[s_idx]) if s_idx < len(velocities) else 1.0
            amplitude = volume * vel

            note_buf = generate_note(
                freq, note_dur, wave, sample_rate, amplitude,
                attack=0.005, decay=0.02, sustain=0.8, release=0.05,
            )
            raw_offset   = s_idx * step_dur * sample_rate
            swing_samples = int(swing_offset * sample_rate)
            offset        = int(raw_offset) + swing_samples
            end           = min(offset + len(note_buf), total_len)
            if offset < total_len:
                track_buf[offset:end] += note_buf[: end - offset]

        # Normalise and convert to stereo
        peak = np.max(np.abs(track_buf))
        if peak > 1.0:
            track_buf /= peak

        from audio_engine import mono_to_stereo
        stereo = mono_to_stereo(track_buf)

        # Build filename:  track_01_A4.wav
        safe_note = note_name.replace("#", "s")   # avoid # in filenames
        filename  = f"track_{t_idx + 1:02d}_{safe_note}{file_format}"
        out_path  = str(out_dir / filename)

        export_audio(stereo, sample_rate, out_path)
        written.append(out_path)

    return written
