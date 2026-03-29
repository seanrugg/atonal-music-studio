"""
audio_engine.py — Atonal Music Studio
======================================
Core audio synthesis engine.

Provides:
  - Waveform generators  : sine, sawtooth, square, triangle
  - ADSR envelope shaping
  - Note generation
  - Schroeder reverb (comb + allpass network, no external deps)
  - Stereo mixing utilities
  - AudioEngine class for playback (sounddevice-backed)

All audio is represented as float32 NumPy arrays normalised to [-1, 1].
Sample rate defaults to 44 100 Hz (CD quality).
"""

from __future__ import annotations

import threading
import numpy as np
from scipy import signal
import sounddevice as sd
import soundfile as sf

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = 44_100

_CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES: list[str] = [f"{n}{o}" for o in range(9) for n in _CHROMATIC]

WAVE_TYPES: list[str] = ["Sine", "Sawtooth", "Square", "Triangle"]


# ─────────────────────────────────────────────────────────────────────────────
# Pitch helpers
# ─────────────────────────────────────────────────────────────────────────────

def midi_to_freq(midi_note: float) -> float:
    """Convert a MIDI note number (0–127) to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def freq_to_midi(freq: float) -> float:
    """Convert a frequency in Hz to the nearest MIDI note number."""
    return 69 + 12 * np.log2(max(freq, 1e-9) / 440.0)


def note_name_to_midi(note_name: str) -> int:
    """
    Convert a note name such as 'A4', 'C#3', or 'Bb2' to a MIDI note number.
    Raises ValueError for unrecognised names.
    """
    _NOTE_MAP = {
        "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
        "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
        "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
    }
    name = note_name.strip()
    if len(name) >= 2 and name[1] in "#b":
        note_part = name[:2].upper().replace("B", "B")
        octave_part = name[2:]
    else:
        note_part = name[0].upper()
        octave_part = name[1:]
    note_part_key = note_part.replace("b", "B")
    if note_part_key not in _NOTE_MAP:
        raise ValueError(f"Unrecognised note: '{note_name}'")
    octave = int(octave_part)
    return (octave + 1) * 12 + _NOTE_MAP[note_part_key]


def nearest_note_name(freq: float) -> str:
    """Return the name of the note nearest to *freq* Hz."""
    midi = int(round(freq_to_midi(freq)))
    midi = max(0, min(len(NOTE_NAMES) - 1, midi))
    return NOTE_NAMES[midi]


# ─────────────────────────────────────────────────────────────────────────────
# Waveform generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_sine(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amplitude: float = 0.7,
) -> np.ndarray:
    """Return a float32 sine wave of *duration* seconds."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def generate_sawtooth(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amplitude: float = 0.7,
) -> np.ndarray:
    """Return a float32 sawtooth wave of *duration* seconds."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (amplitude * signal.sawtooth(2.0 * np.pi * freq * t)).astype(np.float32)


def generate_square(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amplitude: float = 0.7,
    duty: float = 0.5,
) -> np.ndarray:
    """Return a float32 square wave of *duration* seconds."""
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    return (amplitude * signal.square(2.0 * np.pi * freq * t, duty=duty)).astype(np.float32)


def generate_triangle(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amplitude: float = 0.7,
) -> np.ndarray:
    """
    Return a float32 triangle wave of *duration* seconds.

    A triangle wave has softer harmonics than square or sawtooth (only odd
    harmonics, falling off at 1/n²) — sits tonally between sine and sawtooth.
    """
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    # scipy sawtooth with width=0.5 gives a triangle wave
    return (amplitude * signal.sawtooth(2.0 * np.pi * freq * t, width=0.5)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ADSR envelope
# ─────────────────────────────────────────────────────────────────────────────

def apply_adsr(
    buf: np.ndarray,
    sr: int,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
) -> np.ndarray:
    """
    Apply an ADSR volume envelope to *buf* (1-D float32 array).

    Parameters
    ----------
    attack   : ramp-up time in seconds
    decay    : drop-to-sustain time in seconds
    sustain  : sustain level  0.0 – 1.0
    release  : fade-out time in seconds
    """
    n = len(buf)
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = max(0, n - a - d - r)

    env = np.empty(n, dtype=np.float32)
    pos = 0

    end = min(pos + a, n)
    env[pos:end] = np.linspace(0.0, 1.0, end - pos, dtype=np.float32)
    pos = end

    end = min(pos + d, n)
    env[pos:end] = np.linspace(1.0, sustain, end - pos, dtype=np.float32)
    pos = end

    end = min(pos + s, n)
    env[pos:end] = sustain
    pos = end

    end = min(pos + r, n)
    env[pos:end] = np.linspace(sustain, 0.0, end - pos, dtype=np.float32)
    pos = end

    if pos < n:
        env[pos:] = 0.0

    return buf * env


# ─────────────────────────────────────────────────────────────────────────────
# High-level note generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_note(
    freq: float,
    duration: float,
    wave_type: str = "sine",
    sr: int = SAMPLE_RATE,
    amplitude: float = 0.7,
    attack: float = 0.01,
    decay: float = 0.05,
    sustain: float = 0.8,
    release: float = 0.1,
    duty: float = 0.5,
) -> np.ndarray:
    """
    Generate a single synthesised note with an ADSR envelope.

    Parameters
    ----------
    wave_type : 'sine' | 'sawtooth' | 'square' | 'triangle'
    duty      : square-wave duty cycle (ignored for other wave types)
    """
    wt = wave_type.lower()
    if wt == "sine":
        buf = generate_sine(freq, duration, sr, amplitude)
    elif wt == "sawtooth":
        buf = generate_sawtooth(freq, duration, sr, amplitude)
    elif wt == "square":
        buf = generate_square(freq, duration, sr, amplitude, duty)
    elif wt == "triangle":
        buf = generate_triangle(freq, duration, sr, amplitude)
    else:
        buf = generate_sine(freq, duration, sr, amplitude)

    return apply_adsr(buf, sr, attack, decay, sustain, release)


# ─────────────────────────────────────────────────────────────────────────────
# Schroeder Reverb
# ─────────────────────────────────────────────────────────────────────────────

# Schroeder reverb delay times (in seconds) — classic Moorer/Schroeder values
# scaled slightly for a musical room feel.
_COMB_DELAYS_S   = [0.02971, 0.03372, 0.03671, 0.04013]
_ALLPASS_DELAYS_S = [0.005, 0.0017]
_COMB_GAIN       = 0.84   # feedback gain for comb filters
_ALLPASS_GAIN    = 0.7    # allpass coefficient


def _comb_filter(
    x: np.ndarray,
    delay_samples: int,
    gain: float,
) -> np.ndarray:
    """Single feedback comb filter."""
    y = np.zeros_like(x)
    buf = np.zeros(delay_samples, dtype=np.float32)
    ptr = 0
    for i in range(len(x)):
        delayed = buf[ptr]
        y[i] = x[i] + gain * delayed
        buf[ptr] = x[i] + gain * delayed
        ptr = (ptr + 1) % delay_samples
    return y


def _allpass_filter(
    x: np.ndarray,
    delay_samples: int,
    gain: float,
) -> np.ndarray:
    """Single Schroeder allpass filter."""
    y = np.zeros_like(x)
    buf = np.zeros(delay_samples, dtype=np.float32)
    ptr = 0
    for i in range(len(x)):
        delayed = buf[ptr]
        v = x[i] + gain * delayed
        y[i] = -gain * v + delayed
        buf[ptr] = v
        ptr = (ptr + 1) % delay_samples
    return y


def apply_reverb(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    room_size: float = 0.5,
    wet: float = 0.3,
) -> np.ndarray:
    """
    Apply a Schroeder reverb to *audio*.

    Works on both mono (N,) and stereo (N, 2) float32 arrays.
    Returns the same shape as input.

    Parameters
    ----------
    room_size : 0.0–1.0 — scales comb filter feedback gain.
                0.0 = very short / tight room,  1.0 = large / long tail.
    wet       : 0.0–1.0 — mix between dry (0) and wet (1) signal.
                Typical useful range is 0.1–0.5.
    """
    room_size = float(np.clip(room_size, 0.0, 1.0))
    wet       = float(np.clip(wet,       0.0, 1.0))

    # Scale comb gain by room_size  (0.5 → gain, 1.0 → gain*1.12, 0.0 → gain*0.6)
    gain = _COMB_GAIN * (0.6 + 0.52 * room_size)

    stereo_input = audio.ndim == 2
    if stereo_input:
        # Process each channel independently
        left  = apply_reverb(audio[:, 0], sr, room_size, wet)
        right = apply_reverb(audio[:, 1], sr, room_size, wet)
        return np.column_stack([left, right]).astype(np.float32)

    mono = audio.astype(np.float32)

    # Parallel comb filters
    comb_out = np.zeros_like(mono)
    for delay_s in _COMB_DELAYS_S:
        d = max(1, int(delay_s * sr))
        comb_out += _comb_filter(mono, d, gain)
    comb_out /= len(_COMB_DELAYS_S)

    # Series allpass filters
    ap_out = comb_out
    for delay_s in _ALLPASS_DELAYS_S:
        d = max(1, int(delay_s * sr))
        ap_out = _allpass_filter(ap_out, d, _ALLPASS_GAIN)

    result = (1.0 - wet) * mono + wet * ap_out
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Stereo utilities
# ─────────────────────────────────────────────────────────────────────────────

def mono_to_stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """
    Convert a mono float32 array to a stereo (N, 2) float32 array.

    pan : -1.0 = hard left, 0.0 = centre, +1.0 = hard right
    """
    pan = float(np.clip(pan, -1.0, 1.0))
    left_gain  = float(np.sqrt(max(0.0, (1.0 - pan) / 2.0)))
    right_gain = float(np.sqrt(max(0.0, (1.0 + pan) / 2.0)))
    return np.column_stack(
        [mono * left_gain, mono * right_gain]
    ).astype(np.float32)


def mix_to_stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """Alias kept for backward compatibility."""
    return mono_to_stereo(mono, pan)


def mix_buffers(
    buffers: list[np.ndarray],
    volumes: list[float] | None = None,
    pans: list[float] | None = None,
) -> np.ndarray:
    """
    Mix a list of mono float32 buffers into a single stereo (N, 2) buffer.

    Lengths need not match; shorter buffers are zero-padded.
    The output is normalised if the peak exceeds 1.0.
    """
    if not buffers:
        return np.zeros((SAMPLE_RATE, 2), dtype=np.float32)

    if volumes is None:
        volumes = [1.0] * len(buffers)
    if pans is None:
        pans = [0.0] * len(buffers)

    max_len = max(len(b) for b in buffers)
    mixed = np.zeros((max_len, 2), dtype=np.float32)

    for buf, vol, pan in zip(buffers, volumes, pans):
        stereo = mono_to_stereo(buf * float(vol), pan)
        mixed[: len(stereo)] += stereo

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    return mixed


def normalise(buf: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalise *buf* so its peak amplitude equals *target_peak*."""
    peak = np.max(np.abs(buf))
    if peak < 1e-9:
        return buf
    return (buf / peak * target_peak).astype(np.float32)


def fade_in_out(
    buf: np.ndarray,
    sr: int = SAMPLE_RATE,
    fade_s: float = 0.01,
) -> np.ndarray:
    """Apply a short linear fade-in and fade-out to avoid clicks."""
    fade_n = int(fade_s * sr)
    if len(buf) < fade_n * 2:
        return buf
    result = buf.copy()
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    if result.ndim == 1:
        result[:fade_n]  *= ramp
        result[-fade_n:] *= ramp[::-1]
    else:
        result[:fade_n]  *= ramp[:, np.newaxis]
        result[-fade_n:] *= ramp[::-1, np.newaxis]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AudioEngine — playback manager
# ─────────────────────────────────────────────────────────────────────────────

class AudioEngine:
    """
    Thin wrapper around sounddevice that provides non-blocking playback
    with loop support and thread-safe stop control.

    Usage::

        engine = AudioEngine()
        engine.play(my_buffer)
        engine.play(my_buffer, loop=True)
        engine.stop()
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._stream: sd.OutputStream | None = None
        self._buffer: np.ndarray | None = None
        self._pos: int = 0
        self._loop: bool = False
        self._playing: bool = False
        self._lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, audio_data: np.ndarray, loop: bool = False) -> None:
        """
        Play *audio_data* (mono or stereo float32).

        If *loop* is True the buffer plays continuously until stop() is called.
        """
        self.stop()

        if audio_data.ndim == 1:
            audio_data = mono_to_stereo(audio_data)
        buf = np.clip(audio_data.astype(np.float32), -1.0, 1.0)

        with self._lock:
            self._buffer = buf
            self._pos = 0
            self._loop = loop
            self._playing = True

        self._start_stream()

    def stop(self) -> None:
        """Stop playback immediately."""
        with self._lock:
            self._playing = False
            self._buffer = None

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            if not self._playing or self._buffer is None:
                outdata[:] = 0.0
                raise sd.CallbackStop()

            buf = self._buffer
            remaining = len(buf) - self._pos

            if remaining <= 0:
                if self._loop:
                    self._pos = 0
                    remaining = len(buf)
                else:
                    outdata[:] = 0.0
                    self._playing = False
                    raise sd.CallbackStop()

            to_copy = min(frames, remaining)
            outdata[:to_copy] = buf[self._pos : self._pos + to_copy]
            if to_copy < frames:
                outdata[to_copy:] = 0.0
            self._pos += to_copy

    def _start_stream(self) -> None:
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                blocksize=1024,
                callback=self._callback,
            )
            self._stream.start()
        except Exception:
            self._playing = False
            raise

    def __del__(self) -> None:
        self.stop()
