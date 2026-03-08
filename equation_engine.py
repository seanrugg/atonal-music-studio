"""
equation_engine.py — Atonal Music Studio
==========================================
Mathematical sequence generators and frequency-mapping logic for the
equation synthesiser.

Each generator returns a list of raw float values.  Those values are then
passed through ``sequence_to_frequencies()`` which maps them to audible
frequencies according to the chosen base frequency, mapping mode, and
optional scale quantisation.

Supported sequences
-------------------
  Fibonacci              : 1, 1, 2, 3, 5, 8, 13 …
  Pi (π) digits          : 3, 1, 4, 1, 5, 9, 2, 6 …
  Euler's e digits       : 2, 7, 1, 8, 2, 8, 1, 8 …
  Golden Ratio (φ) powers: φ⁰, φ¹, φ², φ³ …
  Square Roots           : √1, √2, √3, √4 …
  Prime Numbers          : 2, 3, 5, 7, 11 …
  Harmonic Series (1/n)  : 1, 1/2, 1/3, 1/4 …
  Natural Harmonics (nx) : 1, 2, 3, 4 … (overtone series)
  Logistic Map (chaos)   : xₙ₊₁ = r·xₙ·(1−xₙ)
  Custom Expression      : any formula using i, pi, e, phi, sqrt, …

Mapping modes
-------------
  ratio    : freq = base_freq × |value|  (folded into audible range)
  semitone : semitone offset = round(|value|) % (12 × octave_range)
  modular  : |value| → scale degree (wraps within chosen scale)
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from audio_engine import generate_note, mix_to_stereo, SAMPLE_RATE

# ─────────────────────────────────────────────────────────────────────────────
# Scale definitions  (semitone intervals from root)
# ─────────────────────────────────────────────────────────────────────────────

SCALES: dict[str, list[int] | None] = {
    "Chromatic":           list(range(12)),
    "Major":               [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor":       [0, 2, 3, 5, 7, 8, 10],
    "Pentatonic Major":    [0, 2, 4, 7, 9],
    "Pentatonic Minor":    [0, 3, 5, 7, 10],
    "Whole Tone":          [0, 2, 4, 6, 8, 10],
    "Diminished":          [0, 2, 3, 5, 6, 8, 9, 11],
    "Augmented":           [0, 3, 4, 7, 8, 11],
    "Lydian":              [0, 2, 4, 6, 7, 9, 11],
    "Phrygian":            [0, 1, 3, 5, 7, 8, 10],
    "Locrian":             [0, 1, 3, 5, 6, 8, 10],
    "Free (Atonal)":       None,   # No quantisation — pure mathematical output
}

# ─────────────────────────────────────────────────────────────────────────────
# Scale quantisation helper
# ─────────────────────────────────────────────────────────────────────────────

def quantize_to_scale(
    freq: float,
    base_freq: float,
    scale_name: str,
    octave_range: int = 3,
) -> float:
    """
    Snap *freq* to the nearest pitch in the given scale.

    If *scale_name* is ``"Free (Atonal)"`` the frequency is returned unchanged.
    """
    intervals = SCALES.get(scale_name)
    if intervals is None:
        return freq

    # Build candidate frequencies across the requested octave range
    candidates: list[float] = []
    for oct_off in range(-1, octave_range + 1):
        for semitones in intervals:
            f = base_freq * (2.0 ** ((oct_off * 12 + semitones) / 12.0))
            if f > 0:
                candidates.append(f)

    if not candidates:
        return freq

    log_freq = math.log2(freq)
    log_cands = np.log2(np.array(candidates))
    idx = int(np.argmin(np.abs(log_cands - log_freq)))
    return candidates[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Individual sequence generators
# ─────────────────────────────────────────────────────────────────────────────

def fibonacci_sequence(n: int) -> list[float]:
    """Return the first *n* Fibonacci numbers as floats."""
    if n <= 0:
        return []
    seq = [1.0, 1.0]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def pi_sequence(n: int) -> list[float]:
    """
    Return *n* values derived from the decimal digits of π.
    Each digit + 1 is used so that no value is zero.
    """
    pi_digits = (
        "14159265358979323846264338327950288419716939937510"
        "58209749445923078164062862089986280348253421170679"
        "82148086513282306647093844609550582231725359408128"
        "48111745028410270193852110555964462294895493038196"
    )
    digits = [int(d) + 1 for d in pi_digits if d.isdigit()]
    return [float(d) for d in digits[:n]]


def e_sequence(n: int) -> list[float]:
    """
    Return *n* values derived from the decimal digits of Euler's number e.
    Each digit + 1 is used so that no value is zero.
    """
    e_digits = (
        "71828182845904523536028747135266249775724709369995"
        "95749669676277240766303535475945713821785251664274"
        "27466391932003059921817413596629043572900334295261"
        "79881236826966189208674273943066882264803032825095"
    )
    digits = [int(d) + 1 for d in e_digits if d.isdigit()]
    return [float(d) for d in digits[:n]]


def phi_powers(n: int) -> list[float]:
    """Return φ⁰, φ¹, φ², … φⁿ⁻¹ (golden ratio powers)."""
    phi = (1.0 + math.sqrt(5)) / 2.0
    return [phi ** i for i in range(n)]


def sqrt_sequence(n: int) -> list[float]:
    """Return √1, √2, √3, …, √n."""
    return [math.sqrt(i + 1) for i in range(n)]


def prime_sequence(n: int) -> list[float]:
    """Return the first *n* prime numbers."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return [float(p) for p in primes]


def harmonic_series(n: int) -> list[float]:
    """Return 1, 1/2, 1/3, … 1/n (sub-harmonic / harmonic series)."""
    return [1.0 / (i + 1) for i in range(n)]


def natural_harmonics(n: int) -> list[float]:
    """Return 1, 2, 3, … n (overtone series)."""
    return [float(i + 1) for i in range(n)]


def logistic_map(
    n: int,
    r: float = 3.7,
    x0: float = 0.5,
) -> list[float]:
    """
    Return *n* values from the logistic map  xₙ₊₁ = r·xₙ·(1−xₙ).

    With r ≈ 3.57–4.0 the system is chaotic, producing aperiodic sequences.
    Values are in (0, 1); multiply up in the mapping stage.
    """
    x = x0
    result = []
    for _ in range(n):
        result.append(x)
        x = r * x * (1.0 - x)
    return result


def custom_expression_sequence(n: int, expr: str) -> list[float]:
    """
    Evaluate *expr* for i = 0, 1, …, n-1 and return the results.

    Available names inside the expression:
      i    — current index (integer)
      pi   — π
      e    — Euler's number
      phi  — golden ratio  (1 + √5) / 2
      sqrt, sin, cos, tan, log, log2, exp, abs, pow, floor, ceil

    Raises ``ValueError`` if the expression fails for every index.
    """
    safe_env: dict = {
        "pi":    math.pi,
        "e":     math.e,
        "phi":   (1 + math.sqrt(5)) / 2,
        "sqrt":  math.sqrt,
        "sin":   math.sin,
        "cos":   math.cos,
        "tan":   math.tan,
        "log":   math.log,
        "log2":  math.log2,
        "exp":   math.exp,
        "abs":   abs,
        "pow":   pow,
        "floor": math.floor,
        "ceil":  math.ceil,
        "__builtins__": {},
    }

    results: list[float] = []
    errors = 0
    for i in range(n):
        safe_env["i"] = i
        try:
            val = float(eval(expr, safe_env))  # noqa: S307
            results.append(val)
        except Exception:
            results.append(1.0)
            errors += 1

    if errors == n:
        raise ValueError(
            f"Expression '{expr}' failed for all {n} values. "
            "Check syntax and available names."
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Registry — maps UI display names to generator callables
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE_GENERATORS: dict[str, Callable[[int], list[float]] | None] = {
    "Fibonacci":                fibonacci_sequence,
    "Pi (π) digits":            pi_sequence,
    "Euler's e digits":         e_sequence,
    "Golden Ratio (φ) powers":  phi_powers,
    "Square Roots (√n)":        sqrt_sequence,
    "Prime Numbers":            prime_sequence,
    "Harmonic Series (1/n)":    harmonic_series,
    "Natural Harmonics (n×)":   natural_harmonics,
    "Logistic Map (chaos)":     logistic_map,
    "Custom Expression":        None,   # Handled separately in UI
}


# ─────────────────────────────────────────────────────────────────────────────
# Sequence → frequency mapping
# ─────────────────────────────────────────────────────────────────────────────

def sequence_to_frequencies(
    sequence: list[float],
    base_freq: float,
    mode: str = "ratio",
    scale_name: str = "Chromatic",
    octave_range: int = 3,
) -> list[tuple[float, str]]:
    """
    Map a raw numerical sequence to audible frequencies.

    Parameters
    ----------
    sequence    : raw values from a generator
    base_freq   : reference frequency in Hz (root note)
    mode        : mapping strategy (see below)
    scale_name  : name from ``SCALES``; ``"Free (Atonal)"`` skips quantisation
    octave_range: how many octaves above/below base_freq to allow

    Modes
    -----
    ``ratio``    — freq = base_freq × |value|, folded into (20 Hz, 8 000 Hz)
    ``semitone`` — offset = round(|value|) % (12 × octave_range) semitones
    ``modular``  — map |value| to a scale degree (wraps the scale)

    Returns
    -------
    List of (frequency_hz, label_string) tuples.
    """
    results: list[tuple[float, str]] = []

    for val in sequence:
        abs_val = abs(float(val))

        if mode == "ratio":
            freq = base_freq * max(abs_val, 1e-9)
            # Fold into a sensible audible range
            while freq > 8_000.0:
                freq /= 2.0
            while freq < 20.0:
                freq *= 2.0

        elif mode == "semitone":
            max_semi = 12 * octave_range
            semitones = int(round(abs_val)) % max_semi
            freq = base_freq * (2.0 ** (semitones / 12.0))

        elif mode == "modular":
            intervals = SCALES.get(scale_name) or list(range(12))
            sl = len(intervals)
            idx = int(abs_val) % sl
            oct_off = int(abs_val) // sl
            freq = base_freq * (2.0 ** ((intervals[idx] + oct_off * 12) / 12.0))

        else:
            freq = base_freq * max(abs_val, 1e-9)

        # Quantise to scale if requested (only meaningful in ratio mode)
        if mode == "ratio" and scale_name != "Free (Atonal)":
            freq = quantize_to_scale(freq, base_freq, scale_name, octave_range)

        # Hard clamp to audible range
        freq = float(np.clip(freq, 20.0, 20_000.0))
        results.append((freq, f"{freq:.2f} Hz"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Audio rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_sequence_audio(
    freq_list: list[float],
    note_duration: float,
    wave_type: str = "sine",
    gap: float = 0.02,
    amplitude: float = 0.7,
    attack: float = 0.01,
    decay: float = 0.05,
    sustain: float = 0.8,
    release: float = 0.1,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Synthesise *freq_list* as a sequential melody.

    Each frequency is rendered as a note of *note_duration* seconds,
    optionally separated by *gap* seconds of silence.

    Returns a 1-D float32 numpy array ready for playback or export.
    """
    if not freq_list:
        return np.zeros(sr, dtype=np.float32)

    parts: list[np.ndarray] = []
    silence = np.zeros(int(gap * sr), dtype=np.float32) if gap > 0 else None

    for freq in freq_list:
        note = generate_note(
            freq, note_duration, wave_type, sr,
            amplitude, attack, decay, sustain, release,
        )
        parts.append(note)
        if silence is not None:
            parts.append(silence.copy())

    return np.concatenate(parts).astype(np.float32)
