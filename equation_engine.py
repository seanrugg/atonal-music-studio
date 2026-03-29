"""
equation_engine.py — Atonal Music Studio
==========================================
Mathematical sequence generators and frequency-mapping logic for the
equation synthesiser, plus the iterated-function engine for the
Iterated Function tab.

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
  Collatz Sequence       : 3n+1 conjecture path to 1
  Triangular Numbers     : 1, 3, 6, 10, 15 …
  Catalan Numbers        : 1, 1, 2, 5, 14, 42 …
  Van der Corput         : low-discrepancy base-2 sequence in (0,1)
  Custom Expression      : any formula using i, pi, e, phi, sqrt, …

Mapping modes
-------------
  ratio    : freq = base_freq × |value|  (folded into audible range)
  semitone : semitone offset = round(|value|) % (12 × octave_range)
  modular  : |value| → scale degree (wraps within chosen scale)

Iterated Function engine
------------------------
  iterate_function(expr, seed, n, …) runs f(x) repeatedly where each
  output becomes the next input.  Stops early if the frequency leaves the
  audible range, returning the partial orbit and a termination reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
# Sequence metadata  (shown as tooltips / descriptions in the UI)
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE_METADATA: dict[str, str] = {
    "Fibonacci": (
        "Each term is the sum of the two preceding terms: 1, 1, 2, 3, 5, 8 …\n"
        "Ratios converge to the golden ratio φ ≈ 1.618, producing near-just intervals."
    ),
    "Pi (π) digits": (
        "Decimal digits of π (plus 1 so no zeros): 3,1,4,1,5,9,2,6 …\n"
        "Statistically normal — no pattern, good for pseudo-random atonal sequences."
    ),
    "Euler's e digits": (
        "Decimal digits of e (plus 1): 2,7,1,8,2,8,1,8 …\n"
        "Like π digits but with a different statistical distribution."
    ),
    "Golden Ratio (φ) powers": (
        "Successive powers of φ = (1+√5)/2 ≈ 1.618: 1, 1.618, 2.618, 4.236 …\n"
        "In ratio mode these produce the harmonic series of the golden ratio."
    ),
    "Square Roots (√n)": (
        "√1, √2, √3, √4 … — irrational for non-perfect squares.\n"
        "Produces a slowly rising, compressed frequency curve."
    ),
    "Prime Numbers": (
        "2, 3, 5, 7, 11, 13 … — no closed-form pattern.\n"
        "In semitone mode creates angular, unpredictable melodic leaps."
    ),
    "Harmonic Series (1/n)": (
        "1, 1/2, 1/3, 1/4 … — the sub-harmonic series.\n"
        "In ratio mode with a high base frequency creates descending overtone clouds."
    ),
    "Natural Harmonics (n×)": (
        "1, 2, 3, 4 … — the overtone series.\n"
        "Intervals between harmonics get smaller as n grows (octave, fifth, fourth …)."
    ),
    "Logistic Map (chaos)": (
        "xₙ₊₁ = r·xₙ·(1−xₙ).  With r > 3.57 the system is chaotic.\n"
        "Values stay in (0,1) — multiply by octave range to spread across pitches."
    ),
    "Collatz Sequence": (
        "Start from n: if even → n/2, if odd → 3n+1.  Repeat until reaching 1.\n"
        "All tested starting values eventually reach 1 (unproven in general).\n"
        "Produces dramatic rises and falls — musically unpredictable."
    ),
    "Triangular Numbers": (
        "1, 3, 6, 10, 15, 21 … — T(n) = n(n+1)/2.\n"
        "Grows quadratically; in semitone mode produces an accelerating rise."
    ),
    "Catalan Numbers": (
        "1, 1, 2, 5, 14, 42 … — C(n) = C(2n,n)/(n+1).\n"
        "Counts many combinatorial structures.  Rapid exponential growth;\n"
        "use modular mode to keep pitches in range."
    ),
    "Van der Corput": (
        "Low-discrepancy sequence in (0,1) — base-2 bit reversal.\n"
        "More evenly distributed than pseudo-random; fills pitch space uniformly\n"
        "without clustering.  Good for systematic atonal exploration."
    ),
    "Custom Expression": (
        "Enter any expression using i (step index 0…n-1), pi, e, phi,\n"
        "sqrt, sin, cos, tan, log, log2, exp, abs, pow, floor, ceil.\n"
        "Example:  sin(i * pi / 4) + 1"
    ),
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
    """Return *n* values derived from the decimal digits of π (digit + 1)."""
    pi_digits = (
        "14159265358979323846264338327950288419716939937510"
        "58209749445923078164062862089986280348253421170679"
        "82148086513282306647093844609550582231725359408128"
        "48111745028410270193852110555964462294895493038196"
    )
    digits = [int(d) + 1 for d in pi_digits if d.isdigit()]
    return [float(d) for d in digits[:n]]


def e_sequence(n: int) -> list[float]:
    """Return *n* values derived from the decimal digits of e (digit + 1)."""
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
    """Return 1, 1/2, 1/3, … 1/n."""
    return [1.0 / (i + 1) for i in range(n)]


def natural_harmonics(n: int) -> list[float]:
    """Return 1, 2, 3, … n (overtone series)."""
    return [float(i + 1) for i in range(n)]


def logistic_map(n: int, r: float = 3.7, x0: float = 0.5) -> list[float]:
    """
    Return *n* values from the logistic map  xₙ₊₁ = r·xₙ·(1−xₙ).

    With r ≈ 3.57–4.0 the system is chaotic.
    """
    x = x0
    result = []
    for _ in range(n):
        result.append(x)
        x = r * x * (1.0 - x)
    return result


def collatz_sequence(n: int, start: int = 27) -> list[float]:
    """
    Return up to *n* steps of the Collatz sequence starting from *start*.

    Rule: if k is even → k/2, if odd → 3k+1.  Stops at 1.
    The sequence is padded with 1.0 if it terminates before *n* steps.
    """
    seq: list[float] = []
    k = max(2, start)
    while len(seq) < n:
        seq.append(float(k))
        if k == 1:
            break
        k = k // 2 if k % 2 == 0 else 3 * k + 1
    # Pad with 1.0 if terminated early
    while len(seq) < n:
        seq.append(1.0)
    return seq[:n]


def triangular_numbers(n: int) -> list[float]:
    """Return T(1), T(2), … T(n)  where T(k) = k(k+1)/2."""
    return [float(k * (k + 1) // 2) for k in range(1, n + 1)]


def catalan_numbers(n: int) -> list[float]:
    """
    Return the first *n* Catalan numbers.

    C(n) = binomial(2n, n) / (n+1).
    Growth is exponential (~4^n / n^1.5) so use modular mapping mode.
    """
    result: list[float] = []
    c = 1
    for k in range(n):
        result.append(float(c))
        c = c * 2 * (2 * k + 1) // ((k + 2))
    return result


def van_der_corput(n: int, base: int = 2) -> list[float]:
    """
    Return *n* values of the van der Corput low-discrepancy sequence in (0, 1).

    Uses base-2 (binary) bit reversal by default.  Values are offset by a
    small epsilon so that no value is exactly zero.
    """
    result: list[float] = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        k = i
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        result.append(max(r, 1e-6))
    return result


def custom_expression_sequence(n: int, expr: str) -> list[float]:
    """
    Evaluate *expr* for i = 0, 1, …, n-1 and return the results.

    Available names: i, pi, e, phi, sqrt, sin, cos, tan, log, log2,
    exp, abs, pow, floor, ceil.
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
    "Collatz Sequence":         collatz_sequence,
    "Triangular Numbers":       triangular_numbers,
    "Catalan Numbers":          catalan_numbers,
    "Van der Corput":           van_der_corput,
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
    mode        : 'ratio' | 'semitone' | 'modular'
    scale_name  : name from ``SCALES``; ``"Free (Atonal)"`` skips quantisation
    octave_range: octaves above/below base_freq to allow

    Returns list of (frequency_hz, label_string) tuples.
    """
    results: list[tuple[float, str]] = []

    for val in sequence:
        abs_val = abs(float(val))

        if mode == "ratio":
            freq = base_freq * max(abs_val, 1e-9)
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

        if mode == "ratio" and scale_name != "Free (Atonal)":
            freq = quantize_to_scale(freq, base_freq, scale_name, octave_range)

        freq = float(np.clip(freq, 20.0, 20_000.0))
        results.append((freq, f"{freq:.2f} Hz"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Iterated Function Engine
# ─────────────────────────────────────────────────────────────────────────────

# Safe math environment shared by the iterated function evaluator
_ITER_SAFE_BASE: dict = {
    "pi":    math.pi,
    "e":     math.e,
    "phi":   (1.0 + math.sqrt(5)) / 2.0,
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

FREQ_MIN = 20.0
FREQ_MAX = 20_000.0


@dataclass
class IterationResult:
    """
    Result of running an iterated function on a seed frequency.

    Attributes
    ----------
    frequencies     : list of frequencies generated (including seed)
    steps_generated : number of steps actually produced
    requested_steps : number of steps that were requested
    terminated_early: True if the sequence stopped before *requested_steps*
    termination_reason: human-readable explanation of why it stopped
    raw_values      : the raw f(x) values before any scale quantisation
    """
    frequencies:       list[float]
    steps_generated:   int
    requested_steps:   int
    terminated_early:  bool
    termination_reason: str
    raw_values:        list[float]


def iterate_function(
    expr: str,
    seed_freq: float,
    n_steps: int,
    scale_name: str = "Free (Atonal)",
    octave_range: int = 3,
) -> IterationResult:
    """
    Apply f(x) iteratively, using each output as the next input.

    The expression *expr* is evaluated with ``x`` bound to the current
    frequency.  The result is the next frequency.  Iteration stops if:

    - The output frequency is below FREQ_MIN (20 Hz)
    - The output frequency is above FREQ_MAX (20 000 Hz)
    - A math error occurs (division by zero, domain error, etc.)
    - *n_steps* have been successfully generated

    The seed frequency is included as step 0 (so up to n_steps + 1 values
    are returned when the full run completes).

    Available names in *expr*:
      x    — current frequency (Hz)
      pi, e, phi, sqrt, sin, cos, tan, log, log2, exp, abs, pow, floor, ceil

    Parameters
    ----------
    expr        : mathematical expression string, e.g. ``"2*x + 5"``
    seed_freq   : starting frequency in Hz
    n_steps     : maximum number of iterations (not counting the seed)
    scale_name  : optional scale quantisation applied to each output
    octave_range: octave range for scale quantisation

    Returns
    -------
    IterationResult dataclass
    """
    env = dict(_ITER_SAFE_BASE)

    frequencies: list[float] = [float(seed_freq)]
    raw_values:  list[float] = [float(seed_freq)]
    terminated_early = False
    termination_reason = f"Completed all {n_steps} requested steps."

    x = float(seed_freq)

    for step in range(n_steps):
        env["x"] = x
        try:
            raw = float(eval(expr, env))  # noqa: S307
        except ZeroDivisionError:
            terminated_early = True
            termination_reason = (
                f"Stopped at step {step + 1}: division by zero "
                f"(x = {x:.4f} Hz)."
            )
            break
        except Exception as exc:
            terminated_early = True
            termination_reason = (
                f"Stopped at step {step + 1}: math error — {exc} "
                f"(x = {x:.4f} Hz)."
            )
            break

        raw_values.append(raw)

        # Out-of-range check BEFORE quantisation
        if raw < FREQ_MIN:
            terminated_early = True
            termination_reason = (
                f"Stopped at step {step + 1}: f(x) = {raw:.4f} Hz "
                f"dropped below audible minimum ({FREQ_MIN} Hz)."
            )
            break
        if raw > FREQ_MAX:
            terminated_early = True
            termination_reason = (
                f"Stopped at step {step + 1}: f(x) = {raw:.4f} Hz "
                f"exceeded audible maximum ({FREQ_MAX} Hz)."
            )
            break

        # Optional scale quantisation
        if scale_name != "Free (Atonal)":
            quantised = quantize_to_scale(raw, seed_freq, scale_name, octave_range)
        else:
            quantised = raw

        frequencies.append(quantised)
        x = quantised   # feed quantised value forward (musical consistency)

    return IterationResult(
        frequencies=frequencies,
        steps_generated=len(frequencies) - 1,   # exclude seed
        requested_steps=n_steps,
        terminated_early=terminated_early,
        termination_reason=termination_reason,
        raw_values=raw_values,
    )


def validate_iter_expression(expr: str, seed_freq: float = 440.0) -> str | None:
    """
    Validate *expr* by evaluating it once at *seed_freq*.

    Returns ``None`` if valid, or an error message string if not.
    """
    env = dict(_ITER_SAFE_BASE)
    env["x"] = seed_freq
    try:
        result = float(eval(expr, env))  # noqa: S307
        if not math.isfinite(result):
            return f"Expression returned a non-finite value: {result}"
        return None
    except Exception as exc:
        return str(exc)


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
