"""
main.py — Atonal Music Studio
===============================
Cross-platform desktop application for non-traditional music composition.

Tabs
----
  Synth             : Sine / sawtooth / square / triangle oscillator + ADSR + reverb
  Equation          : Mathematical sequence synthesiser (Fibonacci, π, e, φ, √, primes,
                      Collatz, Triangular, Catalan, Van der Corput, custom …)
  Sequencer         : 16-step × 8-track beat sequencer with per-step velocity,
                      swing, and per-track pattern length
  Samples           : Audio file loader and player
  Iterated Function : f(x) iterated function synthesiser with cobweb diagram

Requires: PyQt6, numpy, scipy, sounddevice, soundfile
Optional: pydub + FFmpeg  (for MP3 / M4A / AAC export)
"""

from __future__ import annotations

import sys
import os
import math
import copy
import threading
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QGroupBox, QCheckBox, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem, QToolBar, QStatusBar,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QLineEdit, QSizePolicy, QDialog, QDialogButtonBox,
    QFrame, QMenu,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject, QPoint
from PyQt6.QtGui import (
    QAction, QPainter, QPen, QColor, QFont, QBrush,
    QPalette, QKeySequence,
)

from audio_engine import (
    AudioEngine, generate_note, mix_buffers, mono_to_stereo,
    SAMPLE_RATE, NOTE_NAMES, WAVE_TYPES, midi_to_freq, note_name_to_midi,
    normalise, fade_in_out, apply_reverb,
)
from equation_engine import (
    SEQUENCE_GENERATORS, SEQUENCE_METADATA, SCALES,
    sequence_to_frequencies, render_sequence_audio,
    custom_expression_sequence, iterate_function,
    validate_iter_expression, IterationResult,
)
from project_io import (
    new_project, save_project, load_project,
    export_audio, load_audio_file, batch_export_tracks,
    get_recent_projects, get_recent_projects_from_dir,
    PROJECT_EXTENSION, EXPORT_FILTER, IMPORT_FILTER,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dark colour palette
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_palette(app: QApplication) -> None:
    palette = QPalette()
    bg      = QColor("#1e1e2e")
    surface = QColor("#2a2a3e")
    border  = QColor("#444466")
    text    = QColor("#e0e0f0")
    accent  = QColor("#5577ff")
    accent2 = QColor("#33aaff")

    palette.setColor(QPalette.ColorRole.Window,          bg)
    palette.setColor(QPalette.ColorRole.WindowText,      text)
    palette.setColor(QPalette.ColorRole.Base,            surface)
    palette.setColor(QPalette.ColorRole.AlternateBase,   bg)
    palette.setColor(QPalette.ColorRole.Text,            text)
    palette.setColor(QPalette.ColorRole.BrightText,      QColor("white"))
    palette.setColor(QPalette.ColorRole.Button,          surface)
    palette.setColor(QPalette.ColorRole.ButtonText,      text)
    palette.setColor(QPalette.ColorRole.Highlight,       accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    palette.setColor(QPalette.ColorRole.Link,            accent2)
    palette.setColor(QPalette.ColorRole.Midlight,        border)
    palette.setColor(QPalette.ColorRole.Mid,             border)
    palette.setColor(QPalette.ColorRole.Dark,            bg)
    app.setPalette(palette)


# ─────────────────────────────────────────────────────────────────────────────
# WaveformDisplay
# ─────────────────────────────────────────────────────────────────────────────

class WaveformDisplay(QWidget):
    """Renders a float32 mono or stereo numpy array as a waveform."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 70)
        self._data: np.ndarray | None = None

    def set_data(self, audio: np.ndarray | None) -> None:
        if audio is None or len(audio) == 0:
            self._data = None
        else:
            mono = audio[:, 0] if audio.ndim > 1 else audio
            target = min(len(mono), 2000)
            step = max(1, len(mono) // target)
            self._data = mono[::step][:target].astype(np.float32)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2.0

        p.fillRect(0, 0, w, h, QColor("#12121e"))
        p.setPen(QPen(QColor("#33334a"), 1))
        p.drawLine(0, int(mid), w, int(mid))

        if self._data is None or len(self._data) == 0:
            p.setPen(QColor("#666688"))
            p.drawText(10, int(mid) + 5, "No waveform data")
            return

        p.setPen(QPen(QColor("#4488ff"), 1))
        n = len(self._data)
        scale = (mid - 4) * 0.9
        pts = [
            (int(i * w / n), int(mid - self._data[i] * scale))
            for i in range(n)
        ]
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            p.drawLine(x0, max(0, min(h - 1, y0)),
                       x1, max(0, min(h - 1, y1)))


# ─────────────────────────────────────────────────────────────────────────────
# CobwebDiagram — custom painting widget for iterated function visualisation
# ─────────────────────────────────────────────────────────────────────────────

class CobwebDiagram(QWidget):
    """
    Draws a cobweb / orbit diagram for an iterated function.

    The diagram shows:
      - The curve y = f(x) plotted over the visible frequency range
      - The diagonal y = x
      - The cobweb path tracing the orbit of the seed frequency

    All frequencies are displayed on a log₂ scale so that octaves are
    evenly spaced (matching human pitch perception).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self._result: IterationResult | None = None
        self._expr: str = ""
        self._freq_min: float = 20.0
        self._freq_max: float = 20_000.0

    def set_result(self, result: IterationResult, expr: str) -> None:
        self._result = result
        self._expr = expr
        if result and result.frequencies:
            all_f = result.raw_values + result.frequencies
            valid = [f for f in all_f if 0 < f < 1e9]
            if valid:
                pad = 0.5   # octaves of padding
                self._freq_min = max(10.0, min(valid) * (2 ** -pad))
                self._freq_max = min(30_000.0, max(valid) * (2 ** pad))
        self.update()

    def _to_canvas(self, fx: float, fy: float, w: int, h: int) -> tuple[int, int]:
        """Map (log-freq-x, log-freq-y) to canvas pixel (px, py)."""
        lo = math.log2(self._freq_min)
        hi = math.log2(self._freq_max)
        span = hi - lo if hi > lo else 1.0
        px = int((math.log2(max(fx, 1e-3)) - lo) / span * w)
        py = int(h - (math.log2(max(fy, 1e-3)) - lo) / span * h)
        return px, py

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor("#12121e"))

        if self._result is None or not self._result.frequencies:
            p.setPen(QColor("#666688"))
            p.drawText(10, h // 2, "Generate a sequence to see the cobweb diagram.")
            return

        lo = math.log2(self._freq_min)
        hi = math.log2(self._freq_max)
        span = hi - lo if hi > lo else 1.0

        # Grid lines (octave markers)
        p.setPen(QPen(QColor("#22223a"), 1))
        f = self._freq_min
        while f <= self._freq_max * 1.01:
            px, _ = self._to_canvas(f, self._freq_min, w, h)
            p.drawLine(px, 0, px, h)
            _, py = self._to_canvas(self._freq_min, f, w, h)
            p.drawLine(0, py, w, py)
            f *= 2.0

        # Diagonal y = x
        p.setPen(QPen(QColor("#444466"), 1, Qt.PenStyle.DashLine))
        x0, y0 = self._to_canvas(self._freq_min, self._freq_min, w, h)
        x1, y1 = self._to_canvas(self._freq_max, self._freq_max, w, h)
        p.drawLine(x0, y0, x1, y1)

        # Curve y = f(x) — sample 200 points across log-freq range
        from equation_engine import _ITER_SAFE_BASE
        import math as _math
        env = dict(_ITER_SAFE_BASE)
        curve_pts: list[tuple[int, int]] = []
        n_pts = 200
        for i in range(n_pts + 1):
            t = i / n_pts
            fx = self._freq_min * (2 ** (t * span))
            env["x"] = fx
            try:
                fy = float(eval(self._expr, env))  # noqa: S307
                if 0 < fy < 1e9:
                    px, py = self._to_canvas(fx, fy, w, h)
                    curve_pts.append((px, py))
            except Exception:
                pass

        p.setPen(QPen(QColor("#ff8844"), 2))
        for i in range(1, len(curve_pts)):
            p.drawLine(*curve_pts[i - 1], *curve_pts[i])

        # Cobweb path
        freqs = self._result.frequencies
        if len(freqs) < 2:
            return

        cobweb_pen = QPen(QColor("#44ddff"), 1)
        p.setPen(cobweb_pen)

        x_cur = freqs[0]
        for i in range(1, len(freqs)):
            y_cur = freqs[i]
            # Vertical: (x_cur, x_cur) → (x_cur, y_cur)  [on the curve]
            px0, py0 = self._to_canvas(x_cur, x_cur, w, h)
            px1, py1 = self._to_canvas(x_cur, y_cur, w, h)
            p.drawLine(px0, py0, px1, py1)
            # Horizontal: (x_cur, y_cur) → (y_cur, y_cur)  [to diagonal]
            px2, py2 = self._to_canvas(y_cur, y_cur, w, h)
            p.drawLine(px1, py1, px2, py2)
            x_cur = y_cur

        # Seed dot
        px, py = self._to_canvas(freqs[0], freqs[0], w, h)
        p.setBrush(QBrush(QColor("#ffdd44")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(px - 5, py - 5, 10, 10)

        # Legend
        p.setPen(QColor("#aaaacc"))
        p.setFont(QFont("monospace", 8))
        p.drawText(6, 14, f"f(x) = {self._expr}")
        p.drawText(6, 28, f"seed = {freqs[0]:.2f} Hz   steps = {self._result.steps_generated}")
        p.setPen(QColor("#ff8844"))
        p.drawText(6, h - 20, "— f(x) curve")
        p.setPen(QColor("#44ddff"))
        p.drawText(80, h - 20, "— cobweb path")
        p.setPen(QColor("#ffdd44"))
        p.drawText(170, h - 20, "● seed")


# ─────────────────────────────────────────────────────────────────────────────
# Shared reverb controls group
# ─────────────────────────────────────────────────────────────────────────────

class ReverbGroup(QGroupBox):
    """Reusable reverb controls groupbox."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Reverb", parent)
        form = QFormLayout(self)

        self.enabled_check = QCheckBox("Enable reverb")
        form.addRow("", self.enabled_check)

        self.room_spin = QDoubleSpinBox()
        self.room_spin.setRange(0.0, 1.0)
        self.room_spin.setSingleStep(0.05)
        self.room_spin.setValue(0.5)
        self.room_spin.setToolTip("0 = small/tight room   1 = large/long tail")
        form.addRow("Room size:", self.room_spin)

        self.wet_spin = QDoubleSpinBox()
        self.wet_spin.setRange(0.0, 1.0)
        self.wet_spin.setSingleStep(0.05)
        self.wet_spin.setValue(0.3)
        self.wet_spin.setToolTip("Dry/wet mix.  0 = fully dry   1 = fully wet")
        form.addRow("Wet mix:", self.wet_spin)

        self.enabled_check.toggled.connect(self._on_toggle)
        self._on_toggle(False)

    def _on_toggle(self, checked: bool) -> None:
        self.room_spin.setEnabled(checked)
        self.wet_spin.setEnabled(checked)

    def get_state(self) -> dict:
        return {
            "reverb_enabled": self.enabled_check.isChecked(),
            "reverb_room":    self.room_spin.value(),
            "reverb_wet":     self.wet_spin.value(),
        }

    def set_state(self, state: dict) -> None:
        self.enabled_check.setChecked(bool(state.get("reverb_enabled", False)))
        self.room_spin.setValue(float(state.get("reverb_room", 0.5)))
        self.wet_spin.setValue(float(state.get("reverb_wet", 0.3)))

    def maybe_apply(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Apply reverb to *audio* if enabled; otherwise return unchanged."""
        if self.enabled_check.isChecked():
            return apply_reverb(audio, sr, self.room_spin.value(), self.wet_spin.value())
        return audio


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _form_row(label: str, widget: QWidget, form: QFormLayout) -> None:
    form.addRow(QLabel(label), widget)


def _set_combo(combo: QComboBox, text: str) -> None:
    idx = combo.findText(text, Qt.MatchFlag.MatchFixedString)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def freq_to_midi_safe(freq: float) -> int:
    try:
        midi = 69 + 12 * math.log2(max(freq, 1.0) / 440.0)
        return max(0, min(127, int(round(midi))))
    except Exception:
        return 69


def _make_wave_combo(current: str = "Sine") -> QComboBox:
    """Return a QComboBox pre-populated with all wave types."""
    cb = QComboBox()
    cb.addItems(WAVE_TYPES)
    _set_combo(cb, current)
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# SynthTab
# ─────────────────────────────────────────────────────────────────────────────

class SynthTab(QWidget):
    """Oscillator synthesiser with ADSR envelope and optional reverb."""

    status_message = pyqtSignal(str)

    def __init__(self, engine: AudioEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._last_buffer: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Oscillator
        osc_box = QGroupBox("Oscillator")
        osc_lay = QHBoxLayout(osc_box)
        osc_lay.addWidget(QLabel("Wave:"))
        self.wave_combo = _make_wave_combo("Sine")
        osc_lay.addWidget(self.wave_combo)
        osc_lay.addSpacing(16)
        osc_lay.addWidget(QLabel("Duty Cycle:"))
        self.duty_spin = QDoubleSpinBox()
        self.duty_spin.setRange(0.05, 0.95)
        self.duty_spin.setSingleStep(0.05)
        self.duty_spin.setValue(0.5)
        self.duty_spin.setToolTip(
            "Pulse width for square wave (0.5 = perfect square).\n"
            "Has no effect on sine, sawtooth, or triangle."
        )
        osc_lay.addWidget(self.duty_spin)
        osc_lay.addStretch()
        root.addWidget(osc_box)

        # Pitch
        pitch_box = QGroupBox("Pitch")
        pitch_lay = QGridLayout(pitch_box)
        pitch_lay.addWidget(QLabel("Note:"), 0, 0)
        self.note_combo = QComboBox()
        self.note_combo.addItems(NOTE_NAMES)
        self.note_combo.setCurrentText("A4")
        self.note_combo.currentTextChanged.connect(self._on_note_changed)
        pitch_lay.addWidget(self.note_combo, 0, 1)
        pitch_lay.addWidget(QLabel("Frequency (Hz):"), 0, 2)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20.0, 20_000.0)
        self.freq_spin.setValue(440.0)
        self.freq_spin.setSingleStep(1.0)
        self.freq_spin.setDecimals(3)
        self.freq_spin.valueChanged.connect(self._on_freq_changed)
        pitch_lay.addWidget(self.freq_spin, 0, 3)
        pitch_lay.addWidget(QLabel("Detune (cents):"), 1, 0)
        self.detune_slider = QSlider(Qt.Orientation.Horizontal)
        self.detune_slider.setRange(-200, 200)
        self.detune_slider.setValue(0)
        pitch_lay.addWidget(self.detune_slider, 1, 1, 1, 3)
        root.addWidget(pitch_box)

        # ADSR
        adsr_box = QGroupBox("ADSR Envelope")
        adsr_form = QFormLayout(adsr_box)

        def _s(lo, hi, val):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(0.01)
            s.setValue(val)
            return s

        self.attack_spin  = _s(0.0, 10.0, 0.01)
        self.decay_spin   = _s(0.0, 10.0, 0.05)
        self.sustain_spin = _s(0.0,  1.0, 0.80)
        self.release_spin = _s(0.0, 10.0, 0.10)
        _form_row("Attack (s):",  self.attack_spin,  adsr_form)
        _form_row("Decay (s):",   self.decay_spin,   adsr_form)
        _form_row("Sustain:",     self.sustain_spin, adsr_form)
        _form_row("Release (s):", self.release_spin, adsr_form)
        root.addWidget(adsr_box)

        # Reverb
        self.reverb = ReverbGroup()
        root.addWidget(self.reverb)

        # Playback
        play_box = QGroupBox("Playback")
        play_lay = QHBoxLayout(play_box)
        play_lay.addWidget(QLabel("Duration (s):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.05, 30.0)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setValue(1.0)
        play_lay.addWidget(self.duration_spin)
        play_lay.addSpacing(12)
        play_lay.addWidget(QLabel("Volume:"))
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0.0, 1.0)
        self.volume_spin.setSingleStep(0.05)
        self.volume_spin.setValue(0.7)
        play_lay.addWidget(self.volume_spin)
        play_lay.addSpacing(12)
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setFixedWidth(90)
        self.play_btn.clicked.connect(self._play)
        play_lay.addWidget(self.play_btn)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setFixedWidth(90)
        self.stop_btn.clicked.connect(self.engine.stop)
        play_lay.addWidget(self.stop_btn)
        self.export_btn = QPushButton("Export Note…")
        self.export_btn.clicked.connect(self._export)
        play_lay.addWidget(self.export_btn)
        play_lay.addStretch()
        root.addWidget(play_box)

        root.addWidget(QLabel("Waveform preview:"))
        self.waveform = WaveformDisplay()
        self.waveform.setMinimumHeight(100)
        root.addWidget(self.waveform)
        root.addStretch()

    def _on_note_changed(self, name: str) -> None:
        try:
            freq = midi_to_freq(note_name_to_midi(name))
            self.freq_spin.blockSignals(True)
            self.freq_spin.setValue(freq)
            self.freq_spin.blockSignals(False)
        except Exception:
            pass

    def _on_freq_changed(self, freq: float) -> None:
        midi = freq_to_midi_safe(freq)
        note_idx = max(0, min(len(NOTE_NAMES) - 1, midi - 12))
        nearest = NOTE_NAMES[note_idx]
        self.note_combo.blockSignals(True)
        self.note_combo.setCurrentText(nearest)
        self.note_combo.blockSignals(False)

    def _params(self) -> dict:
        base = self.freq_spin.value()
        detune = self.detune_slider.value()
        freq = base * (2.0 ** (detune / 1200.0))
        return dict(
            freq=freq,
            wave_type=self.wave_combo.currentText().lower(),
            duration=self.duration_spin.value(),
            amplitude=self.volume_spin.value(),
            attack=self.attack_spin.value(),
            decay=self.decay_spin.value(),
            sustain=self.sustain_spin.value(),
            release=self.release_spin.value(),
            duty=self.duty_spin.value(),
        )

    def _play(self) -> None:
        try:
            p = self._params()
            buf = generate_note(**p)
            buf = self.reverb.maybe_apply(buf)
            self._last_buffer = buf
            self.waveform.set_data(buf)
            self.engine.play(buf)
            self.status_message.emit(
                f"Playing {p['freq']:.2f} Hz  |  {p['wave_type'].capitalize()} wave"
            )
        except Exception as e:
            self.status_message.emit(f"Synth error: {e}")

    def _export(self) -> None:
        if self._last_buffer is None:
            QMessageBox.information(self, "Export", "Play a note first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Note", "note.wav", EXPORT_FILTER
        )
        if not path:
            return
        try:
            audio = self._last_buffer
            if audio.ndim == 1:
                audio = mono_to_stereo(audio)
            export_audio(audio, SAMPLE_RATE, path)
            self.status_message.emit(f"Exported → {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def get_state(self) -> dict:
        return {
            "wave_type": self.wave_combo.currentText().lower(),
            "frequency": self.freq_spin.value(),
            "duration":  self.duration_spin.value(),
            "amplitude": self.volume_spin.value(),
            "attack":    self.attack_spin.value(),
            "decay":     self.decay_spin.value(),
            "sustain":   self.sustain_spin.value(),
            "release":   self.release_spin.value(),
            "duty":      self.duty_spin.value(),
            **self.reverb.get_state(),
        }

    def set_state(self, state: dict) -> None:
        _set_combo(self.wave_combo, state.get("wave_type", "sine").capitalize())
        self.freq_spin.setValue(float(state.get("frequency", 440.0)))
        self.duration_spin.setValue(float(state.get("duration", 1.0)))
        self.volume_spin.setValue(float(state.get("amplitude", 0.7)))
        self.attack_spin.setValue(float(state.get("attack", 0.01)))
        self.decay_spin.setValue(float(state.get("decay", 0.05)))
        self.sustain_spin.setValue(float(state.get("sustain", 0.8)))
        self.release_spin.setValue(float(state.get("release", 0.1)))
        self.duty_spin.setValue(float(state.get("duty", 0.5)))
        self.reverb.set_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# EquationTab
# ─────────────────────────────────────────────────────────────────────────────

class EquationTab(QWidget):
    """Mathematical sequence synthesiser."""

    status_message = pyqtSignal(str)

    def __init__(self, engine: AudioEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._last_buffer: np.ndarray | None = None
        self._last_freqs: list[float] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # LEFT controls
        left = QWidget()
        left.setMaximumWidth(350)
        left_v = QVBoxLayout(left)
        left_v.setSpacing(8)

        seq_box = QGroupBox("Mathematical Sequence")
        seq_form = QFormLayout(seq_box)

        self.seq_combo = QComboBox()
        self.seq_combo.addItems(list(SEQUENCE_GENERATORS.keys()))
        self.seq_combo.currentTextChanged.connect(self._on_seq_type_changed)
        # Show metadata as tooltip
        self.seq_combo.currentTextChanged.connect(self._update_seq_tooltip)
        _form_row("Sequence:", self.seq_combo, seq_form)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 64)
        self.n_spin.setValue(16)
        _form_row("Number of notes:", self.n_spin, seq_form)

        self.logistic_r_spin = QDoubleSpinBox()
        self.logistic_r_spin.setRange(0.01, 4.0)
        self.logistic_r_spin.setValue(3.7)
        self.logistic_r_spin.setSingleStep(0.01)
        self._logistic_r_label = QLabel("Logistic r:")
        seq_form.addRow(self._logistic_r_label, self.logistic_r_spin)

        self.expr_label = QLabel("Expression (use i):")
        self.expr_edit = QLineEdit("sin(i * pi / 4) + 1")
        seq_form.addRow(self.expr_label, self.expr_edit)

        self.seq_desc_label = QLabel("")
        self.seq_desc_label.setWordWrap(True)
        self.seq_desc_label.setStyleSheet("color: #8888aa; font-size: 10px;")
        seq_form.addRow("", self.seq_desc_label)

        left_v.addWidget(seq_box)
        self._on_seq_type_changed(self.seq_combo.currentText())
        self._update_seq_tooltip(self.seq_combo.currentText())

        map_box = QGroupBox("Frequency Mapping")
        map_form = QFormLayout(map_box)

        freq_row = QWidget()
        freq_h = QHBoxLayout(freq_row)
        freq_h.setContentsMargins(0, 0, 0, 0)
        self.base_freq_spin = QDoubleSpinBox()
        self.base_freq_spin.setRange(20.0, 4000.0)
        self.base_freq_spin.setValue(220.0)
        freq_h.addWidget(self.base_freq_spin)
        freq_h.addWidget(QLabel("or note:"))
        self.base_note_combo = QComboBox()
        self.base_note_combo.addItems(NOTE_NAMES)
        self.base_note_combo.setCurrentText("A3")
        self.base_note_combo.currentTextChanged.connect(self._on_base_note_changed)
        freq_h.addWidget(self.base_note_combo)
        map_form.addRow("Base frequency:", freq_row)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(list(SCALES.keys()))
        _form_row("Scale / quantisation:", self.scale_combo, map_form)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["ratio", "semitone", "modular"])
        _form_row("Mapping mode:", self.mode_combo, map_form)

        self.octave_spin = QSpinBox()
        self.octave_spin.setRange(1, 6)
        self.octave_spin.setValue(3)
        _form_row("Octave range:", self.octave_spin, map_form)
        left_v.addWidget(map_box)

        synth_box = QGroupBox("Synthesis")
        synth_form = QFormLayout(synth_box)
        self.eq_wave_combo = _make_wave_combo("Sine")
        _form_row("Wave type:", self.eq_wave_combo, synth_form)
        self.note_dur_spin = QDoubleSpinBox()
        self.note_dur_spin.setRange(0.02, 10.0)
        self.note_dur_spin.setSingleStep(0.05)
        self.note_dur_spin.setValue(0.3)
        _form_row("Note duration (s):", self.note_dur_spin, synth_form)
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.0, 2.0)
        self.gap_spin.setSingleStep(0.01)
        self.gap_spin.setValue(0.02)
        _form_row("Gap (s):", self.gap_spin, synth_form)
        self.eq_attack_spin = QDoubleSpinBox()
        self.eq_attack_spin.setRange(0.0, 5.0)
        self.eq_attack_spin.setValue(0.01)
        _form_row("Attack (s):", self.eq_attack_spin, synth_form)
        self.eq_release_spin = QDoubleSpinBox()
        self.eq_release_spin.setRange(0.0, 5.0)
        self.eq_release_spin.setValue(0.1)
        _form_row("Release (s):", self.eq_release_spin, synth_form)
        left_v.addWidget(synth_box)

        self.reverb = ReverbGroup()
        left_v.addWidget(self.reverb)

        btn_row = QHBoxLayout()
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.clicked.connect(self._generate)
        btn_row.addWidget(self.gen_btn)
        self.eq_play_btn = QPushButton("▶  Play")
        self.eq_play_btn.clicked.connect(self._play)
        btn_row.addWidget(self.eq_play_btn)
        self.eq_stop_btn = QPushButton("■  Stop")
        self.eq_stop_btn.clicked.connect(self.engine.stop)
        btn_row.addWidget(self.eq_stop_btn)
        left_v.addLayout(btn_row)
        self.eq_export_btn = QPushButton("Export Sequence…")
        self.eq_export_btn.clicked.connect(self._export)
        left_v.addWidget(self.eq_export_btn)
        left_v.addStretch()
        root.addWidget(left)

        # RIGHT table + waveform
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.addWidget(QLabel("Generated sequence:"))
        self.seq_table = QTableWidget(0, 3)
        self.seq_table.setHorizontalHeaderLabels(["Step", "Raw value", "Frequency (Hz)"])
        self.seq_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.seq_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.seq_table.setAlternatingRowColors(True)
        right_v.addWidget(self.seq_table)
        right_v.addWidget(QLabel("Waveform preview:"))
        self.eq_waveform = WaveformDisplay()
        self.eq_waveform.setMinimumHeight(100)
        right_v.addWidget(self.eq_waveform)
        root.addWidget(right, 1)

    def _on_seq_type_changed(self, name: str) -> None:
        is_custom   = name == "Custom Expression"
        is_logistic = name == "Logistic Map (chaos)"
        self.expr_label.setVisible(is_custom)
        self.expr_edit.setVisible(is_custom)
        self._logistic_r_label.setVisible(is_logistic)
        self.logistic_r_spin.setVisible(is_logistic)

    def _update_seq_tooltip(self, name: str) -> None:
        desc = SEQUENCE_METADATA.get(name, "")
        self.seq_desc_label.setText(desc)

    def _on_base_note_changed(self, name: str) -> None:
        try:
            self.base_freq_spin.setValue(midi_to_freq(note_name_to_midi(name)))
        except Exception:
            pass

    def _generate(self) -> bool:
        seq_type = self.seq_combo.currentText()
        n = self.n_spin.value()
        try:
            if seq_type == "Custom Expression":
                raw = custom_expression_sequence(n, self.expr_edit.text())
            elif seq_type == "Logistic Map (chaos)":
                from equation_engine import logistic_map
                raw = logistic_map(n, r=self.logistic_r_spin.value())
            else:
                gen_fn = SEQUENCE_GENERATORS[seq_type]
                raw = gen_fn(n)

            base_freq = self.base_freq_spin.value()
            freq_pairs = sequence_to_frequencies(
                raw, base_freq,
                self.mode_combo.currentText(),
                self.scale_combo.currentText(),
                self.octave_spin.value(),
            )
            self._last_freqs = [f for f, _ in freq_pairs]

            self.seq_table.setRowCount(len(raw))
            for i, (rv, (freq, _)) in enumerate(zip(raw, freq_pairs)):
                self.seq_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.seq_table.setItem(i, 1, QTableWidgetItem(f"{float(rv):.5g}"))
                self.seq_table.setItem(i, 2, QTableWidgetItem(f"{freq:.3f}"))

            self.status_message.emit(
                f"Generated {len(self._last_freqs)}-note {seq_type} sequence "
                f"(base {base_freq:.1f} Hz, {self.scale_combo.currentText()})"
            )
            return True
        except Exception as e:
            QMessageBox.warning(self, "Generation Error", str(e))
            return False

    def _play(self) -> None:
        if not self._last_freqs:
            if not self._generate():
                return
        try:
            buf = render_sequence_audio(
                self._last_freqs,
                self.note_dur_spin.value(),
                self.eq_wave_combo.currentText().lower(),
                self.gap_spin.value(),
                0.7,
                self.eq_attack_spin.value(),
                0.05, 0.8,
                self.eq_release_spin.value(),
            )
            buf = self.reverb.maybe_apply(buf)
            self._last_buffer = buf
            self.eq_waveform.set_data(buf)
            self.engine.play(buf)
            self.status_message.emit("Playing equation sequence…")
        except Exception as e:
            self.status_message.emit(f"Playback error: {e}")

    def _export(self) -> None:
        if self._last_buffer is None:
            QMessageBox.information(self, "Export", "Generate and play a sequence first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Sequence", "sequence.wav", EXPORT_FILTER
        )
        if not path:
            return
        try:
            audio = self._last_buffer
            if audio.ndim == 1:
                audio = mono_to_stereo(audio)
            export_audio(audio, SAMPLE_RATE, path)
            self.status_message.emit(f"Exported → {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def get_state(self) -> dict:
        return {
            "sequence_type": self.seq_combo.currentText(),
            "custom_expr":   self.expr_edit.text(),
            "n_notes":       self.n_spin.value(),
            "base_freq":     self.base_freq_spin.value(),
            "scale":         self.scale_combo.currentText(),
            "mode":          self.mode_combo.currentText(),
            "octave_range":  self.octave_spin.value(),
            "wave_type":     self.eq_wave_combo.currentText().lower(),
            "note_duration": self.note_dur_spin.value(),
            "gap":           self.gap_spin.value(),
            "attack":        self.eq_attack_spin.value(),
            "release":       self.eq_release_spin.value(),
            **self.reverb.get_state(),
        }

    def set_state(self, state: dict) -> None:
        _set_combo(self.seq_combo, state.get("sequence_type", "Fibonacci"))
        self.expr_edit.setText(state.get("custom_expr", "sin(i * pi / 4) + 1"))
        self.n_spin.setValue(int(state.get("n_notes", 16)))
        self.base_freq_spin.setValue(float(state.get("base_freq", 220.0)))
        _set_combo(self.scale_combo, state.get("scale", "Chromatic"))
        _set_combo(self.mode_combo, state.get("mode", "ratio"))
        self.octave_spin.setValue(int(state.get("octave_range", 3)))
        _set_combo(self.eq_wave_combo, state.get("wave_type", "sine").capitalize())
        self.note_dur_spin.setValue(float(state.get("note_duration", 0.3)))
        self.gap_spin.setValue(float(state.get("gap", 0.02)))
        self.eq_attack_spin.setValue(float(state.get("attack", 0.01)))
        self.eq_release_spin.setValue(float(state.get("release", 0.1)))
        self.reverb.set_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# SequencerTab
# ─────────────────────────────────────────────────────────────────────────────

NUM_STEPS  = 16
NUM_TRACKS = 8

_NOTE_SUBSET = [
    "C2","D2","E2","F2","G2","A2","B2",
    "C3","D3","E3","F3","G3","A3","B3",
    "C4","D4","E4","F4","G4","A4","B4",
    "C5","D5","E5","F5","G5","A5","B5",
    "C6","D6","E6",
] + [n for n in NOTE_NAMES if "#" in n and n[-1] in "2345"]

_PATTERN_LENGTHS = [4, 8, 16]


class VelocityDialog(QDialog):
    """Small dialog for setting per-step velocity (right-click on a step button)."""

    def __init__(self, current_vel: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Step Velocity")
        self.setFixedSize(220, 100)
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Velocity (0.0 – 1.0):"))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 1.0)
        self.spin.setSingleStep(0.05)
        self.spin.setDecimals(2)
        self.spin.setValue(current_vel)
        row.addWidget(self.spin)
        layout.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def velocity(self) -> float:
        return self.spin.value()


class SequencerTab(QWidget):
    """16-step × 8-track step sequencer with velocity, swing, and variable pattern length."""

    status_message = pyqtSignal(str)

    def __init__(self, engine: AudioEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._steps:      list[list[bool]]  = [[False] * NUM_STEPS for _ in range(NUM_TRACKS)]
        self._velocities: list[list[float]] = [[1.0]   * NUM_STEPS for _ in range(NUM_TRACKS)]
        self._pat_lens:   list[int]         = [NUM_STEPS] * NUM_TRACKS
        self._notes:      list[str]         = ["A4"]  * NUM_TRACKS
        self._waves:      list[str]         = ["sine"] * NUM_TRACKS
        self._volumes:    list[float]       = [0.7]   * NUM_TRACKS
        self._muted:      list[bool]        = [False]  * NUM_TRACKS
        self._last_buffer: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Transport bar
        trans = QHBoxLayout()
        trans.addWidget(QLabel("BPM:"))
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(20, 300)
        self.bpm_spin.setValue(120)
        trans.addWidget(self.bpm_spin)

        trans.addSpacing(12)
        trans.addWidget(QLabel("Swing:"))
        self.swing_slider = QSlider(Qt.Orientation.Horizontal)
        self.swing_slider.setRange(0, 50)
        self.swing_slider.setValue(0)
        self.swing_slider.setFixedWidth(100)
        self.swing_slider.setToolTip(
            "Swing delay: pushes every second 16th-note step forward.\n"
            "0 = straight, 50 = maximum swing (~triplet feel)."
        )
        trans.addWidget(self.swing_slider)
        self.swing_label = QLabel("0%")
        self.swing_slider.valueChanged.connect(
            lambda v: self.swing_label.setText(f"{v}%")
        )
        trans.addWidget(self.swing_label)

        trans.addSpacing(12)
        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(True)
        trans.addWidget(self.loop_check)

        trans.addSpacing(12)
        self.seq_play_btn = QPushButton("▶  Play")
        self.seq_play_btn.setFixedWidth(90)
        self.seq_play_btn.clicked.connect(self._play_sequence)
        trans.addWidget(self.seq_play_btn)

        self.seq_stop_btn = QPushButton("■  Stop")
        self.seq_stop_btn.setFixedWidth(90)
        self.seq_stop_btn.clicked.connect(self._stop_sequence)
        trans.addWidget(self.seq_stop_btn)

        trans.addSpacing(12)
        self.seq_export_btn = QPushButton("Export Mix…")
        self.seq_export_btn.clicked.connect(self._export)
        trans.addWidget(self.seq_export_btn)

        self.batch_export_btn = QPushButton("Batch Export Tracks…")
        self.batch_export_btn.clicked.connect(self._batch_export)
        trans.addWidget(self.batch_export_btn)

        clr_btn = QPushButton("Clear All")
        clr_btn.clicked.connect(self._clear_all)
        trans.addWidget(clr_btn)
        trans.addStretch()
        root.addLayout(trans)

        # Step grid
        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(3)

        for step in range(NUM_STEPS):
            lbl = QLabel(f"{step + 1}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedWidth(38)
            grid.addWidget(lbl, 0, step + 5)

        self._step_btns: list[list[QPushButton]] = []
        for track in range(NUM_TRACKS):
            row_btns: list[QPushButton] = []

            lbl = QLabel(f"T{track + 1}")
            lbl.setFixedWidth(24)
            grid.addWidget(lbl, track + 1, 0)

            note_cb = QComboBox()
            note_cb.addItems(_NOTE_SUBSET)
            note_cb.setCurrentText("A4")
            note_cb.setFixedWidth(64)
            note_cb.currentTextChanged.connect(
                lambda text, t=track: self._set_note(t, text)
            )
            grid.addWidget(note_cb, track + 1, 1)

            wave_cb = _make_wave_combo("Sine")
            wave_cb.setFixedWidth(84)
            wave_cb.currentTextChanged.connect(
                lambda text, t=track: self._set_wave(t, text)
            )
            grid.addWidget(wave_cb, track + 1, 2)

            # Pattern length dropdown
            plen_cb = QComboBox()
            plen_cb.addItems([str(pl) for pl in _PATTERN_LENGTHS])
            plen_cb.setCurrentText("16")
            plen_cb.setFixedWidth(50)
            plen_cb.setToolTip("Pattern length for this track (enables polyrhythm)")
            plen_cb.currentTextChanged.connect(
                lambda text, t=track: self._set_pat_len(t, int(text))
            )
            grid.addWidget(plen_cb, track + 1, 3)

            mute_cb = QCheckBox("M")
            mute_cb.setToolTip("Mute this track")
            mute_cb.toggled.connect(
                lambda checked, t=track: self._set_muted(t, checked)
            )
            grid.addWidget(mute_cb, track + 1, 4)

            for step in range(NUM_STEPS):
                btn = QPushButton()
                btn.setCheckable(True)
                btn.setFixedSize(36, 36)
                btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                btn.customContextMenuRequested.connect(
                    lambda pos, t=track, s=step, b=btn: self._on_step_right_click(t, s, b)
                )
                self._update_step_style(btn, False, 1.0)
                btn.toggled.connect(
                    lambda checked, t=track, s=step: self._set_step(t, s, checked)
                )
                grid.addWidget(btn, track + 1, step + 5)
                row_btns.append(btn)

            self._step_btns.append(row_btns)

        grid_scroll.setWidget(grid_widget)
        root.addWidget(grid_scroll, 1)

        root.addWidget(QLabel("Rendered sequence preview:"))
        self.seq_waveform = WaveformDisplay()
        self.seq_waveform.setMinimumHeight(80)
        root.addWidget(self.seq_waveform)

    # ── Step button styling ────────────────────────────────────────────────

    def _update_step_style(self, btn: QPushButton, active: bool, velocity: float) -> None:
        """Colour a step button based on active state and velocity."""
        if not active:
            btn.setStyleSheet(
                "QPushButton { background: #2a2a3e; border: 1px solid #444466; border-radius: 3px; }"
                "QPushButton:checked { background: #4488ff; border: 1px solid #6699ff; }"
            )
            return
        # Scale the blue component by velocity (darker = quieter)
        v = float(np.clip(velocity, 0.0, 1.0))
        r = int(20  + v * 40)
        g = int(100 + v * 56)
        b = int(180 + v * 75)
        color = f"#{r:02x}{g:02x}{b:02x}"
        btn.setStyleSheet(
            f"QPushButton {{ background: {color}; border: 1px solid #6699ff; border-radius: 3px; }}"
            f"QPushButton:checked {{ background: {color}; border: 1px solid #88bbff; }}"
        )

    # ── Slots ──────────────────────────────────────────────────────────────

    def _set_step(self, track: int, step: int, on: bool) -> None:
        self._steps[track][step] = on
        vel = self._velocities[track][step]
        self._update_step_style(self._step_btns[track][step], on, vel)

    def _on_step_right_click(self, track: int, step: int, btn: QPushButton) -> None:
        """Open velocity dialog on right-click."""
        current_vel = self._velocities[track][step]
        dlg = VelocityDialog(current_vel, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_vel = dlg.velocity()
            self._velocities[track][step] = new_vel
            self._update_step_style(btn, self._steps[track][step], new_vel)

    def _set_note(self, track: int, note: str) -> None:
        self._notes[track] = note

    def _set_wave(self, track: int, wave: str) -> None:
        self._waves[track] = wave.lower()

    def _set_muted(self, track: int, muted: bool) -> None:
        self._muted[track] = muted

    def _set_pat_len(self, track: int, length: int) -> None:
        self._pat_lens[track] = length

    # ── Render ─────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray | None:
        bpm     = self.bpm_spin.value()
        swing   = self.swing_slider.value() / 100.0   # 0.0 – 0.5
        step_dur = 60.0 / bpm / 4.0

        track_buffers: list[np.ndarray] = []
        track_volumes: list[float]      = []

        # LCM of all active pattern lengths → total loop length
        active_lens = [
            self._pat_lens[t]
            for t in range(NUM_TRACKS)
            if not self._muted[t] and any(self._steps[t][:self._pat_lens[t]])
        ]
        if not active_lens:
            return None

        from math import gcd
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b)
        total_steps = active_lens[0]
        for l in active_lens[1:]:
            total_steps = lcm(total_steps, l)
        total_steps = min(total_steps, 256)   # guard against huge LCMs

        total_len = int(step_dur * total_steps * SAMPLE_RATE)

        for t in range(NUM_TRACKS):
            if self._muted[t]:
                continue
            pat_len = self._pat_lens[t]
            if not any(self._steps[t][:pat_len]):
                continue

            try:
                freq = midi_to_freq(note_name_to_midi(self._notes[t]))
            except Exception:
                freq = 440.0

            wave     = self._waves[t]
            note_dur = step_dur * 0.9
            track_buf = np.zeros(total_len, dtype=np.float32)

            for step_abs in range(total_steps):
                step_idx = step_abs % pat_len
                if not self._steps[t][step_idx]:
                    continue

                swing_offset = 0.0
                if step_abs % 2 == 1 and swing > 0.0:
                    swing_offset = step_dur * swing

                vel       = self._velocities[t][step_idx]
                amplitude = self._volumes[t] * vel

                note_buf = generate_note(
                    freq, note_dur, wave, SAMPLE_RATE, amplitude,
                    attack=0.005, decay=0.02, sustain=0.8, release=0.05,
                )
                raw_offset    = step_abs * step_dur * SAMPLE_RATE
                swing_samples = int(swing_offset * SAMPLE_RATE)
                offset        = int(raw_offset) + swing_samples
                end           = min(offset + len(note_buf), total_len)
                if offset < total_len:
                    track_buf[offset:end] += note_buf[:end - offset]

            track_buffers.append(track_buf)
            track_volumes.append(self._volumes[t])

        if not track_buffers:
            return None

        return mix_buffers(track_buffers, track_volumes)

    def _play_sequence(self) -> None:
        try:
            buf = self._render()
            if buf is None:
                self.status_message.emit("No active steps — nothing to play.")
                return
            self._last_buffer = buf
            self.seq_waveform.set_data(buf)
            self.engine.play(buf, loop=self.loop_check.isChecked())
            self.status_message.emit(
                f"Playing sequencer at {self.bpm_spin.value()} BPM  "
                f"swing {self.swing_slider.value()}%"
                + ("  (looping)" if self.loop_check.isChecked() else "")
            )
        except Exception as e:
            self.status_message.emit(f"Sequencer error: {e}")

    def _stop_sequence(self) -> None:
        self.engine.stop()
        self.status_message.emit("Stopped.")

    def _clear_all(self) -> None:
        for t in range(NUM_TRACKS):
            for s in range(NUM_STEPS):
                self._steps[t][s] = False
                self._step_btns[t][s].blockSignals(True)
                self._step_btns[t][s].setChecked(False)
                self._step_btns[t][s].blockSignals(False)
                self._update_step_style(self._step_btns[t][s], False, 1.0)
        self.status_message.emit("Sequencer cleared.")

    def _export(self) -> None:
        buf = self._render()
        if buf is None:
            QMessageBox.information(self, "Export", "No active steps to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mix", "sequence.wav", EXPORT_FILTER
        )
        if not path:
            return
        try:
            export_audio(buf, SAMPLE_RATE, path)
            self.status_message.emit(f"Exported → {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _batch_export(self) -> None:
        """Export each active track to a separate WAV file."""
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select output folder for track files"
        )
        if not out_dir:
            return
        try:
            state = self.get_state()
            written = batch_export_tracks(state, out_dir, SAMPLE_RATE, ".wav")
            if written:
                self.status_message.emit(
                    f"Batch export: {len(written)} track(s) written to {out_dir}"
                )
                QMessageBox.information(
                    self,
                    "Batch Export Complete",
                    f"Exported {len(written)} track(s):\n" + "\n".join(Path(p).name for p in written)
                )
            else:
                QMessageBox.information(self, "Batch Export", "No active tracks to export.")
        except Exception as e:
            QMessageBox.warning(self, "Batch Export Error", str(e))

    # ── Project state ──────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "bpm":   self.bpm_spin.value(),
            "swing": self.swing_slider.value() / 100.0,
            "tracks": [
                {
                    "note":           self._notes[t],
                    "wave_type":      self._waves[t],
                    "volume":         self._volumes[t],
                    "muted":          self._muted[t],
                    "steps":          list(self._steps[t]),
                    "velocities":     list(self._velocities[t]),
                    "pattern_length": self._pat_lens[t],
                }
                for t in range(NUM_TRACKS)
            ],
        }

    def set_state(self, state: dict) -> None:
        self.bpm_spin.setValue(int(state.get("bpm", 120)))
        swing_pct = int(float(state.get("swing", 0.0)) * 100)
        self.swing_slider.setValue(swing_pct)

        for t, track in enumerate(state.get("tracks", [])[:NUM_TRACKS]):
            self._notes[t]     = track.get("note", "A4")
            self._waves[t]     = track.get("wave_type", "sine")
            self._volumes[t]   = float(track.get("volume", 0.7))
            self._muted[t]     = bool(track.get("muted", False))
            self._pat_lens[t]  = int(track.get("pattern_length", NUM_STEPS))
            steps      = track.get("steps",      [False] * NUM_STEPS)
            velocities = track.get("velocities", [1.0]   * NUM_STEPS)
            for s in range(NUM_STEPS):
                val = bool(steps[s]) if s < len(steps) else False
                vel = float(velocities[s]) if s < len(velocities) else 1.0
                self._steps[t][s]      = val
                self._velocities[t][s] = vel
                self._step_btns[t][s].blockSignals(True)
                self._step_btns[t][s].setChecked(val)
                self._step_btns[t][s].blockSignals(False)
                self._update_step_style(self._step_btns[t][s], val, vel)


# ─────────────────────────────────────────────────────────────────────────────
# SamplesTab
# ─────────────────────────────────────────────────────────────────────────────

class SamplesTab(QWidget):
    """Load audio files and play them back with basic transport controls."""

    status_message = pyqtSignal(str)

    def __init__(self, engine: AudioEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._samples: dict[str, tuple[np.ndarray, int]] = {}
        self._current_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left = QWidget()
        left.setMaximumWidth(280)
        left_v = QVBoxLayout(left)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Sample…")
        load_btn.clicked.connect(self._load_sample)
        btn_row.addWidget(load_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_sample)
        btn_row.addWidget(remove_btn)
        left_v.addLayout(btn_row)

        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self._on_sample_selected)
        left_v.addWidget(self.sample_list)
        root.addWidget(left)

        right = QWidget()
        right_v = QVBoxLayout(right)

        self.info_label = QLabel("No sample loaded.")
        right_v.addWidget(self.info_label)

        self.sample_waveform = WaveformDisplay()
        self.sample_waveform.setMinimumHeight(120)
        right_v.addWidget(self.sample_waveform)

        ctrl_box = QGroupBox("Playback Controls")
        ctrl_form = QFormLayout(ctrl_box)
        self.s_volume_spin = QDoubleSpinBox()
        self.s_volume_spin.setRange(0.0, 2.0)
        self.s_volume_spin.setSingleStep(0.05)
        self.s_volume_spin.setValue(1.0)
        _form_row("Volume:", self.s_volume_spin, ctrl_form)
        self.s_speed_spin = QDoubleSpinBox()
        self.s_speed_spin.setRange(0.1, 4.0)
        self.s_speed_spin.setSingleStep(0.05)
        self.s_speed_spin.setValue(1.0)
        _form_row("Speed / pitch (×):", self.s_speed_spin, ctrl_form)
        self.s_loop_check = QCheckBox("Loop")
        ctrl_form.addRow("", self.s_loop_check)
        right_v.addWidget(ctrl_box)

        play_row = QHBoxLayout()
        self.s_play_btn = QPushButton("▶  Play")
        self.s_play_btn.clicked.connect(self._play_sample)
        play_row.addWidget(self.s_play_btn)
        self.s_stop_btn = QPushButton("■  Stop")
        self.s_stop_btn.clicked.connect(self.engine.stop)
        play_row.addWidget(self.s_stop_btn)
        self.s_export_btn = QPushButton("Export Sample…")
        self.s_export_btn.clicked.connect(self._export_sample)
        play_row.addWidget(self.s_export_btn)
        play_row.addStretch()
        right_v.addLayout(play_row)
        right_v.addStretch()
        root.addWidget(right, 1)

    def _on_sample_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self.sample_list.item(row)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        self._current_path = path
        data, sr = self._samples[path]
        dur = len(data) / sr
        ch = data.shape[1] if data.ndim > 1 else 1
        self.info_label.setText(
            f"{Path(path).name}   |   {sr} Hz  ·  {ch}ch  ·  {dur:.2f} s"
        )
        self.sample_waveform.set_data(data)

    def _load_sample(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Audio Sample", "", IMPORT_FILTER
        )
        for path in paths:
            if path in self._samples:
                continue
            try:
                data, sr = load_audio_file(path)
                self._samples[path] = (data, sr)
                item = QListWidgetItem(Path(path).name)
                item.setData(Qt.ItemDataRole.UserRole, path)
                self.sample_list.addItem(item)
                self.status_message.emit(f"Loaded: {Path(path).name}")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"{Path(path).name}\n\n{e}")

    def _remove_sample(self) -> None:
        row = self.sample_list.currentRow()
        if row < 0:
            return
        item = self.sample_list.takeItem(row)
        if item:
            path = item.data(Qt.ItemDataRole.UserRole)
            self._samples.pop(path, None)
            if self._current_path == path:
                self._current_path = None
                self.info_label.setText("No sample loaded.")
                self.sample_waveform.set_data(None)

    def _play_sample(self) -> None:
        if self._current_path is None:
            self.status_message.emit("Select a sample from the list first.")
            return
        data, sr = self._samples[self._current_path]
        vol   = self.s_volume_spin.value()
        speed = self.s_speed_spin.value()
        if abs(speed - 1.0) > 0.01:
            from scipy import signal as sig
            n_out = int(len(data) / speed)
            if data.ndim == 1:
                data = sig.resample(data, n_out).astype(np.float32)
            else:
                data = np.column_stack([
                    sig.resample(data[:, c], n_out).astype(np.float32)
                    for c in range(data.shape[1])
                ])
        buf = np.clip((data * vol).astype(np.float32), -1.0, 1.0)
        self.engine.play(buf, loop=self.s_loop_check.isChecked())
        self.status_message.emit(
            f"Playing {Path(self._current_path).name}  ×{speed:.2f}"
        )

    def _export_sample(self) -> None:
        if self._current_path is None:
            QMessageBox.information(self, "Export", "Select a sample first.")
            return
        data, sr = self._samples[self._current_path]
        default_name = Path(self._current_path).stem + "_export.wav"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Sample", default_name, EXPORT_FILTER
        )
        if not path:
            return
        try:
            export_audio(data, sr, path)
            self.status_message.emit(f"Exported → {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def get_state(self) -> dict:
        return {"samples": list(self._samples.keys())}

    def set_state(self, state: dict) -> None:
        for path in state.get("samples", []):
            if path and os.path.isfile(path) and path not in self._samples:
                try:
                    data, sr = load_audio_file(path)
                    self._samples[path] = (data, sr)
                    item = QListWidgetItem(Path(path).name)
                    item.setData(Qt.ItemDataRole.UserRole, path)
                    self.sample_list.addItem(item)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# IteratedFunctionTab
# ─────────────────────────────────────────────────────────────────────────────

class IteratedFunctionTab(QWidget):
    """
    Iterated Function synthesiser.

    The user enters f(x), a seed frequency, and a step count.
    f(x) is applied repeatedly: each output becomes the next x.
    The resulting frequency orbit is played as a melody and visualised
    as a cobweb diagram.
    """

    status_message = pyqtSignal(str)

    def __init__(self, engine: AudioEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._last_buffer: np.ndarray | None = None
        self._last_result: IterationResult | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ── LEFT: controls ────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(360)
        left_v = QVBoxLayout(left)
        left_v.setSpacing(8)

        # Function editor
        fn_box = QGroupBox("Iterated Function  f(x)")
        fn_form = QFormLayout(fn_box)

        self.expr_edit = QLineEdit("x * 1.5")
        self.expr_edit.setPlaceholderText("e.g.  x * 1.5   or   x + 50   or   x * sin(x/100) + x")
        self.expr_edit.setToolTip(
            "Enter a function of x (current frequency in Hz).\n"
            "f(x) is applied repeatedly; each output becomes the next x.\n\n"
            "Available: x, pi, e, phi, sqrt, sin, cos, tan,\n"
            "           log, log2, exp, abs, pow, floor, ceil"
        )
        fn_form.addRow("f(x) =", self.expr_edit)

        self.validate_label = QLabel("")
        self.validate_label.setStyleSheet("color: #88cc88; font-size: 10px;")
        fn_form.addRow("", self.validate_label)
        self.expr_edit.textChanged.connect(self._validate_expr)

        self.seed_spin = QDoubleSpinBox()
        self.seed_spin.setRange(20.0, 20_000.0)
        self.seed_spin.setValue(220.0)
        self.seed_spin.setSingleStep(1.0)
        self.seed_spin.setDecimals(3)
        self.seed_spin.setToolTip("Starting frequency x₀ in Hz")
        fn_form.addRow("Seed frequency (Hz):", self.seed_spin)

        self.seed_note_combo = QComboBox()
        self.seed_note_combo.addItems(NOTE_NAMES)
        self.seed_note_combo.setCurrentText("A3")
        self.seed_note_combo.currentTextChanged.connect(self._on_seed_note_changed)
        fn_form.addRow("  or note:", self.seed_note_combo)

        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(1, 128)
        self.n_steps_spin.setValue(16)
        fn_form.addRow("Max steps:", self.n_steps_spin)

        left_v.addWidget(fn_box)

        # Scale / quantisation
        scale_box = QGroupBox("Scale Quantisation")
        scale_form = QFormLayout(scale_box)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(list(SCALES.keys()))
        _set_combo(self.scale_combo, "Free (Atonal)")
        scale_form.addRow("Scale:", self.scale_combo)
        self.octave_spin = QSpinBox()
        self.octave_spin.setRange(1, 6)
        self.octave_spin.setValue(3)
        scale_form.addRow("Octave range:", self.octave_spin)
        left_v.addWidget(scale_box)

        # Synthesis
        synth_box = QGroupBox("Synthesis")
        synth_form = QFormLayout(synth_box)
        self.wave_combo = _make_wave_combo("Sine")
        synth_form.addRow("Wave type:", self.wave_combo)
        self.note_dur_spin = QDoubleSpinBox()
        self.note_dur_spin.setRange(0.02, 10.0)
        self.note_dur_spin.setSingleStep(0.05)
        self.note_dur_spin.setValue(0.3)
        synth_form.addRow("Note duration (s):", self.note_dur_spin)
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.0, 2.0)
        self.gap_spin.setSingleStep(0.01)
        self.gap_spin.setValue(0.05)
        synth_form.addRow("Gap (s):", self.gap_spin)
        self.attack_spin = QDoubleSpinBox()
        self.attack_spin.setRange(0.0, 5.0)
        self.attack_spin.setValue(0.01)
        synth_form.addRow("Attack (s):", self.attack_spin)
        self.release_spin = QDoubleSpinBox()
        self.release_spin.setRange(0.0, 5.0)
        self.release_spin.setValue(0.1)
        synth_form.addRow("Release (s):", self.release_spin)
        left_v.addWidget(synth_box)

        self.reverb = ReverbGroup()
        left_v.addWidget(self.reverb)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run f(x)")
        self.run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.run_btn)
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.clicked.connect(self._play)
        btn_row.addWidget(self.play_btn)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.clicked.connect(self.engine.stop)
        btn_row.addWidget(self.stop_btn)
        left_v.addLayout(btn_row)

        self.export_btn = QPushButton("Export Sequence…")
        self.export_btn.clicked.connect(self._export)
        left_v.addWidget(self.export_btn)

        # Termination info
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #aaaacc; font-size: 10px;")
        left_v.addWidget(self.info_label)
        left_v.addStretch()
        root.addWidget(left)

        # ── RIGHT: cobweb diagram + table ─────────────────────────────────────
        right = QWidget()
        right_v = QVBoxLayout(right)

        right_v.addWidget(QLabel("Cobweb / orbit diagram:"))
        self.cobweb = CobwebDiagram()
        self.cobweb.setMinimumHeight(320)
        right_v.addWidget(self.cobweb, 2)

        right_v.addWidget(QLabel("Generated frequencies:"))
        self.freq_table = QTableWidget(0, 3)
        self.freq_table.setHorizontalHeaderLabels(["Step", "Frequency (Hz)", "Note"])
        self.freq_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.freq_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.freq_table.setAlternatingRowColors(True)
        right_v.addWidget(self.freq_table, 1)

        root.addWidget(right, 1)

        # Initial validation
        self._validate_expr(self.expr_edit.text())

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_seed_note_changed(self, name: str) -> None:
        try:
            self.seed_spin.setValue(midi_to_freq(note_name_to_midi(name)))
        except Exception:
            pass

    def _validate_expr(self, expr: str) -> None:
        err = validate_iter_expression(expr, self.seed_spin.value())
        if err is None:
            self.validate_label.setText("✓ Expression looks valid")
            self.validate_label.setStyleSheet("color: #88cc88; font-size: 10px;")
        else:
            self.validate_label.setText(f"✗ {err}")
            self.validate_label.setStyleSheet("color: #ff6666; font-size: 10px;")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _run(self) -> bool:
        expr     = self.expr_edit.text().strip()
        seed     = self.seed_spin.value()
        n_steps  = self.n_steps_spin.value()
        scale    = self.scale_combo.currentText()
        oct_range = self.octave_spin.value()

        err = validate_iter_expression(expr, seed)
        if err:
            QMessageBox.warning(self, "Invalid Expression", err)
            return False

        try:
            result = iterate_function(expr, seed, n_steps, scale, oct_range)
        except Exception as e:
            QMessageBox.warning(self, "Iteration Error", str(e))
            return False

        self._last_result = result

        # Populate table (skip index 0 = seed)
        freqs = result.frequencies
        self.freq_table.setRowCount(len(freqs))
        for i, f in enumerate(freqs):
            from audio_engine import nearest_note_name
            label = "seed" if i == 0 else str(i)
            self.freq_table.setItem(i, 0, QTableWidgetItem(label))
            self.freq_table.setItem(i, 1, QTableWidgetItem(f"{f:.3f}"))
            self.freq_table.setItem(i, 2, QTableWidgetItem(nearest_note_name(f)))

        # Update cobweb
        self.cobweb.set_result(result, expr)

        # Status / termination info
        if result.terminated_early:
            msg = (
                f"⚠  Sequence stopped early at step {result.steps_generated} / {n_steps}.\n"
                f"{result.termination_reason}"
            )
            self.info_label.setStyleSheet("color: #ffaa44; font-size: 10px;")
        else:
            msg = f"✓  {result.steps_generated} steps generated.  {result.termination_reason}"
            self.info_label.setStyleSheet("color: #88cc88; font-size: 10px;")
        self.info_label.setText(msg)

        self.status_message.emit(
            f"Iterated f(x)={expr}  seed={seed:.2f} Hz  "
            f"→ {result.steps_generated} frequencies"
        )
        return True

    def _play(self) -> None:
        if self._last_result is None:
            if not self._run():
                return
        if self._last_result is None:
            return

        freqs = self._last_result.frequencies
        if len(freqs) < 2:
            self.status_message.emit("Not enough frequencies to play.")
            return

        try:
            buf = render_sequence_audio(
                freqs,
                self.note_dur_spin.value(),
                self.wave_combo.currentText().lower(),
                self.gap_spin.value(),
                0.7,
                self.attack_spin.value(),
                0.05, 0.8,
                self.release_spin.value(),
            )
            buf = self.reverb.maybe_apply(buf)
            self._last_buffer = buf
            self.engine.play(buf)
            self.status_message.emit(
                f"Playing iterated function sequence  ({len(freqs)} notes)"
            )
        except Exception as e:
            self.status_message.emit(f"Playback error: {e}")

    def _export(self) -> None:
        if self._last_buffer is None:
            QMessageBox.information(self, "Export", "Run and play the sequence first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Iterated Sequence", "iterated.wav", EXPORT_FILTER
        )
        if not path:
            return
        try:
            audio = self._last_buffer
            if audio.ndim == 1:
                audio = mono_to_stereo(audio)
            export_audio(audio, SAMPLE_RATE, path)
            self.status_message.emit(f"Exported → {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # ── Project state ─────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "expr":           self.expr_edit.text(),
            "seed_freq":      self.seed_spin.value(),
            "n_steps":        self.n_steps_spin.value(),
            "scale":          self.scale_combo.currentText(),
            "octave_range":   self.octave_spin.value(),
            "wave_type":      self.wave_combo.currentText().lower(),
            "note_duration":  self.note_dur_spin.value(),
            "gap":            self.gap_spin.value(),
            "attack":         self.attack_spin.value(),
            "release":        self.release_spin.value(),
            **self.reverb.get_state(),
        }

    def set_state(self, state: dict) -> None:
        self.expr_edit.setText(state.get("expr", "x * 1.5"))
        self.seed_spin.setValue(float(state.get("seed_freq", 220.0)))
        self.n_steps_spin.setValue(int(state.get("n_steps", 16)))
        _set_combo(self.scale_combo, state.get("scale", "Free (Atonal)"))
        self.octave_spin.setValue(int(state.get("octave_range", 3)))
        _set_combo(self.wave_combo, state.get("wave_type", "sine").capitalize())
        self.note_dur_spin.setValue(float(state.get("note_duration", 0.3)))
        self.gap_spin.setValue(float(state.get("gap", 0.05)))
        self.attack_spin.setValue(float(state.get("attack", 0.01)))
        self.release_spin.setValue(float(state.get("release", 0.1)))
        self.reverb.set_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# MainWindow
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Top-level application window."""

    APP_NAME = "Atonal Music Studio"

    def __init__(self) -> None:
        super().__init__()
        self.engine = AudioEngine()
        self._project: dict = new_project()
        self._project_path: str | None = None
        self._dirty = False

        self.setWindowTitle(self.APP_NAME)
        self.setMinimumSize(1020, 720)
        self._build_ui()
        self._build_menus()
        self.statusBar().showMessage("Ready.")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        self.synth_tab    = SynthTab(self.engine)
        self.equation_tab = EquationTab(self.engine)
        self.sequencer_tab = SequencerTab(self.engine)
        self.samples_tab  = SamplesTab(self.engine)
        self.iter_tab     = IteratedFunctionTab(self.engine)

        self.tabs.addTab(self.synth_tab,     "🎛  Synthesiser")
        self.tabs.addTab(self.equation_tab,  "∞  Equation Synth")
        self.tabs.addTab(self.sequencer_tab, "⚡  Sequencer")
        self.tabs.addTab(self.samples_tab,   "🎵  Samples")
        self.tabs.addTab(self.iter_tab,      "🔁  Iterated f(x)")

        layout.addWidget(self.tabs)

        for tab in (self.synth_tab, self.equation_tab,
                    self.sequencer_tab, self.samples_tab, self.iter_tab):
            tab.status_message.connect(self.statusBar().showMessage)

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # ── File ──────────────────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        new_act = QAction("&New Project", self)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.triggered.connect(self._action_new)
        file_menu.addAction(new_act)

        open_act = QAction("&Open Project…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._action_open)
        file_menu.addAction(open_act)

        save_act = QAction("&Save Project", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._action_save)
        file_menu.addAction(save_act)

        saveas_act = QAction("Save Project &As…", self)
        saveas_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        saveas_act.triggered.connect(self._action_save_as)
        file_menu.addAction(saveas_act)

        file_menu.addSeparator()

        # Recent projects sub-menu (populated dynamically on show)
        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._recent_menu.aboutToShow.connect(self._populate_recent_menu)

        file_menu.addSeparator()

        export_act = QAction("&Export Audio…", self)
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self._action_export)
        file_menu.addAction(export_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # ── Transport ─────────────────────────────────────────────────────────
        trans_menu = mb.addMenu("&Transport")
        stop_all_act = QAction("■  Stop All Audio", self)
        stop_all_act.setShortcut(QKeySequence("Space"))
        stop_all_act.triggered.connect(self.engine.stop)
        trans_menu.addAction(stop_all_act)

        # ── Help ──────────────────────────────────────────────────────────────
        help_menu = mb.addMenu("&Help")
        about_act = QAction("&About…", self)
        about_act.triggered.connect(self._action_about)
        help_menu.addAction(about_act)
        shortcut_act = QAction("Keyboard &Shortcuts…", self)
        shortcut_act.triggered.connect(self._action_shortcuts)
        help_menu.addAction(shortcut_act)

    # ── Recent projects ───────────────────────────────────────────────────────

    def _populate_recent_menu(self) -> None:
        self._recent_menu.clear()
        recents: list[str] = []

        if self._project_path:
            recents = get_recent_projects(self._project_path)
        else:
            # No project open — scan home directory for any sidecars
            recents = get_recent_projects_from_dir(str(Path.home()))

        if not recents:
            no_act = QAction("(no recent projects)", self)
            no_act.setEnabled(False)
            self._recent_menu.addAction(no_act)
            return

        for path in recents:
            act = QAction(Path(path).name + f"  —  {path}", self)
            act.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
            self._recent_menu.addAction(act)

    def _open_recent(self, path: str) -> None:
        if not self._confirm_discard():
            return
        try:
            p = load_project(path)
            self._project = p
            self._project_path = path
            self._apply_project(p)
            self._set_dirty(False)
            self.statusBar().showMessage(f"Opened: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Open Failed", str(e))

    # ── Project helpers ───────────────────────────────────────────────────────

    def _collect_project(self) -> dict:
        p = copy.deepcopy(self._project)
        p["synth"]             = self.synth_tab.get_state()
        p["equation"]          = self.equation_tab.get_state()
        p["sequencer"]         = self.sequencer_tab.get_state()
        p["samples"]           = self.samples_tab.get_state().get("samples", [])
        p["iterated_function"] = self.iter_tab.get_state()
        return p

    def _apply_project(self, p: dict) -> None:
        self.synth_tab.set_state(p.get("synth", {}))
        self.equation_tab.set_state(p.get("equation", {}))
        self.sequencer_tab.set_state(p.get("sequencer", {}))
        self.samples_tab.set_state({"samples": p.get("samples", [])})
        self.iter_tab.set_state(p.get("iterated_function", {}))

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Discard and continue?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Discard

    def _set_dirty(self, dirty: bool = True) -> None:
        self._dirty = dirty
        title = self.APP_NAME
        if self._project_path:
            title += f" — {Path(self._project_path).name}"
        if dirty:
            title += " *"
        self.setWindowTitle(title)

    # ── Menu actions ──────────────────────────────────────────────────────────

    def _action_new(self) -> None:
        if not self._confirm_discard():
            return
        self._project = new_project()
        self._project_path = None
        self._apply_project(self._project)
        self._set_dirty(False)
        self.statusBar().showMessage("New project created.")

    def _action_open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            f"Atonal Music Studio (*{PROJECT_EXTENSION});;All Files (*)",
        )
        if not path:
            return
        try:
            p = load_project(path)
            self._project = p
            self._project_path = path
            self._apply_project(p)
            self._set_dirty(False)
            self.statusBar().showMessage(f"Opened: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Open Failed", str(e))

    def _action_save(self) -> None:
        if self._project_path is None:
            self._action_save_as()
        else:
            self._do_save(self._project_path)

    def _action_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            self._project.get("name", "Untitled") + PROJECT_EXTENSION,
            f"Atonal Music Studio (*{PROJECT_EXTENSION})",
        )
        if path:
            self._do_save(path)

    def _do_save(self, path: str) -> None:
        try:
            p = self._collect_project()
            save_project(path, p)
            self._project = p
            self._project_path = path
            self._set_dirty(False)
            self.statusBar().showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def _action_export(self) -> None:
        tab = self.tabs.currentWidget()
        if hasattr(tab, "_export"):
            tab._export()  # type: ignore[attr-defined]
        elif hasattr(tab, "_export_sample"):
            tab._export_sample()  # type: ignore[attr-defined]
        else:
            self.statusBar().showMessage("Nothing to export on this tab.")

    def _action_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {self.APP_NAME}",
            f"<h2>Atonal Music Studio</h2>"
            f"<p>A cross-platform creative tool for non-traditional music composition.</p>"
            f"<p><b>Features</b></p><ul>"
            f"<li>Sine, sawtooth, square, and <b>triangle</b> wave synthesiser with ADSR + reverb</li>"
            f"<li>Equation synth: Fibonacci, π, e, φ, √n, primes, Collatz, Triangular, "
            f"Catalan, Van der Corput, logistic map, custom expressions</li>"
            f"<li>16-step × 8-track sequencer with per-step velocity, swing, "
            f"per-track pattern length (polyrhythm), and batch track export</li>"
            f"<li>Multi-format audio sample player</li>"
            f"<li><b>Iterated Function</b> tab: explore f(x) orbits with cobweb diagram</li>"
            f"<li>Project save/load, recent projects, multi-format audio export</li>"
            f"</ul>"
            f"<p>Supported formats: WAV · FLAC · OGG · AIFF<br>"
            f"With FFmpeg: MP3 · M4A · AAC</p>"
            f"<p><small>Python · NumPy · SciPy · sounddevice · PyQt6</small></p>",
        )

    def _action_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "Ctrl+N          New project\n"
            "Ctrl+O          Open project\n"
            "Ctrl+S          Save project\n"
            "Ctrl+Shift+S    Save As\n"
            "Ctrl+E          Export audio (current tab)\n"
            "Space           Stop all audio\n"
            "Ctrl+Q          Quit\n\n"
            "Sequencer:\n"
            "  Right-click a step button → set velocity",
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._confirm_discard():
            self.engine.stop()
            event.accept()
        else:
            event.ignore()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Atonal Music Studio")
    app.setOrganizationName("AtonalStudio")
    _apply_dark_palette(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
