# -*- mode: python ; coding: utf-8 -*-
#
# atonal.spec — PyInstaller build spec for Atonal Music Studio
#
# Usage (from the repo root, with venv active):
#   pyinstaller atonal.spec
#
# Output lands in dist/AtonalMusicStudio/
#
# Platform notes
# --------------
# Build on the TARGET platform — PyInstaller does not cross-compile.
#   Windows  →  run build_win.bat
#   macOS    →  run build_mac.sh
#   Linux    →  run build_linux.sh
#
# The spec is intentionally kept platform-agnostic so that the same file
# drives all three builds.  Platform-specific packaging (Inno Setup / DMG /
# tarball) is handled by the build scripts.

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── Collect binary/data files that PyInstaller misses by default ─────────────

# soundfile ships a native libsndfile binary
soundfile_binaries = collect_dynamic_libs("soundfile")
soundfile_datas    = collect_data_files("soundfile")

# sounddevice ships PortAudio binaries on Windows and macOS
sounddevice_binaries = collect_dynamic_libs("sounddevice")
sounddevice_datas    = collect_data_files("sounddevice")

# scipy uses compiled Fortran/C extensions — collect_dynamic_libs handles them
scipy_binaries = collect_dynamic_libs("scipy")

# PyQt6 plugins (platform, imageformats, etc.) — critical for the UI to render
pyqt6_datas = collect_data_files("PyQt6", includes=["Qt/plugins/*"])

# App icon — included as a data file so the app can reference it at runtime
icon_datas = []
if os.path.isfile("icon.png"):
    icon_datas = [("icon.png", ".")]

all_datas    = soundfile_datas + sounddevice_datas + pyqt6_datas + icon_datas
all_binaries = soundfile_binaries + sounddevice_binaries + scipy_binaries

# ── Analysis ──────────────────────────────────────────────────────────────────

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=[
        # soundfile / sounddevice internals
        "soundfile",
        "sounddevice",
        "cffi",
        "_cffi_backend",
        # scipy submodules that are imported at runtime but not statically visible
        "scipy.signal",
        "scipy.signal.windows",
        "scipy.signal._upfirdn",
        "scipy.special",
        "scipy.special._ufuncs",
        "scipy.linalg",
        "scipy.fft",
        "scipy._lib.array_api_compat.numpy.fft",
        # numpy
        "numpy",
        "numpy.core._multiarray_umath",
        "numpy.core._multiarray_tests",
        # PyQt6
        "PyQt6",
        "PyQt6.QtWidgets",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        # pydub is optional (MP3/M4A export) — include so import doesn't crash
        # if pydub is installed; FFmpeg still needs to be on PATH at runtime
        "pydub",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Things we definitely don't need — keeps the bundle smaller
        "tkinter",
        "unittest",
        "email",
        "html",
        "http",
        "urllib",
        "xmlrpc",
        "xml",
        "pydoc",
        "doctest",
        "difflib",
        "calendar",
        "ftplib",
        "test",
        "IPython",
        "matplotlib",
    ],
    noarchive=False,
    optimize=1,
)

# ── PYZ archive ───────────────────────────────────────────────────────────────

pyz = PYZ(a.pure)

# ── EXE / bundle ──────────────────────────────────────────────────────────────

# Resolve icon path per platform
if sys.platform == "win32":
    icon_path = "icon.ico" if os.path.isfile("icon.ico") else None
elif sys.platform == "darwin":
    icon_path = "icon.icns" if os.path.isfile("icon.icns") else None
else:
    icon_path = "icon.png" if os.path.isfile("icon.png") else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # one-folder mode
    name="AtonalMusicStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                       # compress binaries where available
    console=False,                  # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,         # add signing identity here when ready
    entitlements_file=None,
    icon=icon_path,
)

# ── COLLECT (one-folder layout) ───────────────────────────────────────────────

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AtonalMusicStudio",
)

# ── macOS .app bundle ─────────────────────────────────────────────────────────

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="AtonalMusicStudio.app",
        icon=icon_path,
        bundle_identifier="com.atonalstudio.app",
        info_plist={
            "CFBundleDisplayName":        "Atonal Music Studio",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion":            "1.0.0",
            "NSHighResolutionCapable":    True,
            "NSMicrophoneUsageDescription":
                "Atonal Music Studio uses audio output only.",
            # Suppress the quarantine warning as much as possible for unsigned
            "LSFileQuarantineEnabled": False,
        },
    )
