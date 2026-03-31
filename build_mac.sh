#!/bin/bash
# =============================================================
# build_mac.sh — Build Atonal Music Studio for macOS
# =============================================================
# Prerequisites (run once):
#   python3 -m venv venv
#   source venv/bin/activate
#   pip install -r requirements.txt -r requirements-dev.txt
#
# Optional — for the .dmg disk image:
#   brew install create-dmg
#
# Optional — for MP3/M4A/AAC export:
#   brew install ffmpeg
#
# After this script completes:
#   dist/AtonalMusicStudio.app        ← drag to /Applications manually
#   dist/AtonalMusicStudio.dmg        ← distributable disk image
#     (only if create-dmg is installed)
# =============================================================

set -e   # exit on first error

APP_NAME="AtonalMusicStudio"
APP_VERSION="1.0.0"
DMG_NAME="${APP_NAME}-${APP_VERSION}-mac"

echo ""
echo "============================================="
echo " Atonal Music Studio — macOS Build"
echo "============================================="
echo ""

# ── Activate venv if not already active ──────────────────────────────────────
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "ERROR: No virtual environment found."
        echo "Run:  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt"
        exit 1
    fi
fi

# ── Clean previous build ─────────────────────────────────────────────────────
echo "Cleaning previous build output..."
rm -rf "build/${APP_NAME}"
rm -rf "dist/${APP_NAME}.app"
rm -f  "dist/${DMG_NAME}.dmg"

# ── Convert icon if needed ───────────────────────────────────────────────────
# PyInstaller wants a .icns on macOS.
if [ ! -f "icon.icns" ] && [ -f "icon.png" ]; then
    echo "Converting icon.png to icon.icns..."
    if command -v sips &>/dev/null && command -v iconutil &>/dev/null; then
        ICONSET="icon.iconset"
        mkdir -p "$ICONSET"
        for SIZE in 16 32 64 128 256 512; do
            sips -z $SIZE $SIZE icon.png --out "${ICONSET}/icon_${SIZE}x${SIZE}.png" &>/dev/null
            DOUBLE=$((SIZE * 2))
            sips -z $DOUBLE $DOUBLE icon.png --out "${ICONSET}/icon_${SIZE}x${SIZE}@2x.png" &>/dev/null
        done
        iconutil -c icns "$ICONSET" -o icon.icns
        rm -rf "$ICONSET"
        echo "  icon.icns created."
    else
        echo "  [warning] sips/iconutil not available — building without .icns"
    fi
fi

# ── Run PyInstaller ──────────────────────────────────────────────────────────
echo ""
echo "Running PyInstaller..."
echo ""
pyinstaller atonal.spec --noconfirm

echo ""
echo "PyInstaller build complete."
echo "Output: dist/${APP_NAME}.app"
echo ""

# ── DMG creation ─────────────────────────────────────────────────────────────
if command -v create-dmg &>/dev/null; then
    echo "Building DMG disk image..."
    create-dmg \
        --volname "Atonal Music Studio" \
        --volicon "icon.icns" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "${APP_NAME}.app" 175 190 \
        --hide-extension "${APP_NAME}.app" \
        --app-drop-link 425 190 \
        --no-internet-enable \
        "dist/${DMG_NAME}.dmg" \
        "dist/${APP_NAME}.app"
    echo "  DMG ready: dist/${DMG_NAME}.dmg"
else
    echo "  [info] create-dmg not found — skipping DMG creation."
    echo "  Install with:  brew install create-dmg"
    echo "  The .app bundle is usable directly: dist/${APP_NAME}.app"
fi

echo ""
echo "============================================="
echo " Build finished."
echo "============================================="
echo ""

# ── macOS unsigned app note ───────────────────────────────────────────────────
echo "NOTE: This app is unsigned."
echo "Users will need to right-click → Open on first launch to bypass Gatekeeper."
echo "To add signing later: codesign --deep -s 'Developer ID Application: ...' dist/${APP_NAME}.app"
echo ""
