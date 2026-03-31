#!/bin/bash
# =============================================================
# build_linux.sh — Build Atonal Music Studio for Linux
# =============================================================
# Prerequisites (run once):
#   sudo apt install python3 python3-venv python3-pip libportaudio2 libxcb-cursor0
#   python3 -m venv venv
#   source venv/bin/activate
#   pip install -r requirements.txt -r requirements-dev.txt
#
# Optional — for MP3/M4A/AAC export:
#   sudo apt install ffmpeg
#
# After this script completes:
#   dist/AtonalMusicStudio/           ← folder containing the app
#   dist/AtonalMusicStudio.tar.gz     ← distributable archive
# =============================================================

set -e

APP_NAME="AtonalMusicStudio"
APP_VERSION="1.0.0"
TARBALL="${APP_NAME}-${APP_VERSION}-linux-x86_64.tar.gz"

echo ""
echo "============================================="
echo " Atonal Music Studio — Linux Build"
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

# ── System dependency check ───────────────────────────────────────────────────
echo "Checking system dependencies..."
MISSING=""
for pkg in libportaudio2 libxcb-cursor0; do
    if ! dpkg -l "$pkg" &>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done
if [ -n "$MISSING" ]; then
    echo "  [warning] Missing system packages:$MISSING"
    echo "  Install with:  sudo apt install$MISSING"
fi

# ── Clean previous build ─────────────────────────────────────────────────────
echo "Cleaning previous build output..."
rm -rf "build/${APP_NAME}"
rm -rf "dist/${APP_NAME}"
rm -f  "dist/${TARBALL}"

# ── Run PyInstaller ──────────────────────────────────────────────────────────
echo ""
echo "Running PyInstaller..."
echo ""
pyinstaller atonal.spec --noconfirm

echo ""
echo "PyInstaller build complete."
echo "Output: dist/${APP_NAME}/"
echo ""

# ── Install desktop entry and icon into the bundle ───────────────────────────
BUNDLE_DIR="dist/${APP_NAME}"

# Copy icon
if [ -f "icon.png" ]; then
    cp icon.png "${BUNDLE_DIR}/icon.png"
fi

# Write a launcher shell script so users can double-click from a file manager
cat > "${BUNDLE_DIR}/AtonalMusicStudio.sh" << 'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/AtonalMusicStudio" "$@"
LAUNCHER
chmod +x "${BUNDLE_DIR}/AtonalMusicStudio.sh"

# Write a .desktop file inside the bundle for easy system integration
cat > "${BUNDLE_DIR}/atonal-music-studio.desktop" << DESKTOP
[Desktop Entry]
Name=Atonal Music Studio
Comment=Non-traditional music composition tool
Exec=${HOME}/${APP_NAME}/AtonalMusicStudio
Icon=${HOME}/${APP_NAME}/icon.png
Type=Application
Terminal=false
Categories=Audio;Music;
StartupNotify=true
DESKTOP

echo "  Desktop entry written to ${BUNDLE_DIR}/atonal-music-studio.desktop"
echo "  (Users should edit the Exec= and Icon= paths after extracting)"

# ── Create tarball ────────────────────────────────────────────────────────────
echo ""
echo "Creating distributable tarball..."
cd dist
tar -czf "${TARBALL}" "${APP_NAME}/"
cd ..
echo "  Tarball ready: dist/${TARBALL}"

echo ""
echo "============================================="
echo " Build finished."
echo "============================================="
echo ""
echo "Distribution:"
echo "  Share dist/${TARBALL}"
echo "  Users extract it and run:  ./${APP_NAME}/${APP_NAME}"
echo ""
echo "Optional system integration (users can run after extracting):"
echo "  cp ${APP_NAME}/atonal-music-studio.desktop ~/.local/share/applications/"
echo "  (Edit Exec= and Icon= paths in the .desktop file first)"
echo ""
