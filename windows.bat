@echo off
:: ============================================================
:: build_win.bat — Build Atonal Music Studio for Windows
:: ============================================================
:: Prerequisites (run once):
::   python -m venv venv
::   venv\Scripts\activate
::   pip install -r requirements.txt
::   pip install -r requirements-dev.txt
::
:: Optional — for MP3/M4A/AAC export support:
::   Install FFmpeg and add it to PATH
::
:: After this script completes:
::   dist\AtonalMusicStudio\AtonalMusicStudio.exe  ← runnable directly
::   installer\AtonalMusicStudio-Setup.exe          ← Inno Setup installer
::     (only if Inno Setup 6 is installed)
:: ============================================================

setlocal enabledelayedexpansion

echo.
echo =============================================
echo  Atonal Music Studio — Windows Build
echo =============================================
echo.

:: ── Activate venv if not already active ──────────────────────────────────────
if not defined VIRTUAL_ENV (
    if exist "venv\Scripts\activate.bat" (
        echo Activating virtual environment...
        call venv\Scripts\activate.bat
    ) else (
        echo ERROR: No virtual environment found.
        echo Run:  python -m venv venv  ^&^&  venv\Scripts\activate  ^&^&  pip install -r requirements.txt -r requirements-dev.txt
        exit /b 1
    )
)

:: ── Clean previous build ─────────────────────────────────────────────────────
echo Cleaning previous build output...
if exist build\AtonalMusicStudio rmdir /s /q build\AtonalMusicStudio
if exist dist\AtonalMusicStudio  rmdir /s /q dist\AtonalMusicStudio

:: ── Convert icon if needed ───────────────────────────────────────────────────
:: PyInstaller wants a .ico on Windows.  If only icon.png exists, convert it
:: using Pillow (installed as part of requirements or requirements-dev).
if not exist icon.ico (
    if exist icon.png (
        echo Converting icon.png to icon.ico...
        python -c "from PIL import Image; img=Image.open('icon.png'); img.save('icon.ico', sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])" 2>nul
        if errorlevel 1 (
            echo   [warning] Pillow not available — building without .ico file.
        ) else (
            echo   icon.ico created.
        )
    )
)

:: ── Run PyInstaller ──────────────────────────────────────────────────────────
echo.
echo Running PyInstaller...
echo.
pyinstaller atonal.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller failed.  See output above.
    exit /b 1
)

echo.
echo PyInstaller build complete.
echo Output: dist\AtonalMusicStudio\AtonalMusicStudio.exe
echo.

:: ── Inno Setup ───────────────────────────────────────────────────────────────
:: Looks for Inno Setup 6 in the default install location.
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if exist "%ISCC%" (
    echo Building Inno Setup installer...
    "%ISCC%" installer\atonal_setup.iss
    if errorlevel 1 (
        echo   [warning] Inno Setup compilation failed.  Check installer\atonal_setup.iss
    ) else (
        echo   Installer ready: installer\AtonalMusicStudio-Setup.exe
    )
) else (
    echo   [info] Inno Setup not found — skipping installer creation.
    echo   Download from https://jrsoftware.org/isinfo.php to build an installer.
    echo   The raw app folder is still usable: dist\AtonalMusicStudio\
)

echo.
echo =============================================
echo  Build finished.
echo =============================================
echo.
endlocal
