; =============================================================
; installer/atonal_setup.iss — Inno Setup 6 script
; =============================================================
; Prerequisites:
;   - Inno Setup 6 installed (https://jrsoftware.org/isinfo.php)
;   - PyInstaller build completed:  build_win.bat
;     (produces dist\AtonalMusicStudio\)
;
; Run:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\atonal_setup.iss
;   OR just run build_win.bat — it calls ISCC automatically.
;
; Output:
;   installer\AtonalMusicStudio-Setup.exe
; =============================================================

#define AppName      "Atonal Music Studio"
#define AppVersion   "1.0.0"
#define AppPublisher "Atonal Studio"
#define AppExeName   "AtonalMusicStudio.exe"
#define SourceDir    "..\dist\AtonalMusicStudio"
#define OutputDir    "."

[Setup]
AppId={{8A3F2C1D-4B7E-4F9A-B2D6-1C5E8F3A7D2B}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/seanrugg/atonal-music-studio
AppSupportURL=https://github.com/seanrugg/atonal-music-studio/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir={#OutputDir}
OutputBaseFilename=AtonalMusicStudio-Setup
SetupIconFile=..\icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
; No admin rights required — installs to user's AppData or Program Files
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Allow running on Windows 10 and later
MinVersion=10.0

; Unsigned app — this suppresses the worst of SmartScreen for now.
; When you add a code signing cert, set SignTool here.
; SignTool=signtool sign /fd sha256 /t http://timestamp.digicert.com $f

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Include everything PyInstaller produced
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name: "{group}\{#AppName}";         Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
; Optional desktop shortcut
Name: "{autodesktop}\{#AppName}";   Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
; Offer to launch the app after installation
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up .ams project sidecar files left in AppData if any
Type: filesandordirs; Name: "{localappdata}\{#AppName}"

[Code]
// Show a friendly note about unsigned app on first launch page
procedure InitializeWizard();
begin
  WizardForm.WelcomeLabel2.Caption :=
    'This will install ' + ExpandConstant('{#AppName}') + ' version ' +
    ExpandConstant('{#AppVersion}') + ' on your computer.' + #13#10 + #13#10 +
    'Note: This application is currently unsigned. Windows may show a ' +
    '"Windows protected your PC" SmartScreen warning when you first run ' +
    'the installer. Click "More info" then "Run anyway" to proceed.' + #13#10 + #13#10 +
    'Click Next to continue.';
end;
