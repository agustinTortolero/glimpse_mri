; Glimpse MRI — Inno Setup Script (Windows x64)
; Build with: Inno Setup 6.x (ISCC.exe or IDE)

#define MyAppName       "Glimpse MRI"
#define MyAppVersion    "1.0.0.0"
#define MyAppPublisher  "Agustin Tortolero"
#define MyAppExeName    "glimpseMRI.exe"

; Adjust these two paths if your tree changes:
#define ProjRoot        "C:\AgustinTortolero_repos\portafolio\GlimpseMRI\gui"
#define Dist            ProjRoot + "\dist\GlimpseMRI_Release"

; Optional app icon (bundled into install, used for shortcuts)
#define AppIcon         ProjRoot + "\assets\images\icons\mri.ico"

[Setup]
; Use a real GUID (fixed for upgrades/patches). Generate once and keep it stable.
AppId={{8E06C1F2-8C99-4DA7-8C24-5D0F5B7E0A03}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/agustinTortolero
AppSupportURL=https://github.com/agustinTortolero
AppUpdatesURL=https://github.com/agustinTortolero
DefaultDirName={autopf64}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputBaseFilename=GlimpseMRI_{#MyAppVersion}_Setup
OutputDir={#ProjRoot}\dist\installer
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
Compression=lzma2/ultra64
SolidCompression=yes
CompressionThreads=auto
WizardStyle=modern
SetupLogging=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile={#AppIcon}
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
ChangesAssociations=yes
UsePreviousAppDir=yes
DisableDirPage=no
DisableProgramGroupPage=yes
CloseApplications=force

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
Name: "assocdcm"; Description: "Associate .dcm files with Glimpse MRI"; GroupDescription: "File associations:"

[Files]
; Package everything from dist, but skip deploy logs
Source: "{#Dist}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "windeployqt*.log"

; Keep a copy of the icon for shortcuts & file association
Source: "{#AppIcon}"; DestDir: "{app}\assets\icons"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\icons\mri.ico"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\icons\mri.ico"; Tasks: desktopicon

; Associate .dcm with Glimpse MRI (optional task)
[Registry]
Root: HKCR; Subkey: ".dcm"; ValueType: string; ValueName: ""; ValueData: "GlimpseMRI.DICOM"; Flags: uninsdeletevalue; Tasks: assocdcm
Root: HKCR; Subkey: "GlimpseMRI.DICOM"; ValueType: string; ValueName: ""; ValueData: "DICOM file"; Flags: uninsdeletekey; Tasks: assocdcm
Root: HKCR; Subkey: "GlimpseMRI.DICOM\DefaultIcon"; ValueType: string; ValueData: "{app}\assets\icons\mri.ico,0"; Tasks: assocdcm
Root: HKCR; Subkey: "GlimpseMRI.DICOM\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: assocdcm

[Run]
; VC++ redist (if present in dist)
Filename: "{app}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; \
    Flags: shellexec runhidden waituntilterminated; Check: VCNeeded; \
    StatusMsg: "Installing Visual C++ Runtime..."

; Offer to launch app after install
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Glimpse MRI"; Flags: nowait postinstall skipifsilent

[Code]
function VCNeeded(): Boolean;
begin
  { Visual C++ 2015-2022 uses the 14.0 key path; this covers VS2015..VS2022 redistributables. }
  Result := not RegKeyExists(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64')
        and not RegKeyExists(HKLM, 'SOFTWARE\Microsoft\DevDiv\VC\Servicing\14.0\RuntimeMinimum');
end;
