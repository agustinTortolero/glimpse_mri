; Glimpse MRI — Inno Setup Script (Windows x64)
; Build with: Inno Setup 6.x (ISCC.exe or IDE)

#define MyAppName       "Glimpse MRI"
#define MyAppVersion    "1.0.0.0"
#define MyAppPublisher  "Agustin Tortolero"
#define MyAppExeName    "glimpseMRI.exe"

; Adjust these two paths if your tree changes:
#define ProjRoot        "C:\AgustinTortolero_repos\portafolio\GlimpseMRI\gui"
#define Dist            ProjRoot + "\dist\GlimpseMRI_Release"

; vcpkg bin folder (for DCMTK, ISMRMRD, FFTW, etc.)
#define VcpkgBin        "C:\src\vcpkg\installed\x64-windows\bin"

; Optional app icon (bundled into install, used for shortcuts)
#define AppIcon         ProjRoot + "\assets\images\icons\mri.ico"

[Setup]
; Use a real GUID (fixed for upgrades/patches). Generate once and keep it stable.
AppId={{8E06C1F2-8C99-4DA7-8C24-5D0F5B7E0A03}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}

; Tagline / descriptive name
AppVerName={#MyAppName} - Medical Image R&D Tool
AppComments=Medical Image R&D Tool

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

; Explicitly copy VC++ redistributable from dist (so it's always present in the install dir)
Source: "{#Dist}\vc_redist.x64.exe"; DestDir: "{app}"; Flags: ignoreversion

; Keep a copy of the icon for shortcuts & file association
Source: "{#AppIcon}"; DestDir: "{app}\assets\icons"; Flags: ignoreversion

; Critical runtime DLLs from vcpkg (DCMTK base, ISMRMRD, FFTW)
Source: "{#VcpkgBin}\ofstd.dll";    DestDir: "{app}"; Flags: ignoreversion
Source: "{#VcpkgBin}\ismrmrd.dll";  DestDir: "{app}"; Flags: ignoreversion
Source: "{#VcpkgBin}\fftw3.dll";    DestDir: "{app}"; Flags: ignoreversion
Source: "{#VcpkgBin}\fftw3f.dll";   DestDir: "{app}"; Flags: ignoreversion

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
; VC++ redist (if present in app dir)
Filename: "{app}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; \
    Flags: shellexec runhidden waituntilterminated; Check: VCNeeded; \
    StatusMsg: "Installing Visual C++ Runtime..."

; Offer to launch app after install
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Glimpse MRI"; Flags: nowait postinstall skipifsilent

[Code]

var
  DisclaimerPage: TWizardPage;
  DisclaimerCheckBox: TNewCheckBox;

function VCNeeded(): Boolean;
begin
  { Visual C++ 2015-2022 uses the 14.0 key path; this covers VS2015..VS2022 redistributables. }
  Result := not RegKeyExists(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64')
        and not RegKeyExists(HKLM, 'SOFTWARE\Microsoft\DevDiv\VC\Servicing\14.0\RuntimeMinimum');
end;

procedure InitializeWizard;
var
  DisclaimerText: TNewStaticText;
  Msg: String;
begin
  { Create a custom page right after the license page.
    If you don't have a LicenseFile, you can change wpLicense to wpWelcome. }
  DisclaimerPage :=
    CreateCustomPage(
      wpLicense,
      'Medical Imaging Disclaimer',
      'Please read the following disclaimer before installing Glimpse MRI:');

  Msg :=
    'Non-clinical software: Glimpse MRI is intended solely for research, testing, and software development. It is not a medical device.'#13#10#13#10 +
    'This application must not be used for diagnosis, treatment decisions, or any clinical workflow involving patients. Any images, reconstructions, or measurements produced are for demonstration and technical evaluation only.'#13#10#13#10 +
    'No guarantee is made regarding the correctness, completeness, or suitability of the results for clinical purposes. By using this software, you acknowledge that it is provided "as is", without any warranty of any kind.';

  { Static text with word-wrapped disclaimer }
  DisclaimerText := TNewStaticText.Create(DisclaimerPage);
  DisclaimerText.Parent := DisclaimerPage.Surface;
  DisclaimerText.Left := 0;
  DisclaimerText.Top := 0;
  DisclaimerText.Width := DisclaimerPage.SurfaceWidth;
  { Leave a bit more room at the bottom for a taller, multi-line checkbox }
  DisclaimerText.Height := DisclaimerPage.SurfaceHeight - ScaleY(120);
  DisclaimerText.AutoSize := False;
  DisclaimerText.WordWrap := True;
  DisclaimerText.Caption := Msg;

  { Checkbox the user must tick (shorter multi-line caption + extra height) }
  DisclaimerCheckBox := TNewCheckBox.Create(DisclaimerPage);
  DisclaimerCheckBox.Parent := DisclaimerPage.Surface;
  DisclaimerCheckBox.Left := 0;
  DisclaimerCheckBox.Width := DisclaimerPage.SurfaceWidth;

  { 3 short lines so it doesn't get horizontally clipped on 4K / HiDPI }
  DisclaimerCheckBox.Caption :=
    'I understand that Glimpse MRI is NOT a medical device.';

  { Give the checkbox enough vertical space for 3 lines }
  DisclaimerCheckBox.Height := ScaleY(60);
  DisclaimerCheckBox.Top :=
    DisclaimerText.Top + DisclaimerText.Height + ScaleY(8);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  { Block navigation if user hasn't confirmed the disclaimer }
  if (CurPageID = DisclaimerPage.ID) and (not DisclaimerCheckBox.Checked) then
  begin
    MsgBox(
      'Please confirm that you understand and accept the medical imaging disclaimer before continuing.',
      mbError, MB_OK);
    Result := False;
  end;
end;
