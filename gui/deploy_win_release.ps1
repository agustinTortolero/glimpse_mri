# =========================================
# GlimpseMRI — Windows Release deploy (Qt6, MSVC)
# Creates a portable dist folder ready for testing or packaging.
# =========================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Function Write-Info($msg)  { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
Function Write-Cfg($msg)   { Write-Host "[CFG ] $msg" -ForegroundColor Yellow }
Function Write-Step($msg)  { Write-Host "[STEP] $msg" -ForegroundColor Green }
Function Write-Warn($msg)  { Write-Warning $msg }
Function Fail($msg)        { throw "[ERR ] $msg" }

# ---- CONFIG (adjust if your paths differ) -----------------------------------
$QtBin         = "C:\Qt\6.10.0\msvc2022_64\bin"
$WinDeployQt   = Join-Path $QtBin "windeployqt.exe"

$ProjRoot      = "C:\AgustinTortolero_repos\portafolio\GlimpseMRI\gui"
$BuildRel      = Join-Path $ProjRoot "build\Desktop_Qt_6_10_0_MSVC2022_64bit-Release\release"
$ExeName       = "glimpseMRI.exe"
$ExePath       = Join-Path $BuildRel $ExeName

# Destination (clean each run)
$DistRoot      = Join-Path $ProjRoot "dist"
$DistDir       = Join-Path $DistRoot "GlimpseMRI_Release"

# GUI-local release dir (for clarity; currently only the EXE is used)
$GuiReleaseDir = $BuildRel

# OpenCV (world DLL) built via vcpkg (release)
$OpenCvWorldDir = "C:\src\vcpkg\installed\x64-windows\bin"

# vcpkg (release bin dir)
$VcpkgRoot     = "C:\src\vcpkg\installed\x64-windows"
$VcpkgBin      = Join-Path $VcpkgRoot "bin"

Write-Cfg "QtBin         = $QtBin"
Write-Cfg "BuildRel      = $BuildRel"
Write-Cfg "ExePath       = $ExePath"
Write-Cfg "DistDir       = $DistDir"
Write-Cfg "GUI release   = $GuiReleaseDir"
Write-Cfg "OpenCV bin    = $OpenCvWorldDir"
Write-Cfg "vcpkg bin     = $VcpkgBin"
Write-Host ""

# ---- Sanity checks ----------------------------------------------------------
if (!(Test-Path $ExePath))      { Fail "EXE not found: $ExePath" }
if (!(Test-Path $WinDeployQt))  { Fail "windeployqt not found: $WinDeployQt" }

# ---- Clean dist -------------------------------------------------------------
if (Test-Path $DistDir) {
    Write-Step "Removing existing $DistDir ..."
    Remove-Item -Recurse -Force $DistDir
}
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

# ---- Stage main EXE ---------------------------------------------------------
Write-Step "Copy EXE -> $DistDir"
Copy-Item $ExePath $DistDir -Force

# ---- Run windeployqt --------------------------------------------------------
Write-Step "Running windeployqt ..."
$outLog = Join-Path $DistDir "windeployqt.out.log"
$errLog = Join-Path $DistDir "windeployqt.err.log"

$spArgs = @{
    FilePath               = $WinDeployQt
    ArgumentList           = @(
        "--release",
        "--no-translations",
        "--no-compiler-runtime",
        "--plugindir", (Join-Path $DistDir "plugins"),
        "--dir", $DistDir,
        (Join-Path $DistDir $ExeName)
    )
    WorkingDirectory       = $DistDir
    RedirectStandardOutput = $outLog
    RedirectStandardError  = $errLog
    Wait                   = $true
    PassThru               = $true
}
$proc = Start-Process @spArgs

Write-Info ("windeployqt exit code: " + $proc.ExitCode)
if ($proc.ExitCode -ne 0) {
    Write-Warn "=== windeployqt.err.log (first 200 lines) ==="
    Get-Content $errLog -ErrorAction SilentlyContinue |
        Select-Object -First 200 |
        ForEach-Object { Write-Host $_ }
    Fail "windeployqt failed"
}

# ---- Helper: copy-first-found ----------------------------------------------
function Copy-FirstFound {
    param(
        [Parameter(Mandatory)][string[]] $Candidates,
        [Parameter(Mandatory)][string]   $DestFullPath,
        [Parameter(Mandatory)][string]   $Label,
        [switch] $NormalizeName
    )
    Write-Step ("Searching " + $Label + " in:")
    foreach ($c in $Candidates) {
        $exists = Test-Path $c
        Write-Info (" - " + $c + "  " + ($(if ($exists) {"OK"} else {"(missing)"})))
    }
    $found = $null
    foreach ($c in $Candidates) {
        if (Test-Path $c) { $found = $c; break }
    }
    if ($null -eq $found) {
        Write-Warn ("No candidate found for " + $Label)
        return $false
    }

    if ($NormalizeName) {
        $destDir  = Split-Path $DestFullPath -Parent
        $destName = Split-Path $DestFullPath -Leaf
        $destReal = Join-Path $destDir $destName
        Write-Step ("Copy " + $Label + " -> " + $destReal)
        Copy-Item $found $destReal -Force
        Write-Info  ("=> Wrote: " + $destReal)
    } else {
        Write-Step ("Copy " + $Label + " -> " + $DestFullPath)
        Copy-Item $found $DestFullPath -Force
        Write-Info ("=> Wrote: " + $DestFullPath)
    }
    return $true
}

# ---- Copy MRI engine + DICOM DLLs ------------------------------------------
$engineCandidates = @(
    "C:\AgustinTortolero_repos\portafolio\GlimpseMRI\engine\build\Release\mri_engine.dll",
    "C:\AgustinTortolero_repos\portafolio\GlimpseMRI\engine\build\RelWithDebInfo\mri_engine.dll"
)

# UPDATED: use the actual dicom_io_lib paths you found
$dicomCandidates  = @(
    "C:\AgustinTortolero_repos\portafolio\GlimpseMRI\dicom_io_lib\build\Desktop_Qt_6_10_0_MSVC2022_64bit-Release\dicom_io_lib.dll",
    "C:\AgustinTortolero_repos\portafolio\GlimpseMRI\dicom_io_lib\bin\dicom_io_lib.dll",
    "C:\AgustinTortolero_repos\portafolio\GlimpseMRI\gui\build\Desktop_Qt_6_10_0_MSVC2022_64bit-Release\release\dicom_io_lib.dll"
)

[void](Copy-FirstFound -Candidates $engineCandidates `
                       -DestFullPath (Join-Path $DistDir "mri_engine.dll") `
                       -Label "mri_engine.dll" `
                       -NormalizeName)

[void](Copy-FirstFound -Candidates $dicomCandidates `
                       -DestFullPath (Join-Path $DistDir "dicom_io_lib.dll") `
                       -Label "dicom_io_lib.dll" `
                       -NormalizeName)

# Small safeguard in case old name slipped in
$dd = Join-Path $DistDir "dicom__io_lib.dll"
$dn = Join-Path $DistDir "dicom_io_lib.dll"
if ((Test-Path $dd) -and -not (Test-Path $dn)) {
    Write-Step "Renaming dicom__io_lib.dll -> dicom_io_lib.dll"
    Rename-Item $dd $dn -Force
}

# ---- Copy OpenCV world DLL from vcpkg --------------------------------------
if (!(Test-Path $OpenCvWorldDir)) {
    Write-Warn ("OpenCV path not found: " + $OpenCvWorldDir)
} else {
    $OpenCvWorld = Get-ChildItem -Path $OpenCvWorldDir -Filter "opencv_world*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $OpenCvWorld) {
        Write-Step ("Copy OpenCV world DLL: " + $OpenCvWorld.Name)
        Write-Info ("    Source: " + $OpenCvWorld.FullName)
        Write-Info ("    Dest  : " + $DistDir)
        Copy-Item $OpenCvWorld.FullName $DistDir -Force
    } else {
        Write-Warn ("No opencv_world*.dll found under " + $OpenCvWorldDir)
    }
}

# ---- Copy vcpkg-dependent DLLs ---------------------------------------------
if (!(Test-Path $VcpkgBin)) {
    Write-Warn ("vcpkg bin not found: " + $VcpkgBin)
} else {
    $copyMasks = @(
        "fftw3f*.dll",
        "fftw3*.dll",
        "pugixml*.dll",
        "hdf5*.dll",
        "zlib*.dll",
        "szip*.dll",
        "*aec*.dll",
        "charls*.dll",
        "dcm*.dll",
        "of*.dll",
        "ijg*.dll"
    )

    foreach ($mask in $copyMasks) {
        $files = Get-ChildItem -Path $VcpkgBin -Filter $mask -ErrorAction SilentlyContinue
        foreach ($f in $files) {
            if ($f.Name -match "_D\.dll$") { continue }
            if ($f.Name -match "d\.dll$")  { continue }
            if ($f.Name -ieq "zlibd1.dll") { continue }
            Write-Step ("Copy " + $f.Name + " from vcpkg/bin")
            Copy-Item $f.FullName $DistDir -Force
        }
    }
}

# ---- Purge DEBUG artifacts accidentally present -----------------------------
$debugPatterns = @("*_D.dll","*d.dll","zlibd1.dll")
foreach ($pat in $debugPatterns) {
    $matches = Get-ChildItem -Path $DistDir -Filter $pat -ErrorAction SilentlyContinue
    foreach ($m in $matches) {
        Write-Step ("PURGE debug DLL: " + $m.Name)
        Remove-Item $m.FullName -Force
    }
}

# ---- CUDA runtime (release; optional but recommended for GPU path) ----------
$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if (Test-Path $CudaBin) {
    foreach ($mask in @('cudart64*.dll','cufft64*.dll','cublas64*.dll','cublasLt64*.dll')) {
        Get-ChildItem -Path $CudaBin -Filter $mask -ErrorAction SilentlyContinue |
            ForEach-Object {
                Write-Step ("Copy CUDA runtime: " + $_.Name)
                Copy-Item $_.FullName $DistDir -Force
            }
    }
} else {
    Write-Warn ("CUDA bin not found at " + $CudaBin + " - GPU path will require system installed CUDA runtime.")
}

# ---- VC++ redistributable (optional; bundled for installer) -----------------
$VCRedistCandidates = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.44.35112\vc_redist.x64.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\v143\vc_redist.x64.exe"
)

$vcDest = Join-Path $DistDir "vc_redist.x64.exe"

Write-Step "Checking for VC++ redistributable to copy into dist..."
$vcFound = $false
foreach ($c in $VCRedistCandidates) {
    $exists = Test-Path $c
    Write-Info (" - Candidate: " + $c + "  " + ($(if ($exists) {"OK"} else {"(missing)"})))
    if ($exists -and -not $vcFound) {
        Write-Step ("Copy VC++ redistributable from: " + $c)
        Copy-Item $c $vcDest -Force
        Write-Info ("=> Wrote: " + $vcDest)
        $vcFound = $true
    }
}

if (-not $vcFound) {
    Write-Warn "VC++ redistributable not found; installer will not bundle vc_redist.x64.exe"
}

# ---- Final sanity check -----------------------------------------------------
$mustHave = @(
    (Join-Path $DistDir "mri_engine.dll"),
    (Join-Path $DistDir "dicom_io_lib.dll")
)
$opencvAny = Get-ChildItem -Path $DistDir -Filter "opencv_world*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1

$missing = @()
foreach ($p in $mustHave) { if (!(Test-Path $p)) { $missing += $p } }
if ($null -eq $opencvAny) { $missing += "(opencv_world*.dll)" }

if ($missing.Count -gt 0) {
    Write-Host ""
    $nl = [Environment]::NewLine
    $msg = "Missing critical runtime(s):" + $nl + " - " + ($missing -join ($nl + " - "))
    Write-Warn $msg
    Fail "Packaging incomplete. See warnings above."
}

# ---- Final listing ----------------------------------------------------------
Write-Host ""
Write-Info ("[LIST] " + $DistDir)
Get-ChildItem $DistDir |
    Select-Object Name, Length |
    Sort-Object Name |
    Format-Table -AutoSize

Write-Host ""
Write-Step ("Deploy complete. You can now run: " + (Join-Path $DistDir $ExeName))
