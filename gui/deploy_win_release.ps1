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

# Your GUI-local release libs (next to project, not build)
$GuiReleaseDir = Join-Path $ProjRoot "release"
$EngineDllPRJ  = Join-Path $GuiReleaseDir "mri_engine_v_1_1.dll"
$DicomDllPRJ   = Join-Path $GuiReleaseDir "dicom_io_lib.dll"

# Build output candidates
$EngineDllBLD  = Join-Path $BuildRel "mri_engine_v_1_1.dll"
$DicomDllBLD_1 = Join-Path $BuildRel "dicom_io_lib.dll"
$DicomDllBLD_2 = Join-Path $BuildRel "dicom__io_lib.dll"  # double underscore (seen previously)

# OpenCV world (release)
$OpenCvWorldDir = "C:\opencv\opencv\build\x64\vc16\bin"
$OpenCvWorld    = Join-Path $OpenCvWorldDir "opencv_world490.dll"

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

# ---- Run windeployqt (Qt Widgets app) --------------------------------------
Write-Step "Run windeployqt --release --compiler-runtime --verbose 2 ..."
$WinDepArgs = @(
    "--release",
    "--compiler-runtime",
    "--verbose", "2",
    "--no-quick-import",
    "--no-opengl-sw",
    "--dir", $DistDir,
    (Join-Path $DistDir $ExeName)
)

$outLog = Join-Path $DistDir "windeployqt.out.log"
$errLog = Join-Path $DistDir "windeployqt.err.log"

Write-Info "Redirecting output to:"
Write-Info " - $outLog"
Write-Info " - $errLog"

$spArgs = @{
    FilePath               = $WinDeployQt
    ArgumentList           = $WinDepArgs
    NoNewWindow            = $true
    Wait                   = $true
    PassThru               = $true
    RedirectStandardOutput = $outLog
    RedirectStandardError  = $errLog
}
$proc = Start-Process @spArgs

Write-Info ("windeployqt exit code: " + $proc.ExitCode)
if ($proc.ExitCode -ne 0) {
    Write-Warn "=== windeployqt.err.log (first 200 lines) ==="
    Get-Content $errLog -ErrorAction SilentlyContinue | Select-Object -First 200 | ForEach-Object { Write-Host $_ }
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
    if ($null -ne $found) {
        if ($NormalizeName) {
            Write-Step ("Copy " + $Label + " from: " + $found)
            Copy-Item $found $DestFullPath -Force
            Write-Info  ("=> Wrote: " + $DestFullPath)
        } else {
            $destDir = Split-Path -Parent $DestFullPath
            Write-Step ("Copy " + $Label + " from: " + $found)
            Copy-Item $found $destDir -Force
            Write-Info  ("=> Wrote: " + (Join-Path $destDir (Split-Path -Leaf $found)))
        }
        return $true
    }
    Write-Warn ("No candidate found for " + $Label + ".")
    return $false
}

# ---- Copy engine (release) --------------------------------------------------
$EngineOk = Copy-FirstFound -Candidates @($EngineDllPRJ, $EngineDllBLD) `
                            -DestFullPath (Join-Path $DistDir "mri_engine_v_1_1.dll") `
                            -Label "Engine DLL" -NormalizeName

# ---- Copy DICOM (release, robust) ------------------------------------------
$DicomOk = Copy-FirstFound -Candidates @($DicomDllPRJ, $DicomDllBLD_1, $DicomDllBLD_2) `
                           -DestFullPath (Join-Path $DistDir "dicom_io_lib.dll") `
                           -Label "DICOM DLL" -NormalizeName

if (-not $DicomOk) {
    $wild = @(
        (Get-ChildItem -Path $BuildRel      -Filter "dicom*io*lib*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1),
        (Get-ChildItem -Path $GuiReleaseDir -Filter "dicom*io*lib*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1)
    ) | Where-Object { $_ -ne $null }
    if ($wild.Count -gt 0) {
        $first = $wild[0].FullName
        Write-Step ("Wildcard hit for DICOM DLL: " + $first)
        Copy-Item $first (Join-Path $DistDir "dicom_io_lib.dll") -Force
        Write-Info  ("=> Wrote: " + (Join-Path $DistDir "dicom_io_lib.dll"))
        $DicomOk = $true
    } else {
        Write-Warn "Wildcard search found no dicom*io*lib*.dll in BuildRel/GUI release."
    }
}

# If both dicom names land in dist, drop the odd one
$dd = Join-Path $DistDir "dicom__io_lib.dll"
$dn = Join-Path $DistDir "dicom_io_lib.dll"
if ((Test-Path $dd) -and (Test-Path $dn)) {
    Write-Step ("REMOVE duplicate: " + (Split-Path -Leaf $dd))
    Remove-Item $dd -Force
}

# ---- OpenCV world (release) -------------------------------------------------
if (Test-Path $OpenCvWorld) {
    Write-Step ("Copy OpenCV world: " + $OpenCvWorld)
    Copy-Item $OpenCvWorld $DistDir -Force
} else {
    $candidate = Get-ChildItem -Path $OpenCvWorldDir -Filter "opencv_world*.dll" -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -notmatch "d\.dll$" } |
                 Sort-Object Name -Descending | Select-Object -First 1
    if ($null -ne $candidate) {
        Write-Step ("Copy OpenCV world (fallback): " + $candidate.FullName)
        Copy-Item $candidate.FullName $DistDir -Force
    } else {
        Write-Warn ("OpenCV world DLL not found at " + $OpenCvWorldDir)
    }
}

# ---- vcpkg runtime DLLs (release) -------------------------------------------
$copyMasks = @(
    "ismrmrd*.dll",
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

# ---- Purge DEBUG artifacts accidentally present -----------------------------
$debugPatterns = @("*_D.dll","*d.dll","zlibd1.dll")
foreach ($pat in $debugPatterns) {
    $matches = Get-ChildItem -Path $DistDir -Filter $pat -ErrorAction SilentlyContinue
    foreach ($m in $matches) {
        Write-Step ("PURGE debug DLL: " + $m.Name)
        Remove-Item $m.FullName -Force
    }
}

# ---- Optional: tidy non-runtime dev files if any slipped in -----------------
$trash = @("*.pdb","*.ilk","*.ipdb","*.iobj")
foreach ($t in $trash) {
    $m = Get-ChildItem -Path $DistDir -Filter $t -ErrorAction SilentlyContinue
    foreach ($x in $m) {
        Write-Step ("REMOVE dev artifact: " + $x.Name)
        Remove-Item $x.FullName -Force
    }
}

# ---- FFTW (release) ---------------------------------------------------------
$FftwBin = "C:\src\vcpkg\installed\x64-windows\bin"
foreach ($mask in @('libfftw3-3.dll','libfftw3f-3.dll','fftw3.dll','libfftw3.dll')) {
    $hits = Get-ChildItem -Path $FftwBin -Filter $mask -ErrorAction SilentlyContinue
    foreach ($h in $hits) {
        Write-Step ("Copy FFTW: " + $h.Name)
        Copy-Item $h.FullName $DistDir -Force
    }
}

# ---- CUDA runtime (release; optional but recommended for GPU path) ----------
$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if (Test-Path $CudaBin) {
    foreach ($mask in @('cudart64*.dll','cufft64*.dll','cublas64*.dll','cublasLt64*.dll')) {
        Get-ChildItem -Path $CudaBin -Filter $mask -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Step ("Copy CUDA runtime: " + $_.Name)
            Copy-Item $_.FullName $DistDir -Force
        }
    }
} else {
    Write-Warn ("CUDA bin not found at " + $CudaBin + " - GPU path will require system installed CUDA runtime.")
}

# ---- Final sanity check -----------------------------------------------------
$mustHave = @(
    (Join-Path $DistDir "mri_engine_v_1_1.dll"),
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
Get-ChildItem $DistDir | Select-Object Name, Length | Sort-Object Name | Format-Table -AutoSize

Write-Host ""
Write-Step ("Deploy complete. You can now run: " + (Join-Path $DistDir $ExeName))

