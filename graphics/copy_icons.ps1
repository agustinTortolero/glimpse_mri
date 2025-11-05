# === copy_icons.ps1 — copy generated PNGs into Qt project's icons folder =====
$ErrorActionPreference = "Stop"

# Paths
$srcDir = Join-Path $PWD 'build_icons'
$dstDir = 'C:\AgustinTortolero_repos\portafolio\GlimpseMRI\gui\assets\images\icons'

# Sizes used in assets.qrc (/icons prefix)
$sizes = 16,20,24,32,40,48,64,72,96,128,192,256,512

Write-Host "[DBG] Source dir :" $srcDir
Write-Host "[DBG] Target dir :" $dstDir
if (-not (Test-Path $srcDir)) { throw "Source dir not found: $srcDir. Run make_icons.ps1 first." }
New-Item -ItemType Directory -Force -Path $dstDir | Out-Null

$copied = 0
foreach ($s in $sizes) {
  $src = Join-Path $srcDir ("mri_{0}.png" -f $s)
  if (Test-Path $src) {
    $dst = Join-Path $dstDir (Split-Path $src -Leaf)
    Write-Host "[DBG] Copy $src -> $dst"
    Copy-Item -Force $src $dst
    $copied++
  } else {
    Write-Warning "[MISS] $src not found (skipping)"
  }
}

# (Optional) copy ICO for Windows EXE if you ever enable RC_ICONS in .pro
$icoSrc = Join-Path $srcDir 'mri.ico'
if (Test-Path $icoSrc) {
  $icoDst = Join-Path $dstDir 'mri.ico'
  Write-Host "[DBG] Copy (optional) $icoSrc -> $icoDst"
  Copy-Item -Force $icoSrc $icoDst
}

Write-Host "[OK] Copied $copied PNG(s) to $dstDir"
Write-Host "[HINT] Re-run qmake/rebuild so the QRC picks up any changes if paths changed."
# ============================================================================
