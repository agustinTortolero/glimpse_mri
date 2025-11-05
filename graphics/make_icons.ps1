# === make_icons.ps1 — PowerShell-only (no ImageMagick required) ==============
$ErrorActionPreference = "Stop"

# 0) Locate Inkscape CLI (adjust if needed)
$INK = @(
  (Get-Command inkscape.com -ErrorAction SilentlyContinue).Source
  (Get-Command inkscape     -ErrorAction SilentlyContinue).Source
  "$Env:ProgramFiles\Inkscape\bin\inkscape.com"
  "$Env:ProgramFiles\Inkscape\inkscape.exe"
  "$Env:ProgramFiles(x86)\Inkscape\bin\inkscape.com"
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $INK) {
  throw "Inkscape CLI not found. Try installing or use this path if you have it:
  C:\Program Files\Inkscape\bin\inkscape.com
  (winget install Inkscape.Inkscape)"
}
& $INK -V

# 1) Inputs / outputs
$src = Join-Path $PWD "mri-icon.svg"
if (-not (Test-Path $src)) { throw "Source SVG not found: $src" }

$out = Join-Path $PWD "build_icons"
$sizesMinimal = 16,24,32,48,64,128,256,512
$sizesExtras  = 20,40,72,96,192
$sizes = ($sizesMinimal + $sizesExtras) | Sort-Object -Unique

Write-Host "[DBG] Source SVG: $src"
Write-Host "[DBG] Output dir : $out"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# 2) Export PNGs (solid black background)
foreach ($sz in $sizes) {
  $png = Join-Path $out ("mri_{0}.png" -f $sz)
  Write-Host "[DBG] Exporting ${sz}x${sz} → $png"
  & $INK `
    $src `
    --export-filename="$png" `
    -w $sz -h $sz `
    --export-background="#000000" `
    --export-background-opacity 1
}

# 3) Build ICO (pure PowerShell, embeds PNGs)
function New-IcoFromPngs {
  param([string[]]$PngPaths, [string]$OutIco)

  Write-Host "[DBG] ICO: reading PNGs…"

  $items = @()
  foreach ($p in $PngPaths) {
    if (-not (Test-Path $p)) { Write-Warning "[MISS] $p not found (skipping)"; continue }
    $bytes = [IO.File]::ReadAllBytes($p)
    # PNG signature check
    $sig = ($bytes[0..7] -join ',')
    if ($sig -ne '137,80,78,71,13,10,26,10') { throw "Not a PNG: $p" }

    # IHDR chunk (width/height big-endian at offsets 16..23)
    $wArr = [byte[]]($bytes[16..19])
    $hArr = [byte[]]($bytes[20..23])
    [Array]::Reverse($wArr)
    [Array]::Reverse($hArr)
    $w = [BitConverter]::ToUInt32($wArr,0)
    $h = [BitConverter]::ToUInt32($hArr,0)

    $items += [PSCustomObject]@{ Path=$p; Bytes=$bytes; W=$w; H=$h; Size=$bytes.Length }
  }

  if ($items.Count -eq 0) { throw "No PNGs available to build ICO." }

  # Sort by width for a neat directory table
  $imgs = $items | Sort-Object -Property W

  $ms = New-Object IO.MemoryStream
  $bw = New-Object IO.BinaryWriter($ms)

  # ICONDIR
  $bw.Write([UInt16]0)                 # reserved
  $bw.Write([UInt16]1)                 # type: 1=icon
  $bw.Write([UInt16]$imgs.Count)       # count

  # Directory entries
  $offset = 6 + (16 * $imgs.Count)
  foreach ($img in $imgs) {
    $bw.Write([byte]([Math]::Min($img.W,255)))  # 0 means 256
    $bw.Write([byte]([Math]::Min($img.H,255)))
    $bw.Write([byte]0)        # colors
    $bw.Write([byte]0)        # reserved
    $bw.Write([UInt16]1)      # planes
    $bw.Write([UInt16]32)     # bitcount (assume RGBA)
    $bw.Write([UInt32]$img.Size)     # bytes in resource
    $bw.Write([UInt32]$offset)       # offset to PNG data
    $offset += $img.Size
  }

  # Image data (PNG blobs)
  foreach ($img in $imgs) { $bw.Write($img.Bytes) }

  $bw.Flush()
  [IO.File]::WriteAllBytes($OutIco, $ms.ToArray())
  $bw.Dispose(); $ms.Dispose()
}

$ico = Join-Path $out "mri.ico"
$icoSizes = 16,24,32,48,64,128,256
$icoPngs  = foreach ($s in $icoSizes) { Join-Path $out ("mri_{0}.png" -f $s) }
Write-Host "[DBG] Building ICO → $ico"
New-IcoFromPngs -PngPaths $icoPngs -OutIco $ico

# 4) Verify PNG sizes (no external tools)
function Get-PngDims([string]$Path){
  $b=[IO.File]::ReadAllBytes($Path)
  $wArr=[byte[]]($b[16..19]); [Array]::Reverse($wArr)
  $hArr=[byte[]]($b[20..23]); [Array]::Reverse($hArr)
  $w=[BitConverter]::ToUInt32($wArr,0)
  $h=[BitConverter]::ToUInt32($hArr,0)
  return "$([IO.Path]::GetFileName($Path)) -> ${w}x${h}px"
}

Write-Host "[DBG] Verifying PNG dimensions:"
Get-ChildItem (Join-Path $out "mri_*.png") |
  Sort-Object Name |
  ForEach-Object { Write-Host "       $(Get-PngDims $_.FullName)" }

Write-Host "[OK] Done. PNGs + ICO in: $out"
# ============================================================================
