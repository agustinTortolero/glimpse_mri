#!/usr/bin/env bash
set -euo pipefail

# pack_jetson_bundle_auto_rpath.sh
# Same as pack_jetson_bundle.sh but automatically patches RUNPATH/RPATH when patchelf is installed.

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }
warn() { echo "[$(ts)][WRN] $*" >&2; }
die() { echo "[$(ts)][ERR] $*" >&2; exit 1; }

BUILD_TYPE="Release"
GUI_EXE=""
OUT_ROOT=""
KEEP_DIR=0

# Default runpath: bundle libs first, keep CUDA on common Jetson path.
RPATH_STR='$ORIGIN/../lib:/usr/local/cuda/lib64'

usage() {
  cat <<EOF
Usage: $0 [--release|--debug] [--exe /path/to/gui_exe] [--out /path/to/dist_root] [--keep-dir] [--rpath "STR"]

Options:
  --release          Bundle Release build (default)
  --debug            Bundle Debug build
  --exe PATH         Explicit GUI executable path (otherwise auto-detect inside build_gui_<TYPE>/)
  --out DIR          Output root directory (default: ./dist_jetson)
  --keep-dir         Do NOT delete the output directory if it already exists
  --rpath STR        Override patched RUNPATH/RPATH (default: $ORIGIN/../lib:/usr/local/cuda/lib64)

Notes:
  - If patchelf is NOT installed, we cannot patch RUNPATH; run.sh will still set LD_LIBRARY_PATH.
  - Recommended:
      sudo apt-get update && sudo apt-get install -y patchelf
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) BUILD_TYPE="Release"; shift ;;
    --debug)   BUILD_TYPE="Debug"; shift ;;
    --exe)     GUI_EXE="${2:-}"; shift 2 ;;
    --out)     OUT_ROOT="${2:-}"; shift 2 ;;
    --keep-dir) KEEP_DIR=1; shift ;;
    --rpath)   RPATH_STR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1 (use --help)" ;;
  esac
done

command -v tar >/dev/null 2>&1 || die "tar not found"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"
command -v file >/dev/null 2>&1 || die "file not found (sudo apt-get install -y file)"
command -v ldd >/dev/null 2>&1 || die "ldd not found"
command -v awk >/dev/null 2>&1 || die "awk not found"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/dist_jetson}"

GUI_BLD="${ROOT}/build_gui_${BUILD_TYPE}"
ENG_BLD="${ROOT}/build_engine_${BUILD_TYPE}"
DICOM_BLD="${ROOT}/build_dicom_${BUILD_TYPE}"

[[ -d "$GUI_BLD" ]] || die "Missing GUI build dir: $GUI_BLD (run ./build_jetson.shh first)"
[[ -d "$ENG_BLD" ]] || die "Missing engine build dir: $ENG_BLD"
[[ -d "$DICOM_BLD" ]] || die "Missing dicom build dir: $DICOM_BLD"

GIT_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "nosha")"
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo "unknown-jetson")"
ARCH="$(uname -m)"
STAMP="$(date +"%Y%m%d_%H%M%S")"

OUT_NAME="glimpse_mri_${ARCH}_${BUILD_TYPE}_${GIT_SHA}_${STAMP}"
OUT_DIR="${OUT_ROOT}/${OUT_NAME}"
BIN_DIR="${OUT_DIR}/bin"
LIB_DIR="${OUT_DIR}/lib"
DIAG_DIR="${OUT_DIR}/diag"

log "ROOT=$ROOT"
log "BUILD_TYPE=$BUILD_TYPE"
log "MODEL=$MODEL"
log "ARCH=$ARCH"
log "GIT_SHA=$GIT_SHA"
log "OUT_DIR=$OUT_DIR"
log "RPATH_STR=$RPATH_STR"

if [[ -d "$OUT_DIR" && "$KEEP_DIR" -eq 0 ]]; then
  log "Removing existing OUT_DIR: $OUT_DIR"
  rm -rf "$OUT_DIR"
fi

mkdir -p "$BIN_DIR" "$LIB_DIR" "$DIAG_DIR"

# Detect GUI executable
if [[ -n "$GUI_EXE" ]]; then
  [[ -f "$GUI_EXE" ]] || die "--exe not found: $GUI_EXE"
else
  log "Auto-detecting GUI executable under: $GUI_BLD"
  while IFS= read -r f; do
    info="$(file "$f" 2>/dev/null || true)"
    if [[ "$info" == *"ELF 64-bit"* && "$info" == *"aarch64"* && "$info" == *"executable"* ]]; then
      GUI_EXE="$f"
      break
    fi
  done < <(find "$GUI_BLD" -maxdepth 8 -type f ! -name "*.so*" | sort)
fi
[[ -n "$GUI_EXE" ]] || die "Could not auto-detect GUI executable. Re-run with: --exe /full/path/to/gui_binary"

log "GUI_EXE=$GUI_EXE"
cp -v "$GUI_EXE" "$BIN_DIR/"
GUI_BASENAME="$(basename "$GUI_EXE")"

# Copy shared libs from build trees (if any)
copy_libs_from_dir() {
  local src="$1"
  local label="$2"
  log "Copying ${label} .so* from: $src"
  find "$src" -maxdepth 10 \( -type f -o -type l \) -name "*.so*" -print0 2>/dev/null \
    | sort -z \
    | while IFS= read -r -d '' f; do
        cp -vL "$f" "$LIB_DIR/"
      done
}
copy_libs_from_dir "$ENG_BLD" "engine" || true
copy_libs_from_dir "$DICOM_BLD" "dicom" || true

# Copy repo-local deps discovered by ldd (this catches gui/release/libmri_engine.so.1)
log "Scanning ldd for repo-local dependencies (paths under: $ROOT)"
ldd_out="$(ldd "$BIN_DIR/$GUI_BASENAME" 2>/dev/null || true)"
echo "$ldd_out" | awk '
  $2 == "=>" && $3 ~ /^\// {print $3}
  $1 ~ /^\// {print $1}
' | sort -u | while IFS= read -r p; do
  [[ -z "$p" ]] && continue
  if [[ "$p" == "$ROOT/"* ]]; then
    log "  Bundling repo-local dep: $p"
    cp -vL "$p" "$LIB_DIR/"
  fi
done

# Patch RUNPATH/RPATH automatically (if patchelf exists)
if command -v patchelf >/dev/null 2>&1; then
  log "patchelf found -> patching GUI RUNPATH/RPATH to: $RPATH_STR"
  patchelf --set-rpath "$RPATH_STR" "$BIN_DIR/$GUI_BASENAME" || warn "patchelf failed for GUI"

  log "Setting RPATH on bundled libs to \$ORIGIN (best-effort)"
  for so in "$LIB_DIR"/*.so*; do
    [[ -f "$so" ]] || continue
    patchelf --set-rpath '$ORIGIN' "$so" || true
  done
else
  warn "patchelf NOT found -> cannot patch RUNPATH/RPATH. run.sh will rely on LD_LIBRARY_PATH."
fi

# Diagnostics
log "Writing dependency diagnostics (ldd) ..."
ldd "$BIN_DIR/$GUI_BASENAME" | tee "${DIAG_DIR}/ldd_${GUI_BASENAME}.txt" >/dev/null || true
(
  cd "$OUT_DIR"
  env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$LIB_DIR" ldd "$BIN_DIR/$GUI_BASENAME" \
    | tee "${DIAG_DIR}/ldd_${GUI_BASENAME}_with_bundle_lib.txt" >/dev/null || true
)
if command -v readelf >/dev/null 2>&1; then
  readelf -d "$BIN_DIR/$GUI_BASENAME" | egrep -i "rpath|runpath" \
    | tee "${DIAG_DIR}/readelf_rpath_runpath.txt" >/dev/null || true
fi

# run wrapper
cat > "${OUT_DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[\$(ts)][DBG] \$*"; }

HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
BIN="\${HERE}/bin"
LIB="\${HERE}/lib"

log "HERE=\$HERE"
log "BIN=\$BIN"
log "LIB=\$LIB"
log "ARCH=\$(uname -m)"

export LD_LIBRARY_PATH="\${LIB}:\${LD_LIBRARY_PATH:-}"
log "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH"

exec "\${BIN}/${GUI_BASENAME}" "\$@"
EOF
chmod +x "${OUT_DIR}/run.sh"

# include prerequisites.sh if present
if [[ -f "${ROOT}/prerequisites.sh" ]]; then
  cp -v "${ROOT}/prerequisites.sh" "${OUT_DIR}/install_deps.sh"
  chmod +x "${OUT_DIR}/install_deps.sh" || true
fi

# README
cat > "${OUT_DIR}/README_JETSON_BINARY.md" <<EOF
# GlimpseMRI Jetson Binary Bundle

**Build:** ${BUILD_TYPE}  
**Git:** ${GIT_SHA}  
**Arch:** ${ARCH}  

## Runpath patch
If \`patchelf\` was available on the build machine, the GUI binary was patched to:
\`\`\`
${RPATH_STR}
\`\`\`

## Run
\`\`\`bash
chmod +x ./run.sh
./run.sh
\`\`\`
EOF

# tarball + sha
TARBALL="${OUT_ROOT}/${OUT_NAME}.tar.gz"
log "Creating tarball: $TARBALL"
tar -C "$OUT_ROOT" -czf "$TARBALL" "$OUT_NAME"
( cd "$OUT_ROOT" && sha256sum "$(basename "$TARBALL")" | tee "$(basename "$TARBALL").sha256" )

log "DONE"
log "Bundle dir : $OUT_DIR"
log "Tarball    : $TARBALL"
