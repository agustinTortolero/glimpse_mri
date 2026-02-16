#!/usr/bin/env bash
set -euo pipefail

# GlimpseMRI Jetson (aarch64) bundle packer:
# - Copies GUI binary into bundle/bin
# - Bundles project .so (engine, dicom) if present
# - Bundles repo-local deps discovered by ldd (e.g. gui/release/libmri_engine.so.1)
# - Bundles pinned ISMRMRD SONAME (libismrmrd.so.1.4*) if required by engine
# - Patches RUNPATH to prefer bundled libs ($ORIGIN/../lib)
# - Writes diagnostics + creates tarball + sha256

# -------------------------
# Logging (to stderr so command-substitutions stay clean)
# -------------------------
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log()  { echo "[$(ts)][DBG] $*" >&2; }
warn() { echo "[$(ts)][WRN] $*" >&2; }
die()  { echo "[$(ts)][ERR] $*" >&2; exit 1; }

# -------------------------
# Args
# -------------------------
BUILD_TYPE="Release"
GUI_EXE=""
KEEP_DIR=0

usage() {
  cat >&2 <<'EOF'
Usage:
  ./pack_jetson_bundle.sh [--release|--debug] [--exe /full/path/to/glimpseMRI] [--keep]

Notes:
  - If --exe is not provided, the script will try to auto-detect an aarch64 ELF executable in build_gui_<Type>.
  - Output: dist_jetson/glimpse_mri_aarch64_<Type>_<gitsha>_<timestamp>/{bin,lib,diag,run.sh,install_deps.sh}
  - Also creates: dist_jetson/<bundle_name>.tar.gz and .tar.gz.sha256
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) BUILD_TYPE="Release"; shift;;
    --debug)   BUILD_TYPE="Debug"; shift;;
    --exe)     GUI_EXE="${2:-}"; shift 2;;
    --keep)    KEEP_DIR=1; shift;;
    -h|--help) usage; exit 0;;
    *) die "Unknown arg: $1 (use --help)";;
  esac
done

# -------------------------
# Paths + metadata
# -------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$ROOT/dist_jetson"

GUI_BLD="$ROOT/build_gui_${BUILD_TYPE}"
ENG_BLD="$ROOT/build_engine_${BUILD_TYPE}"
DICOM_BLD="$ROOT/build_dicom_${BUILD_TYPE}"

ARCH="$(uname -m)"
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo "unknown")"
GIT_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "nogit")"
STAMP="$(date +%Y%m%d_%H%M%S)"

[[ "$ARCH" == "aarch64" ]] || warn "ARCH=$ARCH (expected aarch64 on Jetson)"

OUT_DIR="$DIST_DIR/glimpse_mri_${ARCH}_${BUILD_TYPE}_${GIT_SHA}_${STAMP}"
BIN_DIR="$OUT_DIR/bin"
LIB_DIR="$OUT_DIR/lib"
DIAG_DIR="$OUT_DIR/diag"

log "ROOT=$ROOT"
log "BUILD_TYPE=$BUILD_TYPE"
log "MODEL=$MODEL"
log "ARCH=$ARCH"
log "GIT_SHA=$GIT_SHA"
log "OUT_DIR=$OUT_DIR"

mkdir -p "$BIN_DIR" "$LIB_DIR" "$DIAG_DIR"

if [[ -d "$OUT_DIR" && "$KEEP_DIR" -eq 0 ]]; then
  rm -rf "$OUT_DIR"
  mkdir -p "$BIN_DIR" "$LIB_DIR" "$DIAG_DIR"
fi

# -------------------------
# GUI exe: explicit or auto-detect
# -------------------------
if [[ -n "$GUI_EXE" ]]; then
  [[ -f "$GUI_EXE" ]] || die "--exe not found: $GUI_EXE"
else
  [[ -d "$GUI_BLD" ]] || die "GUI build dir missing: $GUI_BLD (build first or pass --exe)"
  log "Auto-detecting GUI executable under: $GUI_BLD"
  while IFS= read -r f; do
    # skip shared libs
    [[ "$f" == *.so* ]] && continue
    info="$(file "$f" 2>/dev/null || true)"
    # robust: order-independent checks
    echo "$info" | grep -qi "ELF 64-bit" || continue
    echo "$info" | grep -qi "executable" || continue
    echo "$info" | grep -qiE "aarch64|ARM aarch64" || continue
    GUI_EXE="$f"
    break
  done < <(find "$GUI_BLD" -maxdepth 8 -type f | sort)
fi

[[ -n "$GUI_EXE" ]] || die "Could not auto-detect GUI executable. Re-run with: --exe /full/path/to/gui_binary"

log "GUI_EXE=$GUI_EXE"
cp -v "$GUI_EXE" "$BIN_DIR/"
GUI_BASENAME="$(basename "$GUI_EXE")"

# -------------------------
# Copy .so libs from build trees (if any)
# -------------------------
copy_libs_from_dir() {
  local src="$1"
  local label="$2"
  [[ -d "$src" ]] || { warn "Missing $label dir: $src"; return 0; }
  log "Copying ${label} *.so* from: $src"
  find "$src" -maxdepth 10 \( -type f -o -type l \) -name "*.so*" -print0 2>/dev/null \
    | sort -z \
    | while IFS= read -r -d '' f; do
        cp -vL "$f" "$LIB_DIR/" || true
      done
}

copy_libs_from_dir "$ENG_BLD" "engine" || true
copy_libs_from_dir "$DICOM_BLD" "dicom" || true

# -------------------------
# Copy repo-local deps discovered by ldd (captures gui/release/libmri_engine.so.1)
# -------------------------
log "Scanning ldd for repo-local dependencies (paths under: $ROOT)"
ldd "$BIN_DIR/$GUI_BASENAME" > "$DIAG_DIR/ldd_${GUI_BASENAME}.txt" || true

REPO_LOCAL_COUNT=0
while read -r line; do
  # typical ldd line:
  #   libfoo.so.1 => /path/to/libfoo.so.1 (0x...)
  p="$(echo "$line" | awk '{print $3}' | tr -d '()' || true)"
  [[ -z "$p" ]] && continue
  if [[ "$p" == "$ROOT/"* ]]; then
    log "  Bundling repo-local dep: $p"
    cp -vL "$p" "$LIB_DIR/" || true
    REPO_LOCAL_COUNT=$((REPO_LOCAL_COUNT+1))
  fi
done < <(cat "$DIAG_DIR/ldd_${GUI_BASENAME}.txt" | grep "=> " || true)

log "Repo-local deps copied from ldd: $REPO_LOCAL_COUNT"

# -------------------------
# Bundle pinned ISMRMRD SONAME if engine needs it (libismrmrd.so.1.4*)
# -------------------------
ENGINE_IN_BUNDLE="$LIB_DIR/libmri_engine.so.1"
if [[ -f "$ENGINE_IN_BUNDLE" ]]; then
  if readelf -d "$ENGINE_IN_BUNDLE" 2>/dev/null | grep -q "libismrmrd.so.1.4"; then
    log "Engine depends on libismrmrd.so.1.4 -> bundling ISMRMRD 1.4 into tarball"
    ISM_PATH="$(ldd "$ENGINE_IN_BUNDLE" | awk '/libismrmrd\.so\.1\.4/ {print $3; exit}' || true)"
    if [[ -n "${ISM_PATH:-}" && -e "$ISM_PATH" ]]; then
      log "Resolved from ldd: libismrmrd.so.1.4 -> $ISM_PATH"
      ISM_DIR="$(dirname "$ISM_PATH")"
      log "Bundling system dep family: $ISM_DIR/libismrmrd.so.1.4* -> $LIB_DIR"
      cp -av "$ISM_DIR"/libismrmrd.so.1.4* "$LIB_DIR/" || true
    else
      warn "ldd could not resolve libismrmrd.so.1.4. Trying ldconfig..."
      ISM_PATH2="$(ldconfig -p 2>/dev/null | awk '/libismrmrd\.so\.1\.4/ {print $NF; exit}' || true)"
      if [[ -n "${ISM_PATH2:-}" && -e "$ISM_PATH2" ]]; then
        log "Resolved from ldconfig: libismrmrd.so.1.4 -> $ISM_PATH2"
        ISM_DIR2="$(dirname "$ISM_PATH2")"
        log "Bundling system dep family: $ISM_DIR2/libismrmrd.so.1.4* -> $LIB_DIR"
        cp -av "$ISM_DIR2"/libismrmrd.so.1.4* "$LIB_DIR/" || true
      else
        warn "Could not resolve libismrmrd.so.1.4 on this machine. Fresh-Jetson run may fail."
      fi
    fi

    if ls -lah "$LIB_DIR"/libismrmrd.so.1.4* >/dev/null 2>&1; then
      log "ISMRMRD bundled OK:"
      ls -lah "$LIB_DIR"/libismrmrd.so.1.4*
    else
      warn "ISMRMRD still not present in bundle/lib after attempted copy."
    fi
  else
    log "Engine does not list libismrmrd.so.1.4 in NEEDED (skipping ISMRMRD bundling)"
  fi
else
  warn "Engine not found in bundle at: $ENGINE_IN_BUNDLE (cannot bundle ISMRMRD)"
fi

# -------------------------
# RUNPATH patching (prefer bundle libs)
# -------------------------
PATCH_RUNPATH='$ORIGIN/../lib:/usr/local/cuda/lib64'

if command -v patchelf >/dev/null 2>&1; then
  log "patchelf found -> patching RUNPATH to use bundled libs first"
  patchelf --set-rpath "$PATCH_RUNPATH" "$BIN_DIR/$GUI_BASENAME" || true
  # also patch all .so in bundle/lib
  find "$LIB_DIR" -maxdepth 1 \( -type f -o -type l \) -name "*.so*" -print0 2>/dev/null \
    | while IFS= read -r -d '' so; do
        patchelf --set-rpath "$PATCH_RUNPATH" "$so" || true
      done
else
  warn "patchelf NOT found -> run.sh will rely on LD_LIBRARY_PATH (recommended: sudo apt-get install -y patchelf)"
fi

# -------------------------
# Diagnostics
# -------------------------
log "Writing dependency diagnostics (ldd) ..."
ldd "$BIN_DIR/$GUI_BASENAME" > "$DIAG_DIR/ldd_${GUI_BASENAME}.txt" || true
LD_LIBRARY_PATH="$LIB_DIR" ldd "$BIN_DIR/$GUI_BASENAME" > "$DIAG_DIR/ldd_${GUI_BASENAME}_with_bundle_lib.txt" || true
if [[ -f "$ENGINE_IN_BUNDLE" ]]; then
  ldd "$ENGINE_IN_BUNDLE" > "$DIAG_DIR/ldd_libmri_engine.txt" || true
  LD_LIBRARY_PATH="$LIB_DIR" ldd "$ENGINE_IN_BUNDLE" > "$DIAG_DIR/ldd_libmri_engine_with_bundle_lib.txt" || true
fi

# -------------------------
# Include prerequisites.sh as install_deps.sh (if present)
# -------------------------
if [[ -f "$ROOT/prerequisites.sh" ]]; then
  log "Including prerequisites.sh as install_deps.sh"
  cp -v "$ROOT/prerequisites.sh" "$OUT_DIR/install_deps.sh"
else
  warn "No prerequisites.sh found at repo root; install_deps.sh will not be included."
fi

# -------------------------
# run.sh
# -------------------------
cat > "$OUT_DIR/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
BIN="\$HERE/bin"
LIB="\$HERE/lib"
export LD_LIBRARY_PATH="\$LIB:\${LD_LIBRARY_PATH:-}"
echo "[DBG] HERE=\$HERE"
echo "[DBG] BIN=\$BIN"
echo "[DBG] LIB=\$LIB"
echo "[DBG] ARCH=\$(uname -m)"
echo "[DBG] LD_LIBRARY_PATH=\$LD_LIBRARY_PATH"
exec "\$BIN/$GUI_BASENAME" "\$@"
EOF
chmod +x "$OUT_DIR/run.sh"

# -------------------------
# Tarball + sha256
# -------------------------
TARBALL="${OUT_DIR}.tar.gz"
log "Creating tarball: $TARBALL"
tar -C "$DIST_DIR" -czf "$TARBALL" "$(basename "$OUT_DIR")"

log "Writing SHA256"
( cd "$DIST_DIR" && sha256sum "$(basename "$TARBALL")" | tee "$(basename "$TARBALL").sha256" )

log "DONE"
log "Bundle dir : $OUT_DIR"
log "Tarball    : $TARBALL"
log "SHA256     : ${TARBALL}.sha256"
