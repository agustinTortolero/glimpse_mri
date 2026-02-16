#!/usr/bin/env bash
set -euo pipefail

# pack_jetson_bundle.sh
# Create a GitHub-release friendly tar.gz bundle for Jetson Orin Nano (aarch64).
#
# What it bundles:
#   - GUI executable (auto-detected from build_gui_<TYPE>/)
#   - Your project shared libs from:
#       build_engine_<TYPE>/*.so*
#       build_dicom_<TYPE>/*.so*
#   - run.sh wrapper that sets LD_LIBRARY_PATH so the app can find bundled libs
#   - README_JETSON_BINARY.md + SHA256
#
# What it does NOT bundle:
#   - JetPack / CUDA / NVIDIA driver stack (must already be installed on target Jetson)
#   - System libs like Qt6, HDF5, FFTW, DCMTK (installed via prerequisites.sh)

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }
warn() { echo "[$(ts)][WRN] $*" >&2; }
die() { echo "[$(ts)][ERR] $*" >&2; exit 1; }

BUILD_TYPE="Release"
GUI_EXE=""
OUT_ROOT=""
KEEP_DIR=0

usage() {
  cat <<EOF
Usage: $0 [--release|--debug] [--exe /path/to/gui_exe] [--out /path/to/dist_root] [--keep-dir]

Options:
  --release          Bundle Release build (default)
  --debug            Bundle Debug build
  --exe PATH         Explicit GUI executable path (otherwise auto-detect inside build_gui_<TYPE>/)
  --out DIR          Output root directory (default: ./dist_jetson)
  --keep-dir         Do NOT delete the output directory if it already exists (append timestamped name anyway)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) BUILD_TYPE="Release"; shift ;;
    --debug)   BUILD_TYPE="Debug"; shift ;;
    --exe)     GUI_EXE="${2:-}"; shift 2 ;;
    --out)     OUT_ROOT="${2:-}"; shift 2 ;;
    --keep-dir) KEEP_DIR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1 (use --help)" ;;
  esac
done

command -v tar >/dev/null 2>&1 || die "tar not found"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"
command -v file >/dev/null 2>&1 || die "file not found (sudo apt-get install -y file)"
command -v ldd >/dev/null 2>&1 || die "ldd not found"

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

if [[ -d "$OUT_DIR" && "$KEEP_DIR" -eq 0 ]]; then
  log "Removing existing OUT_DIR: $OUT_DIR"
  rm -rf "$OUT_DIR"
fi

mkdir -p "$BIN_DIR" "$LIB_DIR" "$DIAG_DIR"

# --------------------------
# Detect GUI executable
# --------------------------
if [[ -n "$GUI_EXE" ]]; then
  [[ -f "$GUI_EXE" ]] || die "--exe not found: $GUI_EXE"
else
  log "Auto-detecting GUI executable under: $GUI_BLD"
  # NOTE: On Jetson, `file` often reports "pie executable, ARM aarch64, ..."
  # so we must not assume ordering of "aarch64" vs "executable".
  CANDIDATES=()
  while IFS= read -r f; do
    CANDIDATES+=("$f")
    info="$(file "$f" 2>/dev/null || true)"
    if [[ "$info" == *"ELF 64-bit"* && "$info" == *"aarch64"* && "$info" == *"executable"* ]]; then
      GUI_EXE="$f"
      break
    fi
  done < <(find "$GUI_BLD" -maxdepth 6 -type f ! -name "*.so*" | sort)

  if [[ -z "$GUI_EXE" ]]; then
    warn "Could not auto-detect GUI executable."
    warn "Hint: this is often because `file` output ordering differs (e.g., 'pie executable, ARM aarch64')."
    warn "You can always re-run with: --exe /full/path/to/gui_binary"
    warn "First 10 candidates + 'file' output for debugging:"
    n=0
    for f in "${CANDIDATES[@]}"; do
      [[ -f "$f" ]] || continue
      info="$(file "$f" 2>/dev/null || true)"
      warn "  - $f :: $info"
      n=$((n+1))
      [[ $n -ge 10 ]] && break
    done
  fi
fi

[[ -n "$GUI_EXE" ]] || die "Could not auto-detect GUI executable. Re-run with: --exe /full/path/to/gui_binary"

log "GUI_EXE=$GUI_EXE"
cp -v "$GUI_EXE" "$BIN_DIR/"

GUI_BASENAME="$(basename "$GUI_EXE")"

# -------------------------
# Bundle icons/assets (optional but recommended for desktop launcher)
# -------------------------
ICON_SRC_DIR="$ROOT/gui/assets/images/icons"
ASSETS_DIR="$OUT_DIR/assets"
ICONS_DIR="$ASSETS_DIR/icons"
if [[ -d "$ICON_SRC_DIR" ]]; then
  mkdir -p "$ICONS_DIR"
  log "Copying app icons from: $ICON_SRC_DIR"
  # copy all PNG sizes (small footprint, handy for launchers)
  cp -v "$ICON_SRC_DIR"/mri_*.png "$ICONS_DIR/" 2>/dev/null || true
  # also copy .ico (harmless on Linux, useful if you later repackage)
  cp -v "$ICON_SRC_DIR"/mri.ico "$ICONS_DIR/" 2>/dev/null || true

  # provide a stable "icon.png" path for launchers
  if [[ -f "$ICONS_DIR/mri_256.png" ]]; then
    cp -v "$ICONS_DIR/mri_256.png" "$ASSETS_DIR/icon.png" || true
  elif [[ -f "$ICONS_DIR/mri_128.png" ]]; then
    cp -v "$ICONS_DIR/mri_128.png" "$ASSETS_DIR/icon.png" || true
  elif [[ -f "$ICONS_DIR/mri_96.png" ]]; then
    cp -v "$ICONS_DIR/mri_96.png" "$ASSETS_DIR/icon.png" || true
  fi
else
  warn "Icon source dir not found (skipping icons): $ICON_SRC_DIR"
fi


# --------------------------
# Copy your project shared libs (engine + dicom)
# --------------------------
log "Copying engine .so from: $ENG_BLD"
find "$ENG_BLD" -maxdepth 5 -type f -name "*.so*" -print -exec cp -v {} "$LIB_DIR/" \; || true

log "Copying dicom .so from: $DICOM_BLD"
find "$DICOM_BLD" -maxdepth 5 -type f -name "*.so*" -print -exec cp -v {} "$LIB_DIR/" \; || true

# Safety check: lib dir not empty
if ! compgen -G "${LIB_DIR}/*.so*" > /dev/null; then
  warn "No .so files were copied into ${LIB_DIR}. If your project produces only static libs, this can be OK."
  warn "But if the GUI expects your engine/dicom .so at runtime, you probably want them bundled."
fi

# --------------------------
# Optional: set RPATH so bin finds ../lib without env vars
# --------------------------
if command -v patchelf >/dev/null 2>&1; then
  log "patchelf found -> setting RPATH on GUI to \$ORIGIN/../lib"
  patchelf --set-rpath '$ORIGIN/../lib' "$BIN_DIR/$GUI_BASENAME" || warn "patchelf failed for GUI"

  log "Setting RPATH on bundled libs to \$ORIGIN (best-effort)"
  for so in "$LIB_DIR"/*.so*; do
    [[ -f "$so" ]] || continue
    patchelf --set-rpath '$ORIGIN' "$so" || true
  done
else
  warn "patchelf NOT found -> run.sh will rely on LD_LIBRARY_PATH (recommended: sudo apt-get install -y patchelf)"
fi

# --------------------------
# Diagnostics: ldd outputs (very helpful for testing)
# --------------------------
log "Writing dependency diagnostics (ldd) ..."
ldd "$BIN_DIR/$GUI_BASENAME" | tee "${DIAG_DIR}/ldd_${GUI_BASENAME}.txt" >/dev/null || true

for so in "$LIB_DIR"/*.so*; do
  [[ -f "$so" ]] || continue
  base="$(basename "$so")"
  ldd "$so" | tee "${DIAG_DIR}/ldd_${base}.txt" >/dev/null || true
done

# --------------------------
# Create run wrapper
# --------------------------
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
log "LD_LIBRARY_PATH(before)=\${LD_LIBRARY_PATH:-<empty>}"

export LD_LIBRARY_PATH="\${LIB}:\${LD_LIBRARY_PATH:-}"
log "LD_LIBRARY_PATH(after)=\$LD_LIBRARY_PATH"

# If you ever decide to bundle Qt plugins, you can add:
# export QT_PLUGIN_PATH="\${HERE}/qtplugins"
# export QT_QPA_PLATFORM_PLUGIN_PATH="\${HERE}/qtplugins/platforms"
# export QT_DEBUG_PLUGINS=1

EXE="${GUI_BASENAME}"
log "Launching: \${BIN}/\${EXE}"
exec "\${BIN}/\${EXE}" "\$@"
EOF
chmod +x "${OUT_DIR}/run.sh"

# desktop launcher helper (creates a clickable Desktop icon)
cat > "${OUT_DIR}/create_desktop_shortcut.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }

APPNAME="GlimpseMRI"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# try to discover the actual Desktop folder (works on Ubuntu)
DESKTOP_DIR="${HOME}/Desktop"
if command -v xdg-user-dir >/dev/null 2>&1; then
  DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "${DESKTOP_DIR}")"
fi
mkdir -p "${DESKTOP_DIR}"

ICON_PATH="${BUNDLE_DIR}/assets/icon.png"
if [[ ! -f "${ICON_PATH}" ]]; then
  # fallback to a bundled icon size if icon.png wasn't created
  if [[ -f "${BUNDLE_DIR}/assets/icons/mri_256.png" ]]; then ICON_PATH="${BUNDLE_DIR}/assets/icons/mri_256.png"; fi
  if [[ -f "${BUNDLE_DIR}/assets/icons/mri_128.png" ]]; then ICON_PATH="${BUNDLE_DIR}/assets/icons/mri_128.png"; fi
fi
if [[ ! -f "${ICON_PATH}" ]]; then
  ICON_PATH="applications-science"
fi

DESK_FILE="${DESKTOP_DIR}/${APPNAME}.desktop"
log "Creating launcher: ${DESK_FILE}"
cat > "${DESK_FILE}" <<EOL
[Desktop Entry]
Type=Application
Version=1.0
Name=${APPNAME}
Comment=High-performance MRI reconstruction viewer
Path=${BUNDLE_DIR}
Exec=${BUNDLE_DIR}/run.sh
Icon=${ICON_PATH}
Terminal=false
Categories=Science;MedicalSoftware;
EOL

chmod +x "${DESK_FILE}"

# Mark as trusted on GNOME (prevents the "untrusted" warning on some Ubuntu setups)
if command -v gio >/dev/null 2>&1; then
  gio set "${DESK_FILE}" metadata::trusted true 2>/dev/null || true
fi

log "Done. If you don't see the icon immediately, log out/in or run: nautilus -q"
EOF
chmod +x "${OUT_DIR}/create_desktop_shortcut.sh"



# --------------------------
# Bundle dependency installer (so the tarball can be used on a fresh Jetson)
# --------------------------
PREREQ_SRC="${ROOT}/prerequisites.sh"
if [[ -f "${PREREQ_SRC}" ]]; then
  log "Including prerequisites.sh as install_deps.sh"
  cp -v "${PREREQ_SRC}" "${OUT_DIR}/install_deps.sh"
  chmod +x "${OUT_DIR}/install_deps.sh"
else
  warn "prerequisites.sh not found at ${PREREQ_SRC} -> generating minimal install_deps.sh"
  cat > "${OUT_DIR}/install_deps.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
TS() { date +"%Y-%m-%d %H:%M:%S"; }
DBG(){ echo "[$(TS)][DBG] $*"; }
WRN(){ echo "[$(TS)][WRN] $*" >&2; }
ERR(){ echo "[$(TS)][ERR] $*" >&2; }

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  ERR "Run with sudo: sudo $0"
  exit 1
fi

try_install() {
  local pkgs=("$@")
  [[ ${#pkgs[@]} -gt 0 ]] || return 0
  DBG "APT install: ${pkgs[*]}"
  if ! apt-get install -y "${pkgs[@]}"; then
    WRN "APT failed for: ${pkgs[*]} (continuing)"
  fi
}

DBG "Updating apt indexes..."
apt-get update

# Minimal runtime deps for GlimpseMRI Jetson bundle (Ubuntu 22.04 / JetPack 6.x)
try_install \
  libhdf5-dev \
  libdcmtk-dev \
  libpugixml1v5 \
  libfftw3-dev

# Qt6 GUI runtime/dev (Qt 6.2 on Ubuntu 22.04)
try_install \
  qt6-base-dev \
  qt6-multimedia-dev \
  libqt6svg6-dev

# OpenCV runtime/dev (binary links against libopencv_* libs)
try_install libopencv-dev

# Convenience: lets us patch RUNPATH in future or debug
try_install patchelf

DBG "DONE"
EOF
  chmod +x "${OUT_DIR}/install_deps.sh"
fi


# --------------------------
# Bundle README
# --------------------------
cat > "${OUT_DIR}/README_JETSON_BINARY.md" <<EOF
# GlimpseMRI Jetson Binary Bundle

**Build:** ${BUILD_TYPE}  
**Git:** ${GIT_SHA}  
**Arch:** ${ARCH}  
**Device (build machine):** ${MODEL}

## What this bundle contains
- \`bin/${GUI_BASENAME}\` (GUI executable)
- \`lib/\` (project .so libraries from engine/dicom builds)
- \`run.sh\` launcher
- \`diag/\` dependency dumps (\`ldd\` output)

## Assumptions / Requirements (target Jetson)
1. **JetPack is already installed** (CUDA + NVIDIA drivers + system integration).
2. System runtime deps are installed (Qt6, HDF5, FFTW, DCMTK, etc.).
   - Use the bundled installer script:
     \`\`\`bash
     chmod +x ./install_deps.sh
sudo ./install_deps.sh
     \`\`\`

> Compatibility note: dynamic linking means you should target the **same JetPack major** (and ideally the same Ubuntu release)
> as the machine that produced this bundle.

## Run
\`\`\`bash
chmod +x ./run.sh
./run.sh
\`\`\`

## Troubleshooting
- Missing libraries:
  - Check \`diag/ldd_*.txt\` for "not found"
  - Install missing packages (usually via \`./prerequisites.sh\`)
- Qt plugin issues:
  - Try:
    \`\`\`bash
    QT_DEBUG_PLUGINS=1 ./run.sh
    \`\`\`
EOF

# --------------------------
# Create tarball + checksum
# --------------------------
TARBALL="${OUT_ROOT}/${OUT_NAME}.tar.gz"
log "Creating tarball: $TARBALL"
mkdir -p "$OUT_ROOT"
tar -C "$OUT_ROOT" -czf "$TARBALL" "$OUT_NAME"

log "Writing SHA256"
( cd "$OUT_ROOT" && sha256sum "$(basename "$TARBALL")" | tee "$(basename "$TARBALL").sha256" )

log "DONE"
log "Bundle dir : $OUT_DIR"
log "Tarball    : $TARBALL"
