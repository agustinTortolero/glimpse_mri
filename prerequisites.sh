#!/usr/bin/env bash
set -euo pipefail

# GlimpseMRI Jetson prerequisites installer
# - Detects missing packages and installs them
# - Tries to ensure CUDA toolchain is present (JetPack)
# - Installs Qt6 dev packages for GUI

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }
warn() { echo "[$(ts)][WRN] $*" >&2; }
die() { echo "[$(ts)][ERR] $*" >&2; exit 1; }

if [[ "${EUID}" -eq 0 ]]; then
  warn "You are running as root. It's fine, but usually run as your user with sudo available."
fi

log "PWD=$(pwd)"
log "ARCH=$(uname -m)"
log "KERNEL=$(uname -r)"
log "OS:"
(lsb_release -a || true)

if [[ -f /etc/nv_tegra_release ]]; then
  log "Jetson detected: /etc/nv_tegra_release exists"
  log "nv_tegra_release:"
  cat /etc/nv_tegra_release || true
else
  warn "This doesn't look like a Jetson (no /etc/nv_tegra_release). Script will still try Ubuntu deps."
fi

if ! command -v sudo >/dev/null 2>&1; then
  die "sudo is not available. Install sudo or run with root privileges."
fi

# --- helpers ---
is_pkg_installed() {
  local pkg="$1"
  dpkg -s "$pkg" >/dev/null 2>&1
}

ensure_apt_pkg() {
  local pkg="$1"
  if is_pkg_installed "$pkg"; then
    log "APT OK: ${pkg}"
  else
    log "APT MISSING: ${pkg} -> installing..."
    sudo apt-get install -y "$pkg"
  fi
}

ensure_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    log "CMD OK: ${cmd} -> $(command -v "$cmd")"
  else
    warn "CMD missing: ${cmd}"
  fi
}

log "Updating apt indexes..."
sudo apt-get update -y

# Core build tooling
BASE_PKGS=(
  build-essential
  git
  cmake
  ninja-build
  pkg-config
  python3
  python3-pip
)

# Common native libs (safe / frequently needed)
NATIVE_PKGS=(
  libssl-dev
  zlib1g-dev
  libzstd-dev
  liblz4-dev
  libpng-dev
  libjpeg-dev
  libtiff-dev
  libopenblas-dev
  libeigen3-dev
)

# Qt6 GUI deps (adjust if your GUI uses extra modules)
QT6_PKGS=(
  qt6-base-dev
  qt6-base-dev-tools
  qt6-tools-dev
  qt6-tools-dev-tools
  qt6-multimedia-dev
  qt6-multimedia-dev-tools
  qt6-svg-dev
  libgl1-mesa-dev
  libxkbcommon-dev
)

# DICOM / HDF5 / FFTW are common in MRI pipelines (keep even if some modules don’t use them today)
MRI_PKGS=(
  libdcmtk-dev
  libhdf5-dev
  libfftw3-dev
)

log "Installing base build packages..."
for p in "${BASE_PKGS[@]}"; do ensure_apt_pkg "$p"; done

log "Installing native libraries..."
for p in "${NATIVE_PKGS[@]}"; do ensure_apt_pkg "$p"; done

log "Installing Qt6 GUI packages..."
for p in "${QT6_PKGS[@]}"; do ensure_apt_pkg "$p"; done

log "Installing MRI-related packages..."
for p in "${MRI_PKGS[@]}"; do ensure_apt_pkg "$p"; done

# --- CUDA / JetPack check ---
CUDA_DIR="/usr/local/cuda"
if [[ -x "${CUDA_DIR}/bin/nvcc" ]]; then
  log "CUDA OK: ${CUDA_DIR}/bin/nvcc exists"
  "${CUDA_DIR}/bin/nvcc" --version || true
else
  warn "CUDA nvcc not found at ${CUDA_DIR}/bin/nvcc"
  warn "Attempting to install JetPack meta package (this can be large)..."

  # On Jetson, this is usually the correct way to pull CUDA, cuDNN, TensorRT, etc.
  # If your system is already JetPack-based but nvcc missing, this may fix it.
  sudo apt-get install -y nvidia-jetpack || warn "Failed to install nvidia-jetpack (maybe repo not configured)."

  if [[ -x "${CUDA_DIR}/bin/nvcc" ]]; then
    log "CUDA OK after install: nvcc found"
    "${CUDA_DIR}/bin/nvcc" --version || true
  else
    warn "CUDA still not found. If you are on Jetson, verify JetPack is installed and CUDA path is correct."
    warn "You can check: dpkg -l | grep -i jetpack"
  fi
fi

# Helpful sanity checks
log "Sanity checks:"
ensure_cmd cmake
ensure_cmd ninja
ensure_cmd make
ensure_cmd qmake6
ensure_cmd qt6-qmake
ensure_cmd pkg-config

log "DONE. If anything still fails during build, paste the last ~60 lines of the build log."

