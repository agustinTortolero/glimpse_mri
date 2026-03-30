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
    WRN "APT failed for: ${pkgs[*]}"
    return 1
  fi
}

try_install_any() {
  local label="$1"
  shift
  local found=0

  for pkg in "$@"; do
    if try_install "$pkg"; then
      found=1
      DBG "Installed ${label} package: ${pkg}"
      break
    fi
  done

  if [[ "$found" -ne 1 ]]; then
    WRN "Could not install any package candidate for ${label}: $*"
  fi
}

DBG "Raspberry Pi 5 ARM64 build prerequisites"
DBG "ARCH=$(uname -m)"
DBG "OS:"; (lsb_release -a 2>/dev/null || cat /etc/os-release || true)

DBG "Updating apt indexes..."
apt-get update

DBG "Installing core build tools..."
try_install \
  build-essential \
  git \
  cmake \
  ninja-build \
  pkg-config \
  make \
  file \
  patchelf \
  dos2unix \
  curl

DBG "Installing core dependencies (CPU-only; no CUDA)..."
try_install \
  libhdf5-dev \
  libdcmtk-dev \
  libfftw3-dev \
  libopencv-dev \
  libpugixml-dev

# ISMRMRD package names may vary by distro/repo.
try_install_any "ismrmrd-dev" libismrmrd-dev

DBG "Installing Qt6 development packages for CMake..."
try_install_any "qt6-base-dev" qt6-base-dev
try_install_any "qt6-base-dev-tools" qt6-base-dev-tools
try_install_any "qt6-tools-dev" qt6-tools-dev
try_install_any "qt6-tools-dev-tools" qt6-tools-dev-tools
try_install_any "qt6-multimedia-dev" qt6-multimedia-dev
try_install_any "qt6-svg-dev" libqt6svg6-dev qt6-svg-dev

DBG "DONE"
DBG "Sanity checks:"
DBG "  cmake : $(command -v cmake || echo missing)"
DBG "  ninja : $(command -v ninja || echo missing)"
DBG "  gcc   : $(command -v gcc   || echo missing)"
DBG "  g++   : $(command -v g++   || echo missing)"
