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
  for pkg in "$@"; do
    if try_install "$pkg"; then
      DBG "Installed ${label} package: ${pkg}"
      return 0
    fi
  done
  WRN "Could not install any package candidate for ${label}: $*"
  return 1
}

DBG "Raspberry Pi 5 ARM64 runtime prerequisites"
DBG "ARCH=$(uname -m)"
DBG "OS:"; (lsb_release -a 2>/dev/null || cat /etc/os-release || true)

DBG "Updating apt indexes..."
apt-get update

try_install \
  libhdf5-dev \
  libdcmtk-dev \
  libfftw3-dev \
  libopencv-dev \
  libpugixml1v5 \
  patchelf

try_install_any "ismrmrd-runtime" libismrmrd1.4 libismrmrd1
try_install_any "qt6-base" libqt6core6 libqt6gui6 libqt6widgets6
try_install_any "qt6-multimedia" libqt6multimedia6
try_install_any "qt6-svg" libqt6svg6

DBG "DONE"
