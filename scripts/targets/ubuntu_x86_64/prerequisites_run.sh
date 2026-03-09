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

DBG "ARCH=$(uname -m)"
DBG "OS:"; (lsb_release -a 2>/dev/null || cat /etc/os-release || true)

DBG "Updating apt indexes..."
apt-get update

try_install \
  libhdf5-dev \
  libdcmtk-dev \
  libpugixml1v5 \
  libfftw3-dev \
  libopencv-dev \
  qt6-base-dev \
  qt6-multimedia-dev \
  libqt6svg6-dev \
  patchelf

DBG "DONE"