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

DBG "ARCH=$(uname -m)"
DBG "OS:"; (lsb_release -a 2>/dev/null || cat /etc/os-release || true)

DBG "Enabling Ubuntu universe repo..."
if command -v add-apt-repository >/dev/null 2>&1; then
  add-apt-repository -y universe || true
else
  apt-get update
  apt-get install -y software-properties-common
  add-apt-repository -y universe || true
fi

DBG "Updating apt indexes..."
apt-get update

DBG "Installing core build tools..."
apt-get install -y \
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

DBG "Installing library dependencies..."
apt-get install -y \
  libhdf5-dev \
  libdcmtk-dev \
  libpugixml-dev \
  libfftw3-dev \
  libopencv-dev

DBG "Installing Qt6 development packages for CMake..."
apt-get install -y \
  qt6-base-dev \
  qt6-base-dev-tools \
  qt6-tools-dev \
  qt6-tools-dev-tools \
  qt6-multimedia-dev \
  libqt6svg6-dev

DBG "DONE"
DBG "Sanity checks:"
DBG "  cmake : $(command -v cmake || echo missing)"
DBG "  ninja : $(command -v ninja || echo missing)"
DBG "  gcc   : $(command -v gcc   || echo missing)"
DBG "  g++   : $(command -v g++   || echo missing)"