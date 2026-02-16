#!/usr/bin/env bash
set -euo pipefail

# prerequisites_fixed_v4.sh
# Runtime deps installer for GlimpseMRI Jetson binary bundle.
# - Targets Ubuntu 22.04 (JetPack 6.x)
# - Uses apt
# - Tries to be resilient: if a package name doesn't exist, it warns and continues.

TS() { date +"%Y-%m-%d %H:%M:%S"; }
DBG(){ echo "[$(TS)][DBG] $*"; }
WRN(){ echo "[$(TS)][WRN] $*" >&2; }
ERR(){ echo "[$(TS)][ERR] $*" >&2; }

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  ERR "Run with sudo: sudo $0"
  exit 1
fi

try_install() {
  # Usage: try_install pkg1 pkg2 ...
  local pkgs=("$@")
  [[ ${#pkgs[@]} -gt 0 ]] || return 0

  DBG "APT install: ${pkgs[*]}"
  if ! apt-get install -y "${pkgs[@]}"; then
    WRN "APT failed for: ${pkgs[*]} (continuing)"
    return 0
  fi
}

DBG "PWD=$(pwd)"
DBG "ARCH=$(uname -m)"
DBG "KERNEL=$(uname -r)"
DBG "OS:"; (lsb_release -a 2>/dev/null || true)

if [[ -f /etc/nv_tegra_release ]]; then
  DBG "Jetson detected: /etc/nv_tegra_release exists"
  DBG "nv_tegra_release:"; cat /etc/nv_tegra_release || true
else
  WRN "This does not look like a Jetson (no /etc/nv_tegra_release)."
fi

DBG "Updating apt indexes..."
apt-get update

# Core runtime deps for GlimpseMRI (based on missing libs we saw during testing)
# Note: Some are -dev packages; that's okay for now (simpler and pulls runtime libs).
try_install \
  libhdf5-dev \
  libdcmtk-dev \
  libpugixml1v5 \
  libfftw3-dev

# Qt6 runtime / dev (Qt 6.2 on Ubuntu 22.04)
try_install \
  qt6-base-dev \
  qt6-multimedia-dev \
  libqt6svg6-dev

# This package name is NOT present on many Jetson images; keep it optional.
try_install qt6-multimedia-dev-tools

# OpenCV (your binary links against libopencv_*; this ensures OpenCV libs exist)
try_install libopencv-dev

DBG "DONE"
