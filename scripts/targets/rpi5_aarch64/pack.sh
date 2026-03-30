#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${THIS_DIR}/../../core" && pwd)"

source "${CORE_DIR}/common.sh"
source "${CORE_DIR}/pack_bundle.sh"

BUILD_TYPE="Release"
OUT_ROOT=""
GUI_EXE=""
KEEP_DIR=0

usage() {
  cat <<USAGE_EOF
Usage: $0 [--debug] [--out /path/to/dist_root] [--exe /path/to/gui_exe] [--keep-dir]

Package Raspberry Pi 5 ARM64 tarball from Linux CMake build outputs.
The produced bundle includes run.sh, install_deps.sh, and install.sh.
USAGE_EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) BUILD_TYPE="Debug"; shift ;;
    --out) OUT_ROOT="${2:-}"; shift 2 ;;
    --exe) GUI_EXE="${2:-}"; shift 2 ;;
    --keep-dir) KEEP_DIR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

ROOT="$(repo_root_from_this_script "${BASH_SOURCE[0]}")"
RUNTIME_SCRIPT="${THIS_DIR}/prerequisites_run.sh"

log "Raspberry Pi 5 aarch64 pack wrapper"
log "ROOT=$ROOT"
log "BUILD_TYPE=$BUILD_TYPE"
log "OUT_ROOT=${OUT_ROOT:-<default>}"
log "GUI_EXE=${GUI_EXE:-<auto>}"
log "KEEP_DIR=$KEEP_DIR"
log "RUNTIME_SCRIPT=$RUNTIME_SCRIPT"

pack_glimpse_bundle "$ROOT" "$BUILD_TYPE" "$OUT_ROOT" "$GUI_EXE" "$KEEP_DIR" "$RUNTIME_SCRIPT"
