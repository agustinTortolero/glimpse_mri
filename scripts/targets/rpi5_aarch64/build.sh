#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${THIS_DIR}/../../core" && pwd)"

source "${CORE_DIR}/common.sh"
source "${CORE_DIR}/build_modules.sh"

CLEAN=0
BUILD_TYPE="Release"
RUN_TESTS=0
JOBS="$(default_jobs)"
CUDA_MODE="off"

usage() {
  cat <<USAGE_EOF
Usage: $0 [--clean] [--debug] [--tests] [--jobs N]

Build GlimpseMRI for Raspberry Pi 5 (aarch64) using Linux CMake-only flow.
CPU-only target: CUDA is forced OFF.

Options:
  --clean           Remove build dirs before configuring
  --debug           Build Debug instead of Release
  --tests           Run ctest where available
  --jobs N          Parallel jobs (default: nproc)
USAGE_EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --debug) BUILD_TYPE="Debug"; shift ;;
    --tests) RUN_TESTS=1; shift ;;
    --jobs) JOBS="${2:-$JOBS}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

ROOT="$(repo_root_from_this_script "${BASH_SOURCE[0]}")"

log "Raspberry Pi 5 aarch64 build wrapper"
log "ROOT=$ROOT"
log "BUILD_TYPE=$BUILD_TYPE"
log "JOBS=$JOBS"
log "RUN_TESTS=$RUN_TESTS"
log "CLEAN=$CLEAN"
log "CUDA_MODE(forced)=$CUDA_MODE"

build_glimpse_repo "$ROOT" "$BUILD_TYPE" "$JOBS" "$RUN_TESTS" "$CLEAN" "$CUDA_MODE"
