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
CUDA_MODE="${CUDA_MODE:-auto}"

usage() {
  cat <<EOF
Usage: $0 [--clean] [--debug] [--tests] [--jobs N] [--cuda auto|on|off]

Build GlimpseMRI for Ubuntu x86_64 using Linux CMake-only flow.

Options:
  --clean           Remove build dirs before configuring
  --debug           Build Debug instead of Release
  --tests           Run ctest where available
  --jobs N          Parallel jobs (default: nproc)
  --cuda MODE       auto | on | off
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --debug) BUILD_TYPE="Debug"; shift ;;
    --tests) RUN_TESTS=1; shift ;;
    --jobs) JOBS="${2:-$JOBS}"; shift 2 ;;
    --cuda) CUDA_MODE="${2:-$CUDA_MODE}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

ROOT="$(repo_root_from_this_script "${BASH_SOURCE[0]}")"

log "Ubuntu x86_64 build wrapper"
log "ROOT=$ROOT"
log "BUILD_TYPE=$BUILD_TYPE"
log "JOBS=$JOBS"
log "RUN_TESTS=$RUN_TESTS"
log "CLEAN=$CLEAN"
log "CUDA_MODE=$CUDA_MODE"

build_glimpse_repo "$ROOT" "$BUILD_TYPE" "$JOBS" "$RUN_TESTS" "$CLEAN" "$CUDA_MODE"