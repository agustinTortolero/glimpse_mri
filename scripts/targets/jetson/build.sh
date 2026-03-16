#!/usr/bin/env bash
set -euo pipefail

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }
err() { echo "[$(ts)][ERR] $*" >&2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE_BUILD_DIR="${ROOT_DIR}/build/engine-jetson"
DICOM_BUILD_DIR="${ROOT_DIR}/build/dicom-jetson"
GUI_BUILD_DIR="${ROOT_DIR}/build/gui-jetson"

BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_MODE="${GLIMPSE_CUDA_MODE:-ON}"
CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-87}"

log "ROOT_DIR=${ROOT_DIR}"
log "ENGINE_BUILD_DIR=${ENGINE_BUILD_DIR}"
log "DICOM_BUILD_DIR=${DICOM_BUILD_DIR}"
log "GUI_BUILD_DIR=${GUI_BUILD_DIR}"
log "BUILD_TYPE=${BUILD_TYPE}"
log "CUDA_MODE=${CUDA_MODE}"
log "CUDA_ARCH=${CUDA_ARCH}"

if [[ "$(uname -m)" != "aarch64" ]]; then
    err "This script is intended for Jetson/aarch64. Current arch: $(uname -m)"
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    err "cmake not found in PATH"
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    err "nvcc not found in PATH"
    exit 1
fi

log "nvcc path: $(command -v nvcc)"
log "cmake path: $(command -v cmake)"
log "nvcc version:"
nvcc --version || true
log "cmake version:"
cmake --version || true

mkdir -p "${ENGINE_BUILD_DIR}" "${DICOM_BUILD_DIR}" "${GUI_BUILD_DIR}"

log "=== Step 1/3: Configure engine ==="
cmake -S "${ROOT_DIR}/engine" -B "${ENGINE_BUILD_DIR}" \
    -DGLIMPSE_CUDA_MODE="${CUDA_MODE}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

log "=== Step 1/3: Build engine ==="
cmake --build "${ENGINE_BUILD_DIR}" -j"$(nproc)"

if [[ ! -f "${ENGINE_BUILD_DIR}/libmri_engine.so" ]]; then
    err "Engine build completed but libmri_engine.so was not found at ${ENGINE_BUILD_DIR}"
    exit 1
fi
log "Engine library found: ${ENGINE_BUILD_DIR}/libmri_engine.so"

log "=== Step 2/3: Configure dicom_io_lib ==="
cmake -S "${ROOT_DIR}/dicom_io_lib" -B "${DICOM_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

log "=== Step 2/3: Build dicom_io_lib ==="
cmake --build "${DICOM_BUILD_DIR}" -j"$(nproc)"

if [[ ! -f "${DICOM_BUILD_DIR}/libdicom_io_lib.so" ]]; then
    err "DICOM build completed but libdicom_io_lib.so was not found at ${DICOM_BUILD_DIR}"
    exit 1
fi
log "DICOM library found: ${DICOM_BUILD_DIR}/libdicom_io_lib.so"

log "=== Step 3/3: Configure GUI ==="
cmake -S "${ROOT_DIR}/gui" -B "${GUI_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGLIMPSE_ENGINE_BUILD_DIR="${ENGINE_BUILD_DIR}" \
    -DGLIMPSE_DICOM_BUILD_DIR="${DICOM_BUILD_DIR}"

log "=== Step 3/3: Build GUI ==="
cmake --build "${GUI_BUILD_DIR}" -j"$(nproc)"

if [[ ! -f "${GUI_BUILD_DIR}/glimpseMRI" ]]; then
    err "GUI build completed but glimpseMRI was not found at ${GUI_BUILD_DIR}"
    exit 1
fi
log "GUI executable found: ${GUI_BUILD_DIR}/glimpseMRI"

log "=== Build summary ==="
log "Engine : ${ENGINE_BUILD_DIR}/libmri_engine.so"
log "DICOM  : ${DICOM_BUILD_DIR}/libdicom_io_lib.so"
log "GUI    : ${GUI_BUILD_DIR}/glimpseMRI"
log "Jetson build completed successfully"
