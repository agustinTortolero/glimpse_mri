#!/usr/bin/env bash
set -euo pipefail

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }
warn() { echo "[$(ts)][WRN] $*" >&2; }
err() { echo "[$(ts)][ERR] $*" >&2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_ROOT="${ROOT_DIR}/build"

ENGINE_BUILD_DIR="${ENGINE_BUILD_DIR:-${BUILD_ROOT}/engine-jetson}"
DICOM_BUILD_DIR="${DICOM_BUILD_DIR:-${BUILD_ROOT}/dicom-jetson}"
GUI_BUILD_DIR="${GUI_BUILD_DIR:-${BUILD_ROOT}/gui-jetson}"

APP_NAME="${APP_NAME:-glimpseMRI}"
BUNDLE_NAME="${BUNDLE_NAME:-glimpseMRI-jetson}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/dist}"
STAGE_DIR="${OUT_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${OUT_DIR}/${BUNDLE_NAME}.tar.gz"

ENGINE_LIB="${ENGINE_BUILD_DIR}/libmri_engine.so"
DICOM_LIB="${DICOM_BUILD_DIR}/libdicom_io_lib.so"
GUI_EXE="${GUI_BUILD_DIR}/${APP_NAME}"
INSTALL_DEPS_SOURCE="${ROOT_DIR}/scripts/targets/jetson/install_deps.sh"
INSTALL_DESKTOP_SOURCE="${ROOT_DIR}/scripts/targets/jetson/install_desktop.sh"
ICON_SOURCE="${ROOT_DIR}/gui/assets/images/icons/mri_256.png"

RUN_SCRIPT="${STAGE_DIR}/run.sh"
README_FILE="${STAGE_DIR}/README.txt"
INSTALL_DEPS_STAGED="${STAGE_DIR}/tools/install_deps.sh"
INSTALL_DESKTOP_STAGED="${STAGE_DIR}/tools/install_desktop.sh"
ICON_STAGED="${STAGE_DIR}/icons/mri_256.png"

log "ROOT_DIR=${ROOT_DIR}"
log "BUILD_ROOT=${BUILD_ROOT}"
log "ENGINE_BUILD_DIR=${ENGINE_BUILD_DIR}"
log "DICOM_BUILD_DIR=${DICOM_BUILD_DIR}"
log "GUI_BUILD_DIR=${GUI_BUILD_DIR}"
log "OUT_DIR=${OUT_DIR}"
log "STAGE_DIR=${STAGE_DIR}"
log "ARCHIVE_PATH=${ARCHIVE_PATH}"

if [[ "$(uname -m)" != "aarch64" ]]; then
    err "This packaging script is intended for Jetson/aarch64. Current arch: $(uname -m)"
    exit 1
fi

for path in "${ENGINE_LIB}" "${DICOM_LIB}" "${GUI_EXE}"; do
    if [[ ! -f "${path}" ]]; then
        err "Required build artifact not found: ${path}"
        err "Run ./scripts/targets/jetson/build.sh first, or override the build directories."
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/bin" "${STAGE_DIR}/lib" "${STAGE_DIR}/tools" "${STAGE_DIR}/icons"

log "Copying GUI executable"
cp -av "${GUI_EXE}" "${STAGE_DIR}/bin/"

log "Copying engine library"
cp -av "${ENGINE_LIB}" "${STAGE_DIR}/lib/"

log "Copying DICOM library"
cp -av "${DICOM_LIB}" "${STAGE_DIR}/lib/"

if [[ -f "${INSTALL_DEPS_SOURCE}" ]]; then
    log "Copying dependency helper"
    cp -av "${INSTALL_DEPS_SOURCE}" "${INSTALL_DEPS_STAGED}"
    chmod +x "${INSTALL_DEPS_STAGED}"
else
    warn "Dependency helper not found at ${INSTALL_DEPS_SOURCE}; bundle README will reference a missing file"
fi

if [[ -f "${INSTALL_DESKTOP_SOURCE}" ]]; then
    log "Copying desktop integration helper"
    cp -av "${INSTALL_DESKTOP_SOURCE}" "${INSTALL_DESKTOP_STAGED}"
    chmod +x "${INSTALL_DESKTOP_STAGED}"
else
    warn "Desktop integration helper not found at ${INSTALL_DESKTOP_SOURCE}; bundle README will reference a missing file"
fi

if [[ -f "${ICON_SOURCE}" ]]; then
    log "Copying desktop icon"
    cp -av "${ICON_SOURCE}" "${ICON_STAGED}"
else
    warn "Desktop icon not found at ${ICON_SOURCE}; desktop helper will fall back to a generic icon"
fi

log "Writing run.sh"
cat > "${RUN_SCRIPT}" <<'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${HERE}/lib:${LD_LIBRARY_PATH:-}"

exec "${HERE}/bin/glimpseMRI" "$@"
RUNEOF
chmod +x "${RUN_SCRIPT}"

log "Writing README.txt"
cat > "${README_FILE}" <<'READEOF'
Glimpse MRI — Jetson bundle

Contents
--------
- bin/glimpseMRI
- lib/libmri_engine.so
- lib/libdicom_io_lib.so
- run.sh
- tools/install_deps.sh
- tools/install_desktop.sh
- icons/mri_256.png

How to run
----------
From inside this bundle directory:

  ./run.sh

What this bundle includes
-------------------------
This bundle includes the Glimpse MRI executable and the project-built shared libraries:

- libmri_engine.so
- libdicom_io_lib.so

What this bundle does NOT include
---------------------------------
This bundle does not try to vendor the full Jetson system stack.
It expects the target Jetson system to already provide the required runtime dependencies,
such as CUDA, Qt6, OpenCV, DCMTK, HDF5, FFTW, and ISMRMRD.

Before first launch on a fresh Jetson, install the expected system prerequisites:

  ./tools/install_deps.sh --runtime

To add a desktop launcher for the extracted bundle:

  ./tools/install_desktop.sh

If you also plan to build locally on the Jetson instead of only running the bundle:

  ./tools/install_deps.sh --build

Typical runtime-provided dependencies on the validated Jetson system included:
- Qt6 runtime libraries
- CUDA runtime / cuFFT
- OpenCV
- DCMTK
- HDF5
- FFTW3f
- ISMRMRD

Recommended target
------------------
A Jetson system configured similarly to the validated development machine.

Notes
-----
This is a Jetson-native bundle intended for development/research distribution.
It is not a fully self-contained universal Linux package.
READEOF

log "Writing dependency report"
ldd "${STAGE_DIR}/bin/${APP_NAME}" > "${STAGE_DIR}/ldd_glimpseMRI.txt" || warn "ldd failed for ${APP_NAME}"
ldd "${STAGE_DIR}/lib/libmri_engine.so" > "${STAGE_DIR}/ldd_libmri_engine.txt" || warn "ldd failed for libmri_engine.so"
ldd "${STAGE_DIR}/lib/libdicom_io_lib.so" > "${STAGE_DIR}/ldd_libdicom_io_lib.txt" || warn "ldd failed for libdicom_io_lib.so"

log "Writing SHA256 checksums"
(
    cd "${STAGE_DIR}"
    sha256sum \
        "bin/${APP_NAME}" \
        "lib/libmri_engine.so" \
        "lib/libdicom_io_lib.so" \
        "run.sh" \
        "tools/install_deps.sh" \
        "tools/install_desktop.sh" \
        "icons/mri_256.png" \
        "README.txt" \
        > SHA256SUMS.txt
)

log "Creating tar.gz archive"
rm -f "${ARCHIVE_PATH}"
tar -C "${OUT_DIR}" -czf "${ARCHIVE_PATH}" "${BUNDLE_NAME}"

log "Package staging complete"
log "Bundle dir : ${STAGE_DIR}"
log "Archive    : ${ARCHIVE_PATH}"

log "Bundle contents:"
find "${STAGE_DIR}" -maxdepth 2 -type f | sort

log "Packaging completed successfully"
