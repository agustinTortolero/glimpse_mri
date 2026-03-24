#!/usr/bin/env bash
set -euo pipefail

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }
warn() { echo "[$(ts)][WRN] $*" >&2; }
err() { echo "[$(ts)][ERR] $*" >&2; }

usage() {
    cat <<'EOF'
Usage:
  ./scripts/targets/jetson/install_deps.sh [--runtime|--build|--all] [--print-only] [--no-update]

Modes:
  --runtime    Install packages needed to run a packaged GlimpseMRI bundle on Jetson.
  --build      Install packages needed to build GlimpseMRI locally on Jetson.
  --all        Install both runtime and build prerequisites (default).

Options:
  --print-only Print the package groups and apt command, but do not install anything.
  --no-update  Skip apt-get update before install.
  -h, --help   Show this help.

Notes:
  - This script keeps the current Jetson packaging philosophy: GlimpseMRI bundles only
    project-built binaries, while Jetson/Ubuntu libraries remain system prerequisites.
  - CUDA / JetPack components are expected to come from the Jetson environment.
  - Some package names may need small adjustments across JetPack / Ubuntu releases.
    The package arrays below are intentionally easy to audit and edit.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Required command not found: $1"
        exit 1
    fi
}

join_by() {
    local delimiter="$1"
    shift
    local first=1
    for item in "$@"; do
        if (( first )); then
            printf "%s" "$item"
            first=0
        else
            printf "%s%s" "$delimiter" "$item"
        fi
    done
}

append_unique() {
    local value
    for value in "${@:2}"; do
        if [[ "$value" == "$1" ]]; then
            return 0
        fi
    done
    return 1
}

INSTALL_RUNTIME=0
INSTALL_BUILD=0
PRINT_ONLY=0
RUN_UPDATE=1

if [[ $# -eq 0 ]]; then
    INSTALL_RUNTIME=1
    INSTALL_BUILD=1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime)
            INSTALL_RUNTIME=1
            ;;
        --build)
            INSTALL_BUILD=1
            ;;
        --all)
            INSTALL_RUNTIME=1
            INSTALL_BUILD=1
            ;;
        --print-only)
            PRINT_ONLY=1
            ;;
        --no-update)
            RUN_UPDATE=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if (( ! INSTALL_RUNTIME && ! INSTALL_BUILD )); then
    err "No install mode selected. Use --runtime, --build, or --all."
    exit 1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
    warn "This helper is intended for Jetson/aarch64. Current arch: $(uname -m)"
fi

require_cmd apt-get

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    log "Detected OS: ${PRETTY_NAME:-unknown}"
else
    warn "/etc/os-release not found; continuing without distro metadata"
fi

if command -v nvcc >/dev/null 2>&1; then
    log "Detected nvcc at $(command -v nvcc)"
else
    log "nvcc not found in PATH. That is acceptable for --runtime, but local CUDA builds need JetPack/CUDA installed."
fi

# Runtime packages for executing the packaged bundle on a fresh Jetson.
# Keep this list focused on direct runtime needs with relatively stable package names.
# More distro-sensitive library packages are handled via build_packages below, or via
# the optional_runtime_packages section if you want to install them explicitly.
runtime_packages=(
    libqt6core6
    libqt6gui6
    libqt6widgets6
    libqt6dbus6
    libfftw3-single3
    libpugixml1v5
    qt6-qpa-plugins
)

# Optional / extended runtime packages.
# These are plausible runtime providers for non-bundled libraries used by the app, but
# exact package names can vary by Ubuntu / JetPack release. We do not install them by
# default in --runtime mode; keep this list easy to adjust per validated Jetson image.
optional_runtime_packages=(
    libopencv-core-dev
    libopencv-imgproc-dev
    libopencv-imgcodecs-dev
    libdcmtk-dev
    libhdf5-dev
    libismrmrd-dev
)

# Build packages for compiling on Jetson. These intentionally include dev/meta packages
# and will usually satisfy most runtime dependencies transitively on the build machine.
# This matches the current philosophy: use system packages for Jetson prerequisites
# rather than bundling the full system stack into the archive.
build_packages=(
    build-essential
    cmake
    ninja-build
    pkg-config
    qt6-base-dev
    qt6-base-dev-tools
    qt6-tools-dev-tools
    libopencv-dev
    libdcmtk-dev
    libhdf5-dev
    libfftw3-dev
    libpugixml-dev
    libismrmrd-dev
)

selected_packages=()

if (( INSTALL_RUNTIME )); then
    log "Selected mode: runtime prerequisites"
    for pkg in "${runtime_packages[@]}"; do
        if ! append_unique "$pkg" "${selected_packages[@]}"; then
            selected_packages+=("$pkg")
        fi
    done
fi

if (( INSTALL_BUILD )); then
    log "Selected mode: build prerequisites"
    for pkg in "${build_packages[@]}"; do
        if ! append_unique "$pkg" "${selected_packages[@]}"; then
            selected_packages+=("$pkg")
        fi
    done
fi

log "CUDA / JetPack note: this script does not install CUDA, cuFFT, cuBLAS, or JetPack."
log "Those components are expected to come from the target Jetson environment."

log "Runtime package list ($((${#runtime_packages[@]})) packages):"
for pkg in "${runtime_packages[@]}"; do
    log "  - ${pkg}"
done

log "Optional / extended runtime packages ($((${#optional_runtime_packages[@]})) packages):"
for pkg in "${optional_runtime_packages[@]}"; do
    log "  - ${pkg}"
done

log "Build package list ($((${#build_packages[@]})) packages):"
for pkg in "${build_packages[@]}"; do
    log "  - ${pkg}"
done

log "Selected install set ($((${#selected_packages[@]})) packages):"
for pkg in "${selected_packages[@]}"; do
    log "  - ${pkg}"
done

apt_install_cmd=(sudo apt-get install -y "${selected_packages[@]}")

if (( PRINT_ONLY )); then
    log "--print-only requested; no changes will be made"
    if (( RUN_UPDATE )); then
        log "Would run: sudo apt-get update"
    fi
    log "Would run: $(join_by ' ' "${apt_install_cmd[@]}")"
    log "Optional runtime packages are not included automatically in --runtime mode."
    log "If your validated Jetson image needs them explicitly, append the relevant package names manually."
    exit 0
fi

if (( RUN_UPDATE )); then
    log "Running apt-get update"
    sudo apt-get update
else
    log "Skipping apt-get update (--no-update)"
fi

log "Installing selected packages"
"${apt_install_cmd[@]}"

log "Dependency installation completed"
log "Optional runtime packages were not installed automatically."
log "If ldd still reports missing non-Qt libraries, review optional_runtime_packages in this script for your Jetson image."
log "Suggested next steps:"
if (( INSTALL_RUNTIME )); then
    log "  - Validate runtime libs: ldd ./dist/glimpseMRI-jetson/bin/glimpseMRI | grep 'not found' || true"
fi
if (( INSTALL_BUILD )); then
    log "  - Build: ./scripts/targets/jetson/build.sh"
    log "  - Package: ./scripts/targets/jetson/pack.sh"
fi
