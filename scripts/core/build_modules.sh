#!/usr/bin/env bash
set -euo pipefail

CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CORE_DIR}/common.sh"
source "${CORE_DIR}/detect_buildsystem.sh"
source "${CORE_DIR}/detect_cuda.sh"

build_cmake_module() {
  local name="$1"
  local src="$2"
  local bld="$3"
  local build_type="$4"
  local jobs="$5"
  local run_tests="$6"
  local clean="$7"
  shift 7
  local extra_args=("$@")

  need_cmd cmake
  local gen
  gen="$(pick_generator)"

  maybe_clean_dir "$bld" "$clean"
  ensure_dir "$bld"

  log "----------------------------------------"
  log "CMAKE MODULE: $name"
  log "SRC=$src"
  log "BLD=$bld"
  log "GEN=$gen"
  log "BUILD_TYPE=$build_type"
  log "EXTRA_ARGS=${extra_args[*]:-<none>}"
  log "----------------------------------------"

  run_cmd cmake -S "$src" -B "$bld" -G "$gen" \
    -DCMAKE_BUILD_TYPE="$build_type" \
    "${extra_args[@]}"

  run_cmd cmake --build "$bld" -j "$jobs"

  if [[ "$run_tests" == "1" ]]; then
    if [[ -f "${bld}/CTestTestfile.cmake" ]] || [[ -d "${bld}/Testing" ]]; then
      log "Running ctest for module: $name"
      (
        cd "$bld"
        run_cmd ctest --output-on-failure -j "$jobs"
      )
    else
      log "No tests detected for module: $name"
    fi
  fi

  log "Artifacts for $name:"
  find "$bld" -maxdepth 5 -type f \( -executable -o -name "*.so" -o -name "*.a" \) -print || true
}

build_module_cmake_only() {
  local root="$1"
  local name="$2"
  local src="$3"
  local bld="$4"
  local build_type="$5"
  local jobs="$6"
  local run_tests="$7"
  local clean="$8"
  local cuda_mode="$9"

  [[ -d "$src" ]] || die "Missing module directory: $src"
  assert_cmake_module "$src" || die "Module '${name}' is not ready for Linux CMake-only build."

  local extra_args=()

  if [[ "$name" == "engine" ]]; then
    if [[ "$cuda_mode" == "ON" ]]; then
      extra_args+=(
        "-DENABLE_CUDA=ON"
        "-DUSE_CUDA=ON"
        "-DGLIMPSE_ENABLE_CUDA=ON"
      )
    else
      extra_args+=(
        "-DENABLE_CUDA=OFF"
        "-DUSE_CUDA=OFF"
        "-DGLIMPSE_ENABLE_CUDA=OFF"
      )
    fi
  fi

  if [[ "$name" == "gui" ]]; then
    extra_args+=(
      "-DGLIMPSE_ENGINE_BUILD_DIR=${root}/build_engine_${build_type}"
      "-DGLIMPSE_DICOM_BUILD_DIR=${root}/build_dicom_${build_type}"
    )
  fi

  build_cmake_module \
    "$name" "$src" "$bld" "$build_type" "$jobs" "$run_tests" "$clean" \
    "${extra_args[@]}"
}

build_glimpse_repo() {
  local root="$1"
  local build_type="$2"
  local jobs="$3"
  local run_tests="$4"
  local clean="$5"
  local requested_cuda_mode="$6"

  local log_dir="${root}/build_logs"
  ensure_dir "$log_dir"

  local log_file="${log_dir}/build_$(date +"%Y%m%d_%H%M%S")_${build_type}.log"

  exec > >(tee -a "$log_file") 2>&1

  local resolved_cuda_mode
  resolved_cuda_mode="$(resolve_cuda_mode "$requested_cuda_mode")"

  log "ROOT=$root"
  log "BUILD_TYPE=$build_type"
  log "JOBS=$jobs"
  log "RUN_TESTS=$run_tests"
  log "CLEAN=$clean"
  log "CUDA_MODE(requested)=$requested_cuda_mode"
  log "CUDA_MODE(resolved)=$resolved_cuda_mode"
  log "ARCH=$(uname -m)"
  log "KERNEL=$(uname -r)"
  log "OS:"
  (lsb_release -a 2>/dev/null || cat /etc/os-release || true)
  log "GIT_SHA=$(git_sha_or_nosha "$root")"

  log "Updating submodules (if any)..."
  git -C "$root" submodule update --init --recursive || true

  build_module_cmake_only \
    "$root" \
    "dicom_io_lib" \
    "${root}/dicom_io_lib" \
    "${root}/build_dicom_${build_type}" \
    "$build_type" "$jobs" "$run_tests" "$clean" "$resolved_cuda_mode"

  build_module_cmake_only \
    "$root" \
    "engine" \
    "${root}/engine" \
    "${root}/build_engine_${build_type}" \
    "$build_type" "$jobs" "$run_tests" "$clean" "$resolved_cuda_mode"

  build_module_cmake_only \
    "$root" \
    "gui" \
    "${root}/gui" \
    "${root}/build_gui_${build_type}" \
    "$build_type" "$jobs" "$run_tests" "$clean" "$resolved_cuda_mode"

  log "----------------------------------------"
  log "BUILD COMPLETE"
  log "Log file: $log_file"
  log "----------------------------------------"
}