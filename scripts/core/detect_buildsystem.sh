#!/usr/bin/env bash
set -euo pipefail

# Linux policy: CMake only.
# .pro files are ignored by Linux scripts.

detect_build_system() {
  local dir="$1"

  if [[ -f "${dir}/CMakeLists.txt" ]]; then
    echo "cmake"
    return 0
  fi

  echo "none"
}

assert_cmake_module() {
  local dir="$1"
  [[ -f "${dir}/CMakeLists.txt" ]] || {
    echo "[ERR] Missing CMakeLists.txt in ${dir}. Linux build is CMake-only." >&2
    return 1
  }
}

pick_generator() {
  if command -v ninja >/dev/null 2>&1; then
    echo "Ninja"
  else
    echo "Unix Makefiles"
  fi
}