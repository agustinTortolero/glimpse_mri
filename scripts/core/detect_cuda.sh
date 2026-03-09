#!/usr/bin/env bash
set -euo pipefail

cuda_toolkit_present() {
  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi

  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    return 0
  fi

  return 1
}

cuda_runtime_present() {
  if [[ -e /dev/nvidiactl ]] || [[ -e /proc/driver/nvidia/version ]]; then
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >/dev/null 2>&1 && return 0
  fi

  return 1
}

cuda_device_present() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L >/dev/null 2>&1 && return 0
  fi

  [[ -e /dev/nvidia0 ]] && return 0

  return 1
}

resolve_cuda_mode() {
  local requested="${1:-auto}"

  case "${requested}" in
    off)  echo "OFF" ;;
    on)   echo "ON" ;;
    auto)
      if cuda_toolkit_present; then
        echo "ON"
      else
        echo "OFF"
      fi
      ;;
    *)
      echo "OFF"
      ;;
  esac
}