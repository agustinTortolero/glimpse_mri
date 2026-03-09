#!/usr/bin/env bash
set -euo pipefail

ts() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)][DBG] $*"
}

warn() {
  echo "[$(ts)][WRN] $*" >&2
}

die() {
  echo "[$(ts)][ERR] $*" >&2
  exit 1
}

run_cmd() {
  log "RUN: $*"
  "$@"
}

need_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing command: $cmd"
}

ensure_dir() {
  mkdir -p "$1"
}

maybe_clean_dir() {
  local dir="$1"
  local clean="${2:-0}"
  if [[ "$clean" == "1" ]]; then
    log "Cleaning directory: $dir"
    rm -rf "$dir"
  fi
}

default_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  else
    echo 2
  fi
}

git_sha_or_nosha() {
  local root="$1"
  git -C "$root" rev-parse --short HEAD 2>/dev/null || echo "nosha"
}

repo_root_from_this_script() {
  local src="${1:-${BASH_SOURCE[0]}}"
  local dir
  dir="$(cd "$(dirname "$src")" && pwd)"
  if git -C "$dir" rev-parse --show-toplevel >/dev/null 2>&1; then
    git -C "$dir" rev-parse --show-toplevel
    return 0
  fi
  (cd "$dir/../../.." && pwd)
}