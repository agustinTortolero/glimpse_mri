#!/usr/bin/env bash
set -euo pipefail

CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CORE_DIR}/common.sh"

auto_detect_gui_exe() {
  local gui_bld="$1"
  local arch="$2"

  [[ -d "$gui_bld" ]] || return 1

  while IFS= read -r f; do
    local info
    info="$(file "$f" 2>/dev/null || true)"

    if [[ "$info" == *"ELF 64-bit"* && "$info" == *"executable"* ]]; then
      if [[ "$arch" == "x86_64" && "$info" == *"x86-64"* ]]; then
        echo "$f"
        return 0
      fi
      if [[ "$arch" == "aarch64" && ( "$info" == *"aarch64"* || "$info" == *"ARM aarch64"* ) ]]; then
        echo "$f"
        return 0
      fi
      echo "$f"
      return 0
    fi
  done < <(find "$gui_bld" -maxdepth 6 -type f ! -path "*/CMakeFiles/*" | sort)

  return 1
}

pack_glimpse_bundle() {
  local root="$1"
  local build_type="$2"
  local out_root="${3:-}"
  local gui_exe="${4:-}"
  local keep_dir="${5:-0}"
  local runtime_src_override="${6:-}"

  need_cmd tar
  need_cmd sha256sum
  need_cmd file
  need_cmd ldd

  [[ -n "$out_root" ]] || out_root="${root}/dist_linux"

  local gui_bld="${root}/build_gui_${build_type}"
  local eng_bld="${root}/build_engine_${build_type}"
  local dicom_bld="${root}/build_dicom_${build_type}"

  [[ -d "$gui_bld" ]] || die "Missing GUI build dir: $gui_bld"
  [[ -d "$eng_bld" ]] || die "Missing engine build dir: $eng_bld"
  [[ -d "$dicom_bld" ]] || die "Missing dicom build dir: $dicom_bld"

  local git_sha
  git_sha="$(git_sha_or_nosha "$root")"

  local arch
  arch="$(uname -m)"

  local stamp
  stamp="$(date +"%Y%m%d_%H%M%S")"

  local out_name="glimpse_mri_${arch}_${build_type}_${git_sha}_${stamp}"
  local out_dir="${out_root}/${out_name}"
  local bin_dir="${out_dir}/bin"
  local lib_dir="${out_dir}/lib"
  local diag_dir="${out_dir}/diag"

  log "ROOT=$root"
  log "BUILD_TYPE=$build_type"
  log "ARCH=$arch"
  log "OUT_DIR=$out_dir"

  if [[ -d "$out_dir" && "$keep_dir" != "1" ]]; then
    log "Removing existing out dir: $out_dir"
    rm -rf "$out_dir"
  fi

  ensure_dir "$bin_dir"
  ensure_dir "$lib_dir"
  ensure_dir "$diag_dir"

  if [[ -z "$gui_exe" ]]; then
    gui_exe="$(auto_detect_gui_exe "$gui_bld" "$arch" || true)"
  fi

  [[ -n "$gui_exe" ]] || die "Could not auto-detect GUI executable. Pass --exe /path/to/binary"
  [[ -f "$gui_exe" ]] || die "GUI executable not found: $gui_exe"

  log "GUI_EXE=$gui_exe"
  cp -v "$gui_exe" "$bin_dir/"
  local gui_basename
  gui_basename="$(basename "$gui_exe")"

  log "Copying engine shared libs..."
  find "$eng_bld" -maxdepth 5 -type f -name "*.so*" -print -exec cp -v {} "$lib_dir/" \; || true

  log "Copying dicom shared libs..."
  find "$dicom_bld" -maxdepth 5 -type f -name "*.so*" -print -exec cp -v {} "$lib_dir/" \; || true

  local icon_src_dir="${root}/gui/assets/images/icons"
  local assets_dir="${out_dir}/assets"
  local icons_dir="${assets_dir}/icons"

  if [[ -d "$icon_src_dir" ]]; then
    ensure_dir "$icons_dir"
    cp -v "$icon_src_dir"/mri_*.png "$icons_dir/" 2>/dev/null || true
    cp -v "$icon_src_dir"/mri.ico "$icons_dir/" 2>/dev/null || true

    if [[ -f "$icons_dir/mri_256.png" ]]; then
      cp -v "$icons_dir/mri_256.png" "${assets_dir}/icon.png" || true
    elif [[ -f "$icons_dir/mri_128.png" ]]; then
      cp -v "$icons_dir/mri_128.png" "${assets_dir}/icon.png" || true
    fi
  else
    warn "Icon source dir not found: $icon_src_dir"
  fi

  if command -v patchelf >/dev/null 2>&1; then
    log "Setting RPATH on GUI binary to \$ORIGIN/../lib"
    patchelf --set-rpath '$ORIGIN/../lib' "${bin_dir}/${gui_basename}" || warn "patchelf failed for GUI"
  else
    warn "patchelf not found; run.sh will rely on LD_LIBRARY_PATH"
  fi

  log "Writing ldd diagnostics..."
  ldd "${bin_dir}/${gui_basename}" | tee "${diag_dir}/ldd_${gui_basename}.txt" >/dev/null || true
  for so in "${lib_dir}"/*.so*; do
    [[ -f "$so" ]] || continue
    base="$(basename "$so")"
    ldd "$so" | tee "${diag_dir}/ldd_${base}.txt" >/dev/null || true
  done

  cat > "${out_dir}/run.sh" <<INSTALL_RUN_EOF
#!/usr/bin/env bash
set -euo pipefail
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[\$(ts)][DBG] \$*"; }

HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
BIN="\${HERE}/bin"
LIB="\${HERE}/lib"

log "HERE=\$HERE"
log "BIN=\$BIN"
log "LIB=\$LIB"
log "ARCH=\$(uname -m)"

export LD_LIBRARY_PATH="\${LIB}:\${LD_LIBRARY_PATH:-}"
log "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH"

exec "\${BIN}/${gui_basename}" "\$@"
INSTALL_RUN_EOF
  chmod +x "${out_dir}/run.sh"

  cat > "${out_dir}/create_desktop_shortcut.sh" <<'DESKTOP_EOF'
#!/usr/bin/env bash
set -euo pipefail
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }

APPNAME="GlimpseMRI"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DESKTOP_DIR="${HOME}/Desktop"
if command -v xdg-user-dir >/dev/null 2>&1; then
  DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "${DESKTOP_DIR}")"
fi
mkdir -p "${DESKTOP_DIR}"

ICON_PATH="${BUNDLE_DIR}/assets/icon.png"
if [[ ! -f "${ICON_PATH}" ]]; then
  if [[ -f "${BUNDLE_DIR}/assets/icons/mri_256.png" ]]; then ICON_PATH="${BUNDLE_DIR}/assets/icons/mri_256.png"; fi
  if [[ -f "${BUNDLE_DIR}/assets/icons/mri_128.png" ]]; then ICON_PATH="${BUNDLE_DIR}/assets/icons/mri_128.png"; fi
fi
if [[ ! -f "${ICON_PATH}" ]]; then
  ICON_PATH="applications-science"
fi

DESK_FILE="${DESKTOP_DIR}/${APPNAME}.desktop"
log "Creating launcher: ${DESK_FILE}"

cat > "${DESK_FILE}" <<EOL
[Desktop Entry]
Type=Application
Version=1.0
Name=${APPNAME}
Comment=High-performance MRI reconstruction viewer
Path=${BUNDLE_DIR}
Exec=${BUNDLE_DIR}/run.sh
Icon=${ICON_PATH}
Terminal=false
Categories=Science;MedicalSoftware;
EOL

chmod +x "${DESK_FILE}"

if command -v gio >/dev/null 2>&1; then
  gio set "${DESK_FILE}" metadata::trusted true 2>/dev/null || true
fi

log "Done. If Ubuntu hides it, right-click -> Allow Launching."
DESKTOP_EOF
  chmod +x "${out_dir}/create_desktop_shortcut.sh"

  local runtime_src="${root}/scripts/targets/ubuntu_x86_64/prerequisites_run.sh"
  if [[ -n "$runtime_src_override" ]]; then
    runtime_src="$runtime_src_override"
  fi

  if [[ -f "$runtime_src" ]]; then
    cp -v "$runtime_src" "${out_dir}/install_deps.sh"
    chmod +x "${out_dir}/install_deps.sh"
  else
    cat > "${out_dir}/install_deps.sh" <<'MISSING_DEPS_EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "[ERR] Missing embedded runtime dependency installer."
exit 1
MISSING_DEPS_EOF
    chmod +x "${out_dir}/install_deps.sh"
  fi

  cat > "${out_dir}/install.sh" <<'INSTALL_EOF'
#!/usr/bin/env bash
set -euo pipefail
ts() { date +"%Y-%m-%d %H:%M:%S"; }
dbg() { echo "[$(ts)][DBG] $*"; }
wrn() { echo "[$(ts)][WRN] $*" >&2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_USER_DEST="${HOME}/glimpse_mri"
DEST="${DEFAULT_USER_DEST}"

if [[ "${1:-}" == "--prefix" && -n "${2:-}" ]]; then
  DEST="$2"
  shift 2
fi

dbg "Bundle source: ${HERE}"
dbg "Install destination: ${DEST}"

if [[ -e "${DEST}" ]]; then
  wrn "Destination already exists and will be replaced: ${DEST}"
  rm -rf "${DEST}"
fi

mkdir -p "$(dirname "${DEST}")"
cp -a "${HERE}" "${DEST}"

dbg "Install complete."
if [[ -f "${DEST}/install_deps.sh" ]]; then
  dbg "Optional runtime deps step: sudo ${DEST}/install_deps.sh"
fi
dbg "Run with: ${DEST}/run.sh"
INSTALL_EOF
  chmod +x "${out_dir}/install.sh"

  ensure_dir "$out_root"
  local tarball="${out_root}/${out_name}.tar.gz"

  log "Creating tarball: $tarball"
  tar -C "$out_root" -czf "$tarball" "$out_name"

  log "Writing SHA256..."
  (
    cd "$out_root"
    sha256sum "$(basename "$tarball")" | tee "$(basename "$tarball").sha256"
  )

  log "Bundle dir : $out_dir"
  log "Tarball    : $tarball"
}
