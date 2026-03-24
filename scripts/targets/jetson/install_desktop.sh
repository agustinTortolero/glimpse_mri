#!/usr/bin/env bash
set -euo pipefail

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)][DBG] $*"; }
warn() { echo "[$(ts)][WRN] $*" >&2; }
err() { echo "[$(ts)][ERR] $*" >&2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${HERE}/.." && pwd)"
APP_NAME="${APP_NAME:-GlimpseMRI}"
EXEC_PATH="${BUNDLE_DIR}/run.sh"
ICON_SOURCE="${BUNDLE_DIR}/icons/mri_256.png"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"
DESKTOP_FILE="${DESKTOP_DIR}/glimpsemri.desktop"
INSTALLED_ICON="${ICON_DIR}/glimpsemri.png"

log "HERE=${HERE}"
log "BUNDLE_DIR=${BUNDLE_DIR}"
log "DESKTOP_FILE=${DESKTOP_FILE}"
log "ICON_SOURCE=${ICON_SOURCE}"

if [[ ! -x "${EXEC_PATH}" ]]; then
    err "Expected launcher not found or not executable: ${EXEC_PATH}"
    exit 1
fi

mkdir -p "${DESKTOP_DIR}"
log "Ensured desktop entry directory exists"

if [[ -f "${ICON_SOURCE}" ]]; then
    mkdir -p "${ICON_DIR}"
    cp -av "${ICON_SOURCE}" "${INSTALLED_ICON}"
    log "Installed icon to ${INSTALLED_ICON}"
    ICON_FIELD="${INSTALLED_ICON}"
else
    warn "No staged icon found at ${ICON_SOURCE}; desktop entry will use a generic icon"
    ICON_FIELD="utilities-terminal"
fi

log "Writing desktop entry"
cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=${APP_NAME}
Comment=Launch the GlimpseMRI Jetson bundle
Exec=${EXEC_PATH}
Icon=${ICON_FIELD}
Terminal=false
Categories=Science;MedicalSoftware;Viewer;
StartupNotify=true
EOF

chmod 644 "${DESKTOP_FILE}"
log "Desktop entry written: ${DESKTOP_FILE}"

if command -v update-desktop-database >/dev/null 2>&1; then
    log "Refreshing desktop database"
    update-desktop-database "${DESKTOP_DIR}" || warn "update-desktop-database failed"
else
    log "update-desktop-database not found; skipping refresh"
fi

log "Desktop integration helper completed successfully"
