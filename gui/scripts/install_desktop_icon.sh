#!/usr/bin/env bash
set -euo pipefail

APP_NAME="GlimpseMRI"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GUI_BIN="${REPO_ROOT}/build_gui_Release/glimpseMRI"
ICON_PNG="${REPO_ROOT}/gui/assets/images/icons/mri_128.png"

DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[[ -n "${DESKTOP_DIR}" ]] || DESKTOP_DIR="${HOME}/Desktop"

APP_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"

DESKTOP_FILE="${DESKTOP_DIR}/${APP_NAME}.desktop"
APP_FILE="${APP_DIR}/${APP_NAME}.desktop"

echo "[DBG] REPO_ROOT=${REPO_ROOT}"
echo "[DBG] GUI_BIN=${GUI_BIN}"
echo "[DBG] ICON_PNG=${ICON_PNG}"
echo "[DBG] DESKTOP_DIR=${DESKTOP_DIR}"
echo "[DBG] APP_DIR=${APP_DIR}"
echo "[DBG] DESKTOP_FILE=${DESKTOP_FILE}"
echo "[DBG] APP_FILE=${APP_FILE}"

if [[ ! -x "${GUI_BIN}" ]]; then
  echo "[ERR] GUI binary not found/executable: ${GUI_BIN}"
  echo "[ERR] Build first: ./build_jetson.shh --clean"
  exit 1
fi

if [[ ! -f "${ICON_PNG}" ]]; then
  echo "[ERR] Icon not found: ${ICON_PNG}"
  exit 1
fi

mkdir -p "${DESKTOP_DIR}" "${APP_DIR}"

cat > "${DESKTOP_FILE}" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=${APP_NAME}
Comment=Glimpse MRI GUI (Jetson)
Exec=${GUI_BIN}
Path=${REPO_ROOT}
Icon=${ICON_PNG}
Terminal=false
Categories=Science;MedicalSoftware;
StartupNotify=true
DESKTOP

chmod +x "${DESKTOP_FILE}"
cp -f "${DESKTOP_FILE}" "${APP_FILE}"

# Try to mark trusted on GNOME (ignore if unsupported)
if command -v gio >/dev/null 2>&1; then
  gio set "${DESKTOP_FILE}" metadata::trusted true >/dev/null 2>&1 || true
fi

# Refresh menu db if available
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${APP_DIR}" >/dev/null 2>&1 || true
fi

echo "[DBG] Created launchers:"
echo "  - ${DESKTOP_FILE}"
echo "  - ${APP_FILE}"
echo "[DBG] If the Desktop icon won’t launch: right-click → 'Allow Launching'."
