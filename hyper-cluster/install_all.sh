#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Hyper-Cluster Omega :: one-command installer
#
#   ./install_all.sh
#
# The wizard:
#   1. checks the platform (Linux, macOS, Termux on Android);
#   2. creates the virtualenv and installs the dependencies;
#   3. writes a complete `.env` (secrets included, generated when needed);
#   4. configures the Google Drive and the Hugging Face memory sync;
#   5. installs the auto-boot service (systemd on Linux, Termux on Android);
#   6. prints the QR code of the dashboard -- the phone scans it and the
#      Mission Control opens.
#
# It is idempotent and never destroys anything: an existing `.env` is
# backed up, every file is written atomically and a re-run only refreshes
# what is missing.
#
# Usage:
#   ./install_all.sh                    # the interactive wizard
#   ./install_all.sh --non-interactive  # take the defaults, never ask
#   ./install_all.sh --check            # validate, change nothing
#   ./install_all.sh --uninstall        # remove the service (keep the data)
#   ./install_all.sh --help
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_DIR="${HYPER_VENV:-${PROJECT_ROOT}/.venv}"
BIN_DIR="${VENV_DIR}/bin"
ENV_FILE="${HYPER_ENV_FILE:-${PROJECT_ROOT}/.env}"
ENV_BACKUP="${ENV_FILE}.backup.$(date +%Y%m%d%H%M%S)"
STATE_DIR="${HYPER_STATE_DIR:-${HOME}/.hypercluster}"
SERVICE_NAME="hyper-cluster"
MODE="install"
ASSUME_YES=0

# --- the defaults ----------------------------------------------------------
CLUSTER_NAME="hyper-cluster-omega"
HEAD_HOST="0.0.0.0"
HEAD_PORT="8080"
DASHBOARD_PORT="8082"
DATA_DIR="${STATE_DIR}"
PYTHON_BIN="${PYTHON:-python3}"

# --- the colours -----------------------------------------------------------
if [ -t 1 ] && [ "${NO_COLOR:-}" = "" ]; then
  C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'
  C_GREEN=$'\033[32m'; C_CYAN=$'\033[36m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
else
  C_RESET=""; C_DIM=""; C_BOLD=""; C_GREEN=""; C_CYAN=""; C_YELLOW=""; C_RED=""
fi

say()  { printf '%s\n' "$*"; }
step() { printf '\n%s▸ %s%s\n' "${C_BOLD}${C_CYAN}" "$*" "${C_RESET}"; }
ok()   { printf '  %s✓%s %s\n' "${C_GREEN}" "${C_RESET}" "$*"; }
warn() { printf '  %s!%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*"; }
fail() { printf '  %s✗%s %s\n' "${C_RED}" "${C_RESET}" "$*" >&2; }
dim()  { printf '  %s%s%s\n' "${C_DIM}" "$*" "${C_RESET}"; }

die() {
  fail "$*"
  exit 1
}

have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

# --------------------------------------------------------------------------- #
# The prompts
# --------------------------------------------------------------------------- #
ask() {
  # ask <variable> <question> <default>
  local __var="$1" __question="$2" __default="${3:-}" __answer=""
  if [ "${ASSUME_YES}" = "1" ]; then
    printf '%s' "${__default}"
    return 0
  fi
  if [ -n "${__default}" ]; then
    printf '%s [%s] ' "${__question}" "${__default}" >&2
  else
    printf '%s ' "${__question}" >&2
  fi
  read -r __answer || __answer=""
  if [ -z "${__answer}" ]; then
    __answer="${__default}"
  fi
  printf '%s' "${__answer}"
}

ask_yes() {
  # ask_yes <question> <default: y|n>
  local __question="$1" __default="${2:-y}" __answer=""
  if [ "${ASSUME_YES}" = "1" ]; then
    [ "${__default}" = "y" ] && return 0 || return 1
  fi
  local __hint="y/N"
  [ "${__default}" = "y" ] && __hint="Y/n"
  printf '%s [%s] ' "${__question}" "${__hint}" >&2
  read -r __answer || __answer=""
  [ -z "${__answer}" ] && __answer="${__default}"
  case "${__answer}" in
    [yY]*) return 0 ;;
    *) return 1 ;;
  esac
}

ask_secret() {
  # ask_secret <variable> <question>  (the answer is not echoed)
  local __var="$1" __question="$2" __answer=""
  if [ "${ASSUME_YES}" = "1" ]; then
    printf ''
    return 0
  fi
  printf '%s ' "${__question}" >&2
  if have stty; then
    local __saved
    __saved="$(stty -g 2>/dev/null || true)"
    stty -echo 2>/dev/null || true
    read -r __answer || __answer=""
    [ -n "${__saved}" ] && stty "${__saved}" 2>/dev/null || true
    printf '\n' >&2
  else
    read -r __answer || __answer=""
  fi
  printf '%s' "${__answer}"
}

random_secret() {
  # A 48 character hex secret (no external tool needed).
  if have openssl; then
    openssl rand -hex 24
  else
    LC_ALL=C tr -dc 'a-f0-9' </dev/urandom 2>/dev/null | head -c 48
  fi
}

lan_ip() {
  local ip=""
  if have ip; then
    ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')"
  fi
  if [ -z "${ip}" ] && have ifconfig; then
    ip="$(ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127\.' | head -n1)"
  fi
  if [ -z "${ip}" ] && have hostname; then
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  printf '%s' "${ip:-127.0.0.1}"
}

# --------------------------------------------------------------------------- #
# The environment file
# --------------------------------------------------------------------------- #
write_env() {
  # The `.env` is written atomically: a crash never leaves a half file.
  local tmp
  tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
  cat > "${tmp}" <<EOF
# ---------------------------------------------------------------------------
# Hyper-Cluster Omega -- generated by ${SCRIPT_NAME} on $(date -u '+%Y-%m-%dT%H:%M:%SZ')
# Keep this file private: it holds the keys of the cluster.
# ---------------------------------------------------------------------------

# --- identity --------------------------------------------------------------
HYPER_CLUSTER_NAME=${CLUSTER_NAME}

# --- control plane ---------------------------------------------------------
HYPER_HEAD_HOST=${HEAD_HOST}
HYPER_HEAD_PORT=${HEAD_PORT}
HYPER_PUBLIC_BASE_URL=${PUBLIC_BASE_URL}

# --- authentication --------------------------------------------------------
HYPER_API_KEY=${API_KEY}
HYPER_MESH_TOKEN=${MESH_TOKEN}
HYPER_ADMIN_TOKEN=${ADMIN_TOKEN}

# --- storage ---------------------------------------------------------------
HYPER_DATA_DIR=${DATA_DIR}

# --- mobile mission control (milestone 5) ----------------------------------
HYPER_DASHBOARD_ENABLED=1
HYPER_DASHBOARD_PORT=${DASHBOARD_PORT}
HYPER_DASHBOARD_PUBLIC_URL=${DASHBOARD_PUBLIC_URL}
HYPER_DASHBOARD_HEAD_URL=http://127.0.0.1:${HEAD_PORT}

# --- phone alerts ----------------------------------------------------------
HYPER_NOTIFIER_ENABLED=${NOTIFIER_ENABLED}
HYPER_NOTIFIER_TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
HYPER_NOTIFIER_TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
HYPER_NOTIFIER_WEBHOOK_URL=${WEBHOOK_URL}
HYPER_NOTIFIER_WEBHOOK_SECRET=${WEBHOOK_SECRET}
HYPER_NOTIFIER_EVENTS=${NOTIFIER_EVENTS}
HYPER_NOTIFIER_MIN_INTERVAL_SECONDS=${NOTIFIER_INTERVAL}

# --- memory (milestone 4) --------------------------------------------------
HYPER_MEMORY_ENABLED=1
HYPER_MEMORY_DIR=${MEMORY_DIR}

# --- memory cloud sync -----------------------------------------------------
HYPER_SYNC_ENABLED=${SYNC_ENABLED}
HYPER_DRIVE_TOKEN=${DRIVE_TOKEN}
HYPER_DRIVE_FOLDER=${DRIVE_FOLDER}
HYPER_HUB_TOKEN=${HUB_TOKEN}
HYPER_HUB_REPO=${HUB_REPO}

# --- sleep-time self-evolution ---------------------------------------------
HYPER_EVOLUTION_ENABLED=1
HYPER_EVOLUTION_IDLE_SECONDS=${EVOLUTION_IDLE}
HYPER_EVOLUTION_PREEMPT_BUDGET_SECONDS=2

# --- free inference providers (optional) ----------------------------------
GROQ_API_KEY=${GROQ_API_KEY}
CEREBRAS_API_KEY=${CEREBRAS_API_KEY}
SAMBANOVA_API_KEY=${SAMBANOVA_API_KEY}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
GOOGLE_AI_STUDIO_KEY=${GOOGLE_AI_STUDIO_KEY}
FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

# --- headless compute providers (optional) --------------------------------
KAGGLE_USERNAME=${KAGGLE_USERNAME}
KAGGLE_KEY=${KAGGLE_KEY}
HF_TOKEN=${HF_TOKEN}
GITHUB_TOKEN=${GITHUB_TOKEN}
HYPER_SPAWNER_ENABLED=${SPAWNER_ENABLED}
HYPER_SPAWNER_MIN_WORKERS=1

# --- observability ---------------------------------------------------------
HYPER_LOG_LEVEL=${LOG_LEVEL}
HYPER_LOG_JSON=1
EOF
  chmod 600 "${tmp}"
  mv -f "${tmp}" "${ENV_FILE}"
}

env_value() {
  # env_value <key> [file]
  local key="$1" file="${2:-${ENV_FILE}}" line=""
  [ -f "${file}" ] || return 0
  line="$(grep -E "^${key}=" "${file}" 2>/dev/null | tail -n1 || true)"
  printf '%s' "${line#*=}"
}

# --------------------------------------------------------------------------- #
# The steps
# --------------------------------------------------------------------------- #
detect_platform() {
  step "Platform"
  if [ -n "${TERMUX_VERSION:-}" ] || [ "${PREFIX:-}" = "/data/data/com.termux/files/usr" ]; then
    PLATFORM="termux"
    ok "Termux on Android"
  elif [ "$(uname -s)" = "Darwin" ]; then
    PLATFORM="macos"
    ok "macOS $(sw_vers -productVersion 2>/dev/null || echo '')"
  elif [ "$(uname -s)" = "Linux" ]; then
    PLATFORM="linux"
    ok "Linux $(uname -r)"
  else
    PLATFORM="unknown"
    warn "unknown platform: $(uname -s)"
  fi
  if have systemctl && [ "${PLATFORM}" = "linux" ]; then
    ok "systemd is available"
    HAS_SYSTEMD=1
  else
    HAS_SYSTEMD=0
    dim "no systemd -- the service will not be installed"
  fi
}

check_python() {
  step "Python"
  if ! have "${PYTHON_BIN}"; then
    fail "python3 was not found -- install it and run the wizard again"
    exit 1
  fi
  local version
  version="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
  ok "python ${version} (${PYTHON_BIN})"
  local major minor
  major="$(printf '%s' "${version}" | cut -d. -f1)"
  minor="$(printf '%s' "${version}" | cut -d. -f2)"
  if [ "${major}" -lt 3 ] || { [ "${major}" -eq 3 ] && [ "${minor}" -lt 10 ]; }; then
    die "python 3.10 or newer is required (found ${version})"
  fi
}

create_venv() {
  step "Virtualenv"
  if [ ! -x "${BIN_DIR}/python" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" || die "the virtualenv could not be created"
    ok "created ${VENV_DIR}"
  else
    dim "reusing ${VENV_DIR}"
  fi
  "${BIN_DIR}/python" -m pip install -q --upgrade pip setuptools wheel
  if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    "${BIN_DIR}/python" -m pip install -q -r "${PROJECT_ROOT}/requirements.txt" || die "the dependencies could not be installed"
  fi
  "${BIN_DIR}/python" -m pip install -q pytest pytest-asyncio ruff 2>/dev/null || true
  ok "the dependencies are installed"
}

configure_env() {
  step "Configuration"
  if [ -f "${ENV_FILE}" ]; then
    cp -p "${ENV_FILE}" "${ENV_BACKUP}"
    ok "the existing .env was backed up to $(basename "${ENV_BACKUP}")"
  fi

  API_KEY="$(env_value HYPER_API_KEY)"
  MESH_TOKEN="$(env_value HYPER_MESH_TOKEN)"
  ADMIN_TOKEN="$(env_value HYPER_ADMIN_TOKEN)"
  CLUSTER_NAME="$(env_value HYPER_CLUSTER_NAME)"
  HEAD_PORT="$(env_value HYPER_HEAD_PORT)"
  DASHBOARD_PORT="$(env_value HYPER_DASHBOARD_PORT)"
  DATA_DIR="$(env_value HYPER_DATA_DIR)"
  MEMORY_DIR="$(env_value HYPER_MEMORY_DIR)"
  PUBLIC_BASE_URL="$(env_value HYPER_PUBLIC_BASE_URL)"
  DASHBOARD_PUBLIC_URL="$(env_value HYPER_DASHBOARD_PUBLIC_URL)"
  NOTIFIER_ENABLED="$(env_value HYPER_NOTIFIER_ENABLED)"
  TELEGRAM_TOKEN="$(env_value HYPER_NOTIFIER_TELEGRAM_TOKEN)"
  TELEGRAM_CHAT_ID="$(env_value HYPER_NOTIFIER_TELEGRAM_CHAT_ID)"
  WEBHOOK_URL="$(env_value HYPER_NOTIFIER_WEBHOOK_URL)"
  WEBHOOK_SECRET="$(env_value HYPER_NOTIFIER_WEBHOOK_SECRET)"
  NOTIFIER_EVENTS="$(env_value HYPER_NOTIFIER_EVENTS)"
  NOTIFIER_INTERVAL="$(env_value HYPER_NOTIFIER_MIN_INTERVAL_SECONDS)"
  SYNC_ENABLED="$(env_value HYPER_SYNC_ENABLED)"
  DRIVE_TOKEN="$(env_value HYPER_DRIVE_TOKEN)"
  DRIVE_FOLDER="$(env_value HYPER_DRIVE_FOLDER)"
  HUB_TOKEN="$(env_value HYPER_HUB_TOKEN)"
  HUB_REPO="$(env_value HYPER_HUB_REPO)"
  EVOLUTION_IDLE="$(env_value HYPER_EVOLUTION_IDLE_SECONDS)"
  GROQ_API_KEY="$(env_value GROQ_API_KEY)"
  CEREBRAS_API_KEY="$(env_value CEREBRAS_API_KEY)"
  SAMBANOVA_API_KEY="$(env_value SAMBANOVA_API_KEY)"
  OPENROUTER_API_KEY="$(env_value OPENROUTER_API_KEY)"
  GOOGLE_AI_STUDIO_KEY="$(env_value GOOGLE_AI_STUDIO_KEY)"
  FIREWORKS_API_KEY="$(env_value FIREWORKS_API_KEY)"
  KAGGLE_USERNAME="$(env_value KAGGLE_USERNAME)"
  KAGGLE_KEY="$(env_value KAGGLE_KEY)"
  HF_TOKEN="$(env_value HF_TOKEN)"
  GITHUB_TOKEN="$(env_value GITHUB_TOKEN)"
  SPAWNER_ENABLED="$(env_value HYPER_SPAWNER_ENABLED)"
  LOG_LEVEL="$(env_value HYPER_LOG_LEVEL)"

  [ -z "${CLUSTER_NAME}" ] && CLUSTER_NAME="hyper-cluster-omega"
  [ -z "${HEAD_PORT}" ] && HEAD_PORT="8080"
  [ -z "${DASHBOARD_PORT}" ] && DASHBOARD_PORT="8082"
  [ -z "${DATA_DIR}" ] && DATA_DIR="${STATE_DIR}"
  [ -z "${MEMORY_DIR}" ] && MEMORY_DIR="${DATA_DIR}/memory"
  [ -z "${EVOLUTION_IDLE}" ] && EVOLUTION_IDLE="300"
  [ -z "${LOG_LEVEL}" ] && LOG_LEVEL="INFO"
  [ -z "${NOTIFIER_EVENTS}" ] && NOTIFIER_EVENTS="worker_down,worker_up,evolution_cycle,training_complete,cluster_degraded"
  [ -z "${NOTIFIER_INTERVAL}" ] && NOTIFIER_INTERVAL="0"
  [ -z "${DRIVE_FOLDER}" ] && DRIVE_FOLDER="Hyper-Cluster Memory"
  [ -z "${SPAWNER_ENABLED}" ] && SPAWNER_ENABLED="0"

  if [ "${ASSUME_YES}" != "1" ]; then
    say ""
    dim  "The wizard asks a handful of questions. Press Enter to accept the default."
    say ""
    CLUSTER_NAME="$(ask "Cluster name" "${CLUSTER_NAME}")"
    HEAD_PORT="$(ask "Head node port" "${HEAD_PORT}")"
    DASHBOARD_PORT="$(ask "Mission Control port" "${DASHBOARD_PORT}")"

    if [ -z "${API_KEY}" ]; then
      API_KEY="$(random_secret)"
      dim  "A fresh API key was generated for the phone."
    fi
    if [ -z "${MESH_TOKEN}" ]; then
      MESH_TOKEN="$(random_secret)"
    fi
    if [ -z "${ADMIN_TOKEN}" ]; then
      ADMIN_TOKEN="$(random_secret)"
    fi

    say ""
    dim  "Provider keys (optional -- the cluster works without them)."
    GROQ_API_KEY="$(ask "Groq key" "${GROQ_API_KEY}")"
    CEREBRAS_API_KEY="$(ask "Cerebras key" "${CEREBRAS_API_KEY}")"
    OPENROUTER_API_KEY="$(ask "OpenRouter key" "${OPENROUTER_API_KEY}")"
    GOOGLE_AI_STUDIO_KEY="$(ask "Google AI Studio key" "${GOOGLE_AI_STUDIO_KEY}")"
  else
    [ -z "${API_KEY}" ] && API_KEY="$(random_secret)"
    [ -z "${MESH_TOKEN}" ] && MESH_TOKEN="$(random_secret)"
    [ -z "${ADMIN_TOKEN}" ] && ADMIN_TOKEN="$(random_secret)"
  fi

  # the public address of the dashboard (the phone needs it)
  local ip
  ip="$(lan_ip)"
  local default_dash="http://${ip}:${DASHBOARD_PORT}"
  DASHBOARD_PUBLIC_URL="${DASHBOARD_PUBLIC_URL:-${default_dash}}"
  PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-http://${ip}:${HEAD_PORT}}"
  if [ "${ASSUME_YES}" != "1" ]; then
    DASHBOARD_PUBLIC_URL="$(ask "Address the phone uses" "${DASHBOARD_PUBLIC_URL}")"
  fi

  write_env
  ok ".env written (mode 600)"
}

configure_sync() {
  step "Memory sync (Google Drive / Hugging Face)"
  if [ "${ASSUME_YES}" != "1" ]; then
    if ask_yes "Mirror the memory to Google Drive?" "${SYNC_ENABLED:-n}"; then
      SYNC_ENABLED=1
    else
      SYNC_ENABLED=0
    fi
    if [ "${SYNC_ENABLED}" = "1" ]; then
      DRIVE_TOKEN="$(ask_secret "Drive token" "Google Drive access token: ")"
      DRIVE_FOLDER="$(ask "Drive folder" "${DRIVE_FOLDER}")"
    fi
    if ask_yes "Mirror the memory to the Hugging Face Hub?" "${SYNC_ENABLED:-n}"; then
      HUB_ENABLED=1
    else
      HUB_ENABLED=0
    fi
    if [ "${HUB_ENABLED}" = "1" ]; then
      HUB_TOKEN="$(ask_secret "Hub token" "Hugging Face token (write): ")"
      HUB_REPO="$(ask "Hub repository" "${HUB_REPO:-$(id -un)/hyper-cluster-memory}")"
    fi
  fi
  [ "${HUB_ENABLED:-0}" = "1" ] && SYNC_ENABLED=1
  if [ "${SYNC_ENABLED}" = "1" ]; then
    ok "the memory snapshots are mirrored"
  else
    dim "the memory stays on this machine"
  fi
}

configure_notifier() {
  step "Phone alerts"
  if [ "${ASSUME_YES}" != "1" ]; then
    if ask_yes "Push the cluster events to the phone?" "${NOTIFIER_ENABLED:-n}"; then
      NOTIFIER_ENABLED=1
    else
      NOTIFIER_ENABLED=0
    fi
    if [ "${NOTIFIER_ENABLED}" = "1" ]; then
      TELEGRAM_TOKEN="$(ask_secret "Telegram token" "Telegram bot token: ")"
      TELEGRAM_CHAT_ID="$(ask "Telegram chat id" "${TELEGRAM_CHAT_ID}")"
      WEBHOOK_URL="$(ask "Webhook URL (ntfy, a relay, ...)" "${WEBHOOK_URL}")"
      WEBHOOK_SECRET="$(ask_secret "Webhook secret" "Webhook secret: ")"
    fi
  fi
  if [ "${NOTIFIER_ENABLED}" = "1" ]; then
    ok "Telegram and webhook alerts are enabled"
  else
    dim "no phone alerts"
  fi
}

install_service() {
  step "Auto-boot service"
  install_launcher
  case "${PLATFORM}" in
    termux)
      install_termux_service
      ;;
    linux)
      if [ "${HAS_SYSTEMD}" = "1" ]; then
        install_systemd_service
      else
        ok "the launcher is enough: ${LAUNCHER_SCRIPT}"
      fi
      ;;
    *)
      ok "the launcher is enough: ${LAUNCHER_SCRIPT}"
      ;;
  esac
}

install_launcher() {
  # The launcher lives in the repository: one command starts the control
  # plane *and* the Mission Control (one address for the phone).
  LAUNCHER_SCRIPT="${PROJECT_ROOT}/run_hyper_cluster.sh"
  chmod +x "${LAUNCHER_SCRIPT}" 2>/dev/null || true
  ok "the launcher is ready (${LAUNCHER_SCRIPT})"
}

install_systemd_service() {
  local unit="/etc/systemd/system/${SERVICE_NAME}.service"
  local tmp
  tmp="$(mktemp)"
  cat > "${tmp}" <<EOF
[Unit]
Description=Hyper-Cluster Omega (head node + Mission Control)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_ROOT}
Environment=HYPER_ENV_FILE=${ENV_FILE}
ExecStart=${LAUNCHER_SCRIPT}
Restart=always
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF
  if [ "$(id -u)" = "0" ]; then
    install -m 644 "${tmp}" "${unit}"
  elif have sudo; then
    sudo install -m 644 "${tmp}" "${unit}"
  else
    warn "no root and no sudo -- the service was written to ${PROJECT_ROOT}/${SERVICE_NAME}.service"
    install -m 644 "${tmp}" "${PROJECT_ROOT}/${SERVICE_NAME}.service"
    return 0
  fi
  rm -f "${tmp}"
  if have sudo && [ "$(id -u)" != "0" ]; then
    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
    sudo systemctl restart "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
  else
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
    systemctl restart "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
  fi
  ok "systemd unit installed (${SERVICE_NAME}.service)"
}

install_termux_service() {
  local boot_dir="${HOME}/.termux/boot"
  local script="${boot_dir}/start-hyper-cluster.sh"
  mkdir -p "${boot_dir}"
  local tmp
  tmp="$(mktemp)"
  cat > "${tmp}" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
# Hyper-Cluster Omega -- started by Termux:BOOT
exec "${LAUNCHER_SCRIPT}"
EOF
  chmod +x "${tmp}"
  mv -f "${tmp}" "${script}"
  ok "the Termux boot script is installed (${script})"
  dim  "install the Termux:BOOT app to start the cluster with the phone"
}

install_cli() {
  step "The phone command line"
  local target="${BIN_DIR}/hc"
  local tmp
  tmp="$(mktemp)"
  cat > "${tmp}" <<EOF
#!/usr/bin/env bash
# Hyper-Cluster Omega -- the Termux/CLI entry point
export PYTHONPATH="${PROJECT_ROOT}\${PYTHONPATH:+:\${PYTHONPATH}}"
exec "${BIN_DIR}/python" -m mobile.cli "\$@"
EOF
  chmod +x "${tmp}"
  mv -f "${tmp}" "${target}"
  ok "the 'hc' command is installed (${target})"
}

print_qr() {
  step "Mission Control"
  local url="${DASHBOARD_PUBLIC_URL}"
  if [ -z "${url}" ]; then
    url="http://$(lan_ip):${DASHBOARD_PORT}"
  fi
  HYPER_PROJECT_ROOT="${PROJECT_ROOT}" HYPER_QR_URL="${url}" "${BIN_DIR}/python" - <<'PY' || warn "the QR code could not be drawn"
import os
import sys

sys.path.insert(0, os.environ.get("HYPER_PROJECT_ROOT", os.getcwd()))
from mobile.qr import render_terminal

url = os.environ.get("HYPER_QR_URL", "")
print(render_terminal(url))
PY
  printf '\n  %sOpen this address on the phone:%s\n' "${C_BOLD}" "${C_RESET}"
  printf '    %s%s%s\n' "${C_CYAN}" "${url}" "${C_RESET}"
  printf '  %sScan the code above with the camera of the phone.%s\n' "${C_DIM}" "${C_RESET}"
}

verify() {
  step "Verification"
  if [ ! -x "${BIN_DIR}/python" ]; then
    fail "the virtualenv is missing"
    return 1
  fi
  ( cd "${PROJECT_ROOT}" && HYPER_ENV_FILE="${ENV_FILE}" "${BIN_DIR}/python" -m head_node.serve --check ) \
    && ok "the configuration is valid" \
    || warn "the configuration reported a problem (see above)"
  ( cd "${PROJECT_ROOT}" && HYPER_ENV_FILE="${ENV_FILE}" "${BIN_DIR}/python" -c "import mobile.dashboard, mobile.cli, mobile.qr, mobile.notifier" ) \
    && ok "the mobile layer imports" \
    || fail "the mobile layer does not import"
}

print_summary() {
  local ip
  ip="$(lan_ip)"
  cat <<EOF

${C_BOLD}${C_GREEN}Hyper-Cluster Omega is installed.${C_RESET}

  ${C_BOLD}Launch${C_RESET}
    cd ${PROJECT_ROOT} && ./run_hyper_cluster.sh

  ${C_BOLD}Mission Control${C_RESET}
    http://${ip}:${DASHBOARD_PORT}      (scan the QR code above)

  ${C_BOLD}The phone command line${C_RESET}
    ${BIN_DIR}/hc status
    ${BIN_DIR}/hc chat "what can you do?"

  ${C_BOLD}The head node${C_RESET}
    http://${ip}:${HEAD_PORT}/v1/status

EOF
}

# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #
main() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --non-interactive|--yes|-y) ASSUME_YES=1 ;;
      --check) MODE="check" ;;
      --uninstall) MODE="uninstall" ;;
      --help|-h) usage ;;
      *) die "unknown option: $1 (try --help)" ;;
    esac
    shift
  done

  printf '%s\n' "${C_BOLD}Hyper-Cluster Omega :: installer${C_RESET}"

  if [ "${MODE}" = "uninstall" ]; then
    if [ "${HAS_SYSTEMD:-0}" = "1" ] && have systemctl; then
      sudo systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
      rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
      ok "the service was removed"
    fi
    rm -f "${HOME}/.termux/boot/start-hyper-cluster.sh"
    ok "the Termux boot script was removed"
    exit 0
  fi

  detect_platform
  check_python

  if [ "${MODE}" = "check" ]; then
    verify
    exit 0
  fi

  create_venv
  configure_env
  configure_sync
  configure_notifier
  write_env
  install_service
  install_cli
  verify
  print_qr
  print_summary
}

main "$@"
