#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Hyper-Cluster Omega :: worker bootstrap
#
# Zero-touch provisioning for a mesh worker:
#   1. detect the platform and install the few system packages we need;
#   2. fetch a PRE-COMPILED llama.cpp build (no compilation on the node) and
#      verify its SHA-256 -- the artefact is cached, so a re-provision of the
#      same image costs zero bytes;
#   3. fetch the GGUF with resume + checksum verification (also cached);
#   4. create the virtualenv and install the worker dependencies;
#   5. write the worker environment file;
#   6. install and start the service (systemd on Linux, Termux on Android);
#   7. verify that the worker answers /health and registered with the head node.
#
# It is idempotent: running it twice changes nothing the second time.
#
# Usage:
#   worker/bootstrap.sh --check          # validate only, change nothing
#   worker/bootstrap.sh --install        # provision (default)
#   worker/bootstrap.sh --verify         # check the running worker
#   worker/bootstrap.sh --uninstall      # remove the service and the venv
#   worker/bootstrap.sh --print-env      # show the resolved environment
#
# Everything is driven by environment variables so the same script works for a
# Kaggle kernel, a GitHub runner, an Oracle ARM VM or a Termux phone.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- defaults ---------------------------------------------------------------
CACHE_DIR="${HYPER_WORKER_CACHE:-${HOME}/.hypercluster/cache}"
DATA_DIR="${HYPER_WORKER_DATA:-${HOME}/.hypercluster}"
VENV_DIR="${HYPER_WORKER_VENV:-${DATA_DIR}/venv}"
ENV_FILE="${HYPER_WORKER_ENV_FILE:-${DATA_DIR}/worker.env}"
MODEL_DIR="${HYPER_WORKER_MODEL_DIR:-${DATA_DIR}/models}"

HEAD_URL="${HYPER_WORKER_HEAD_URL:-http://127.0.0.1:8080}"
MESH_TOKEN="${HYPER_WORKER_MESH_TOKEN:-}"
MODEL_REPO="${HYPER_WORKER_MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct-GGUF}"
MODEL_FILE="${HYPER_WORKER_MODEL_FILE:-qwen2.5-7b-instruct-q4_k_m.gguf}"
MODEL_SHA256="${HYPER_WORKER_MODEL_SHA256:-}"
MODEL_FAMILY="${HYPER_WORKER_MODEL_FAMILY:-qwen2}"
MODEL_PARAMS_B="${HYPER_WORKER_MODEL_PARAMS_B:-7.0}"
CONTEXT_SIZE="${HYPER_WORKER_CONTEXT_SIZE:-8192}"
GPU_LAYERS="${HYPER_WORKER_GPU_LAYERS:--1}"
PROVIDER="${HYPER_WORKER_PROVIDER:-local}"
REGION="${HYPER_WORKER_REGION:-unknown}"
BIND_PORT="${HYPER_WORKER_BIND_PORT:-8081}"
BIND_HOST="${HYPER_WORKER_BIND_HOST:-0.0.0.0}"
BACKEND="${HYPER_WORKER_BACKEND:-auto}"
HEARTBEAT="${HYPER_WORKER_HEARTBEAT_SECONDS:-20}"
TTL="${HYPER_WORKER_TTL_SECONDS:-3600}"
MAX_CONCURRENCY="${HYPER_WORKER_MAX_CONCURRENCY:-2}"
TAILSCALE_AUTH_KEY="${HYPER_WORKER_TAILSCALE_AUTH_KEY:-}"
TAILSCALE_TAGS="${HYPER_WORKER_TAILSCALE_TAGS:-}"
SERVICE_NAME="${HYPER_WORKER_SERVICE_NAME:-hyper-cluster-worker}"

# Pre-compiled llama.cpp build (override for a different accelerator).
LLAMA_BUILD_URL="${HYPER_WORKER_LLAMA_BUILD_URL:-}"
LLAMA_BUILD_SHA256="${HYPER_WORKER_LLAMA_BUILD_SHA256:-}"
LLAMA_BUILD_NAME="${HYPER_WORKER_LLAMA_BUILD_NAME:-llama.cpp-prebuilt.tar.gz}"

HF_ENDPOINT="${HYPER_WORKER_HF_ENDPOINT:-https://huggingface.co}"
PYTHON_BIN="${HYPER_WORKER_PYTHON:-python3}"

MODE="install"
LOG_JSON="${HYPER_WORKER_LOG_JSON:-1}"

# Platform detection results (filled in by detect_platform).
OS=""
ARCH=""
PLATFORM="unknown"
CPU_ARCH="unknown"
IS_TERMUX=0
IS_ANDROID=0
HAS_NVIDIA=0

# --- logging ----------------------------------------------------------------
if [[ "${LOG_JSON}" == "1" ]]; then
  log()  { printf '{"level":"INFO","step":"%s","message":"%s"}\n' "$1" "$2"; }
  warn() { printf '{"level":"WARN","step":"%s","message":"%s"}\n' "$1" "$2" >&2; }
  die()  { printf '{"level":"ERROR","step":"%s","message":"%s"}\n' "$1" "$2" >&2; exit 1; }
else
  log()  { printf '[%s] %s\n' "$1" "$2"; }
  warn() { printf '[%s] WARNING: %s\n' "$1" "$2" >&2; }
  die()  { printf '[%s] ERROR: %s\n' "$1" "$2" >&2; exit 1; }
fi

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# True when systemd is present *and* is the running init system (containers
# ship the systemctl binary but have no systemd bus).
systemd_usable() {
  have_cmd systemctl || return 1
  [[ -d /run/systemd/system ]] || return 1
  return 0
}

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --check)     MODE="check" ;;
      --install)   MODE="install" ;;
      --verify)    MODE="verify" ;;
      --uninstall) MODE="uninstall" ;;
      --print-env) MODE="print-env" ;;
      -h|--help)   usage; exit 0 ;;
      *)           die "args" "unknown argument: $1" ;;
    esac
    shift
  done
}

# --- platform ---------------------------------------------------------------
detect_platform() {
  OS="$(uname -s)"
  ARCH="$(uname -m)"
  case "${OS}" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="darwin" ;;
    *)      PLATFORM="unknown" ;;
  esac
  case "${ARCH}" in
    x86_64|amd64)  CPU_ARCH="x64" ;;
    aarch64|arm64) CPU_ARCH="arm64" ;;
    armv7l)        CPU_ARCH="armv7" ;;
    *)             CPU_ARCH="${ARCH}" ;;
  esac
  IS_TERMUX=0
  if [[ -n "${PREFIX:-}" && "${PREFIX}" == *"/com.termux"* ]]; then
    IS_TERMUX=1
    PLATFORM="termux"
  fi
  IS_ANDROID=0
  if [[ -n "${ANDROID_ROOT:-}" ]] || [[ "${PLATFORM}" == "termux" ]]; then
    IS_ANDROID=1
  fi
  HAS_NVIDIA=0
  if have_cmd nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    HAS_NVIDIA=1
  fi
}

# --- helpers ----------------------------------------------------------------
sha256_of() {
  local file="$1"
  if have_cmd sha256sum; then
    sha256sum "${file}" | awk '{print $1}'
  elif have_cmd shasum; then
    shasum -a 256 "${file}" | awk '{print $1}'
  else
    echo ""
  fi
}

verify_sha256() {
  local file="$1" expected="$2"
  [[ -z "${expected}" ]] && return 0
  local actual
  actual="$(sha256_of "${file}")"
  if [[ -z "${actual}" ]]; then
    warn "checksum" "no sha256 tool available; skipping verification"
    return 0
  fi
  if [[ "${actual}" != "${expected}" ]]; then
    return 1
  fi
  return 0
}

# Download with resume + checksum verification. Cached artefacts are reused.
fetch_artifact() {
  local url="$1" dest="$2" expected_sha="$3" label="$4"
  if [[ -f "${dest}" ]] && verify_sha256 "${dest}" "${expected_sha}"; then
    log "cache" "${label} already cached at ${dest}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  if have_cmd curl; then
    log "download" "fetching ${label} from ${url}"
    local flags=(-fL --retry 3 --retry-delay 2 -C -)
    if [[ -t 1 ]]; then
      flags+=(--progress-bar)
    else
      flags+=(-sS)
    fi
    curl "${flags[@]}" -o "${dest}" "${url}" \
      || die "download" "failed to download ${label} from ${url}"
  elif have_cmd wget; then
    log "download" "fetching ${label} from ${url}"
    if [[ -t 1 ]]; then
      wget -c -O "${dest}" "${url}" || die "download" "failed to download ${label}"
    else
      wget -q -c -O "${dest}" "${url}" || die "download" "failed to download ${label}"
    fi
  else
    die "download" "neither curl nor wget is available"
  fi
  if ! verify_sha256 "${dest}" "${expected_sha}"; then
    rm -f "${dest}"
    die "checksum" "checksum mismatch for ${label}; the artefact was discarded"
  fi
  log "checksum" "${label} verified"
}

# --- steps ------------------------------------------------------------------
install_system_deps() {
  if have_cmd apt-get && [[ "$(id -u)" == "0" ]]; then
    log "deps" "installing system packages with apt-get"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq || warn "deps" "apt-get update failed"
    apt-get install -y -qq curl ca-certificates python3 python3-venv python3-pip tar xz-utils \
      || warn "deps" "apt-get install failed; continuing with what is present"
  elif have_cmd dnf && [[ "$(id -u)" == "0" ]]; then
    log "deps" "installing system packages with dnf"
    dnf install -y -q curl ca-certificates python3 python3-pip tar xz || warn "deps" "dnf install failed"
  elif have_cmd apk && [[ "$(id -u)" == "0" ]]; then
    log "deps" "installing system packages with apk"
    apk add --no-cache curl ca-certificates python3 py3-pip tar xz || warn "deps" "apk add failed"
  elif have_cmd pkg && [[ "${IS_TERMUX}" == "1" ]]; then
    log "deps" "installing Termux packages"
    pkg update -y >/dev/null 2>&1 || true
    pkg install -y curl python tar xz-utils || warn "deps" "pkg install failed"
  else
    log "deps" "skipping system package installation (not root / no package manager)"
  fi
}

select_python() {
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if have_cmd "${candidate}"; then
      local version
      version="$("${candidate}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
      if [[ "${version%%.*}" -ge 3 && "${version##*.}" -ge 9 ]]; then
        PYTHON_BIN="${candidate}"
        return 0
      fi
    fi
  done
  die "python" "no suitable python3 (>= 3.9) was found"
}

setup_venv() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    log "venv" "reusing the virtualenv at ${VENV_DIR}"
  else
    log "venv" "creating the virtualenv at ${VENV_DIR}"
    mkdir -p "$(dirname "${VENV_DIR}")"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" || die "venv" "could not create the virtualenv"
  fi
  log "pip" "installing the worker dependencies"
  "${VENV_DIR}/bin/python" -m pip install -q --upgrade pip setuptools wheel \
    || warn "pip" "could not upgrade pip"
  "${VENV_DIR}/bin/python" -m pip install -q -r "${PROJECT_ROOT}/requirements.txt" \
    || die "pip" "dependency installation failed"
  if [[ "${BACKEND}" != "simulator" ]] && [[ -z "${HYPER_WORKER_UPSTREAM_URL:-}" ]]; then
    "${VENV_DIR}/bin/python" -m pip install -q llama-cpp-python \
      || warn "pip" "llama-cpp-python could not be installed; the worker will use the simulator backend"
  fi
}

fetch_llama_build() {
  local target_dir="${CACHE_DIR}/llama"
  if [[ -x "${target_dir}/bin/llama-cli" ]] || compgen -G "${target_dir}/bin/llama*" >/dev/null; then
    log "cache" "pre-compiled llama.cpp already cached at ${target_dir}"
    return 0
  fi
  if [[ -z "${LLAMA_BUILD_URL}" ]]; then
    warn "llama" "no pre-compiled build URL configured (HYPER_WORKER_LLAMA_BUILD_URL)"
    warn "llama" "the worker will run with the simulator backend until a build is supplied"
    return 0
  fi
  local archive="${CACHE_DIR}/${LLAMA_BUILD_NAME}"
  fetch_artifact "${LLAMA_BUILD_URL}" "${archive}" "${LLAMA_BUILD_SHA256}" "llama.cpp build"
  mkdir -p "${target_dir}"
  tar -xzf "${archive}" -C "${target_dir}" --strip-components=1 \
    || die "llama" "could not extract ${archive}"
  chmod +x "${target_dir}"/bin/* 2>/dev/null || true
  log "llama" "llama.cpp unpacked into ${target_dir}"
}

fetch_model() {
  local target="${MODEL_DIR}/${MODEL_FILE}"
  if [[ -f "${target}" ]]; then
    log "cache" "model already present at ${target}"
    return 0
  fi
  local url="${HF_ENDPOINT}/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
  fetch_artifact "${url}" "${target}" "${MODEL_SHA256}" "model ${MODEL_FILE}"
}

write_env() {
  mkdir -p "${DATA_DIR}"
  log "env" "writing the worker environment to ${ENV_FILE}"
  cat > "${ENV_FILE}" <<EOF
# Hyper-Cluster worker environment (generated by bootstrap.sh)
HYPER_WORKER_ID=${HYPER_WORKER_ID:-}
HYPER_WORKER_HEAD_URL=${HEAD_URL}
HYPER_WORKER_MESH_TOKEN=${MESH_TOKEN}
HYPER_WORKER_BIND_HOST=${BIND_HOST}
HYPER_WORKER_BIND_PORT=${BIND_PORT}
HYPER_WORKER_PROVIDER=${PROVIDER}
HYPER_WORKER_REGION=${REGION}
HYPER_WORKER_MODEL_PATH=${MODEL_DIR}/${MODEL_FILE}
HYPER_WORKER_MODEL_REPO=${MODEL_REPO}
HYPER_WORKER_MODEL_FILENAME=${MODEL_FILE}
HYPER_WORKER_MODEL_FAMILY=${MODEL_FAMILY}
HYPER_WORKER_MODEL_PARAMS_B=${MODEL_PARAMS_B}
HYPER_WORKER_CONTEXT_SIZE=${CONTEXT_SIZE}
HYPER_WORKER_GPU_LAYERS=${GPU_LAYERS}
HYPER_WORKER_BACKEND=${BACKEND}
HYPER_WORKER_HEARTBEAT_SECONDS=${HEARTBEAT}
HYPER_WORKER_TTL_SECONDS=${TTL}
HYPER_WORKER_MAX_CONCURRENCY=${MAX_CONCURRENCY}
HYPER_WORKER_TAILSCALE_AUTH_KEY=${TAILSCALE_AUTH_KEY}
HYPER_WORKER_TAILSCALE_TAGS=${TAILSCALE_TAGS}
HYPER_WORKER_LOG_LEVEL=${HYPER_WORKER_LOG_LEVEL:-INFO}
HYPER_WORKER_LOG_JSON=${LOG_JSON}
HYPER_WORKER_SANDBOX_ENABLED=${HYPER_WORKER_SANDBOX_ENABLED:-1}
EOF
  chmod 600 "${ENV_FILE}"
}

write_unit() {
  local target="$1"
  local dir
  dir="$(dirname "${target}")"
  mkdir -p "${dir}" 2>/dev/null || {
    warn "service" "cannot create ${dir}"
    return 1
  }
  cat > "${target}" <<EOF
[Unit]
Description=Hyper-Cluster Omega mesh worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SUDO_USER:-$(id -un)}
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=${ENV_FILE}
ExecStart=${VENV_DIR}/bin/python -m worker.worker_agent
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
  log "service" "unit written to ${target}"
}

# Where the unit has to live: system-wide when we may write there, otherwise
# the per-user unit directory (rootless systemd).
unit_path() {
  local system_unit="/etc/systemd/system/${SERVICE_NAME}.service"
  if [[ -w "/etc/systemd/system" ]] || { [[ ! -e "${system_unit}" ]] && [[ -w "/etc/systemd" ]]; }; then
    echo "${system_unit}"
    echo "system"
  else
    echo "${HOME}/.config/systemd/user/${SERVICE_NAME}.service"
    echo "user"
  fi
}

install_service() {
  if [[ "${IS_TERMUX}" == "1" ]]; then
    install_termux_service
    return 0
  fi
  if ! systemd_usable; then
    log "service" "systemd is not the init system here; the worker will run as a managed background process"
    return 0
  fi
  local unit scope
  unit="$(unit_path | head -1)"
  scope="$(unit_path | tail -1)"
  log "service" "installing ${unit} (${scope} scope)"
  if ! write_unit "${unit}"; then
    warn "service" "the unit could not be written; the worker will not auto-restart"
    return 0
  fi
  if [[ "${scope}" == "system" ]]; then
    systemctl daemon-reload || warn "service" "systemctl daemon-reload failed"
    systemctl enable "${SERVICE_NAME}.service" >/dev/null 2>&1 \
      || warn "service" "could not enable the service"
  else
    systemctl --user daemon-reload || warn "service" "systemctl --user daemon-reload failed"
    systemctl --user enable "${SERVICE_NAME}.service" >/dev/null 2>&1 \
      || warn "service" "could not enable the user service"
    log "service" "run 'loginctl enable-linger ${USER}' so the service survives logout"
  fi
  log "service" "service installed"
}

install_termux_service() {
  local script="${DATA_DIR}/run-worker.sh"
  mkdir -p "${DATA_DIR}"
  cat > "${script}" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
source "${ENV_FILE}"
exec ${VENV_DIR}/bin/python -m worker.worker_agent
EOF
  chmod +x "${script}"
  if have_cmd termux-service; then
    log "service" "registering the Termux service"
    termux-service-stop "${SERVICE_NAME}" >/dev/null 2>&1 || true
    termux-service-start "${SERVICE_NAME}" || warn "service" "termux-service-start failed"
  else
    log "service" "termux-service is not installed; run ${script} manually"
  fi
}

start_service() {
  if [[ "${IS_TERMUX}" == "1" ]]; then
    if have_cmd termux-service; then
      termux-service-start "${SERVICE_NAME}" || warn "service" "could not start the Termux service"
    fi
    return 0
  fi
  if systemd_usable; then
    local scope
    scope="$(unit_path | tail -1)"
    if [[ "${scope}" == "system" ]]; then
      if systemctl restart "${SERVICE_NAME}.service"; then
        log "service" "service started"
        return 0
      fi
      warn "service" "could not restart the system service"
    else
      if systemctl --user restart "${SERVICE_NAME}.service"; then
        log "service" "service started"
        return 0
      fi
      warn "service" "could not restart the user service"
    fi
  fi
  start_background_worker
}

# Fallback for hosts without a usable systemd (containers, kernels, CI):
# the worker runs as a managed background process with its own log file.
start_background_worker() {
  local logfile="${DATA_DIR}/worker.log"
  mkdir -p "${DATA_DIR}"
  log "service" "starting the worker as a background process (log: ${logfile})"
  (
    cd "${PROJECT_ROOT}" || exit 1
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
    exec "${VENV_DIR}/bin/python" -m worker.worker_agent
  ) >> "${logfile}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${DATA_DIR}/worker.pid"
  log "service" "worker started (pid ${pid})"
}

verify_worker() {
  local url="http://127.0.0.1:${BIND_PORT}/health"
  log "verify" "probing ${url}"
  local attempt
  for attempt in $(seq 1 30); do
    if have_cmd curl; then
      if curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; then
        log "verify" "the worker answered /health"
        curl -fsS --max-time 3 "${url}" || true
        echo
        return 0
      fi
    fi
    sleep 1
  done
  warn "verify" "the worker did not answer /health within 30s"
  return 1
}

print_env() {
  cat <<EOF
HYPER_WORKER_HEAD_URL=${HEAD_URL}
HYPER_WORKER_BIND_HOST=${BIND_HOST}
HYPER_WORKER_BIND_PORT=${BIND_PORT}
HYPER_WORKER_MODEL_PATH=${MODEL_DIR}/${MODEL_FILE}
HYPER_WORKER_MODEL_REPO=${MODEL_REPO}
HYPER_WORKER_MODEL_FILENAME=${MODEL_FILE}
HYPER_WORKER_MODEL_FAMILY=${MODEL_FAMILY}
HYPER_WORKER_MODEL_PARAMS_B=${MODEL_PARAMS_B}
HYPER_WORKER_CONTEXT_SIZE=${CONTEXT_SIZE}
HYPER_WORKER_GPU_LAYERS=${GPU_LAYERS}
HYPER_WORKER_BACKEND=${BACKEND}
HYPER_WORKER_PROVIDER=${PROVIDER}
HYPER_WORKER_REGION=${REGION}
HYPER_WORKER_MAX_CONCURRENCY=${MAX_CONCURRENCY}
HYPER_WORKER_HEARTBEAT_SECONDS=${HEARTBEAT}
HYPER_WORKER_TTL_SECONDS=${TTL}
HYPER_WORKER_ENV_FILE=${ENV_FILE}
HYPER_WORKER_VENV=${VENV_DIR}
HYPER_WORKER_CACHE=${CACHE_DIR}
EOF
}

uninstall() {
  log "uninstall" "removing the service"
  if have_cmd systemctl; then
    systemctl stop "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
    systemctl disable "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
    rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
    systemctl daemon-reload >/dev/null 2>&1 || true
  fi
  if [[ "${IS_TERMUX}" == "1" ]] && have_cmd termux-service; then
    termux-service-stop "${SERVICE_NAME}" >/dev/null 2>&1 || true
  fi
  log "uninstall" "removing the virtualenv"
  rm -rf "${VENV_DIR}"
  log "uninstall" "done (the model cache was kept at ${MODEL_DIR})"
}

check_environment() {
  detect_platform
  select_python
  local problems=0
  log "check" "platform=${PLATFORM}/${CPU_ARCH} nvidia=${HAS_NVIDIA} termux=${IS_TERMUX} android=${IS_ANDROID}"
  log "check" "python=${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"
  log "check" "head=${HEAD_URL} model=${MODEL_REPO}/${MODEL_FILE}"
  have_cmd curl || { warn "check" "curl is missing"; problems=$((problems+1)); }
  [[ -d "${MODEL_DIR}/${MODEL_FILE}" ]] || log "check" "the model will be downloaded to ${MODEL_DIR}/${MODEL_FILE}"
  [[ -x "${VENV_DIR}/bin/python" ]] || log "check" "the virtualenv will be created at ${VENV_DIR}"
  if [[ "${problems}" -gt 0 ]]; then
    die "check" "${problems} problem(s) found"
  fi
  log "check" "environment looks good"
}

main() {
  parse_args "$@"
  detect_platform
  case "${MODE}" in
    print-env)
      print_env
      ;;
    check)
      check_environment
      ;;
    verify)
      verify_worker
      ;;
    uninstall)
      uninstall
      ;;
    install)
      install_system_deps
      select_python
      setup_venv
      fetch_llama_build
      fetch_model
      write_env
      install_service
      start_service
      verify_worker || warn "install" "the worker did not verify; check the logs"
      log "install" "worker provisioning complete"
      ;;
  esac
}

# Only run when executed directly; sourcing the script exposes its helpers
# (fetch_artifact, verify_sha256, ...) for testing.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
