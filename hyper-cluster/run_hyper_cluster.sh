#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Hyper-Cluster Omega :: the launcher
#
#   ./run_hyper_cluster.sh
#
# It starts the two processes that make the cluster usable:
#
#   * the control plane      (:8080) -- the API, the mesh, the memory, the
#                              sleep-time engine;
#   * the Mission Control    (:8082) -- the PWA the phone opens (it proxies
#                              the API, so the phone only needs one address).
#
# The configuration comes from `.env` (the installer writes it).  Ctrl-C
# stops both processes.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${HYPER_VENV:-${SCRIPT_DIR}/.venv}"
BIN_DIR="${VENV_DIR}/bin"
ENV_FILE="${HYPER_ENV_FILE:-${SCRIPT_DIR}/.env}"
HEAD_HOST="${HYPER_HEAD_HOST:-0.0.0.0}"
HEAD_PORT="${HYPER_HEAD_PORT:-8080}"
DASHBOARD_HOST="${HYPER_DASHBOARD_HOST:-0.0.0.0}"
DASHBOARD_PORT="${HYPER_DASHBOARD_PORT:-8082}"
PYTHON="${BIN_DIR}/python"

if [ ! -x "${PYTHON}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  else
    echo "no python found -- run ./install_all.sh first" >&2
    exit 1
  fi
fi

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
  HEAD_HOST="${HYPER_HEAD_HOST:-${HEAD_HOST}}"
  HEAD_PORT="${HYPER_HEAD_PORT:-${HEAD_PORT}}"
  DASHBOARD_HOST="${HYPER_DASHBOARD_HOST:-${DASHBOARD_HOST}}"
  DASHBOARD_PORT="${HYPER_DASHBOARD_PORT:-${DASHBOARD_PORT}}"
fi

export HYPER_ENV_FILE="${ENV_FILE}"

echo "⚡ Hyper-Cluster Omega"
echo "   control plane   http://${HEAD_HOST}:${HEAD_PORT}"
echo "   mission control http://${DASHBOARD_HOST}:${DASHBOARD_PORT}   (open it on the phone)"

"${PYTHON}" -m head_node.serve --host "${HEAD_HOST}" --port "${HEAD_PORT}" --env-file "${ENV_FILE}" &
HEAD_PID=$!

"${PYTHON}" -m mobile.dashboard --host "${DASHBOARD_HOST}" --port "${DASHBOARD_PORT}" --env-file "${ENV_FILE}" &
DASH_PID=$!

cleanup() {
  trap - INT TERM EXIT
  kill "${HEAD_PID}" "${DASH_PID}" 2>/dev/null || true
  wait "${HEAD_PID}" 2>/dev/null || true
  wait "${DASH_PID}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait -n "${HEAD_PID}" "${DASH_PID}" 2>/dev/null || wait "${HEAD_PID}"
cleanup
