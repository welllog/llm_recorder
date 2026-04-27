#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

LISTEN_ADDR="${LISTEN_ADDR:-:1237}"
UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-}"
API_KEY="${API_KEY:-}"
LOG_FILE="${LOG_FILE:-llm_req.log}"
TIMEOUT="${TIMEOUT:-300s}"

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: please set API_KEY before running this script" >&2
  echo "Example: API_KEY=sk-xxx ${0##*/}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
exec go run ./ \
  --listen "${LISTEN_ADDR}" \
  --upstream-base-url "${UPSTREAM_BASE_URL}" \
  --api-key "${API_KEY}" \
  --log-file "${LOG_FILE}" \
  --timeout "${TIMEOUT}"
