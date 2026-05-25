#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

LISTEN_ADDR="${LISTEN_ADDR:-:1237}"
OPENAI_UPSTREAM_BASE_URL="${OPENAI_UPSTREAM_BASE_URL:-https://api.openai.com/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key-placeholder}"
OPENAI_RESPONSES_UPSTREAM_BASE_URL="${OPENAI_RESPONSES_UPSTREAM_BASE_URL:-}"
OPENAI_RESPONSES_API_KEY="${OPENAI_RESPONSES_API_KEY:-}"
ANTHROPIC_UPSTREAM_BASE_URL="${ANTHROPIC_UPSTREAM_BASE_URL:-https://api.anthropic.com}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-your-anthropic-api-key-placeholder}"
LOG_FILE="${LOG_FILE:-llm_req.log}"
TIMEOUT="${TIMEOUT:-300s}"

if [[ -z "${OPENAI_API_KEY}" || "${OPENAI_API_KEY}" == "your-openai-api-key-placeholder" ]]; then
  echo "ERROR: please set OPENAI_API_KEY before running this script" >&2
  echo "Example: OPENAI_API_KEY=sk-openai-key ANTHROPIC_API_KEY=sk-ant-key ${0##*/}" >&2
  exit 1
fi

if [[ -z "${ANTHROPIC_API_KEY}" || "${ANTHROPIC_API_KEY}" == "your-anthropic-api-key-placeholder" ]]; then
  echo "ERROR: please set ANTHROPIC_API_KEY before running this script" >&2
  echo "Example: OPENAI_API_KEY=sk-openai-key ANTHROPIC_API_KEY=sk-ant-key ${0##*/}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

EXTRA_ARGS=()
if [[ -n "${OPENAI_RESPONSES_UPSTREAM_BASE_URL}" ]]; then
  EXTRA_ARGS+=(--openai-responses-upstream-base-url "${OPENAI_RESPONSES_UPSTREAM_BASE_URL}")
fi
if [[ -n "${OPENAI_RESPONSES_API_KEY}" ]]; then
  EXTRA_ARGS+=(--openai-responses-api-key "${OPENAI_RESPONSES_API_KEY}")
fi

exec go run ./ \
  --listen "${LISTEN_ADDR}" \
  --openai-upstream-base-url "${OPENAI_UPSTREAM_BASE_URL}" \
  --openai-api-key "${OPENAI_API_KEY}" \
  --anthropic-upstream-base-url "${ANTHROPIC_UPSTREAM_BASE_URL}" \
  --anthropic-api-key "${ANTHROPIC_API_KEY}" \
  --log-file "${LOG_FILE}" \
  --timeout "${TIMEOUT}" \
  "${EXTRA_ARGS[@]}"
