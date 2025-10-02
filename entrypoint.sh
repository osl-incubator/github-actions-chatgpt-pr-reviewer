#!/usr/bin/env bash
set -euo pipefail

OPENAI_API_KEY="${1:-}"
OPENAI_MODEL="${2:-}"
OPENAI_TEMPERATURE="${3:-}"
OPENAI_MAX_TOKENS="${4:-}"
PROMPT_EXTRA_CRITERIA="${5:-}"
GITHUB_TOKEN_IN="${6:-}"
GITHUB_PR_ID_IN="${7:-}"

export OPENAI_API_KEY
export OPENAI_MODEL
export OPENAI_TEMPERATURE
export OPENAI_MAX_TOKENS
export PROMPT_EXTRA_CRITERIA

export GITHUB_TOKEN="${GITHUB_TOKEN_IN:-${GITHUB_TOKEN:-}}"
export GITHUB_PR_ID="${GITHUB_PR_ID_IN:-${GITHUB_PR_ID:-}}"

: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_PR_ID:?GITHUB_PR_ID is required}"

exec python /main.py
