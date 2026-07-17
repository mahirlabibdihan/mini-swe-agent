#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OPENHANDS_DIR="${OPENHANDS_DIR:-$WORKSPACE_ROOT/openhands}"
EVAL_LIMIT="${EVAL_LIMIT-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
N_RUNS="${N_RUNS:-1}"

ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set. Add it to $ENV_FILE." >&2
  exit 2
fi

if [[ ! -e "$OPENHANDS_DIR/.git" ]]; then
  echo "OpenHands checkout not found at $OPENHANDS_DIR. Run setup.sh first." >&2
  exit 2
fi

if ! command -v docker >/dev/null || ! docker info >/dev/null 2>&1; then
  echo "Docker is not installed or its Linux daemon is not running." >&2
  exit 2
fi

cp "$SCRIPT_DIR/config.toml.example" "$OPENHANDS_DIR/config.toml"
cd "$OPENHANDS_DIR"

export PYTHONPATH="$OPENHANDS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export EXP_NAME="openrouter-gpt5-mini-50step"

./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
  llm.gpt5_mini_swebench \
  HEAD \
  CodeActAgent \
  "$EVAL_LIMIT" \
  50 \
  "$NUM_WORKERS" \
  princeton-nlp/SWE-bench_Verified \
  test \
  "$N_RUNS" \
  swe
