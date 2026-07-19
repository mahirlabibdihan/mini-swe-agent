#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PIER_DIR="$WORKSPACE_ROOT/pier"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
DATASET_DIR="${DATASET_DIR:-$WORKSPACE_ROOT/datasets/swe-bench-verified}"
N_TASKS="${N_TASKS:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
N_CONCURRENT="${N_CONCURRENT:-1}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Run setup.sh after installing uv." >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "Docker is not installed or its Linux daemon is not running." >&2
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.example to .env and add your OpenRouter key." >&2
  exit 2
fi

if [[ ! -d "$DATASET_DIR" ]] || ! find "$DATASET_DIR" -mindepth 2 -name task.toml -print -quit | grep -q .; then
  echo "SWE-bench Verified tasks were not found at $DATASET_DIR. Run setup.sh first." >&2
  exit 2
fi

cd "$WORKSPACE_ROOT"
uv run --project "$PIER_DIR" pier run \
  --config "$CONFIG_FILE" \
  --env-file "$ENV_FILE" \
  --path "$DATASET_DIR" \
  --n-tasks "$N_TASKS" \
  --sample-seed "$SAMPLE_SEED" \
  --n-concurrent "$N_CONCURRENT" \
  --yes
