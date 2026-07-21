#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PIER_DIR="$WORKSPACE_ROOT/pier"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
DATASET_DIR="${DATASET_DIR:-$WORKSPACE_ROOT/datasets/swe-bench-verified}"
N_TASKS="${N_TASKS:-1}"
# Set SAMPLE_SEED only when a deterministic shuffled sample is wanted.
# Omitting it preserves the dataset order (for example, the first 10 tasks).
SAMPLE_SEED="${SAMPLE_SEED:-}"
N_CONCURRENT="${N_CONCURRENT:-1}"
DISABLE_VERIFICATION="${DISABLE_VERIFICATION:-1}"
JOBS_DIR="${JOBS_DIR:-$SCRIPT_DIR/jobs}"
JOB_NAME="${JOB_NAME:-claude-code-gpt5-mini-swebench-verified}"
JOB_DIR="$JOBS_DIR/$JOB_NAME"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$JOB_DIR/predictions.jsonl}"
OVERWRITE_JOB="${OVERWRITE_JOB:-0}"

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

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ ! -d "$DATASET_DIR" ]] || ! find "$DATASET_DIR" -mindepth 2 -name task.toml -print -quit | grep -q .; then
  echo "SWE-bench Verified tasks were not found at $DATASET_DIR. Run setup.sh first." >&2
  exit 2
fi

resume_job=0
if [[ -d "$JOB_DIR" ]]; then
  if [[ "$OVERWRITE_JOB" == "1" ]]; then
    # This is the wrapper's own result directory; refuse broader targets.
    case "$(realpath -m "$JOB_DIR")" in
      "$(realpath -m "$JOBS_DIR")"/*) rm -rf -- "$JOB_DIR" ;;
      *) echo "Refusing to delete a job directory outside $JOBS_DIR." >&2; exit 2 ;;
    esac
  elif [[ -f "$JOB_DIR/config.json" ]]; then
    resume_job=1
  else
    echo "$JOB_DIR exists but is not a resumable Pier job (config.json is missing)." >&2
    echo "Move it aside or set OVERWRITE_JOB=1 to replace it." >&2
    exit 2
  fi
fi

cd "$WORKSPACE_ROOT"
verification_args=()
if [[ "$DISABLE_VERIFICATION" == "1" ]]; then
  verification_args+=(--disable-verification)
elif [[ "$DISABLE_VERIFICATION" != "0" ]]; then
  echo "DISABLE_VERIFICATION must be 0 or 1." >&2
  exit 2
fi

sample_args=()
if [[ -n "$SAMPLE_SEED" ]]; then
  sample_args+=(--sample-seed "$SAMPLE_SEED")
fi

if [[ "$resume_job" == "1" ]]; then
  echo "Resuming $JOB_DIR; completed instances will be skipped."
  uv run --project "$PIER_DIR" pier job resume --job-path "$JOB_DIR"
else
  uv run --project "$PIER_DIR" pier run \
    --config "$CONFIG_FILE" \
    --env-file "$ENV_FILE" \
    --jobs-dir "$JOBS_DIR" \
    --job-name "$JOB_NAME" \
    --path "$DATASET_DIR" \
    --n-tasks "$N_TASKS" \
    "${sample_args[@]}" \
    --n-concurrent "$N_CONCURRENT" \
    "${verification_args[@]}" \
    --yes
fi

uv run --project "$PIER_DIR" python "$SCRIPT_DIR/export_predictions.py" \
  --overwrite \
  "$JOB_DIR" \
  "$PREDICTIONS_PATH"

echo "SWE-bench predictions: $PREDICTIONS_PATH"
