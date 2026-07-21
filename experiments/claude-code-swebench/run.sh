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
# Omitting it preserves alphabetical instance-id order (for example, the first 10 tasks).
SAMPLE_SEED="${SAMPLE_SEED:-}"
N_CONCURRENT="${N_CONCURRENT:-1}"
DISABLE_VERIFICATION="${DISABLE_VERIFICATION:-1}"
JOBS_DIR="${JOBS_DIR:-$SCRIPT_DIR/jobs}"
JOB_NAME="${JOB_NAME:-claude-code-gpt5-mini-swebench-verified}"
JOB_DIR="$JOBS_DIR/$JOB_NAME"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$JOB_DIR/predictions.jsonl}"
OVERWRITE_JOB="${OVERWRITE_JOB:-0}"
SLICE_SPEC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slice)
      [[ $# -ge 2 ]] || { echo "--slice requires a value such as 0:10." >&2; exit 2; }
      SLICE_SPEC="$2"
      shift 2
      ;;
    --slice=*)
      SLICE_SPEC="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

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

# Docker daemon bind mounts can fail on NFS-backed home directories. Pier will
# copy logs and patches out of the container when mounts are disabled.
export PIER_MOUNT_LOGS="${PIER_MOUNT_LOGS:-0}"

if [[ ! -d "$DATASET_DIR" ]] || ! find "$DATASET_DIR" -mindepth 2 -name task.toml -print -quit | grep -q .; then
  echo "SWE-bench Verified tasks were not found at $DATASET_DIR. Run setup.sh first." >&2
  exit 2
fi

slice_args=()
if [[ -n "$SLICE_SPEC" ]]; then
  if [[ -n "$SAMPLE_SEED" ]]; then
    echo "--slice and SAMPLE_SEED cannot be used together." >&2
    exit 2
  fi
  if ! slice_output="$(
    uv run --project "$PIER_DIR" python "$SCRIPT_DIR/select_slice.py" \
      "$DATASET_DIR" "$SLICE_SPEC"
  )"; then
    exit 2
  fi
  mapfile -t sliced_instances <<< "$slice_output"
  for instance_id in "${sliced_instances[@]}"; do
    slice_args+=(--include-task-name "$instance_id")
  done
fi

if [[ -d "$JOB_DIR" ]]; then
  if [[ "$OVERWRITE_JOB" == "1" ]]; then
    answer="y"
  elif [[ -t 0 ]]; then
    read -r -p "Job directory $JOB_DIR already exists. Delete it and start a new job? [y/N] " answer
  else
    echo "Job directory $JOB_DIR already exists. Set OVERWRITE_JOB=1 to replace it." >&2
    exit 2
  fi

  if [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    # This is the wrapper's own result directory; refuse broader targets.
    case "$(realpath -m "$JOB_DIR")" in
      "$(realpath -m "$JOBS_DIR")"/*) rm -rf -- "$JOB_DIR" ;;
      *) echo "Refusing to delete a job directory outside $JOBS_DIR." >&2; exit 2 ;;
    esac
  else
    echo "Keeping the existing job directory; no experiment was started."
    exit 0
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
limit_args=()
if [[ -n "$SLICE_SPEC" ]]; then
  limit_args+=("${slice_args[@]}")
else
  limit_args+=(--n-tasks "$N_TASKS")
fi
if [[ -n "$SAMPLE_SEED" ]]; then
  sample_args+=(--sample-seed "$SAMPLE_SEED")
fi

uv run --project "$PIER_DIR" pier run \
  --config "$CONFIG_FILE" \
  --env-file "$ENV_FILE" \
  --jobs-dir "$JOBS_DIR" \
  --job-name "$JOB_NAME" \
  --path "$DATASET_DIR" \
  "${limit_args[@]}" \
  "${sample_args[@]}" \
  --n-concurrent "$N_CONCURRENT" \
  "${verification_args[@]}" \
  --yes

uv run --project "$PIER_DIR" python "$SCRIPT_DIR/export_predictions.py" \
  --overwrite \
  "$JOB_DIR" \
  "$PREDICTIONS_PATH"

echo "SWE-bench predictions: $PREDICTIONS_PATH"
