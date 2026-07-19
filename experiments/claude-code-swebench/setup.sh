#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PIER_DIR="$WORKSPACE_ROOT/pier"
DATASETS_DIR="${DATASETS_DIR:-$WORKSPACE_ROOT/datasets}"
DATASET_DIR="${DATASET_DIR:-$DATASETS_DIR/swe-bench-verified}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi

uv sync --project "$PIER_DIR" --dev

if [[ -d "$DATASET_DIR" ]] && find "$DATASET_DIR" -mindepth 2 -name task.toml -print -quit | grep -q .; then
  echo "SWE-bench Verified is already available at $DATASET_DIR"
  exit 0
fi

mkdir -p "$DATASETS_DIR"
uv run --project "$PIER_DIR" harbor download swe-bench/swe-bench-verified -o "$DATASETS_DIR"

if [[ ! -d "$DATASET_DIR" ]] || ! find "$DATASET_DIR" -mindepth 2 -name task.toml -print -quit | grep -q .; then
  echo "Harbor finished, but no tasks were found at $DATASET_DIR." >&2
  echo "Set DATASET_DIR to the downloaded dataset path before running run.sh." >&2
  exit 2
fi

echo "Ready: $DATASET_DIR"
