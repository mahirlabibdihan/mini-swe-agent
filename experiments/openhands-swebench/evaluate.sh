#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OPENHANDS_DIR="${OPENHANDS_DIR:-$WORKSPACE_ROOT/openhands}"
TARGET="${1:-}"
FORCE_REEVALUATE="${FORCE_REEVALUATE:-0}"

for command in docker poetry jq realpath find sort mktemp; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is missing: $command" >&2
    exit 2
  fi
done

if [[ -z "$TARGET" ]]; then
  mapfile -t candidates < <(
    find "$OPENHANDS_DIR/evaluation/evaluation_outputs/outputs" \
      -path '*princeton-nlp__SWE-bench_Verified-test*' \
      -type d -name output -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr
  )
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No per-instance SWE-bench Verified output directory was found." >&2
    exit 2
  fi
  OUTPUT_DIR="${candidates[0]#* }"
elif [[ -d "$TARGET" && "$(basename "$TARGET")" == "output" ]]; then
  OUTPUT_DIR="$(realpath "$TARGET")"
elif [[ -d "$TARGET/output" ]]; then
  OUTPUT_DIR="$(realpath "$TARGET/output")"
else
  echo "Pass either a run directory or its output/ directory: $TARGET" >&2
  exit 2
fi

RUN_DIR="$(dirname "$OUTPUT_DIR")"
TOTAL_COUNT="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' -printf '.' | wc -c)"
if [[ "$TOTAL_COUNT" -eq 0 ]]; then
  echo "No instance JSON files found in $OUTPUT_DIR" >&2
  exit 2
fi

EVAL_WORK_DIR="$(mktemp -d "$RUN_DIR/evaluation-pending.XXXXXX")"
EVAL_INPUT="$EVAL_WORK_DIR/pending.jsonl"
PENDING_COUNT=0

while IFS= read -r -d '' instance_file; do
  if [[ "$FORCE_REEVALUATE" == "1" ]] || ! jq -e '
      (.report? | type) == "object" and ((.report? // {}) | length) > 0
    ' "$instance_file" >/dev/null; then
    jq -c . "$instance_file" >> "$EVAL_INPUT"
    ((PENDING_COUNT += 1))
  fi
done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)

if [[ "$PENDING_COUNT" -eq 0 ]]; then
  rmdir "$EVAL_WORK_DIR"
  echo "All $TOTAL_COUNT instances already have evaluation reports."
  echo "Set FORCE_REEVALUATE=1 to evaluate every instance again."
else
  if [[ "$FORCE_REEVALUATE" == "1" ]]; then
    echo "FORCE_REEVALUATE=1: evaluating all $PENDING_COUNT instances."
  else
    echo "Evaluating $PENDING_COUNT of $TOTAL_COUNT instances without an existing report."
  fi

  cd "$OPENHANDS_DIR"
  bash evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
    "$EVAL_INPUT" \
    "" \
    princeton-nlp/SWE-bench_Verified \
    test \
    local

  while IFS= read -r result; do
    instance_id="$(jq -r '.instance_id' <<< "$result")"
    destination="$OUTPUT_DIR/$instance_id.json"
    temporary="$(mktemp "$OUTPUT_DIR/.${instance_id}.XXXXXX.tmp")"
    jq . <<< "$result" > "$temporary"
    mv "$temporary" "$destination"
  done < "$EVAL_INPUT"

  echo "Updated .report in $PENDING_COUNT files under $OUTPUT_DIR"
  echo "Evaluation artifacts: $EVAL_WORK_DIR"
fi

find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' -print0 \
  | xargs -0 jq -r -s '
      {
        submitted: length,
        evaluated: (map(select((.report? | type) == "object" and (.report | length) > 0)) | length),
        resolved: (map(select(.report.resolved == true)) | length)
      }
      | "Resolved: \(.resolved)",
        "Submitted: \(.submitted)",
        "Evaluated: \(.evaluated)",
        "Resolve rate over submitted: \(if .submitted > 0 then (100 * .resolved / .submitted) else 0 end)%"
    '
