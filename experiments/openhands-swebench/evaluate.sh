#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OPENHANDS_DIR="${OPENHANDS_DIR:-$WORKSPACE_ROOT/openhands}"
OUTPUT_JSONL="${1:-}"
FORCE_REEVALUATE="${FORCE_REEVALUATE:-0}"

for command in docker poetry jq realpath; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is missing: $command" >&2
    exit 2
  fi
done

if [[ -z "$OUTPUT_JSONL" ]]; then
  mapfile -t candidates < <(
    find "$OPENHANDS_DIR/evaluation/evaluation_outputs/outputs" \
      -path '*princeton-nlp__SWE-bench_Verified-test*' \
      -name output.jsonl -type f -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr
  )
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No SWE-bench Verified output.jsonl was found." >&2
    exit 2
  fi
  OUTPUT_JSONL="${candidates[0]#* }"
fi

OUTPUT_JSONL="$(realpath "$OUTPUT_JSONL")"
if [[ ! -f "$OUTPUT_JSONL" ]]; then
  echo "Output file does not exist: $OUTPUT_JSONL" >&2
  exit 2
fi

RUN_DIR="$(dirname "$OUTPUT_JSONL")"
EVAL_INPUT="$OUTPUT_JSONL"
PENDING_COUNT=0

if [[ "$FORCE_REEVALUATE" != "1" ]]; then
  PENDING_COUNT="$(jq -c 'select((.report? | type) != "object" or ((.report? // {}) | length) == 0)' "$OUTPUT_JSONL" | wc -l)"

  if [[ "$PENDING_COUNT" -eq 0 ]]; then
    echo "All instances in $OUTPUT_JSONL already have evaluation reports."
    echo "Set FORCE_REEVALUATE=1 to evaluate every instance again."
  elif [[ "$PENDING_COUNT" -lt "$(wc -l < "$OUTPUT_JSONL")" ]]; then
    EVAL_WORK_DIR="$(mktemp -d "$RUN_DIR/evaluation-pending.XXXXXX")"
    EVAL_INPUT="$EVAL_WORK_DIR/output.jsonl"
    jq -c 'select((.report? | type) != "object" or ((.report? // {}) | length) == 0)' \
      "$OUTPUT_JSONL" > "$EVAL_INPUT"
    echo "Evaluating $PENDING_COUNT instances without an existing report."
  else
    echo "Evaluating all $PENDING_COUNT instances."
  fi
else
  echo "FORCE_REEVALUATE=1: evaluating every instance."
  PENDING_COUNT="$(wc -l < "$OUTPUT_JSONL")"
fi

if [[ "$PENDING_COUNT" -gt 0 ]]; then
  cd "$OPENHANDS_DIR"
  bash evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
    "$EVAL_INPUT" \
    "" \
    princeton-nlp/SWE-bench_Verified \
    test \
    local

  if [[ "$EVAL_INPUT" != "$OUTPUT_JSONL" ]]; then
    MERGED_OUTPUT="$(mktemp "$RUN_DIR/.output-merged.XXXXXX")"
    jq -c -s '
      (.[0] | map({key: .instance_id, value: .report}) | from_entries) as $reports
      | .[1][]
      | if $reports[.instance_id] != null
        then .report = $reports[.instance_id]
        else .
        end
    ' "$EVAL_INPUT" "$OUTPUT_JSONL" > "$MERGED_OUTPUT"
    mv "$MERGED_OUTPUT" "$OUTPUT_JSONL"
    echo "Merged new evaluation reports into $OUTPUT_JSONL"
    echo "New evaluation artifacts: $EVAL_WORK_DIR"
  fi
fi

jq -r -s '
  {
    submitted: length,
    evaluated: (map(select((.report? | type) == "object" and (.report | length) > 0)) | length),
    resolved: (map(select(.report.resolved == true)) | length)
  }
  | "Resolved: \(.resolved)",
    "Submitted: \(.submitted)",
    "Evaluated: \(.evaluated)",
    "Resolve rate over submitted: \(if .submitted > 0 then (100 * .resolved / .submitted) else 0 end)%"
' "$OUTPUT_JSONL"
