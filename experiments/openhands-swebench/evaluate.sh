#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OPENHANDS_DIR="${OPENHANDS_DIR:-$WORKSPACE_ROOT/openhands}"
OUTPUT_JSONL="${1:-}"

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

cd "$OPENHANDS_DIR"
bash evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
  "$OUTPUT_JSONL" \
  "" \
  princeton-nlp/SWE-bench_Verified \
  test \
  local

REPORT_JSON="$(dirname "$OUTPUT_JSONL")/report.json"
if [[ ! -f "$REPORT_JSON" ]]; then
  echo "Evaluation finished without producing $REPORT_JSON" >&2
  exit 1
fi

jq -r '
  "Resolved: \(.resolved_instances)",
  "Submitted: \(.submitted_instances)",
  "Completed: \(.completed_instances)",
  "Verified total: \(.total_instances)",
  "Resolve rate over submitted: \(if .submitted_instances > 0 then (100 * .resolved_instances / .submitted_instances) else 0 end)%",
  "Overall Verified accuracy: \(100 * .resolved_instances / .total_instances)%"
' "$REPORT_JSON"

echo "Report: $REPORT_JSON"
