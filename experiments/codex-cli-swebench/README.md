# Codex CLI on SWE-bench Verified with GPT-5 Mini

This experiment runs the official OpenAI Codex CLI through Pier while routing
inference to `openai/gpt-5-mini` through OpenRouter. Pier installs Codex inside
each Harbor task container, captures its ATIF trajectory, and writes the final
repository change as `agent/model.patch`.

Codex uses OpenRouter's OpenAI-compatible Responses API. Its provider
configuration is intentionally different from the Claude Code experiment:

- base URL: `https://openrouter.ai/api/v1`
- provider: `openrouter`
- wire API: `responses`
- credential: `OPENROUTER_API_KEY`
- model slug: `openai/gpt-5-mini`

## Setup

From the repository root:

```bash
bash experiments/codex-cli-swebench/setup.sh
cp experiments/codex-cli-swebench/.env.example \
  experiments/codex-cli-swebench/.env
# Replace the placeholder in .env with your OpenRouter key.
```

Use Linux or WSL 2 with Docker configured for Linux containers. The setup
script installs Pier's locked dependencies and reuses or downloads the Harbor
SWE-bench Verified dataset under `datasets/swe-bench-verified`.

Trial log bind mounts are disabled by default because Docker daemon bind mounts
can fail on NFS-backed cluster home directories. Set `PIER_MOUNT_LOGS=1` in
`.env` only when the workspace is on a local filesystem.

## Run

Smoke-test the alphabetically first task:

```bash
JOB_NAME=codex-cli-smoke N_CONCURRENT=1 \
  bash experiments/codex-cli-swebench/run.sh
```

Run the alphabetically first 10 tasks:

```bash
JOB_NAME=codex-cli-first10 N_TASKS=10 N_CONCURRENT=2 \
  bash experiments/codex-cli-swebench/run.sh
```

Run a fixed slice:

```bash
JOB_NAME=codex-cli-10-20 N_CONCURRENT=2 \
  bash experiments/codex-cli-swebench/run.sh --slice 10:20
```

`--slice` accepts Python `START:STOP` or `START:STOP:STEP` syntax. It cannot be
combined with `SAMPLE_SEED`, and it overrides `N_TASKS`. Set `SAMPLE_SEED` only
for a reproducible shuffled selection.

Jobs are immutable. When `JOB_NAME` already exists, the wrapper asks whether to
delete it and start again. Answer no or use a different name to preserve the
existing results. For a non-interactive replacement, set `OVERWRITE_JOB=1`.

Pier verification is disabled by default because official SWE-bench evaluation
is performed separately. Set `DISABLE_VERIFICATION=0` only when Pier's task
verifier is intentionally required.

## Limits

The Codex adapter requests `high` reasoning effort. Codex CLI does not expose a
turn-count control equivalent to Claude Code's `--max-turns`, so this experiment
does not claim a 50-turn limit. Execution is bounded by the task and Pier
timeouts. Record that distinction when comparing agent budgets.

## Export

The wrapper writes:

```text
experiments/codex-cli-swebench/jobs/<JOB_NAME>/predictions.jsonl
```

It retries export five times for transient NFS failures. Regenerate the default
job's predictions without rerunning trials:

```bash
uv run --project pier python \
  experiments/codex-cli-swebench/export_predictions.py --overwrite \
  experiments/codex-cli-swebench/jobs/codex-cli-gpt5-mini-swebench-verified \
  experiments/codex-cli-swebench/jobs/codex-cli-gpt5-mini-swebench-verified/predictions.jsonl
```

Replace the job name in both paths for named runs.

## Evaluate

From the repository root, when SWE-bench is installed in the active
environment:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path experiments/codex-cli-swebench/jobs/codex-cli-first10/predictions.jsonl \
  --max_workers 5 \
  --run_id codex-cli-first10 \
  --report_dir evaluation \
  --cache_level instance
```

If the harness is available only in the OpenHands environment:

```bash
cd openhands
poetry run python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path ../experiments/codex-cli-swebench/jobs/codex-cli-first10/predictions.jsonl \
  --max_workers 5 \
  --run_id codex-cli-first10 \
  --report_dir ../evaluation \
  --cache_level instance
```

OpenRouter's Codex configuration requires a named provider and
`wire_api = "responses"`; it does not use Claude Code's Anthropic-compatible
gateway settings.
