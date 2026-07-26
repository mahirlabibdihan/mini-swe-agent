# OpenCode on SWE-bench Verified with GPT-5 Mini

This experiment runs the official OpenCode CLI through Pier with
`openai/gpt-5-mini` on OpenRouter. Pier installs OpenCode in each Harbor task,
captures its ATIF trajectory, and writes the final repository change to
`agent/model.patch`.

## Budget

The primary OpenCode `build` agent is configured with exactly 50 agentic steps:

```yaml
agent:
  build:
    steps: 50
    permission:
      task: deny
```

At the limit, OpenCode prevents further tool iterations and instructs the model
to provide its final response. The `task` tool is disabled so subagents cannot
create independent sessions with separate step counters. This makes the
50-step budget apply to the complete agent run.

## Setup

```bash
bash experiments/opencode-swebench/setup.sh
cp experiments/opencode-swebench/.env.example \
  experiments/opencode-swebench/.env
# Replace the placeholder with your OpenRouter key.
```

Use Linux or WSL 2 with Docker configured for Linux containers. Log bind mounts
are disabled by default for NFS-backed cluster home directories.

## Run

Smoke-test one task:

```bash
JOB_NAME=opencode-smoke N_CONCURRENT=1 \
  bash experiments/opencode-swebench/run.sh
```

Run the alphabetically first 10 tasks:

```bash
JOB_NAME=opencode-first10 N_TASKS=10 N_CONCURRENT=2 \
  bash experiments/opencode-swebench/run.sh
```

Run a fixed slice:

```bash
JOB_NAME=opencode-10-20 N_CONCURRENT=2 \
  bash experiments/opencode-swebench/run.sh --slice 10:20
```

`--slice` accepts Python slice syntax, overrides `N_TASKS`, and cannot be
combined with `SAMPLE_SEED`. Jobs are immutable; an existing `JOB_NAME`
triggers a deletion prompt rather than resume.

## Export and Evaluate

Predictions are written to:

```text
experiments/opencode-swebench/jobs/<JOB_NAME>/predictions.jsonl
```

Regenerate predictions without rerunning trials:

```bash
uv run --project pier python \
  experiments/opencode-swebench/export_predictions.py --overwrite \
  experiments/opencode-swebench/jobs/opencode-gpt5-mini-swebench-verified \
  experiments/opencode-swebench/jobs/opencode-gpt5-mini-swebench-verified/predictions.jsonl
```

Evaluate a named run:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path experiments/opencode-swebench/jobs/opencode-first10/predictions.jsonl \
  --max_workers 5 \
  --run_id opencode-first10 \
  --report_dir evaluation \
  --cache_level instance
```

OpenCode uses the built-in `openrouter` provider and reads
`OPENROUTER_API_KEY` from the experiment `.env`.
