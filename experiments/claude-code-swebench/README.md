# Claude Code on SWE-bench Verified with GPT-5 mini

This experiment runs the official Claude Code CLI as the coding agent while
routing every model role through `openai/gpt-5-mini` on OpenRouter. Pier installs
Claude Code inside each Harbor task sandbox and captures its trajectory and
final Git patch. Patch scoring is performed separately with the official
SWE-bench evaluator.

This is intentionally an interoperability experiment. Claude Code is designed
for Claude models, and Anthropic does not support non-Claude models behind a
gateway. OpenRouter exposes an Anthropic Messages-compatible endpoint for
`openai/gpt-5-mini`, but tool-use or prompt-compatibility differences can still
affect reliability and score.

## Prerequisites

Use Linux or WSL 2 with Docker Desktop configured for Linux containers. Install
Git, Docker, `uv`, and obtain a funded OpenRouter API key. A 500-task run is
expensive and can require substantial time, API credit, and Docker storage. The
scripts therefore default to the first dataset task and one worker.

## Setup

From the repository root:

```bash
bash experiments/claude-code-swebench/setup.sh
cp experiments/claude-code-swebench/.env.example \
  experiments/claude-code-swebench/.env
# Replace the placeholder in .env with your OpenRouter key.
```

`setup.sh` installs Pier's locked dependencies and downloads the 500-task
`swe-bench/swe-bench-verified` Harbor dataset to
`datasets/swe-bench-verified`.

## Run

Smoke-test one task:

```bash
bash experiments/claude-code-swebench/run.sh
```

Add the alphabetically first 10 instance IDs to the default job with two workers:

```bash
N_TASKS=10 N_CONCURRENT=2 \
  bash experiments/claude-code-swebench/run.sh
```

If that job previously ran a random sample, its completed instances are kept
and only missing instances from the first 10 are run. The resulting trajectories
and `predictions.jsonl` remain together in the default job directory.

Pier sorts local datasets alphabetically by instance ID. Set `SAMPLE_SEED` only
when you want it to shuffle that stable ordering before selecting the requested
number.

Run all 500 tasks:

```bash
JOB_NAME=claude-code-all500 N_TASKS=500 N_CONCURRENT=4 \
  bash experiments/claude-code-swebench/run.sh
```

The wrapper disables Pier's verifier by default. This avoids host bind mounts
for the trial logs and makes the run patch-generation-only. To re-enable Pier
verification (not needed for the SWE-bench workflow), set
`DISABLE_VERIFICATION=0`.

To run one named instance, call Pier directly:

```bash
uv run --project pier pier run \
  --config experiments/claude-code-swebench/config.yaml \
  --env-file experiments/claude-code-swebench/.env \
  --path datasets/swe-bench-verified \
  --include-task-name 'django__django-11099' --yes
```

Results and ATIF trajectories are written beneath
`experiments/claude-code-swebench/jobs/`. Resume an interrupted job with:

```bash
uv run --project pier pier job resume \
  experiments/claude-code-swebench/jobs/<job-directory>
```

When a job directory already contains a Pier `config.json`, `run.sh` extends its
saved task selection with the requested instances, resumes it, and skips
completed instances. It prints the resolved instance IDs before resuming so the
selection can be audited. Overlapping selections are deduplicated. Set
`JOB_NAME` only when you intentionally want an independent experiment
directory. To discard a named job and start it again, set `OVERWRITE_JOB=1`;
this removes its trajectories, patches, and predictions.

Each completed trial contains `agent/model.patch`. The run wrapper automatically
collects these into `jobs/<JOB_NAME>/predictions.jsonl`,
the official SWE-bench prediction format. Evaluate that file with the existing
SWE-bench evaluator:

```bash
cd openhands
poetry run python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path ../experiments/claude-code-swebench/jobs/claude-code-gpt5-mini-swebench-verified/predictions.jsonl \
  --max_workers 4 \
  --timeout 3600 \
  --run_id claude-code-gpt5-mini
```

## Fixed agent settings

- agent: official Claude Code CLI through Pier's `claude-code` adapter
- model: `openai/gpt-5-mini`
- gateway: `https://openrouter.ai/api` using its Anthropic-compatible API
- maximum turns: 50
- Claude extended thinking: disabled because GPT-5 mini is not a Claude model
- plan mode: disabled
- Pier verification: disabled; official SWE-bench evaluation is run afterward

The adapter assigns GPT-5 mini to Claude Code's Opus, Sonnet, Haiku, and
subagent model aliases, ensuring helper calls do not silently use a Claude model.
