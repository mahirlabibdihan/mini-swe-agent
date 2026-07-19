# Claude Code on SWE-bench Verified with GPT-5 mini

This experiment runs the official Claude Code CLI as the coding agent while
routing every model role through `openai/gpt-5-mini` on OpenRouter. Pier installs
Claude Code inside each Harbor task sandbox, captures its trajectory, and runs
the task verifier.

This is intentionally an interoperability experiment. Claude Code is designed
for Claude models, and Anthropic does not support non-Claude models behind a
gateway. OpenRouter exposes an Anthropic Messages-compatible endpoint for
`openai/gpt-5-mini`, but tool-use or prompt-compatibility differences can still
affect reliability and score.

## Prerequisites

Use Linux or WSL 2 with Docker Desktop configured for Linux containers. Install
Git, Docker, `uv`, and obtain a funded OpenRouter API key. A 500-task run is
expensive and can require substantial time, API credit, and Docker storage. The
scripts therefore default to one deterministic task and one worker.

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

Run a deterministic 10-task sample with two workers:

```bash
N_TASKS=10 N_CONCURRENT=2 SAMPLE_SEED=0 \
  bash experiments/claude-code-swebench/run.sh
```

Run all 500 tasks:

```bash
N_TASKS=500 N_CONCURRENT=4 \
  bash experiments/claude-code-swebench/run.sh
```

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

## Fixed agent settings

- agent: official Claude Code CLI through Pier's `claude-code` adapter
- model: `openai/gpt-5-mini`
- gateway: `https://openrouter.ai/api` using its Anthropic-compatible API
- maximum turns: 50
- Claude extended thinking: disabled because GPT-5 mini is not a Claude model
- plan mode: disabled
- verification: enabled through each Harbor task's verifier

The adapter assigns GPT-5 mini to Claude Code's Opus, Sonnet, Haiku, and
subagent model aliases, ensuring helper calls do not silently use a Claude model.
