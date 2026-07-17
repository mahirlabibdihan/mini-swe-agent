# Running this mini-swe-agent fork on DeepSWE

DeepSWE uses Harbor tasks and program-based verifiers, not Hugging Face
SWE-bench rows. Consequently `mini-extra swebench-ts` is not the correct
runner. Pier prepares each repository, invokes the agent, extracts its Git
patch, and runs the held-out verifier.

## Prerequisites

- Linux Docker containers (Docker Desktop with Linux containers is fine).
- Python 3.12+ and `uv`.
- Use the patched Pier checkout in `../pier`, not the globally installed Pier.
- API variables required by `config/extra/swebench_ts.yaml`. The checked-in
  reward model currently uses `http://10.141.10.34:3000/v1`; that endpoint
  must be reachable from the task container, or change the reward-model config.

The patched Pier installs this fork at the pinned `REVISION` in
`../pier/src/pier/agents/installed/local_mini_swe_agent.py`. Update that
constant after pushing future agent changes.

## Original/base mini-swe-agent

This is the DeepSWE equivalent of your SWE-bench base command:

```bash
mini-extra swebench --subset verified --split test \
  --output output/verified.test.base/deepseek__deepseek-v4-flash/
```

Run the patched Pier source with `uv --project`:

```bash
# Run from mini-swe-agent/deep-swe
uv run --project ../pier pier run -p tasks \
  --agent local-mini-swe-agent \
  --model openrouter/openai/gpt-5-mini \
  --env-file .env
```

Pier creates its own job/trial output tree, so there is no SWE-bench-style
`--output`, `preds.json`, split, or separate evaluation-harness step. Check
`pier run --help` for the output/job-name option exposed by your installed Pier
version if you want a fixed experiment directory name.

Test the base agent on one task:

```bash
uv run --project ../pier pier run -p tasks/igel-persist-feature-schema \
  --agent local-mini-swe-agent \
  --model openrouter/openai/gpt-5-mini \
  --env-file .env
```

Or use a reproducible ten-task base-agent subset:

```bash
uv run --project ../pier pier run -p tasks \
  --agent local-mini-swe-agent \
  --model openrouter/openai/gpt-5-mini \
  --env-file .env --n-tasks 10 --sample-seed 0
```

`local-mini-swe-agent` runs the normal/base `mini` entry point from this fork.
It does not run the tree-search class. Put `OPENROUTER_API_KEY` in `.env`.

## This fork's tree-search agent

Run these commands from this `deep-swe` directory:

```bash
# One task
uv run --project ../pier pier run -p tasks/igel-persist-feature-schema \
  --agent tree-search-mini-swe-agent \
  --model openrouter/openai/gpt-5-mini --env-file .env

# Deterministic smoke subset
uv run --project ../pier pier run -p tasks \
  --agent tree-search-mini-swe-agent \
  --model openrouter/openai/gpt-5-mini \
  --env-file .env --n-tasks 10 --sample-seed 0

# All 113 tasks
uv run --project ../pier pier run -p tasks \
  --agent tree-search-mini-swe-agent \
  --model openrouter/openai/gpt-5-mini --env-file .env
```

Use `--env modal` for parallel Modal sandboxes. Run `pier run --help` for the
installed version's concurrency, output-directory, retry, and filtering flags.
Pier writes per-trial agent and verifier artifacts, including `reward.json`,
`ctrf.json`, test output, and logs.

## Important differences from the SWE-bench experiment

- There is no `--subset verified` or `--split test`.
- `tasks/` is the dataset; a child directory is one task.
- Do not run the SWE-bench evaluation harness afterward. Pier grades each task.
- DeepSWE requires committed work. The adapter commits the agent's changes
  after it exits; `pre_artifacts.sh` then extracts that patch before grading it
  in a pristine verifier container.
