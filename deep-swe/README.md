# Running the tree-search agent on DeepSWE

DeepSWE uses Harbor tasks and program-based verifiers, not Hugging Face
SWE-bench rows. Consequently `mini-extra swebench-ts` is not the correct
runner. Pier prepares each repository, invokes the agent, extracts its Git
patch, and runs the held-out verifier.

## Prerequisites

- Linux Docker containers (Docker Desktop with Linux containers is fine).
- Python 3.11+ and `uv`.
- Pier 0.3.0 or newer: `uv tool install datacurve-pier`.
- API variables required by `config/extra/swebench_ts.yaml`. The checked-in
  reward model currently uses `http://10.141.10.34:3000/v1`; that endpoint
  must be reachable from the task container, or change the reward-model config.

The custom Pier adapter installs this fork at the pinned `REVISION` in
`pier_agent.py`. Update that constant after pushing future agent changes.

## Original/base mini-swe-agent

This is the DeepSWE equivalent of your SWE-bench base command:

```bash
mini-extra swebench --subset verified --split test \
  --output output/verified.test.base/deepseek__deepseek-v4-flash/
```

For DeepSWE, use the custom adapter that installs this repository rather than
Pier's built-in mini-swe-agent package:

```bash
# Run from mini-swe-agent/deep-swe
pier run -p tasks \
  --agent pier_agent:LocalMiniSweAgent \
  --model deepseek/deepseek-v4-flash
```

Pier creates its own job/trial output tree, so there is no SWE-bench-style
`--output`, `preds.json`, split, or separate evaluation-harness step. Check
`pier run --help` for the output/job-name option exposed by your installed Pier
version if you want a fixed experiment directory name.

Before spending on the complete benchmark, test the base agent on one task:

```bash
pier run -p tasks/igel-persist-feature-schema \
  --agent pier_agent:LocalMiniSweAgent \
  --model deepseek/deepseek-v4-flash
```

Or use a reproducible ten-task base-agent subset:

```bash
pier run -p tasks \
  --agent pier_agent:LocalMiniSweAgent \
  --model deepseek/deepseek-v4-flash \
  --n-tasks 10 --sample-seed 0
```

`LocalMiniSweAgent` runs the normal/base `mini` entry point, but installs it
from `REPOSITORY` at the exact `REVISION` declared in `pier_agent.py`. It does
not run the tree-search class. Set `OPENROUTER_API_KEY` before running these
commands.

## This fork's tree-search agent

Run these commands from this `deep-swe` directory:

```bash
# One task
pier run -p tasks/igel-persist-feature-schema \
  --agent pier_agent:TreeSearchMiniSweAgent \
  --model deepseek/deepseek-v4-flash

# Deterministic smoke subset
pier run -p tasks \
  --agent pier_agent:TreeSearchMiniSweAgent \
  --model deepseek/deepseek-v4-flash \
  --n-tasks 10 --sample-seed 0

# All 113 tasks
pier run -p tasks \
  --agent pier_agent:TreeSearchMiniSweAgent \
  --model deepseek/deepseek-v4-flash
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
