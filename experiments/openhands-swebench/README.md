# OpenHands on SWE-bench Verified

This repository includes the original `OpenHands/OpenHands` codebase as a Git
submodule pinned at release
`0.59.0` (commit `d39f7ae0e66174c50bdc714304fed4078b5e3b72`). That release is pinned because
it contains the legacy SWE-bench evaluation runner. The current OpenHands
evaluation work is migrating to the separate `OpenHands/benchmarks` repository.

The fixed experiment settings are:

- dataset: `princeton-nlp/SWE-bench_Verified`, split `test`
- agent: `CodeActAgent`
- model: `openrouter/openai/gpt-5-mini` through OpenRouter
- limit: 50 agent iterations per instance
- browsing and hints: disabled (the runner defaults)

## Prerequisites

Run this on Linux, or on Windows using WSL 2 and Docker Desktop configured for
Linux containers. You also need Git, GNU Make, Python, Node.js, Poetry, and a
funded OpenRouter account. A full 500-instance run can consume substantial API
credits, disk space, and time.

This machine did not have WSL, Docker, or Poetry when this setup was created, so
dependency installation and a live smoke run could not be completed here.

## Setup

From a Linux/WSL shell at the repository root:

```bash
git submodule update --init --recursive
bash experiments/openhands-swebench/setup.sh
cp experiments/openhands-swebench/.env.example \
  experiments/openhands-swebench/.env
# Edit .env and replace the placeholder with your OpenRouter key.
bash experiments/openhands-swebench/run.sh
```

The default run is deliberately a single Verified instance with one worker.
The runner automatically loads `experiments/openhands-swebench/.env`. The real
`.env` is ignored by Git, while `.env.example` is safe to commit. Its contents
should be:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-your-real-key
```

You can still override the file for a particular run with
`ENV_FILE=/path/to/another.env`.

Run a larger sample by overriding the environment variables:

```bash
EVAL_LIMIT=10 NUM_WORKERS=2 bash experiments/openhands-swebench/run.sh
```

Run all 500 Verified instances by passing an empty evaluation limit:

```bash
EVAL_LIMIT= NUM_WORKERS=4 bash experiments/openhands-swebench/run.sh
```

OpenHands writes inference artifacts beneath
`openhands/evaluation/evaluation_outputs/outputs/swe_bench/`. Re-running with
the same settings resumes/skips completed instances according to the harness.

## Evaluate patches

Inference produces predictions; benchmark scoring is a separate, Docker-based
step. Follow the pinned checkout's
`evaluation/benchmarks/swe_bench/README.md` instructions for the generated
output directory. Start with the one-instance smoke result before launching or
scoring the full dataset.
