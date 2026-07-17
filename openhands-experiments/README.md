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
Linux containers. You also need Git, Python 3.12, Poetry, and a funded
OpenRouter account. A full 500-instance run can consume substantial API
credits, disk space, and time.

This machine did not have WSL, Docker, or Poetry when this setup was created, so
dependency installation and a live smoke run could not be completed here.

OpenHands 0.59 requires Python 3.12 and Poetry 1.8 or newer; its project pins
Poetry `^2.1.2`. Poetry creates the isolated Python environment used by the
evaluation runner. Docker is still used separately for SWE-bench task
containers.

Install the pinned Poetry version with `pipx`:

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
# Start a new shell after ensurepath, then:
pipx install poetry==2.1.2
poetry --version
```

## Setup

From a Linux/WSL shell at the repository root:

```bash
git submodule update --init --recursive
bash openhands-experiments/setup.sh
cp openhands-experiments/.env.example \
  openhands-experiments/.env
# Edit .env and replace the placeholder with your OpenRouter key.
bash openhands-experiments/run.sh
```

Do not skip `setup.sh`: it uses Poetry to install OpenHands and its evaluation
dependency group before inference starts.

The setup intentionally does not run OpenHands' full `make build`. That target
installs the web frontend and uses Playwright's `--with-deps` option, which can
request sudo access for Chromium system packages. Browsing is disabled in this
experiment, so those components are unnecessary.

The default run is deliberately a single Verified instance with one worker.
The runner automatically loads `openhands-experiments/.env`. The real
`.env` is ignored by Git, while `.env.example` is safe to commit. Its contents
should be:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-your-real-key
```

You can still override the file for a particular run with
`ENV_FILE=/path/to/another.env`.

Run a larger sample by overriding the environment variables:

```bash
EVAL_LIMIT=10 NUM_WORKERS=2 bash openhands-experiments/run.sh
```

Run all 500 Verified instances by passing an empty evaluation limit:

```bash
EVAL_LIMIT= NUM_WORKERS=4 bash openhands-experiments/run.sh
```

OpenHands writes this experiment's inference artifacts beneath:

```text
openhands/evaluation/evaluation_outputs/outputs/
  princeton-nlp__SWE-bench_Verified-test/
    CodeActAgent/
      gpt-5-mini_maxiter_50_N_0.59.0-no-hint-openrouter-gpt5-mini-50step-run_1/
```

The run directory contains `output.jsonl`, `metadata.json`, per-instance
`infer_logs/`, and per-instance `llm_completions/`. Re-running with the same
settings resumes/skips completed instances according to the harness.

## Evaluate patches

Inference produces predictions; benchmark scoring is a separate, Docker-based
step. The experiment wrapper converts OpenHands output to SWE-bench prediction
format, invokes the official SWE-bench harness, and prints the resolved rate:

```bash
# Automatically score the newest Verified output.jsonl:
bash openhands-experiments/evaluate.sh

# Or score an explicit inference output:
bash openhands-experiments/evaluate.sh \
  openhands/evaluation/evaluation_outputs/outputs/\
princeton-nlp__SWE-bench_Verified-test/CodeActAgent/\
gpt-5-mini_maxiter_50_N_v0.59.0-no-hint-openrouter-gpt5-mini-50step-run_1/\
output.jsonl
```

The evaluator downloads official per-instance SWE-bench Docker images and can
require substantial disk space. It writes `report.json`, `eval_outputs/`, and
an updated `output.jsonl` beside the inference output. For a partial run,
"resolve rate over submitted" is the sample score; "overall Verified accuracy"
uses all 500 Verified tasks as the denominator. A full benchmark result should
have 500 submitted instances.
