# SWE-Xplorer

**Iterative Tree Search with Cross-Path Reconciliation for Autonomous Software Engineering**

SWE-Xplorer is a research framework for exploring test-time search and scaling
strategies for software-engineering agents. It replaces the usual single linear
agent trajectory with structured exploration over a tree of coding trajectories.

The system uses mini-SWE-agent as its lightweight backbone and adds:

- path-level best-first search over complete investigation trajectories;
- iteration-based pruning to balance exploration and exploitation;
- cross-path reconciliation to combine complementary evidence before pruning;
- action-specific rewards for reads, searches, edits, tests, and submissions;
- lightweight Git-based code-state restoration;
- deferred termination and recursive tournament voting for final patch selection.

This repository is the working research artifact for the **SWE-Xplorer** paper.
It also provides reproducible experiment setups for multiple agent scaffolds,
including the SWE-Xplorer tree-search agent, base mini-SWE-agent, OpenHands, and
Claude Code.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/minisweagent/agents/` | SWE-Xplorer search, reward-guided, and base agent implementations |
| `src/minisweagent/run/extra/` | SWE-bench batch and single-instance runners |
| `src/minisweagent/config/extra/` | SWE-Xplorer and SWE-bench experiment configurations |
| `deep-swe/` | Harbor/Pier workflows for base and tree-search agents on DeepSWE tasks |
| `experiments/openhands-swebench/` | OpenHands baseline on SWE-bench Verified |
| `experiments/claude-code-swebench/` | Claude Code baseline using GPT-5 mini on SWE-bench Verified |
| `pier/` | Pier evaluation framework, pinned as a submodule |
| `openhands/` | OpenHands release used by the baseline, pinned as a submodule |
| `swebench/` | SWE-bench evaluation tooling |
| `output/`, `evaluation/` | Local predictions, trajectories, and evaluation artifacts |

## Installation

Linux is recommended. On Windows, use WSL 2 with Docker Desktop configured for
Linux containers.

Requirements:

- Git
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Docker for sandboxed benchmark execution
- provider credentials such as `OPENROUTER_API_KEY`

Clone with the pinned experiment dependencies:

```bash
git clone --recurse-submodules https://github.com/mahirlabibdihan/mini-swe-agent.git
cd mini-swe-agent
```

For an existing clone:

```bash
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

Install the main project:

```bash
uv sync
```

Conda is not required. `uv` manages the project environment directly.

## SWE-bench Experiments

Put the required API credentials in your environment or a local `.env` file.
Tree search and the linear backbone use different inference commands and
different evaluation entry points.

### SWE-Xplorer tree search

The checked-in tree-search configuration is
`src/minisweagent/config/extra/swebench_ts.yaml`. Run tree-search inference on
SWE-bench Verified with:

```bash
mini-extra swebench-ts \
  --subset verified \
  --split test \
  --output output/verified.test.60/Qwen__Qwen2.5-7B-Instruct/
```

For an interactive single-instance run, use `mini-extra swebench-ts-single`.
Tree-search inference writes per-instance tree artifacts beneath the output
directory. Evaluate the directory with the tree-search harness:

```bash
python -m swebench.harness.run_evaluation_ts \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_dir output/verified.test.60/Qwen__Qwen2.5-7B-Instruct/ \
  --max_workers 5 \
  --run_id verified.test.60 \
  --report_dir evaluation \
  --cache_level instance
```

`run_evaluation_ts` evaluates candidate patches stored in the per-instance
`*.tree.json` artifacts and records pass/fail information back into the trees.

### Linear mini-SWE-agent backbone

Run the base agent on SWE-bench Verified with:

```bash
mini-extra swebench \
  --subset verified \
  --split test \
  --output output/verified.test.64/openai__gpt-5-mini/
```

The base runner produces a standard `preds.json`. Evaluate it with the official
SWE-bench harness:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path output/verified.test.35/openai__gpt-5-mini/preds.json \
  --max_workers 5 \
  --run_id verified.test.35 \
  --report_dir evaluation \
  --cache_level instance
```

The run IDs in these examples are experiment labels. Keep each inference output,
evaluation `run_id`, model name, configuration, and step budget aligned in real
runs.

## DeepSWE Experiments

DeepSWE uses Harbor task directories and program-based verifiers rather than
Hugging Face SWE-bench rows. Pier prepares each repository, invokes the agent,
extracts its patch, and executes the task verifier.

From `deep-swe/`, run the SWE-Xplorer tree-search agent:

```bash
uv run --project ../pier pier run -p tasks \
  --agent tree-search-mini-swe-agent \
  --model openrouter/openai/gpt-5-mini \
  --env-file .env
```

Run the linear mini-SWE-agent backbone under the same evaluator:

```bash
uv run --project ../pier pier run -p tasks \
  --agent local-mini-swe-agent \
  --model openrouter/openai/gpt-5-mini \
  --env-file .env
```

See [`deep-swe/README.md`](deep-swe/README.md) for single-task runs, sampling,
network-filesystem notes, artifacts, and verifier behavior.

## OpenHands Baseline

The OpenHands experiment uses `CodeActAgent`, GPT-5 mini through OpenRouter, a
50-iteration limit, and SWE-bench Verified:

```bash
bash experiments/openhands-swebench/setup.sh
cp experiments/openhands-swebench/.env.example \
  experiments/openhands-swebench/.env
# Add OPENROUTER_API_KEY to the new .env file.

EVAL_LIMIT=10 NUM_WORKERS=2 \
  bash experiments/openhands-swebench/run.sh
```

Evaluate its patches with:

```bash
bash experiments/openhands-swebench/evaluate.sh
```

Full instructions are in
[`experiments/openhands-swebench/README.md`](experiments/openhands-swebench/README.md).

## Claude Code Baseline

This experiment runs the official Claude Code CLI through Pier while routing
all model roles to `openai/gpt-5-mini` through OpenRouter. Each task has a
50-turn limit.

```bash
bash experiments/claude-code-swebench/setup.sh
cp experiments/claude-code-swebench/.env.example \
  experiments/claude-code-swebench/.env
# Add OPENROUTER_API_KEY to the new .env file.

N_TASKS=10 N_CONCURRENT=2 \
  bash experiments/claude-code-swebench/run.sh
```

Omit `SAMPLE_SEED` to take tasks in dataset order. Set it explicitly only for
a reproducible shuffled sample. See
[`experiments/claude-code-swebench/README.md`](experiments/claude-code-swebench/README.md)
for prediction export and official SWE-bench evaluation.

## Interactive mini-SWE-agent

The upstream-compatible linear agent remains available for interactive use and
controlled backbone comparisons:

```bash
uv run mini --model openrouter/openai/gpt-5-mini
```

The original mini-SWE-agent documentation is available at
[mini-swe-agent.com](https://mini-swe-agent.com/latest/). This fork retains the
small bash-only agent design while extending it with the SWE-Xplorer search
stack and experiment infrastructure.

## Reproducibility Notes

- Record the parent repository commit and both submodule commits for every run.
- Keep model names, search budgets, step limits, and sampling seeds with results.
- Do not commit `.env` files or API keys.
- Full benchmark runs are costly and can consume substantial Docker storage.
- Use the same task ordering and evaluator version for comparisons.
- Pier writes per-trial trajectories, patches, metrics, and verifier artifacts.


## Acknowledgments

SWE-Xplorer builds on [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent), [SWE-bench](https://github.com/SWE-bench/SWE-bench) and [WebOperator](https://github.com/kagnlp/WebOperator).

## License

This repository retains the licensing terms of the upstream mini-SWE-agent project. Submodules and third-party components are governed by their respective licenses.
