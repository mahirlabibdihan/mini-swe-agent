#!/usr/bin/env python3
"""Extend an existing Pier job with missing tasks from a dataset selection."""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from pier.models.job.config import DatasetConfig, JobConfig
from pier.models.trial.config import TaskConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--n-tasks", type=int, required=True)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--n-concurrent", type=int, required=True)
    parser.add_argument("--disable-verification", action="store_true")
    return parser.parse_args()


async def resolve_saved_tasks(config: JobConfig) -> list[TaskConfig]:
    tasks = [task.model_copy(deep=True) for task in config.tasks]
    for dataset in config.datasets:
        tasks.extend(
            await dataset.get_task_configs(
                disable_verification=config.verifier.disable
            )
        )
    return tasks


def task_name(task: TaskConfig) -> str:
    return task.get_task_id().get_name()


def remove_model_selection_failures(job_dir: Path) -> int:
    failure_text = "There's an issue with the selected model"
    failed_trial_dirs: list[Path] = []
    for output_path in job_dir.glob("*/agent/claude-code.txt"):
        if failure_text in output_path.read_text(errors="replace"):
            failed_trial_dirs.append(output_path.parent.parent)

    for trial_dir in failed_trial_dirs:
        print(f"Removing model-selection failure for retry: {trial_dir.name}")
        shutil.rmtree(trial_dir)
    return len(failed_trial_dirs)


async def extend(args: argparse.Namespace) -> int:
    config_path = args.job_dir / "config.json"
    config = JobConfig.model_validate_json(config_path.read_text())
    remove_model_selection_failures(args.job_dir)
    existing_tasks = await resolve_saved_tasks(config)
    requested_tasks = await DatasetConfig(
        path=args.dataset_dir,
        n_tasks=args.n_tasks,
        sample_seed=args.sample_seed,
    ).get_task_configs(disable_verification=args.disable_verification)

    selection_label = (
        f"seeded selection (seed={args.sample_seed})"
        if args.sample_seed is not None
        else "alphabetical instance-id selection"
    )
    print(f"Requested {selection_label}:")
    for index, task in enumerate(requested_tasks, start=1):
        print(f"  {index:>3}. {task_name(task)}")

    combined = list(existing_tasks)
    seen = {task_name(task) for task in existing_tasks}
    additions = [task for task in requested_tasks if task_name(task) not in seen]
    combined.extend(additions)

    concurrency_changed = config.n_concurrent_trials != args.n_concurrent
    if not additions and not concurrency_changed:
        print("Requested instances are already present in the job.")
        return 0

    config.tasks = combined
    config.datasets = []
    config.n_concurrent_trials = args.n_concurrent

    # Pier recreates this derived manifest from the expanded task union.
    lock_path = args.job_dir / "lock.json"
    if lock_path.exists():
        lock_path.replace(args.job_dir / "lock.pre-extension.json")

    temporary_path = config_path.with_suffix(".json.tmp")
    temporary_path.write_text(config.model_dump_json(indent=4))
    temporary_path.replace(config_path)

    if additions:
        print(
            f"Added {len(additions)} missing instances; "
            f"the job now contains {len(combined)} instances."
        )
    else:
        print(f"Updated concurrency to {args.n_concurrent} workers.")
    return len(additions)


def main() -> None:
    args = parse_args()
    asyncio.run(extend(args))


if __name__ == "__main__":
    main()
