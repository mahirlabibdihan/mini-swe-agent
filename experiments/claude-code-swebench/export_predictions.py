#!/usr/bin/env python3
"""Export Claude Code trial patches into the official SWE-bench JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path, help="Pier job directory")
    parser.add_argument("output_jsonl", type=Path, help="Destination predictions JSONL")
    parser.add_argument(
        "--model-name",
        default="claude-code-gpt5-mini",
        help="model_name_or_path value for each prediction",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing predictions JSONL file",
    )
    return parser.parse_args()


def instance_id_from_trial_dir(trial_dir: Path) -> str:
    try:
        instance_id, _trial_suffix = trial_dir.name.rsplit("__", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"Unexpected trial directory name: {trial_dir.name}") from error
    return instance_id


def main() -> None:
    args = parse_args()
    if not args.job_dir.is_dir():
        raise SystemExit(f"Job directory does not exist: {args.job_dir}")
    if args.output_jsonl.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.output_jsonl}")

    predictions: list[dict[str, str]] = []
    seen_instance_ids: set[str] = set()
    for patch_path in sorted(args.job_dir.glob("*/agent/model.patch")):
        trial_dir = patch_path.parent.parent
        instance_id = instance_id_from_trial_dir(trial_dir)
        if instance_id in seen_instance_ids:
            raise SystemExit(f"Duplicate trial for instance: {instance_id}")
        seen_instance_ids.add(instance_id)
        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": args.model_name,
                "model_patch": patch_path.read_text(encoding="utf-8"),
            }
        )

    if not predictions:
        raise SystemExit(
            "No agent/model.patch files found. Run the updated patch-only workflow first."
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for prediction in predictions:
            output.write(json.dumps(prediction) + "\n")
    print(f"Wrote {len(predictions)} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
