#!/usr/bin/env python3
"""Print alphabetically ordered SWE-bench instance IDs selected by a slice."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_slice(value: str) -> slice:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("use START:STOP or START:STOP:STEP")
    try:
        values = [int(part) if part else None for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError("slice components must be integers") from error
    result = slice(*values)
    if result.step == 0:
        raise argparse.ArgumentTypeError("slice step cannot be zero")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("slice_spec", type=parse_slice)
    args = parser.parse_args()

    instance_ids = sorted(
        path.parent.name for path in args.dataset_dir.glob("*/task.toml")
    )
    selected = instance_ids[args.slice_spec]
    if not selected:
        raise SystemExit(f"Slice selected no instances: {args.slice_spec}")
    print("\n".join(selected))


if __name__ == "__main__":
    main()
