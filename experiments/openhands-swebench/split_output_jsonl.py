#!/usr/bin/env python3
"""Split an OpenHands output.jsonl file into one JSON file per instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="OpenHands output.jsonl file")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("output"),
        help="Directory for <instance_id>.json files (default: ./output)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing per-instance JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise SystemExit(f"Input JSONL file does not exist: {args.input_jsonl}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.input_jsonl.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"Invalid JSON on line {line_number}: {error.msg}"
                ) from error

            instance_id = record.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise SystemExit(f"Missing valid instance_id on line {line_number}")
            if Path(instance_id).name != instance_id:
                raise SystemExit(
                    f"Unsafe instance_id on line {line_number}: {instance_id!r}"
                )

            destination = args.output_dir / f"{instance_id}.json"
            if destination.exists() and not args.overwrite:
                raise SystemExit(
                    f"Refusing to overwrite {destination}. Use --overwrite if intended."
                )

            with destination.open("w", encoding="utf-8") as target:
                json.dump(record, target, indent=2, ensure_ascii=False)
                target.write("\n")
            written += 1

    print(f"Wrote {written} instance files to {args.output_dir}")


if __name__ == "__main__":
    main()
