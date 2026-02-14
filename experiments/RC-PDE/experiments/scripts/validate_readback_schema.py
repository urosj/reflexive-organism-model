#!/usr/bin/env python3
"""Validate Paper-14 ExperienceSample fixture records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Mapping

from readback_schema import SCHEMA_VERSION, validate_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("experiments/scripts/readback_schema_fixtures.json"),
        help="Path to fixture JSON array.",
    )
    parser.add_argument(
        "--non-strict-version",
        action="store_true",
        help="Allow schema_version mismatch while validating structure/types.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.fixtures.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("Fixture file must contain a JSON list of records.")

    records: List[Mapping[str, Any]] = []
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Record {idx} is not a JSON object.")
        records.append(record)

    errors = validate_records(
        records,
        strict_schema_version=not args.non_strict_version,
    )

    if errors:
        print(f"[FAIL] {len(errors)} validation error(s)")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"[OK] validated {len(records)} fixture records "
        f"against schema_version={SCHEMA_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
