# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Normalize DROID LeRobot v3 ``meta/tasks.parquet`` task-text schema.

Some DROID/LeRobot v3 exports accidentally store task text in an unnamed
pandas index. PyArrow materializes that index as ``__index_level_0__`` in
``meta/tasks.parquet`` instead of the semantic ``task`` column expected by the
training dataset loader. This script rewrites only ``meta/tasks.parquet`` to the
canonical schema::

    task_index: int64
    task: string

The original file is moved aside to a timestamped backup before the normalized
file is installed. No metadata outside ``meta/tasks.parquet`` is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


_INDEX_ARTIFACT_COLUMN = "__index_level_0__"
_CANONICAL_TASK_COLUMN = "task"
_TASK_INDEX_COLUMN = "task_index"


def _log(message: str, json_mode: bool = False) -> None:
    if not json_mode:
        print(message, file=sys.stderr)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_backup_path(path: Path, suffix: str | None = None) -> Path:
    stamp = suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.__index_level_0__.backup_{stamp}")
    if backup.exists():
        raise FileExistsError(f"Backup path already exists: {backup}")
    return backup


def _normalize_tasks_table(table: pa.Table) -> tuple[pa.Table, str]:
    columns = table.column_names
    if _TASK_INDEX_COLUMN not in columns:
        raise KeyError(f"tasks.parquet missing '{_TASK_INDEX_COLUMN}'; got columns {columns}")

    if _CANONICAL_TASK_COLUMN in columns:
        task_text_column = _CANONICAL_TASK_COLUMN
    elif _INDEX_ARTIFACT_COLUMN in columns:
        task_text_column = _INDEX_ARTIFACT_COLUMN
    else:
        raise KeyError(
            "tasks.parquet must contain either 'task' or '__index_level_0__' "
            f"as the task text column; got columns {columns}"
        )

    normalized = pa.table(
        {
            _TASK_INDEX_COLUMN: table[_TASK_INDEX_COLUMN],
            _CANONICAL_TASK_COLUMN: table[task_text_column].cast(pa.string()),
        }
    )
    return normalized, task_text_column


def normalize_tasks_parquet(
    dataset_root: Path,
    dry_run: bool = False,
    backup_suffix: str | None = None,
    json_mode: bool = False,
) -> dict:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing {tasks_path}")

    table = pq.read_table(tasks_path)
    normalized, task_text_column = _normalize_tasks_table(table)

    result = {
        "dataset_root": str(dataset_root),
        "tasks_path": str(tasks_path),
        "original_columns": table.column_names,
        "task_text_source_column": task_text_column,
        "normalized_columns": normalized.column_names,
        "row_count": normalized.num_rows,
        "original_size_bytes": tasks_path.stat().st_size,
        "original_sha256": _sha256(tasks_path),
        "changed": task_text_column != _CANONICAL_TASK_COLUMN or table.column_names != normalized.column_names,
        "backup_path": None,
        "new_size_bytes": None,
        "new_sha256": None,
        "sample_rows": normalized.to_pylist()[:3],
    }

    if not result["changed"]:
        _log(f"{tasks_path} already uses canonical columns {normalized.column_names}; no rewrite needed.", json_mode)
        return result

    backup_path = _unique_backup_path(tasks_path, backup_suffix)
    tmp_path = tasks_path.with_name(f"{tasks_path.name}.normalized_{backup_path.name.rsplit('_', 1)[-1]}.tmp")
    result["backup_path"] = str(backup_path)

    if dry_run:
        _log("Dry run: would move original to backup and install normalized tasks.parquet.", json_mode)
        return result

    pq.write_table(normalized, tmp_path)

    # Validate the temp file before swapping it into place.
    check = pq.read_table(tmp_path)
    if check.column_names != [_TASK_INDEX_COLUMN, _CANONICAL_TASK_COLUMN]:
        raise RuntimeError(f"Normalized schema mismatch: {check.column_names}")
    if check.num_rows != table.num_rows:
        raise RuntimeError(f"Row count mismatch: {check.num_rows} != {table.num_rows}")

    # Preserve the original file, then install the normalized file. This uses
    # rename rather than deletion so the original dataset metadata is recoverable.
    os.rename(tasks_path, backup_path)
    os.rename(tmp_path, tasks_path)

    result["new_size_bytes"] = tasks_path.stat().st_size
    result["new_sha256"] = _sha256(tasks_path)
    _log(f"Normalized {tasks_path}", json_mode)
    _log(f"Original backup: {backup_path}", json_mode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize DROID/LeRobot v3 meta/tasks.parquet from pandas index artifact to task column."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the LeRobot dataset split root, e.g. /path/to/Cosmos3-DROID/success.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect schema and report the planned backup/rewrite without changing files.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=None,
        help="Optional deterministic suffix for the backup file name. Defaults to current timestamp.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary for experiment logs.",
    )
    args = parser.parse_args()

    result = normalize_tasks_parquet(
        args.dataset_root,
        dry_run=args.dry_run,
        backup_suffix=args.backup_suffix,
        json_mode=args.json,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"tasks_path: {result['tasks_path']}")
        print(f"original_columns: {result['original_columns']}")
        print(f"task_text_source_column: {result['task_text_source_column']}")
        print(f"normalized_columns: {result['normalized_columns']}")
        print(f"row_count: {result['row_count']}")
        print(f"changed: {result['changed']}")
        if result["backup_path"]:
            print(f"backup_path: {result['backup_path']}")
        if result["new_sha256"]:
            print(f"new_sha256: {result['new_sha256']}")


if __name__ == "__main__":
    main()
