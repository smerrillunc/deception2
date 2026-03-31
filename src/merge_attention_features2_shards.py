#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


SHARD_NAME_RE = re.compile(r"attention_features2_shard_(\d+)_of_(\d+)\.parquet$")


@dataclass(frozen=True)
class ShardFile:
    path: Path
    shard_id: int
    num_shards: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge attention_features2 shard parquet files into a single "
            "attention_features2.parquet dataset."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more dataset directories or attention_features2_shards directories. "
            "When a dataset directory is provided, the script looks for "
            "<dataset_dir>/attention_features2_shards."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output parquet path. Only valid when merging a single input.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing merged parquet output.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        default=False,
        help="Allow merging an incomplete shard set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report shard completeness and output paths without writing merged parquet files.",
    )
    return parser.parse_args(argv)


def resolve_shard_dir(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.name == "attention_features2_shards":
        return path
    return path / "attention_features2_shards"


def parse_shard_file(path: Path) -> Optional[ShardFile]:
    match = SHARD_NAME_RE.fullmatch(path.name)
    if match is None:
        return None
    shard_id = int(match.group(1))
    num_shards = int(match.group(2))
    return ShardFile(path=path, shard_id=shard_id, num_shards=num_shards)


def discover_shard_files(shard_dir: Path) -> list[ShardFile]:
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Shard directory does not exist: {shard_dir}")

    shard_files: list[ShardFile] = []
    ignored_files: list[Path] = []
    for path in sorted(shard_dir.glob("*.parquet")):
        parsed = parse_shard_file(path)
        if parsed is None:
            ignored_files.append(path)
            continue
        shard_files.append(parsed)

    if ignored_files:
        ignored_text = ", ".join(path.name for path in ignored_files[:5])
        suffix = " ..." if len(ignored_files) > 5 else ""
        raise ValueError(f"Found unexpected parquet files in {shard_dir}: {ignored_text}{suffix}")

    if not shard_files:
        raise FileNotFoundError(f"No shard parquet files found in {shard_dir}")

    totals = {shard.num_shards for shard in shard_files}
    if len(totals) != 1:
        raise ValueError(f"Shard files in {shard_dir} disagree on total shard count: {sorted(totals)}")

    seen_ids: set[int] = set()
    duplicates: list[int] = []
    for shard in shard_files:
        if shard.shard_id in seen_ids:
            duplicates.append(shard.shard_id)
        seen_ids.add(shard.shard_id)
    if duplicates:
        dup_text = ", ".join(str(idx) for idx in sorted(set(duplicates)))
        raise ValueError(f"Duplicate shard ids found in {shard_dir}: {dup_text}")

    return sorted(shard_files, key=lambda shard: shard.shard_id)


def expected_output_path(shard_dir: Path) -> Path:
    return shard_dir.parent / "attention_features2.parquet"


def missing_shard_ids(shard_files: Sequence[ShardFile]) -> list[int]:
    if not shard_files:
        return []
    expected_total = shard_files[0].num_shards
    present = {shard.shard_id for shard in shard_files}
    return [shard_id for shard_id in range(expected_total) if shard_id not in present]


def count_shard_rows(shard: ShardFile) -> int:
    metadata = pq.read_metadata(shard.path)
    return int(metadata.num_rows)


def merge_shard_files(
    shard_files: Sequence[ShardFile],
    output_path: Path,
    *,
    overwrite: bool,
) -> int:
    output_path = output_path.expanduser().resolve()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    if tmp_path.exists():
        tmp_path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    total_rows = 0

    try:
        for shard in shard_files:
            parquet_file = pq.ParquetFile(shard.path)
            shard_schema = parquet_file.schema_arrow
            if writer is None:
                schema = shard_schema
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(tmp_path, schema, compression="snappy")
            elif schema is None or not shard_schema.equals(schema, check_metadata=False):
                raise ValueError(f"Schema mismatch while merging shard: {shard.path}")

            for batch in parquet_file.iter_batches():
                writer.write_table(pa.Table.from_batches([batch], schema=schema))
                total_rows += batch.num_rows

        if writer is None:
            raise RuntimeError("No shard data was written.")
        writer.close()
        writer = None

        if output_path.exists():
            output_path.unlink()
        tmp_path.replace(output_path)
    except Exception:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return total_rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.output is not None and len(args.inputs) != 1:
        raise ValueError("--output may only be used when merging a single input.")

    for raw_input in args.inputs:
        shard_dir = resolve_shard_dir(raw_input)
        shard_files = discover_shard_files(shard_dir)
        missing = missing_shard_ids(shard_files)
        output_path = Path(args.output).expanduser().resolve() if args.output else expected_output_path(shard_dir)
        shard_rows = [count_shard_rows(shard) for shard in shard_files]

        print(f"Shard dir: {shard_dir}")
        print(f"Output parquet: {output_path}")
        print(f"Found shards: {len(shard_files)}/{shard_files[0].num_shards}")
        print(f"Total rows across present shards: {sum(shard_rows)}")
        if missing:
            preview = ", ".join(str(shard_id) for shard_id in missing[:12])
            suffix = " ..." if len(missing) > 12 else ""
            print(f"Missing shard ids: {preview}{suffix}")
        else:
            print("Missing shard ids: none")

        if missing and not args.allow_incomplete:
            raise RuntimeError(
                "Refusing to merge incomplete shard set. "
                "Pass --allow-incomplete if you really want a partial merge."
            )

        if args.dry_run:
            print("Dry run only; no merged parquet written.")
            print("")
            continue

        merged_rows = merge_shard_files(
            shard_files,
            output_path,
            overwrite=args.overwrite,
        )
        print(f"Merged rows written: {merged_rows}")
        print("")


if __name__ == "__main__":
    main()
