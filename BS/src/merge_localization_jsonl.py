#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge localization JSONL shards.")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Shard JSONL files.")
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--dedupe", action="store_true", default=False, help="Dedupe by example_id.")
    args = parser.parse_args(argv)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    with out_path.open("w", encoding="utf-8") as out:
        for in_path in args.input:
            for rec in read_jsonl(Path(in_path)):
                if args.dedupe:
                    ex_id = rec.get("example_id")
                    if ex_id in seen:
                        continue
                    seen.add(ex_id)
                out.write(json.dumps(rec) + "\n")

    print(f"Wrote merged JSONL: {out_path}")


if __name__ == "__main__":
    main()
