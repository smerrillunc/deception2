#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reasoning_parser import reasoning_close_span_from_text


def think_close_span_from_text(text):
    return reasoning_close_span_from_text(text)


def trim_history_after_think(record):
    close_span = think_close_span_from_text(record.get("raw_text"))
    history = list(record.get("history") or [])
    if close_span is None:
        return history

    _, close_end = close_span
    trimmed = []
    for item in history:
        span = item.get("char_span")
        if isinstance(span, (list, tuple)) and len(span) == 2 and span[1] is not None:
            try:
                span_end = int(span[1])
            except (TypeError, ValueError):
                trimmed.append(item)
                continue
            if span_end <= close_end:
                trimmed.append(item)
        else:
            trimmed.append(item)
    return trimmed


def trim_result_after_think(record):
    trimmed_history = trim_history_after_think(record)
    trimmed = dict(record)
    raw_text = trimmed.get("raw_text")
    close_span = think_close_span_from_text(raw_text)
    if close_span is not None and isinstance(raw_text, str):
        trimmed["raw_text"] = raw_text[: close_span[1]]
    trimmed["history"] = trimmed_history
    trimmed["full_score"] = trimmed_history[-1] if trimmed_history else None
    trimmed["think_close_span"] = close_span
    trimmed["dropped_probe_count"] = len(record.get("history") or []) - len(trimmed_history)
    return trimmed


def default_out_path(localization_dir: Path) -> Path:
    base = localization_dir.name
    if base.startswith("localization"):
        return localization_dir.parent / f"{base}.jsonl"
    return localization_dir / "localization.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a localization JSONL from per-example localization JSON files."
    )
    parser.add_argument(
        "--localization_dir",
        type=str,
        required=True,
        help="Directory containing sentence_localization_*.json files.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output JSONL path. Defaults to <parent>/<localization_dir_name>.jsonl.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="sentence_localization_*.json",
        help="Glob pattern under localization_dir.",
    )
    parser.add_argument(
        "--no_dedupe",
        action="store_true",
        default=False,
        help="If set, keep duplicate example_id records.",
    )
    args = parser.parse_args(argv)

    localization_dir = Path(args.localization_dir)
    if not localization_dir.exists():
        raise FileNotFoundError(f"Localization directory not found: {localization_dir}")
    if not localization_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {localization_dir}")

    out_path = Path(args.out_path) if args.out_path else default_out_path(localization_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    n_in = 0
    n_out = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for path in sorted(localization_dir.glob(args.pattern)):
            rec = trim_result_after_think(json.loads(path.read_text(encoding="utf-8")))
            n_in += 1
            if not args.no_dedupe:
                ex_id = rec.get("example_id")
                if ex_id in seen:
                    continue
                seen.add(ex_id)
            out_f.write(json.dumps(rec) + "\n")
            n_out += 1

    print(f"Localization dir: {localization_dir}")
    print(f"Output JSONL: {out_path}")
    print(f"Input files: {n_in}")
    print(f"Written records: {n_out}")


if __name__ == "__main__":
    main()
