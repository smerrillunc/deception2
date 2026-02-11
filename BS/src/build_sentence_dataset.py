#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from deception_dataset import iter_deception_records
from sentence_pipeline import build_sentence_records, write_jsonl


def _example_id(rec: Dict) -> str | None:
    if "record_id" in rec:
        return rec["record_id"]
    state_id = rec.get("state_id")
    sample_idx = rec.get("sample_idx")
    run_id = rec.get("run_id", "run")
    if state_id is not None and sample_idx is not None:
        return f"{run_id}/state_{state_id}/sample_{sample_idx}"
    return None


def _write_examples(
    records: Iterable[Dict],
    out_path: Path,
    text_field: str,
    only_deceptive: bool,
    include_messages: bool,
) -> Iterable[Dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            if only_deceptive and rec.get("deceptive") is not True:
                continue
            example_id = _example_id(rec)
            if not example_id:
                continue
            rec["example_id"] = example_id
            if not include_messages and "messages" in rec:
                rec.pop("messages")
            if text_field not in rec:
                continue
            f.write(json.dumps(rec) + "\n")
            yield rec


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build sentence-level dataset from DeceptionMining JSONL.")
    parser.add_argument("--input_root", type=str, required=True, help="Root of DeceptionMining outputs.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--text_field", type=str, default="action_reasoning")
    parser.add_argument("--include_messages", action="store_true", default=False)
    parser.add_argument("--only_deceptive", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    examples_path = out_dir / "examples.jsonl"
    sentences_path = out_dir / "sentences.jsonl"

    records_iter = iter_deception_records(
        args.input_root,
        include_messages=args.include_messages,
        include_action=True,
        flatten_action=True,
        include_meta=True,
        strict_json=True,
    )

    def _limited(records):
        count = 0
        for rec in records:
            if args.limit and count >= args.limit:
                break
            yield rec
            count += 1

    examples_iter = _write_examples(
        _limited(records_iter),
        examples_path,
        text_field=args.text_field,
        only_deceptive=args.only_deceptive,
        include_messages=args.include_messages,
    )

    sentences = build_sentence_records(
        examples_iter,
        text_field=args.text_field,
        example_id_field="example_id",
        include_example_fields=["deceptive", "naturally_deceptive", "current_rank", "truth_context", "prompt", "game_id", "turn_idx", "game_type"],
    )
    write_jsonl(sentences, sentences_path)

    print(f"Wrote examples: {examples_path}")
    print(f"Wrote sentences: {sentences_path}")


if __name__ == "__main__":
    main()
