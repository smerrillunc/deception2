#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from deception_dataset import (
    LABEL_FILTER_CHOICES,
    iter_deception_records,
    keep_record_for_label_filter,
    normalize_label_filter,
)
from sentence_pipeline import build_sentence_records, write_jsonl


def _example_id(rec: Dict) -> Optional[str]:
    if "record_id" in rec:
        return rec["record_id"]
    state_id = rec.get("state_id")
    sample_idx = rec.get("sample_idx")
    run_id = rec.get("run_id", "run")
    if state_id is not None and sample_idx is not None:
        return f"{run_id}/state_{state_id}/sample_{sample_idx}"
    return None


def _pick_text(rec: Dict, primary_field: str, fallback_field: Optional[str]) -> Optional[str]:
    text = rec.get(primary_field)
    if isinstance(text, str) and text.strip():
        return text

    if fallback_field:
        alt = rec.get(fallback_field)
        if isinstance(alt, str) and alt.strip():
            return alt

    return None


def _coerce_deceptive_label(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "deceptive"}:
            return True
        if v in {"0", "false", "f", "no", "n", "truthful", "non-deceptive", "non_deceptive"}:
            return False

    return None


def _write_examples(
    records: Iterable[Dict],
    out_path: Path,
    text_field: str,
    fallback_text_field: Optional[str],
    label_filter: str,
    limit: int,
    include_messages: bool,
    target_deceptive: int,
    target_truthful: int,
) -> Iterable[Dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    deceptive_written = 0
    truthful_written = 0
    use_target_deceptive = target_deceptive > 0 and label_filter != "truthful_only"
    use_target_truthful = target_truthful > 0 and label_filter != "deceptive_only"

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            if not keep_record_for_label_filter(rec, label_filter):
                continue

            deceptive_label = _coerce_deceptive_label(rec.get("deceptive"))
            if (use_target_deceptive or use_target_truthful) and deceptive_label is None:
                continue
            if use_target_deceptive and deceptive_label is True and deceptive_written >= target_deceptive:
                continue
            if use_target_truthful and deceptive_label is False and truthful_written >= target_truthful:
                continue

            example_id = _example_id(rec)
            if not example_id:
                continue

            text = _pick_text(rec, text_field, fallback_text_field)
            if not text:
                continue

            rec["example_id"] = example_id
            rec[text_field] = text

            if not include_messages and "messages" in rec:
                rec.pop("messages")

            f.write(json.dumps(rec) + "\n")
            yield rec
            written += 1

            if deceptive_label is True:
                deceptive_written += 1
            elif deceptive_label is False:
                truthful_written += 1

            if (
                ((not use_target_deceptive) or deceptive_written >= target_deceptive)
                and ((not use_target_truthful) or truthful_written >= target_truthful)
            ):
                break

            if limit and written >= limit:
                break


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build sentence-level dataset from DeceptionMining JSONL.")
    parser.add_argument("--input_root", type=str, required=True, help="Root of DeceptionMining outputs.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--text_field", type=str, default="action_reasoning")
    parser.add_argument("--fallback_text_field", type=str, default="action_raw_text")
    parser.add_argument("--include_messages", action="store_true", default=False)
    parser.add_argument("--label_filter", type=str, choices=LABEL_FILTER_CHOICES, default="all")
    parser.add_argument("--only_deceptive", action="store_true", default=False)
    parser.add_argument("--only_truthful", action="store_true", default=False)
    parser.add_argument("--target_deceptive", type=int, default=3000)
    parser.add_argument("--target_truthful", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    label_filter = normalize_label_filter(
        args.label_filter,
        only_deceptive=args.only_deceptive,
        only_truthful=args.only_truthful,
    )

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
        label_filter=label_filter,
    )

    examples_iter = _write_examples(
        records_iter,
        examples_path,
        text_field=args.text_field,
        fallback_text_field=args.fallback_text_field,
        label_filter=label_filter,
        limit=args.limit,
        include_messages=args.include_messages,
        target_deceptive=args.target_deceptive,
        target_truthful=args.target_truthful,
    )

    sentences = build_sentence_records(
        examples_iter,
        text_field=args.text_field,
        example_id_field="example_id",
        include_example_fields=[
            "deceptive",
            "current_rank",
            "prompt",
            "game_id",
            "turn_idx",
            "game_type",
            "truth_context",
        ],
    )
    write_jsonl(sentences, sentences_path)

    print(f"Wrote examples: {examples_path}")
    print(f"Wrote sentences: {sentences_path}")
    print(f"Label filter: {label_filter}")
    print(f"Target deceptive: {args.target_deceptive}")
    print(f"Target truthful: {args.target_truthful}")


if __name__ == "__main__":
    main()
