#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from sentence_pipeline import read_jsonl, write_jsonl


WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
NEGATION_RE = re.compile(r"\b(no|not|never|n't|none|nothing|neither|nor)\b", re.IGNORECASE)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return mean(values)


def text_features(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        text = ""
    chars = len(text)
    words = WORD_RE.findall(text)
    word_count = len(words)
    digit_count = sum(ch.isdigit() for ch in text)
    alpha_count = sum(ch.isalpha() for ch in text)
    upper_count = sum(ch.isupper() for ch in text)
    upper_ratio = (upper_count / alpha_count) if alpha_count else 0.0
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0
    punct_counts = {
        "punct_period": text.count("."),
        "punct_comma": text.count(","),
        "punct_qmark": text.count("?"),
        "punct_exclaim": text.count("!"),
        "punct_colon": text.count(":"),
        "punct_semicolon": text.count(";"),
    }
    return {
        "char_count": chars,
        "word_count": word_count,
        "digit_count": digit_count,
        "upper_ratio": upper_ratio,
        "avg_word_len": avg_word_len,
        "negation_count": len(NEGATION_RE.findall(text)),
        **punct_counts,
    }


def load_localization_history(loc_source: Optional[str]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Returns: {example_id: {sentence_idx: history_entry}}
    """
    if not loc_source:
        return {}

    loc_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    path = Path(loc_source)
    files: List[Path] = []

    if path.is_file() and path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                example_id = data.get("example_id")
                history = data.get("history") or []
                if not example_id:
                    continue
                loc_map[example_id] = {int(h["sentence_idx"]): h for h in history if "sentence_idx" in h}
        return loc_map

    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        example_id = data.get("example_id")
        history = data.get("history") or []
        if not example_id:
            continue
        loc_map[example_id] = {int(h["sentence_idx"]): h for h in history if "sentence_idx" in h}

    return loc_map


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract sentence-level features for deception prediction.")
    parser.add_argument("--examples_path", type=str, required=True)
    parser.add_argument("--sentences_path", type=str, required=True)
    parser.add_argument("--tags_path", type=str, default=None)
    parser.add_argument("--localization_path", type=str, default=None, help="JSON file or directory of sentence localization outputs.")
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--context_window", type=int, default=1)
    parser.add_argument("--label_strategy", type=str, default="delta_threshold",
                        choices=["delta_threshold", "rate_threshold", "top_k", "none"])
    parser.add_argument("--delta_threshold", type=float, default=0.1)
    parser.add_argument("--rate_threshold", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args(argv)

    examples = {ex["example_id"]: ex for ex in read_jsonl(args.examples_path) if "example_id" in ex}

    sentences_by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in read_jsonl(args.sentences_path):
        if "example_id" in s:
            sentences_by_example[s["example_id"]].append(s)
    for ex_id, items in sentences_by_example.items():
        items.sort(key=lambda x: x.get("sentence_idx", 0))

    tags_by_sentence: Dict[str, Dict[str, Any]] = {}
    if args.tags_path:
        for t in read_jsonl(args.tags_path):
            if "sentence_id" in t:
                tags_by_sentence[t["sentence_id"]] = t

    loc_by_example = load_localization_history(args.localization_path)

    all_rows: List[Dict[str, Any]] = []
    for ex_id, sentences in sentences_by_example.items():
        example = examples.get(ex_id, {})
        total_sentences = len(sentences)
        loc_map = loc_by_example.get(ex_id, {})

        # Precompute localization rates and deltas
        deception_rates = {}
        for s in sentences:
            idx = int(s.get("sentence_idx", 0))
            hist = loc_map.get(idx)
            if hist:
                deception_rates[idx] = hist.get("deception_rate")

        deltas = {}
        for s in sentences:
            idx = int(s.get("sentence_idx", 0))
            rate = deception_rates.get(idx)
            if rate is None:
                deltas[idx] = None
                continue
            if idx == 0:
                prev_rate = 0.0
            else:
                prev_rate = deception_rates.get(idx - 1)
                if prev_rate is None:
                    prev_rate = 0.0
            deltas[idx] = rate - prev_rate

        # Build per-sentence features
        for i, s in enumerate(sentences):
            sentence_id = s.get("sentence_id")
            sentence_text = s.get("sentence_text", "")
            feats = {
                "sentence_id": sentence_id,
                "example_id": ex_id,
                "sentence_idx": s.get("sentence_idx"),
                "sentence_text": sentence_text,
                "start": s.get("start"),
                "end": s.get("end"),
                "total_sentences": total_sentences,
                "sentence_position": (i / (total_sentences - 1)) if total_sentences > 1 else 0.0,
            }

            feats.update(text_features(sentence_text))

            # Example-level metadata
            for key in ("deceptive", "current_rank", "model_name", "seed", "run_id"):
                if key in example:
                    feats[f"example_{key}"] = example[key]

            # Tag features
            tag = tags_by_sentence.get(sentence_id)
            if tag:
                feats["tag_id"] = tag.get("label_id")
                feats["tag_name"] = tag.get("label_name")
                feats["tag_confidence"] = tag.get("confidence")

            # Localization features
            idx = int(s.get("sentence_idx", 0))
            hist = loc_map.get(idx)
            if hist:
                feats["deception_rate"] = hist.get("deception_rate")
                feats["num_truthful"] = hist.get("num_truthful")
                feats["num_valid"] = hist.get("num_valid")
                feats["ci_low"] = hist.get("ci_low")
                feats["ci_high"] = hist.get("ci_high")

            feats["delta_deception_rate"] = deltas.get(idx)
            feats["prev_deception_rate"] = deception_rates.get(idx - 1) if idx > 0 else 0.0
            feats["next_deception_rate"] = deception_rates.get(idx + 1) if idx + 1 < total_sentences else None

            # Context window features
            if args.context_window > 0:
                prev_items = sentences[max(0, i - args.context_window):i]
                next_items = sentences[i + 1:i + 1 + args.context_window]

                prev_lengths = [len(p.get("sentence_text", "")) for p in prev_items]
                next_lengths = [len(n.get("sentence_text", "")) for n in next_items]
                feats["prev_char_mean"] = _safe_mean(prev_lengths)
                feats["next_char_mean"] = _safe_mean(next_lengths)

                if prev_items:
                    prev_tag = tags_by_sentence.get(prev_items[-1].get("sentence_id"))
                    feats["prev_tag_id"] = prev_tag.get("label_id") if prev_tag else None
                if next_items:
                    next_tag = tags_by_sentence.get(next_items[0].get("sentence_id"))
                    feats["next_tag_id"] = next_tag.get("label_id") if next_tag else None

            all_rows.append(feats)

        # Apply top_k labeling per example if requested
        if args.label_strategy == "top_k":
            deltas_sorted = sorted(
                ((idx, d) for idx, d in deltas.items() if d is not None),
                key=lambda x: x[1],
                reverse=True,
            )
            top_ids = {idx for idx, _ in deltas_sorted[: args.top_k]}
            for row in all_rows[-total_sentences:]:
                idx = int(row.get("sentence_idx", 0))
                row["is_deceptive_sentence"] = idx in top_ids

    # Apply threshold labeling if requested
    if args.label_strategy == "delta_threshold":
        for row in all_rows:
            delta = row.get("delta_deception_rate")
            row["is_deceptive_sentence"] = delta is not None and delta >= args.delta_threshold
    elif args.label_strategy == "rate_threshold":
        for row in all_rows:
            rate = row.get("deception_rate")
            row["is_deceptive_sentence"] = rate is not None and rate >= args.rate_threshold
    elif args.label_strategy == "none":
        for row in all_rows:
            row["is_deceptive_sentence"] = None

    write_jsonl(all_rows, args.out_path)
    print(f"Wrote features: {args.out_path}")


if __name__ == "__main__":
    main()
