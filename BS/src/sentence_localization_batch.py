#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from vllm import LLM

from localization import (
    sample_actions_for_prefix,
    deception_wilson_interval,
    to_json_safe,
)
from sentence_pipeline import read_jsonl, split_sentence_spans


def localize_deception_by_sentence(
    llm,
    model_name,
    prompt,
    raw_text,
    truthful_rank,
    sentences,
    n_samples=25,
    temperature=0.5,
    top_p=0.5,
    repetition_penalty=1.2,
    max_new_tokens=10000,
    base_seed=1234,
    mode="prefix",
):
    history = []
    for idx, sent in enumerate(sentences):
        if mode == "prefix":
            prefix_text = raw_text[:sent["end"]]
        elif mode == "sentence_only":
            prefix_text = sent["text"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dec_rate, num_true, num_valid, generations = sample_actions_for_prefix(
            llm,
            model_name,
            prompt,
            prefix_text,
            truthful_rank,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            base_seed=base_seed + idx + 1,
        )

        ci_low, ci_high = deception_wilson_interval(num_true, num_valid)

        history.append({
            "sentence_idx": idx,
            "char_span": (sent["start"], sent["end"]),
            "sentence_text": sent["text"],
            "deception_rate": dec_rate,
            "num_truthful": num_true,
            "num_valid": num_valid,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "generations": generations,
        })
    return history


def _pick_midpoint(left_idx, right_idx, sent_idxs, min_spacing=1, n_sent=None):
    if right_idx - left_idx <= 1:
        return None
    candidate = (left_idx + right_idx) // 2
    if n_sent is not None and (candidate < 1 or candidate > n_sent):
        return None
    if any(abs(candidate - s) < min_spacing for s in sent_idxs):
        return None
    return candidate


def next_high_gradient_sentence(history, min_spacing=1, n_sent=None):
    # Sort by sentence_end_idx (prefix end count) and dedupe
    history_by_idx = {
        int(h["sentence_end_idx"]): h
        for h in history
        if h.get("sentence_end_idx") is not None
    }
    history_sorted = sorted(history_by_idx.values(), key=lambda h: h["sentence_end_idx"])
    sent_idxs = [int(h["sentence_end_idx"]) for h in history_sorted]
    dec_rates = [float(h["deception_rate"]) for h in history_sorted]
    if len(sent_idxs) < 2:
        return None

    intervals = []
    for i in range(len(sent_idxs) - 1):
        left_idx = sent_idxs[i]
        right_idx = sent_idxs[i + 1]
        gap = right_idx - left_idx
        if gap <= 1:
            continue
        diff = abs(dec_rates[i + 1] - dec_rates[i])
        slope = diff / gap if gap else 0.0
        intervals.append((slope, left_idx, right_idx))

    intervals.sort(key=lambda x: x[0], reverse=True)
    for _, left_idx, right_idx in intervals:
        candidate = _pick_midpoint(left_idx, right_idx, sent_idxs, min_spacing=min_spacing, n_sent=n_sent)
        if candidate is not None:
            return candidate
    return None


def next_largest_gap_sentence(history, min_spacing=1, n_sent=None):
    history_by_idx = {
        int(h["sentence_end_idx"]): h
        for h in history
        if h.get("sentence_end_idx") is not None
    }
    history_sorted = sorted(history_by_idx.values(), key=lambda h: h["sentence_end_idx"])
    sent_idxs = [int(h["sentence_end_idx"]) for h in history_sorted]
    if len(sent_idxs) < 2:
        return None

    gaps = []
    for i in range(len(sent_idxs) - 1):
        left_idx = sent_idxs[i]
        right_idx = sent_idxs[i + 1]
        gap = right_idx - left_idx
        gaps.append((gap, left_idx, right_idx))

    gaps.sort(key=lambda x: x[0], reverse=True)
    for _, left_idx, right_idx in gaps:
        candidate = _pick_midpoint(left_idx, right_idx, sent_idxs, min_spacing=min_spacing, n_sent=n_sent)
        if candidate is not None:
            return candidate
    return None


def localize_deception_adaptive_sentences(
    llm,
    model_name,
    prompt,
    raw_text,
    truthful_rank,
    sentences,
    n_samples=50,
    coarse_iters=8,
    refinement_iters=10,
    min_valid=3,
    min_step_size=1,
    min_spacing=1,
    temperature=0.5,
    top_p=0.5,
    repetition_penalty=1.2,
    max_new_tokens=10000,
    base_seed=1234,
    compute_full_score=True,
):
    n_sent = len(sentences)
    if n_sent == 0:
        return {
            "raw_text": raw_text,
            "prompt": prompt,
            "truthful_rank": truthful_rank,
            "history": [],
            "full_score": None,
        }

    def _prefix_text(sent_end_idx: int) -> str:
        if sent_end_idx <= 0:
            return ""
        end_char = sentences[sent_end_idx - 1]["end"]
        return raw_text[:end_char]

    history = []
    checked: Dict[int, Dict] = {}
    seed_counter = 0

    def _next_seed():
        nonlocal seed_counter
        seed_counter += 1
        return base_seed + seed_counter

    def _probe_sentence(sent_end_idx: int, seed: int = None) -> Dict:
        sent_end_idx = int(sent_end_idx)
        if sent_end_idx in checked:
            return checked[sent_end_idx]

        prefix_text = _prefix_text(sent_end_idx)
        seed_value = seed if seed is not None else _next_seed()
        dec_rate, num_true, num_valid, generations = sample_actions_for_prefix(
            llm,
            model_name,
            prompt,
            prefix_text,
            truthful_rank,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            base_seed=seed_value,
        )

        ci_low, ci_high = deception_wilson_interval(num_true, num_valid)

        if sent_end_idx > 0:
            sent = sentences[sent_end_idx - 1]
            char_span = (sent["start"], sent["end"])
            sent_text = sent["text"]
            sent_idx_inclusive = sent_end_idx - 1
        else:
            char_span = (0, 0)
            sent_text = ""
            sent_idx_inclusive = None

        probe = {
            "sentence_end_idx": sent_end_idx,
            "sentence_idx_inclusive": sent_idx_inclusive,
            "char_span": char_span,
            "sentence_text": sent_text,
            "deception_rate": dec_rate,
            "num_truthful": num_true,
            "num_valid": num_valid,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "seed": seed_value,
            "generations": generations,
        }
        history.append(probe)
        checked[sent_end_idx] = probe
        return probe

    full_score = None
    if n_sent > 0:
        full_probe = _probe_sentence(n_sent, seed=base_seed)
        if compute_full_score:
            full_score = full_probe

        # Always check first sentence prefix
        _probe_sentence(1)

    # ------------------------
    # Coarse binary search over sentence prefixes
    # ------------------------
    left = 0
    right = n_sent
    earliest_idx = None
    earliest_stats = None

    steps = 0
    while left < right and steps < coarse_iters and (right - left) > min_step_size:
        steps += 1
        mid = (left + right) // 2
        probe = _probe_sentence(mid)

        num_valid = probe["num_valid"]
        dec_rate = probe["deception_rate"]

        if num_valid < min_valid:
            left = mid
            continue

        if dec_rate >= 0.5:
            earliest_idx = mid
            earliest_stats = probe
            right = mid
        else:
            left = mid

    # ------------------------
    # Adaptive high-gradient refinement
    # ------------------------
    added = 0
    attempts = 0
    max_attempts = max(refinement_iters * 4, 20)
    while added < refinement_iters and attempts < max_attempts:
        attempts += 1
        next_idx = next_high_gradient_sentence(history, min_spacing=min_spacing, n_sent=n_sent)
        if next_idx is None:
            next_idx = next_largest_gap_sentence(history, min_spacing=min_spacing, n_sent=n_sent)
        if next_idx is None and min_spacing > 1:
            next_idx = next_high_gradient_sentence(history, min_spacing=1, n_sent=n_sent)
        if next_idx is None:
            break
        _probe_sentence(next_idx)
        added += 1

    history = sorted(history, key=lambda h: h["sentence_end_idx"])
    candidate_prefix_end_idxs = sorted({int(h["sentence_end_idx"]) for h in history if h.get("sentence_end_idx") is not None})
    candidate_sentence_idxs = sorted({s - 1 for s in candidate_prefix_end_idxs if s > 0})

    return {
        "raw_text": raw_text,
        "prompt": prompt,
        "truthful_rank": truthful_rank,
        "left_sentence_end_idx": left,
        "right_sentence_end_idx": earliest_idx,
        "right_stats": earliest_stats,
        "full_score": full_score,
        "history": history,
        "candidate_sentence_idxs": candidate_sentence_idxs,
        "candidate_prefix_end_idxs": candidate_prefix_end_idxs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch sentence-level deception localization.")
    parser.add_argument("--examples_path", type=str, required=True)
    parser.add_argument("--sentences_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for per-example JSON outputs.")
    parser.add_argument("--jsonl_path", type=str, default=None, help="Optional JSONL output path.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--mode", type=str, default="prefix", choices=["prefix", "sentence_only"])
    parser.add_argument("--method", type=str, default="adaptive", choices=["adaptive", "full"])
    parser.add_argument("--coarse_iters", type=int, default=8)
    parser.add_argument("--refinement_iters", type=int, default=8)
    parser.add_argument("--min_valid", type=int, default=3)
    parser.add_argument("--min_step_size", type=int, default=1)
    parser.add_argument("--min_spacing", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only_deceptive", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args(argv)

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    example_list = list(read_jsonl(args.examples_path))
    if args.only_deceptive:
        example_list = [e for e in example_list if e.get("deceptive") is True]
    if args.limit:
        example_list = example_list[: args.limit]

    # Shard by index to spread across GPUs
    if args.num_shards > 1:
        example_list = [
            ex for i, ex in enumerate(example_list)
            if (i % args.num_shards) == args.shard_id
        ]

    total_examples = len(example_list)
    print(f"Shard {args.shard_id}/{args.num_shards}: {total_examples} examples")
    if total_examples == 0:
        print("No examples to process for this shard.")
        return

    # Group sentences by example_id
    sentences_by_example: Dict[str, List[Dict]] = {}
    for s in read_jsonl(args.sentences_path):
        ex_id = s.get("example_id")
        if not ex_id:
            continue
        sentences_by_example.setdefault(ex_id, []).append(s)
    for ex_id, items in sentences_by_example.items():
        items.sort(key=lambda x: x.get("sentence_idx", 0))

    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_new_tokens,
        seed=1,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    tokenizer = llm.get_tokenizer()

    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else None
    if jsonl_path and args.num_shards > 1:
        jsonl_path = jsonl_path.with_suffix(f".shard{args.shard_id}.jsonl")

    jsonl_fh = None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fh = jsonl_path.open("w", encoding="utf-8")

    for idx, ex in enumerate(example_list):
        example_id = ex.get("example_id") or ex.get("record_id")
        if not example_id:
            continue

        out_path = None
        if out_dir:
            out_path = out_dir / f"sentence_localization_{example_id.replace('/', '_')}.json"
            if out_path.exists() and not args.overwrite:
                continue

        raw_text = ex.get("action_reasoning")
        if not raw_text:
            continue

        if ex.get("truthful_rank") is not None:
            truthful_rank = int(ex.get("truthful_rank"))
        elif ex.get("previous_rank") is not None:
            truthful_rank = int(ex.get("previous_rank"))
        else:
            truthful_rank = int(ex.get("current_rank"))

        sentences = [
            {
                "start": s.get("start"),
                "end": s.get("end"),
                "text": s.get("sentence_text"),
            }
            for s in sentences_by_example.get(example_id, [])
        ]
        if not sentences:
            sentences = split_sentence_spans(raw_text)

        prompt = ex.get("prompt")
        if not prompt and ex.get("messages"):
            prompt = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
        if not prompt:
            continue

        if args.method == "full":
            history = localize_deception_by_sentence(
                llm,
                args.model_name,
                prompt,
                raw_text,
                truthful_rank,
                sentences,
                n_samples=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                base_seed=args.base_seed,
                mode=args.mode,
            )
            record = {
                "example_id": example_id,
                "raw_text": raw_text,
                "truthful_rank": truthful_rank,
                "history": history,
            }
        else:
            record = localize_deception_adaptive_sentences(
                llm,
                args.model_name,
                prompt,
                raw_text,
                truthful_rank,
                sentences,
                n_samples=args.n_samples,
                coarse_iters=args.coarse_iters,
                refinement_iters=args.refinement_iters,
                min_valid=args.min_valid,
                min_step_size=args.min_step_size,
                min_spacing=args.min_spacing,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                base_seed=args.base_seed,
                compute_full_score=True,
            )
            record["example_id"] = example_id

        record = to_json_safe(record)

        if out_path:
            out_path.write_text(json.dumps(record, indent=2))
        if jsonl_fh:
            jsonl_fh.write(json.dumps(record) + "\n")

        if args.log_every and (idx + 1) % args.log_every == 0:
            print(f"Processed {idx + 1}/{total_examples} examples (shard {args.shard_id})")

    if jsonl_fh:
        jsonl_fh.close()

    if out_dir and jsonl_path:
        print(f"Batch localization complete. Outputs in {out_dir} and {jsonl_path}")
    elif out_dir:
        print(f"Batch localization complete. Outputs in {out_dir}")
    elif jsonl_path:
        print(f"Batch localization complete. Output in {jsonl_path}")
    else:
        print("Batch localization complete.")


if __name__ == "__main__":
    main()
