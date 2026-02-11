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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch sentence-level deception localization.")
    parser.add_argument("--examples_path", type=str, required=True)
    parser.add_argument("--sentences_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for per-example JSON outputs.")
    parser.add_argument("--jsonl_path", type=str, default=None, help="Optional JSONL output path.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--mode", type=str, default="prefix", choices=["prefix", "sentence_only"])
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
        elif ex.get("current_rank") is not None:
            truthful_rank = int(ex.get("current_rank"))
        elif ex.get("truth_context") is not None:
            truthful_rank = ex.get("truth_context")
        else:
            continue

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

        record = to_json_safe({
            "example_id": example_id,
            "raw_text": raw_text,
            "truthful_rank": truthful_rank,
            "history": history,
        })

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
