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
    parser = argparse.ArgumentParser(description="Sentence-level deception localization (debug).")
    parser.add_argument("--examples_path", type=str, required=True)
    parser.add_argument("--sentences_path", type=str, required=True)
    parser.add_argument("--example_id", type=str, required=True)
    parser.add_argument("--example_index", type=int, default=None, help="Optional 0-based index into examples.jsonl")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--mode", type=str, default="prefix", choices=["prefix", "sentence_only"])
    parser.add_argument("--out_path", type=str, default="sentence_localization_debug.json")
    args = parser.parse_args(argv)

    example_list = list(read_jsonl(args.examples_path))
    examples = {ex["example_id"]: ex for ex in example_list if "example_id" in ex}

    example = None
    if args.example_index is not None:
        if args.example_index < 0 or args.example_index >= len(example_list):
            raise ValueError(f"example_index out of range: {args.example_index} (total {len(example_list)})")
        example = example_list[args.example_index]
    else:
        example = examples.get(args.example_id)
        if not example and args.example_id.isdigit():
            idx = int(args.example_id)
            if 0 <= idx < len(example_list):
                example = example_list[idx]

    if not example:
        raise ValueError(f"example_id not found: {args.example_id}")

    raw_text = example.get("action_reasoning")
    if not raw_text:
        raise ValueError("example missing action_reasoning")

    if example.get("truthful_rank") is not None:
        truthful_rank = int(example.get("truthful_rank"))
    elif example.get("previous_rank") is not None:
        truthful_rank = int(example.get("previous_rank"))
    elif example.get("current_rank") is not None:
        truthful_rank = int(example.get("current_rank"))
    elif example.get("truth_context") is not None:
        truthful_rank = example.get("truth_context")
    else:
        raise ValueError("Example missing truthful_rank/current_rank/truth_context.")

    sentences = [
        {
            "start": s.get("start"),
            "end": s.get("end"),
            "text": s.get("sentence_text"),
        }
        for s in read_jsonl(args.sentences_path)
        if s.get("example_id") == args.example_id
    ]

    if not sentences:
        sentences = split_sentence_spans(raw_text)

    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_new_tokens,
        seed=1,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    tokenizer = llm.get_tokenizer()

    prompt = example.get("prompt")
    if not prompt and example.get("messages"):
        prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    if not prompt:
        raise ValueError("example missing prompt/messages for prompt construction")

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

    out_path = Path(args.out_path)
    out_path.write_text(json.dumps(to_json_safe({
        "example_id": args.example_id,
        "raw_text": raw_text,
        "truthful_rank": truthful_rank,
        "history": history,
    }), indent=2))
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
