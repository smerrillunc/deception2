#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from vllm import LLM

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from sentence_localization_batch import (
    _extract_eval_context,
    _extract_raw_text,
    _guess_reasoning_model,
    _infer_game,
    localize_deception_by_sentence,
    read_jsonl,
    to_json_safe,
)
from sentence_pipeline import split_sentence_spans
from utils import prepare_messages_for_model


def _find_example(example_list: List[Dict[str, Any]], example_id: Optional[str], example_index: Optional[int]) -> Dict[str, Any]:
    if example_index is not None:
        if example_index < 0 or example_index >= len(example_list):
            raise ValueError(f"example_index out of range: {example_index} (total {len(example_list)})")
        return example_list[example_index]

    if example_id is None:
        raise ValueError("Provide either --example_id or --example_index.")

    for example in example_list:
        if example.get("example_id") == example_id or example.get("record_id") == example_id:
            return example

    if example_id.isdigit():
        idx = int(example_id)
        if 0 <= idx < len(example_list):
            return example_list[idx]

    raise ValueError(f"example not found: {example_id}")


def _render_prompt(tokenizer: Any, example: Dict[str, Any], model_name: str) -> tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    prompt = example.get("prompt")
    prompt_messages = example.get("messages") if isinstance(example.get("messages"), list) else None
    if prompt:
        return prompt, prompt_messages
    if not prompt_messages:
        return None, None

    prepared = prepare_messages_for_model(prompt_messages, model_name=model_name)
    try:
        prompt = tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt, prompt_messages


def _load_example_sentences(sentences_path: Optional[str], target_example_id: str, raw_text: str) -> List[Dict[str, Any]]:
    if sentences_path:
        matches = [
            {
                "start": sentence.get("start"),
                "end": sentence.get("end"),
                "text": sentence.get("sentence_text"),
            }
            for sentence in read_jsonl(sentences_path)
            if sentence.get("example_id") == target_example_id
        ]
        if matches:
            return matches
    return split_sentence_spans(raw_text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Universal sentence-level deception localization debug.")
    parser.add_argument("--game", type=str, default="auto", choices=["auto", "bs", "gridworld", "advisor_audit"])
    parser.add_argument("--examples_path", type=str, required=True)
    parser.add_argument("--sentences_path", type=str, default=None)
    parser.add_argument("--example_id", type=str, default=None)
    parser.add_argument("--example_index", type=int, default=None, help="Optional 0-based index into examples.jsonl")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--text_field", type=str, default="action_reasoning")
    parser.add_argument("--n_samples", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--mode", type=str, default="prefix", choices=["prefix", "sentence_only"])
    parser.add_argument("--out_path", type=str, default="sentence_localization_debug.json")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--is_reasoning_model", action=argparse.BooleanOptionalAction, default=None)
    else:
        parser.add_argument("--is_reasoning_model", action="store_true", default=None)
    args = parser.parse_args(argv)

    if args.example_id is None and args.example_index is None:
        parser.error("Provide either --example_id or --example_index.")

    example_list = list(read_jsonl(args.examples_path))
    example = _find_example(example_list, args.example_id, args.example_index)
    target_example_id = str(example.get("example_id") or example.get("record_id") or args.example_id)

    raw_text = _extract_raw_text(example, args.text_field)
    if not raw_text:
        raise ValueError(f"example missing usable text for text_field={args.text_field}")

    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_new_tokens,
        seed=1,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
    )
    tokenizer = llm.get_tokenizer()

    prompt, prompt_messages = _render_prompt(tokenizer, example, args.model_name)
    if not prompt:
        raise ValueError("example missing prompt/messages for prompt construction")

    game = _infer_game(example, args.game, prompt)
    context = _extract_eval_context(game, example, prompt)
    if context is None:
        raise ValueError(f"could not extract evaluation context for game={game}")

    sentences = _load_example_sentences(args.sentences_path, target_example_id, raw_text)
    use_reasoning_parser = (
        bool(args.is_reasoning_model)
        if args.is_reasoning_model is not None
        else _guess_reasoning_model(args.model_name)
    )

    history = localize_deception_by_sentence(
        llm,
        tokenizer,
        args.model_name,
        game,
        context,
        prompt,
        prompt_messages,
        raw_text,
        sentences,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        base_seed=args.base_seed,
        mode=args.mode,
        use_reasoning_parser=use_reasoning_parser,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            to_json_safe(
                {
                    "example_id": target_example_id,
                    "raw_text": raw_text,
                    "prompt": prompt,
                    "game": game,
                    "eval_context": context,
                    "history": history,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
