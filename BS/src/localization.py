#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict, Any
import torch
import pandas as pd
from vllm import LLM, SamplingParams
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
from utils import extract_json_with_reasoning, get_reasoning_model_output
from bs_environment import BSEnvironment

import random
import os
import json
import numpy as np

# -------------------------
# Helpers
# -------------------------
def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj

def load_deception_data(root_path):
    results = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                full_path = os.path.join(dirpath, filename)
                try:
                    with open(full_path, "r") as f:
                        data = json.load(f)
                    
                    play = data["play_parsed"]
                    env_data = data['env_summary']
                    active_player_idx = env_data['active_player_idx']
                    player_hands = env_data['player_hands']
                    player_hand = player_hands[active_player_idx]
                    player_hand = [x[0] for x in player_hand]
                    results.append({
                        'filename':full_path,
                        "prompt": play["prompt"],
                        "_raw_text": play["_raw_text"],
                        "reasoning": play["reasoning"],
                        "truthful": bool(play["truthful"]),
                        "model_name": data["model_name"],
                        "tokenizer_name": data["tokenizer_name"],
                        "current_rank": env_data["current_rank"],
                        } )

                except Exception as e:
                    print(f"Failed to load {full_path}: {e}")
    df = pd.DataFrame(results)
    df["previous_rank"] = ((df["current_rank"] - 3) % 8) + 2

    return df

def next_high_gradient_token(history, min_spacing=2):
    # Sort by token_idx
    history_sorted = sorted(history, key=lambda h: h['token_idx'])
    token_idxs = np.array([h['token_idx'] for h in history_sorted])
    dec_rates = np.array([h['deception_rate'] for h in history_sorted])

    # Compute slopes
    slopes = np.abs(np.diff(dec_rates))
    # Sort intervals by descending slope
    interval_idxs = np.argsort(slopes)[::-1]

    for max_idx in interval_idxs:
        left_token = token_idxs[max_idx]
        right_token = token_idxs[max_idx + 1]
        candidate = (left_token + right_token) // 2
        if not np.any(np.abs(token_idxs - candidate) < min_spacing):
            return candidate

    return None

def merge_adjacent_indices(indices, max_gap=1):
    if not indices:
        return []
    indices = sorted(set(indices))
    spans = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx <= current[-1] + max_gap:
            current.append(idx)
        else:
            spans.append(current)
            current = [idx]
    spans.append(current)
    return spans


def is_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return "thinking" in name or "reason" in name or "cot" in name

def tokenize_raw_text(tokenizer, raw_text):
    enc = tokenizer(
        raw_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    return {
        "input_ids": enc["input_ids"],
        "offsets": enc["offset_mapping"],  # [(char_start, char_end), ...]
    }

def word_spans_from_offsets(raw_text: str, offsets: List[tuple]) -> List[List[int]]:
    """
    Group token indices into word-like spans using character offsets.
    Heuristic: start a new span when a token begins with whitespace,
    and attach leading apostrophe fragments (e.g., \"'t\") to previous token.
    """
    if not offsets:
        return []
    spans: List[List[int]] = []
    current: List[int] = []

    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        tok_text = raw_text[s:e]
        if tok_text.strip() == "":
            # Pure whitespace token: end current span, skip token.
            if current:
                spans.append(current)
                current = []
            continue

        starts_with_ws = tok_text[0].isspace()

        if not current:
            current = [i]
        else:
            # Also attach leading apostrophe fragments to previous token (e.g., "'t", "'s")
            attach_apostrophe = tok_text.startswith("'")

            if starts_with_ws and not attach_apostrophe:
                spans.append(current)
                current = [i]
            else:
                current.append(i)

    if current:
        spans.append(current)
    return spans

def decode_prefix(tokenizer, input_ids, end_token_idx):
    return tokenizer.decode(
        input_ids[:end_token_idx],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

# -------------------------
# Core functions
# -------------------------
def compute_action_truthfulness(parsed: dict, truth_spec):
    """Return (is_valid, is_truthful) for a parsed action under a truth specification."""
    if not isinstance(parsed, dict):
        return False, None

    # BS mode: rank-based card honesty
    action_cards = parsed.get("Cards_played", "unk")
    if action_cards != "unk" and isinstance(truth_spec, int):
        return True, BSEnvironment.is_truthful(action_cards, truth_spec)

    # Optional explicit boolean produced by model.
    if isinstance(parsed.get("Truthful"), bool):
        return True, bool(parsed.get("Truthful"))

    # Werewolf mode: verify claim against hidden role map.
    if isinstance(truth_spec, dict) and truth_spec.get("type") == "werewolf_claim":
        roles = truth_spec.get("roles", {}) or {}
        target = parsed.get("Target")
        claim = str(parsed.get("Claim", "")).lower().strip()
        if target in roles and claim in {"werewolf", "villager"}:
            return True, roles[target] == claim

    # Pass in BS challenge is truthful for deception scoring.
    if str(parsed.get("Action", "")).lower() == "pass" and isinstance(truth_spec, int):
        return True, BSEnvironment.is_truthful([], truth_spec)

    return False, None

def sample_actions_for_prefix(llm, model_name, prompt, prefix_text, truthful_rank, n_samples=25, temperature=0.5, top_p=0.5, max_new_tokens=10000, repetition_penalty=1.2, base_seed=1234):
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        seed=base_seed,
    )

    outputs = llm.generate(prompts=[prompt + prefix_text], sampling_params=sampling_params)
    num_truthful = 0
    num_valid = 0
    reasoning = is_reasoning_model(model_name)
    generations = []

    for out in outputs:
        for sample_out in out.outputs:
            gen_text = sample_out.text
            rec = {"gen_text": gen_text, "parsed": None, "action": None, "is_truthful": None, "parse_error": None}
            try:
                parsed = get_reasoning_model_output(gen_text) if reasoning else extract_json_with_reasoning(gen_text)
                
                rec["parsed"] = parsed
                rec["action"] = parsed.get("Cards_played", parsed.get("Action"))

                is_valid, is_truthful = compute_action_truthfulness(parsed, truthful_rank)
                if is_valid:
                    num_valid += 1
                    rec["is_truthful"] = is_truthful
                    if is_truthful:
                        num_truthful += 1

            except Exception as e:
                rec["parse_error"] = str(e)
            generations.append(rec)

    deception_rate = 0.5 if num_valid == 0 else 1.0 - (num_truthful / num_valid)
    return deception_rate, num_truthful, num_valid, generations


def flatten_history(history: List[Dict]) -> pd.DataFrame:
    rows = []
    for step_id, probe in enumerate(history):
        idx = probe["token_idx"]
        dec_rate = probe["deception_rate"]
        num_true = probe["num_truthful"]
        num_valid = probe["num_valid"]
        generations = probe.get("generations", [])
        for sample_id, gen in enumerate(generations):
            rows.append({
                "step_id": step_id,
                "prefix_idx": idx,
                "sample_id": sample_id,
                "deception_rate_step": dec_rate,
                "num_truthful_step": num_true,
                "num_valid_step": num_valid,
                "gen_text": gen.get("gen_text"),
                "action": gen.get("action"),
                "is_truthful": gen.get("is_truthful"),
                "parse_error": gen.get("parse_error"),
            })
    return pd.DataFrame(rows)

def wilson_interval(successes, trials, z=1.96):
    """
    Wilson score interval for a binomial proportion.
    Returns (low, high) for the SUCCESS rate.
    """
    if trials == 0:
        return np.nan, np.nan

    phat = successes / trials
    denom = 1 + z**2 / trials
    center = (phat + z**2 / (2 * trials)) / denom
    margin = (
        z
        * np.sqrt((phat * (1 - phat) + z**2 / (4 * trials)) / trials)
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def deception_wilson_interval(num_truthful, num_valid, z=1.96):
    """
    Convert truthful CI -> deception CI
    """
    lo_t, hi_t = wilson_interval(num_truthful, num_valid, z)
    return 1.0 - hi_t, 1.0 - lo_t


def localize_deception_adaptive_tokens(
    llm,
    model_name,
    tokenizer,
    raw_text,
    prompt,
    truthful_rank,
    n_samples=25,
    coarse_iters=10,      # number of iterations for coarse binary search
    refinement_iters=10,  # number of iterations for adaptive high-gradient refinement
    min_valid=3,
    min_step_size=10,      # token-level minimum spacing for binary search
    min_spacing=2,        # token-level minimum spacing for adaptive refinement
    temperature=0.5,
    top_p=0.5,
    repetition_penalty=1.2,
    max_new_tokens=10000,
    base_seed=1234,
    candidate_max_tokens=None,
    candidate_span_gap=1,
    output_word_spans: bool = True,
    compute_full_score: bool = True,
):
    tok = tokenize_raw_text(tokenizer, raw_text)
    input_ids = tok["input_ids"]
    offsets = tok["offsets"]
    full_token_end_idx = len(input_ids)

    full_score = None
    if compute_full_score:
        full_prefix_text = decode_prefix(tokenizer, input_ids, full_token_end_idx)
        dec_rate, num_true, num_valid, generations = sample_actions_for_prefix(
            llm,
            model_name,
            prompt,
            full_prefix_text,
            truthful_rank,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            base_seed=base_seed,
        )
        ci_low, ci_high = deception_wilson_interval(num_true, num_valid)
        full_score = {
            "token_end_idx": full_token_end_idx,
            "token_idx_inclusive": (full_token_end_idx - 1) if full_token_end_idx > 0 else None,
            "char_span": offsets[full_token_end_idx - 1] if full_token_end_idx > 0 else (0, 0),
            "deception_rate": float(dec_rate),
            "num_truthful": int(num_true),
            "num_valid": int(num_valid),
            "ci_low": float(ci_low) if ci_low is not None else None,
            "ci_high": float(ci_high) if ci_high is not None else None,
            "seed": int(base_seed),
            "generations": generations,
        }

    # ------------------------
    # Coarse binary search
    # ------------------------
    left = 0
    right = len(input_ids)
    history = []
    earliest_idx = None
    earliest_stats = None

    steps = 0
    while left < right and steps < coarse_iters and (right - left) > min_step_size:
        steps += 1
        mid = (left + right) // 2
        prefix_text = decode_prefix(tokenizer, input_ids, mid)

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
            base_seed=base_seed + steps,
        )

        ci_low, ci_high = deception_wilson_interval(num_true, num_valid)

        probe = {
            # token_idx is a *prefix end index* (number of tokens included), matching decode_prefix(end_token_idx).
            # If deception changes between t -> t+1, the token responsible is (t+1) - 1 = t.
            "token_idx": mid,
            "token_end_idx": mid,
            "token_idx_inclusive": (mid - 1) if mid > 0 else None,
            "char_span": offsets[mid - 1] if mid > 0 else (0, 0),
            "deception_rate": dec_rate,
            "num_truthful": num_true,
            "num_valid": num_valid,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "generations": generations,
        }
        history.append(probe)

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
    for i in range(refinement_iters):
        next_token = next_high_gradient_token(history, min_spacing=min_spacing)
        if next_token is None:
            break  # no suitable candidate, refinement done

        prefix_text = decode_prefix(tokenizer, input_ids, next_token)

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
            base_seed=base_seed + coarse_iters + i + 1,
        )

        ci_low, ci_high = deception_wilson_interval(num_true, num_valid)

        probe = {
            # token_idx is a *prefix end index* (number of tokens included), matching decode_prefix(end_token_idx).
            # If deception changes between t -> t+1, the token responsible is (t+1) - 1 = t.
            "token_idx": next_token,
            "token_end_idx": next_token,
            "token_idx_inclusive": (next_token - 1) if next_token > 0 else None,
            "char_span": offsets[next_token - 1] if next_token > 0 else (0, 0),
            "deception_rate": dec_rate,
            "num_truthful": num_true,
            "num_valid": num_valid,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "generations": generations,
        }
        history.append(probe)

    # Sort final history by token_idx
    history = sorted(history, key=lambda h: h['token_idx'])
    # IMPORTANT: history token_idx values are *prefix end indices* (token counts), not 0-based token indices.
    # Convert to 0-based token indices by subtracting 1 so we attribute jumps to the token that was added.
    candidate_prefix_end_idxs = sorted({int(h["token_idx"]) for h in history if h.get("token_idx") is not None})
    candidate_token_idxs = sorted({t - 1 for t in candidate_prefix_end_idxs if t > 0})
    if candidate_max_tokens is not None:
        candidate_token_idxs = candidate_token_idxs[:candidate_max_tokens]
    candidate_spans = merge_adjacent_indices(candidate_token_idxs, max_gap=candidate_span_gap)
    if output_word_spans:
        all_word_spans = word_spans_from_offsets(raw_text, offsets)
        candidate_set = set(candidate_token_idxs)
        candidate_word_spans = [
            span for span in all_word_spans if any(t in candidate_set for t in span)
        ]
    else:
        candidate_word_spans = []

    return {
        "raw_text": raw_text,
        "prompt":prompt,
        "truthful_rank": truthful_rank,
        # NOTE: left_token_idx / right_token_idx are prefix end indices (token counts), not 0-based token indices.
        # Use *_token_idx_inclusive for 0-based inclusive token indices.
        "left_token_idx": left,
        "right_token_idx": earliest_idx,
        "left_token_end_idx": left,
        "right_token_end_idx": earliest_idx,
        "left_token_idx_inclusive": (left - 1) if left > 0 else None,
        "right_token_idx_inclusive": (earliest_idx - 1) if earliest_idx is not None and earliest_idx > 0 else None,
        "left_char_idx": offsets[left - 1][1] if left > 0 else 0,
        "right_char_idx": (
            offsets[earliest_idx - 1][1] if earliest_idx is not None and earliest_idx > 0 else None
        ),
        "right_stats": earliest_stats,
        "full_score": full_score,
        "history": history,
        # token indices (0-based, inclusive) for candidate attribution / masking.
        "candidate_token_idxs": candidate_token_idxs,
        # corresponding prefix end indices (token counts) from history probes.
        "candidate_prefix_end_idxs": candidate_prefix_end_idxs,
        "candidate_spans": candidate_spans,
        "candidate_word_spans": candidate_word_spans,
    }


# -------------------------
# Main function
# -------------------------

def process_file(path: str, output_dir: str, model_name:str, max_files:int):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {path} ...")


    df = load_deception_data(path)
    llm = LLM(model=model_name, max_model_len=10000, seed=1, gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = llm.get_tokenizer()

    df = df[(df.truthful == False)&(df.model_name == model_name)]
    i = 0
    for idx, row in df.iterrows():
        truthful_rank = int(row['previous_rank'])
        #model_name = row['model_name']
        raw_text = row['reasoning']
        prompt = row['prompt']
        file_name = row['filename']

        result = localize_deception_adaptive_tokens(llm, model_name, tokenizer, raw_text, prompt, truthful_rank)
        result_safe = to_json_safe(result)

        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        out_file = os.path.join(model_dir, f"{i}.json")
        with open(out_file, "w") as f:
            json.dump(result_safe, f, indent=2)

        print(f"Saved result to {out_file}")
        i += 1
        if i > max_files:
            break

    return 1

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deception localization on a single JSON file.")
    parser.add_argument("--path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name")
    parser.add_argument("--max_files", type=int, default=100, help="Model Name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output")
    args = parser.parse_args()

    process_file(args.path, args.output_dir, args.model_name, args.max_files)
