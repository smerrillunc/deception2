#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

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
    if not loc_source:
        return {}

    def _history_idx(entry: Dict[str, Any]) -> Optional[int]:
        if "sentence_idx" in entry and entry["sentence_idx"] is not None:
            idx = entry["sentence_idx"]
        elif "sentence_idx_inclusive" in entry and entry["sentence_idx_inclusive"] is not None:
            idx = entry["sentence_idx_inclusive"]
        elif "sentence_end_idx" in entry and entry["sentence_end_idx"] is not None:
            idx = int(entry["sentence_end_idx"]) - 1
        else:
            return None
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        if idx < 0:
            return None
        return idx

    def _history_map(history: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        mapped: Dict[int, Dict[str, Any]] = {}
        for item in history:
            idx = _history_idx(item)
            if idx is None:
                continue
            mapped[idx] = item
        return mapped

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
                loc_map[example_id] = _history_map(history)
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
        loc_map[example_id] = _history_map(history)

    return loc_map


def _pick_model_name(example: Dict[str, Any]) -> Optional[str]:
    return example.get("model_name") or example.get("meta_model_name")


def _map_tokens_to_sentences(
    offsets: List[Tuple[int, int]],
    sentence_spans: List[Dict[str, Any]],
) -> List[int]:
    token_to_sentence: List[int] = [-1] * len(offsets)
    if not offsets or not sentence_spans:
        return token_to_sentence

    sent_idx = 0
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        mid = (start + end) / 2.0
        while sent_idx < len(sentence_spans) and mid >= sentence_spans[sent_idx]["end"]:
            sent_idx += 1
        if sent_idx >= len(sentence_spans):
            break
        if sentence_spans[sent_idx]["start"] <= mid < sentence_spans[sent_idx]["end"]:
            token_to_sentence[idx] = sent_idx
    return token_to_sentence


def _kurtosis(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n < 2:
        return None
    mean_val = sum(vals) / n
    m2 = sum((v - mean_val) ** 2 for v in vals) / n
    if m2 == 0:
        return 0.0
    m4 = sum((v - mean_val) ** 4 for v in vals) / n
    return m4 / (m2 ** 2)


def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    n = len(vals)
    mean_val = sum(vals) / n
    var = sum((v - mean_val) ** 2 for v in vals) / n
    std = math.sqrt(var)
    return {
        "mean": mean_val,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "count": n,
    }


def _compute_downstream_attention_features(
    raw_text: str,
    sentence_spans: List[Dict[str, Any]],
    *,
    model,
    tokenizer,
    device: str,
    max_tokens: int = 0,
    add_special_tokens: bool = False,
    min_sentence_distance: int = 4,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    if not raw_text or not sentence_spans:
        return {}, {"attn_truncated": False, "attn_num_tokens": 0}

    import torch

    enc = tokenizer(
        raw_text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()

    truncated = False
    if max_tokens and input_ids.shape[0] > max_tokens:
        input_ids = input_ids[:max_tokens]
        offsets = offsets[:max_tokens]
        truncated = True

    token_to_sentence = _map_tokens_to_sentences(offsets, sentence_spans)
    num_sentences = len(sentence_spans)
    if num_sentences == 0:
        return {}, {"attn_truncated": truncated, "attn_num_tokens": int(input_ids.shape[0])}

    token_sent = torch.tensor(token_to_sentence, device=device, dtype=torch.long)
    valid_mask = token_sent >= 0
    valid_token_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
    if valid_token_idx.numel() == 0:
        return {}, {"attn_truncated": truncated, "attn_num_tokens": int(input_ids.shape[0])}

    valid_sent_ids = token_sent[valid_mask]
    token_counts = torch.bincount(valid_sent_ids, minlength=num_sentences)
    token_counts_safe = token_counts.clone()
    token_counts_safe[token_counts_safe == 0] = 1

    valid_sentence_mask = token_counts > 0
    downstream_rows: List[torch.Tensor] = []
    downstream_sentence_counts: List[int] = []
    downstream_token_counts: List[int] = []
    for col in range(num_sentences):
        rows = [
            i
            for i in range(col + min_sentence_distance, num_sentences)
            if valid_sentence_mask[i].item()
        ]
        downstream_sentence_counts.append(len(rows))
        if rows:
            row_tensor = torch.tensor(rows, device=device, dtype=torch.long)
            downstream_rows.append(row_tensor)
            downstream_token_counts.append(int(token_counts[row_tensor].sum().item()))
        else:
            downstream_rows.append(torch.tensor([], device=device, dtype=torch.long))
            downstream_token_counts.append(0)

    input_ids = input_ids.unsqueeze(0).to(device)
    autocast_context = nullcontext()
    if str(device).startswith("cuda"):
        model_dtype = None
        try:
            for param in model.parameters():
                if param.is_floating_point():
                    model_dtype = param.dtype
                    break
        except Exception:
            model_dtype = None
        if model_dtype in {torch.float16, torch.bfloat16}:
            autocast_context = torch.autocast(device_type="cuda", dtype=model_dtype)
    with torch.no_grad():
        with autocast_context:
            outputs = model(input_ids, output_attentions=True, use_cache=False)
    attn_layers = outputs.attentions

    if not attn_layers:
        return {}, {"attn_truncated": truncated, "attn_num_tokens": int(input_ids.shape[1])}

    num_layers = len(attn_layers)
    sum_all = torch.zeros(num_sentences, device=device, dtype=torch.float32)
    count_all = torch.zeros(num_sentences, device=device, dtype=torch.long)
    max_all = torch.full((num_sentences,), float("-inf"), device=device, dtype=torch.float32)
    sum_last = torch.zeros(num_sentences, device=device, dtype=torch.float32)
    count_last = torch.zeros(num_sentences, device=device, dtype=torch.long)
    max_last = torch.full((num_sentences,), float("-inf"), device=device, dtype=torch.float32)
    total_heads = 0
    head_kurtosis_all: List[float] = []
    head_kurtosis_last: List[float] = []

    for layer_idx, layer_attn in enumerate(attn_layers):
        layer_attn = layer_attn[0].to(dtype=torch.float32)
        num_heads = layer_attn.shape[0]
        total_heads += num_heads
        is_last = layer_idx == (num_layers - 1)

        for head_idx in range(num_heads):
            attn = layer_attn[head_idx]
            attn_valid = attn[:, valid_token_idx]
            sentence_matrix = torch.zeros((attn.shape[0], num_sentences), device=device, dtype=torch.float32)
            sentence_matrix.index_add_(1, valid_sent_ids, attn_valid)
            sentence_matrix = sentence_matrix / token_counts_safe.to(dtype=torch.float32)

            sentence_sentence = torch.zeros((num_sentences, num_sentences), device=device, dtype=torch.float32)
            sentence_matrix_valid = sentence_matrix[valid_token_idx]
            sentence_sentence.index_add_(0, valid_sent_ids, sentence_matrix_valid)
            sentence_sentence = sentence_sentence / token_counts_safe.to(dtype=torch.float32).view(-1, 1)

            downstream = torch.full((num_sentences,), float("nan"), device=device, dtype=torch.float32)
            for col in range(num_sentences):
                if downstream_sentence_counts[col] == 0:
                    continue
                downstream[col] = sentence_sentence[downstream_rows[col], col].mean()

            finite_mask = torch.isfinite(downstream)
            if finite_mask.any():
                sum_all[finite_mask] += downstream[finite_mask]
                count_all[finite_mask] += 1
                max_all[finite_mask] = torch.maximum(max_all[finite_mask], downstream[finite_mask])
                head_kurt = _kurtosis(downstream[finite_mask].tolist())
                if head_kurt is not None:
                    head_kurtosis_all.append(head_kurt)
            if is_last and finite_mask.any():
                sum_last[finite_mask] += downstream[finite_mask]
                count_last[finite_mask] += 1
                max_last[finite_mask] = torch.maximum(max_last[finite_mask], downstream[finite_mask])
                head_kurt = _kurtosis(downstream[finite_mask].tolist())
                if head_kurt is not None:
                    head_kurtosis_last.append(head_kurt)

    if total_heads == 0:
        return {}, {"attn_truncated": truncated, "attn_num_tokens": int(input_ids.shape[1])}

    mean_all = torch.full((num_sentences,), float("nan"), device=device, dtype=torch.float32)
    mean_last = torch.full((num_sentences,), float("nan"), device=device, dtype=torch.float32)
    if count_all.any():
        mean_all = sum_all / torch.clamp(count_all.to(dtype=torch.float32), min=1.0)
    if count_last.any():
        mean_last = sum_last / torch.clamp(count_last.to(dtype=torch.float32), min=1.0)

    features_by_sentence: Dict[int, Dict[str, Any]] = {}
    for sent_idx in range(num_sentences):
        tok_count = int(token_counts[sent_idx].item())
        down_token_count = downstream_token_counts[sent_idx]
        down_sent_count = downstream_sentence_counts[sent_idx]
        if tok_count == 0 or down_sent_count == 0:
            features_by_sentence[sent_idx] = {
                "attn_token_count": tok_count,
                "attn_downstream_token_count": down_token_count,
                "attn_downstream_sentence_count": down_sent_count,
                "attn_downstream_mean": None,
                "attn_downstream_max": None,
                "attn_downstream_last_layer_mean": None,
                "attn_downstream_last_layer_max": None,
            }
            continue
        mean_val = mean_all[sent_idx].item() if torch.isfinite(mean_all[sent_idx]) else None
        last_val = mean_last[sent_idx].item() if torch.isfinite(mean_last[sent_idx]) else None
        max_val = max_all[sent_idx].item() if torch.isfinite(max_all[sent_idx]) else None
        max_last_val = max_last[sent_idx].item() if torch.isfinite(max_last[sent_idx]) else None
        features_by_sentence[sent_idx] = {
            "attn_token_count": tok_count,
            "attn_downstream_token_count": down_token_count,
            "attn_downstream_sentence_count": down_sent_count,
            "attn_downstream_mean": mean_val,
            "attn_downstream_max": max_val,
            "attn_downstream_last_layer_mean": last_val,
            "attn_downstream_last_layer_max": max_last_val,
        }

    head_stats = _summary_stats(head_kurtosis_all)
    head_last_stats = _summary_stats(head_kurtosis_last)

    meta = {
        "attn_truncated": truncated,
        "attn_num_tokens": int(input_ids.shape[1]),
        "attn_min_sentence_distance": min_sentence_distance,
        "attn_head_kurtosis_mean": head_stats["mean"],
        "attn_head_kurtosis_std": head_stats["std"],
        "attn_head_kurtosis_min": head_stats["min"],
        "attn_head_kurtosis_max": head_stats["max"],
        "attn_head_kurtosis_count": head_stats["count"],
        "attn_head_kurtosis_excess_mean": ((head_stats["mean"] - 3.0) if head_stats["mean"] is not None else None),
        "attn_head_kurtosis_excess_std": head_stats["std"],
        "attn_head_kurtosis_excess_min": ((head_stats["min"] - 3.0) if head_stats["min"] is not None else None),
        "attn_head_kurtosis_excess_max": ((head_stats["max"] - 3.0) if head_stats["max"] is not None else None),
        "attn_head_kurtosis_last_layer_mean": head_last_stats["mean"],
        "attn_head_kurtosis_last_layer_std": head_last_stats["std"],
        "attn_head_kurtosis_last_layer_min": head_last_stats["min"],
        "attn_head_kurtosis_last_layer_max": head_last_stats["max"],
        "attn_head_kurtosis_last_layer_count": head_last_stats["count"],
        "attn_head_kurtosis_last_layer_excess_mean": (
            (head_last_stats["mean"] - 3.0) if head_last_stats["mean"] is not None else None
        ),
        "attn_head_kurtosis_last_layer_excess_std": head_last_stats["std"],
        "attn_head_kurtosis_last_layer_excess_min": (
            (head_last_stats["min"] - 3.0) if head_last_stats["min"] is not None else None
        ),
        "attn_head_kurtosis_last_layer_excess_max": (
            (head_last_stats["max"] - 3.0) if head_last_stats["max"] is not None else None
        ),
    }
    return features_by_sentence, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract sentence-level features for deception prediction.")
    parser.add_argument("--examples_path", type=str, required=True)
    parser.add_argument("--sentences_path", type=str, required=True)
    parser.add_argument("--tags_path", type=str, default=None)
    parser.add_argument("--localization_path", type=str, default=None, help="JSON file or directory of sentence localization outputs.")
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--context_window", type=int, default=1)
    parser.add_argument(
        "--label_strategy",
        type=str,
        default="delta_threshold",
        choices=["delta_threshold", "rate_threshold", "top_k", "none"],
    )
    parser.add_argument("--delta_threshold", type=float, default=0.1)
    parser.add_argument("--rate_threshold", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--localization_mode", type=str, default="auto", choices=["auto", "full", "adaptive"])
    parser.add_argument("--only_localized", action="store_true", default=False)
    parser.add_argument("--enable_attention_features", action="store_true", default=False)
    parser.add_argument("--attention_model_name", type=str, default=None)
    parser.add_argument("--attention_device", type=str, default=None)
    parser.add_argument(
        "--attention_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--attention_max_tokens", type=int, default=0)
    parser.add_argument("--attention_min_sentence_distance", type=int, default=4)
    parser.add_argument("--attention_add_special_tokens", action="store_true", default=False)
    parser.add_argument("--attention_trust_remote_code", action="store_true", default=False)
    args = parser.parse_args(argv)

    examples = {ex["example_id"]: ex for ex in read_jsonl(args.examples_path) if "example_id" in ex}

    sentences_by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sentence in read_jsonl(args.sentences_path):
        if "example_id" in sentence:
            sentences_by_example[sentence["example_id"]].append(sentence)
    for _, items in sentences_by_example.items():
        items.sort(key=lambda x: x.get("sentence_idx", 0))

    tags_by_sentence: Dict[str, Dict[str, Any]] = {}
    if args.tags_path:
        for tag in read_jsonl(args.tags_path):
            if "sentence_id" in tag:
                tags_by_sentence[tag["sentence_id"]] = tag

    loc_by_example = load_localization_history(args.localization_path)

    attention_model = None
    attention_tokenizer = None
    attention_device = None
    if args.enable_attention_features:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        attention_model_name = args.attention_model_name
        if not attention_model_name:
            names = {_pick_model_name(ex) for ex in examples.values() if _pick_model_name(ex)}
            if len(names) == 1:
                attention_model_name = next(iter(names))
            else:
                raise ValueError(
                    "Could not infer a single model name. Please pass --attention_model_name explicitly."
                )

        if args.attention_device:
            attention_device = args.attention_device
        else:
            attention_device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if args.attention_dtype == "auto":
            attention_dtype = torch.float16 if attention_device == "cuda" else torch.float32
        else:
            attention_dtype = dtype_map[args.attention_dtype]

        try:
            attention_tokenizer = AutoTokenizer.from_pretrained(
                attention_model_name,
                use_fast=True,
                trust_remote_code=args.attention_trust_remote_code,
            )
        except TypeError:
            attention_tokenizer = AutoTokenizer.from_pretrained(
                attention_model_name,
                trust_remote_code=args.attention_trust_remote_code,
            )

        if not getattr(attention_tokenizer, "is_fast", False):
            raise ValueError("Fast tokenizer required for offset mapping to compute attention features.")

        try:
            attention_model = AutoModelForCausalLM.from_pretrained(
                attention_model_name,
                torch_dtype=attention_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                trust_remote_code=args.attention_trust_remote_code,
            )
        except TypeError:
            attention_model = AutoModelForCausalLM.from_pretrained(
                attention_model_name,
                torch_dtype=attention_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=args.attention_trust_remote_code,
            )

        attention_model.to(attention_device)
        attention_model.eval()

    all_rows: List[Dict[str, Any]] = []
    for ex_id, sentences in sentences_by_example.items():
        example = examples.get(ex_id, {})
        total_sentences = len(sentences)
        loc_map = loc_by_example.get(ex_id, {})

        deception_rates = {}
        for sentence in sentences:
            idx = int(sentence.get("sentence_idx", 0))
            hist = loc_map.get(idx)
            if hist:
                deception_rates[idx] = hist.get("deception_rate")

        localized_indices = sorted(deception_rates.keys())
        prev_localized_idx_map: Dict[int, Optional[int]] = {}
        next_localized_idx_map: Dict[int, Optional[int]] = {}
        prev_idx = None
        for idx in localized_indices:
            prev_localized_idx_map[idx] = prev_idx
            prev_idx = idx
        next_idx = None
        for idx in reversed(localized_indices):
            next_localized_idx_map[idx] = next_idx
            next_idx = idx

        loc_mode = args.localization_mode
        if loc_mode == "auto":
            if localized_indices and len(localized_indices) < total_sentences:
                loc_mode = "adaptive"
            else:
                loc_mode = "full"

        deltas = {}
        for sentence in sentences:
            idx = int(sentence.get("sentence_idx", 0))
            rate = deception_rates.get(idx)
            if rate is None:
                deltas[idx] = None
                continue

            prev_rate = None
            if loc_mode == "full":
                if idx == 0:
                    prev_rate = 0.0
                else:
                    prev_rate = deception_rates.get(idx - 1)
            else:
                prev_localized_idx = prev_localized_idx_map.get(idx)
                if prev_localized_idx is None:
                    prev_rate = 0.0
                else:
                    prev_rate = deception_rates.get(prev_localized_idx)

            deltas[idx] = (rate - prev_rate) if prev_rate is not None else None

        attn_features = {}
        attn_meta: Dict[str, Any] = {}
        if attention_model is not None:
            raw_text = example.get("action_reasoning") if example else None
            sentence_spans = [{"start": s.get("start"), "end": s.get("end")} for s in sentences]
            attn_features, attn_meta = _compute_downstream_attention_features(
                raw_text,
                sentence_spans,
                model=attention_model,
                tokenizer=attention_tokenizer,
                device=attention_device,
                max_tokens=args.attention_max_tokens,
                add_special_tokens=args.attention_add_special_tokens,
                min_sentence_distance=args.attention_min_sentence_distance,
            )

        start_row_idx = len(all_rows)
        example_rows_added = 0
        for idx, sentence in enumerate(sentences):
            sentence_id = sentence.get("sentence_id")
            sentence_text = sentence.get("sentence_text", "")
            feats = {
                "sentence_id": sentence_id,
                "example_id": ex_id,
                "sentence_idx": sentence.get("sentence_idx"),
                "sentence_text": sentence_text,
                "start": sentence.get("start"),
                "end": sentence.get("end"),
                "total_sentences": total_sentences,
                "sentence_position": (idx / (total_sentences - 1)) if total_sentences > 1 else 0.0,
            }

            feats.update(text_features(sentence_text))

            for key in ("deceptive", "current_rank", "model_name", "seed", "run_id"):
                if key in example:
                    feats[f"example_{key}"] = example[key]

            tag = tags_by_sentence.get(sentence_id)
            if tag:
                feats["tag_id"] = tag.get("label_id")
                feats["tag_name"] = tag.get("label_name")
                feats["tag_confidence"] = tag.get("confidence")

            sent_idx = int(sentence.get("sentence_idx", 0))
            hist = loc_map.get(sent_idx)
            if hist:
                feats["deception_rate"] = hist.get("deception_rate")
                feats["num_truthful"] = hist.get("num_truthful")
                feats["num_valid"] = hist.get("num_valid")
                feats["ci_low"] = hist.get("ci_low")
                feats["ci_high"] = hist.get("ci_high")

            is_localized = sent_idx in deception_rates
            feats["is_localized"] = is_localized
            feats["localization_mode"] = loc_mode

            feats["delta_deception_rate"] = deltas.get(sent_idx)
            if loc_mode == "full":
                if sent_idx == 0:
                    feats["prev_deception_rate"] = 0.0
                    feats["prev_localized_idx"] = None
                else:
                    feats["prev_deception_rate"] = deception_rates.get(sent_idx - 1)
                    feats["prev_localized_idx"] = sent_idx - 1 if (sent_idx - 1) in deception_rates else None
                feats["next_deception_rate"] = deception_rates.get(sent_idx + 1)
                feats["next_localized_idx"] = sent_idx + 1 if (sent_idx + 1) in deception_rates else None
            else:
                prev_loc_idx = prev_localized_idx_map.get(sent_idx)
                next_loc_idx = next_localized_idx_map.get(sent_idx)
                feats["prev_localized_idx"] = prev_loc_idx
                feats["next_localized_idx"] = next_loc_idx
                if prev_loc_idx is None:
                    feats["prev_deception_rate"] = 0.0
                else:
                    feats["prev_deception_rate"] = deception_rates.get(prev_loc_idx)
                feats["next_deception_rate"] = deception_rates.get(next_loc_idx) if next_loc_idx is not None else None

            if feats.get("prev_localized_idx") is not None:
                feats["localized_gap"] = sent_idx - int(feats["prev_localized_idx"])
            else:
                feats["localized_gap"] = None

            if attn_meta:
                feats.update(attn_meta)
            if attn_features:
                feats.update(attn_features.get(sent_idx, {}))

            if args.context_window > 0:
                prev_items = sentences[max(0, idx - args.context_window):idx]
                next_items = sentences[idx + 1:idx + 1 + args.context_window]

                prev_lengths = [len(item.get("sentence_text", "")) for item in prev_items]
                next_lengths = [len(item.get("sentence_text", "")) for item in next_items]
                feats["prev_char_mean"] = _safe_mean(prev_lengths)
                feats["next_char_mean"] = _safe_mean(next_lengths)

                if prev_items:
                    prev_tag = tags_by_sentence.get(prev_items[-1].get("sentence_id"))
                    feats["prev_tag_id"] = prev_tag.get("label_id") if prev_tag else None
                if next_items:
                    next_tag = tags_by_sentence.get(next_items[0].get("sentence_id"))
                    feats["next_tag_id"] = next_tag.get("label_id") if next_tag else None

            if args.only_localized and not is_localized:
                continue
            all_rows.append(feats)
            example_rows_added += 1

        if args.label_strategy == "top_k":
            deltas_sorted = sorted(
                ((idx, delta) for idx, delta in deltas.items() if delta is not None),
                key=lambda x: x[1],
                reverse=True,
            )
            top_ids = {idx for idx, _ in deltas_sorted[: args.top_k]}
            example_rows = all_rows[start_row_idx:start_row_idx + example_rows_added]
            for row in example_rows:
                sent_idx = int(row.get("sentence_idx", 0))
                if row.get("delta_deception_rate") is None:
                    row["is_deceptive_sentence"] = None
                else:
                    row["is_deceptive_sentence"] = sent_idx in top_ids

    if args.label_strategy == "delta_threshold":
        for row in all_rows:
            delta = row.get("delta_deception_rate")
            if delta is None:
                row["is_deceptive_sentence"] = None
            else:
                row["is_deceptive_sentence"] = delta >= args.delta_threshold
    elif args.label_strategy == "rate_threshold":
        for row in all_rows:
            rate = row.get("deception_rate")
            if rate is None:
                row["is_deceptive_sentence"] = None
            else:
                row["is_deceptive_sentence"] = rate >= args.rate_threshold
    elif args.label_strategy == "none":
        for row in all_rows:
            row["is_deceptive_sentence"] = None

    write_jsonl(all_rows, args.out_path)
    print(f"Wrote features: {args.out_path}")


if __name__ == "__main__":
    main()
