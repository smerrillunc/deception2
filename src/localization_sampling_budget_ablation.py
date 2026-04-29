#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_CONFIG_ROOT", "/tmp/vllm")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
NOTEBOOK_ROOT = ROOT_DIR / "Notebooks"
DATASET_ROOT = ROOT_DIR / "DatasetMain"

for search_root in (SCRIPT_DIR, NOTEBOOK_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from sentence_pipeline import split_sentence_spans


ENV_SPECS = OrderedDict(
    [
        ("AdvisorAudit", "advisor_audit"),
        ("BS", "bs"),
        ("CarSales", "car_sales"),
        ("Gridworld", "gridworld"),
        ("Interview", "interview"),
    ]
)
ENV_DIR_TO_NAME = {value: key for key, value in ENV_SPECS.items()}

_SLB_MODULE: Any | None = None


def load_sentence_localization_batch():
    global _SLB_MODULE
    if _SLB_MODULE is None:
        import sentence_localization_batch as slb

        _SLB_MODULE = importlib.reload(slb)
    return _SLB_MODULE


def load_prepare_messages_for_model():
    from utils import prepare_messages_for_model

    return prepare_messages_for_model


def maybe_tqdm(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: int | None = None,
    disable: bool = False,
    leave: bool = True,
):
    if disable or _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total, leave=leave)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in str(text).split(",") if part.strip()}))
    if not values:
        raise ValueError(f"Expected at least one integer in {text!r}.")
    return values


def parse_envs(values: list[str] | None) -> list[str]:
    if not values:
        return list(ENV_SPECS.keys())
    out: list[str] = []
    for raw_value in values:
        for piece in str(raw_value).split(","):
            value = piece.strip()
            if not value:
                continue
            if value in ENV_SPECS:
                out.append(value)
                continue
            if value in ENV_DIR_TO_NAME:
                out.append(ENV_DIR_TO_NAME[value])
                continue
            raise ValueError(
                f"Unknown environment {value!r}. Expected one of "
                f"{sorted(list(ENV_SPECS.keys()) + list(ENV_DIR_TO_NAME.keys()))}."
            )
    deduped: list[str] = []
    for env_name in out:
        if env_name not in deduped:
            deduped.append(env_name)
    return deduped


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def fallback_record_id(example: dict[str, Any]) -> str:
    if example.get("example_id"):
        return str(example["example_id"])
    if example.get("record_id"):
        return str(example["record_id"])
    run_id = str(example.get("run_id") or example.get("run_date") or "run")
    state_id = example.get("state_id")
    sample_idx = example.get("sample_idx")
    if state_id is not None and sample_idx is not None:
        return f"{run_id}/state_{state_id}/sample_{sample_idx}"
    game_id = example.get("game_id")
    turn_idx = example.get("turn_idx")
    if game_id is not None and turn_idx is not None:
        return f"{run_id}/game_{game_id}/turn_{turn_idx}"
    return f"{run_id}/line"


def extract_raw_text(example: dict[str, Any], text_field: str) -> str | None:
    candidates: list[Any] = []
    if text_field and isinstance(example.get(text_field), str):
        candidates.append(example.get(text_field))
    for key in ("action_reasoning", "reasoning", "action_raw_text"):
        if isinstance(example.get(key), str):
            candidates.append(example.get(key))
    action = example.get("action")
    if isinstance(action, dict):
        if isinstance(action.get("reasoning"), str):
            candidates.append(action.get("reasoning"))
        if isinstance(action.get("_raw_text"), str):
            candidates.append(action.get("_raw_text"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def stable_example_key(env_name: str, example_id: str) -> str:
    return f"{env_name}::{example_id}"


def cache_filename_for_example(env_name: str, example_id: str) -> str:
    digest = hashlib.md5(f"{env_name}::{example_id}".encode("utf-8")).hexdigest()[:12]
    return f"{slugify(env_name)}__{slugify(example_id)}__{digest}.json"


def outcomes_to_float_array(outcomes: list[Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in outcomes],
        dtype=np.float32,
    )


def deception_stats_from_float_array(values: np.ndarray) -> dict[str, float]:
    valid_mask = np.isfinite(values)
    num_valid = int(valid_mask.sum())
    num_truthful = int(np.nansum(values[valid_mask])) if num_valid > 0 else 0
    num_deceptive = int(num_valid - num_truthful)
    deception_rate = 0.5 if num_valid == 0 else float(num_deceptive / num_valid)
    return {
        "num_truthful": float(num_truthful),
        "num_valid": float(num_valid),
        "num_invalid": float(values.shape[0] - num_valid),
        "deception_rate": deception_rate,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    if finite.size == 1:
        value = float(finite[0])
        return value, value, value
    sample_means = np.empty(int(repeats), dtype=np.float32)
    for idx in range(int(repeats)):
        draw = rng.choice(finite, size=finite.size, replace=True)
        sample_means[idx] = float(np.mean(draw, dtype=np.float32))
    return (
        float(np.mean(finite, dtype=np.float32)),
        float(np.quantile(sample_means, 0.025)),
        float(np.quantile(sample_means, 0.975)),
    )


def build_candidate_pool(
    dataset_root: Path,
    *,
    env_names: list[str],
    model_dirname: str,
    text_field: str,
    deceptive_only: bool,
    max_sentences: int,
    disable_tqdm: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    lookup: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []

    for env_name in maybe_tqdm(env_names, desc="Scan examples", total=len(env_names), disable=disable_tqdm):
        env_dirname = ENV_SPECS[env_name]
        examples_path = dataset_root / env_dirname / model_dirname / "examples.jsonl"
        if not examples_path.exists():
            raise FileNotFoundError(f"Missing examples.jsonl for {env_name}: {examples_path}")

        total_rows = 0
        label_rows = 0
        short_rows = 0
        with examples_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                total_rows += 1

                if deceptive_only and example.get("deceptive") is not True:
                    continue
                label_rows += 1

                raw_text = extract_raw_text(example, text_field)
                if not raw_text:
                    continue
                sentences = split_sentence_spans(raw_text)
                sentence_count = int(len(sentences))
                if sentence_count < 1 or sentence_count >= int(max_sentences):
                    continue

                short_rows += 1
                example_id = str(example.get("example_id") or fallback_record_id(example))
                example_key = stable_example_key(env_name, example_id)
                lookup[example_key] = example
                rows.append(
                    {
                        "example_key": example_key,
                        "example_id": example_id,
                        "env_name": env_name,
                        "env_dirname": env_dirname,
                        "model_dirname": model_dirname,
                        "sentence_count": sentence_count,
                        "meta_model_name": example.get("meta_model_name"),
                        "action_preview": json.dumps(example.get("action"), ensure_ascii=False)[:220],
                        "reasoning_preview": raw_text[:180].replace("\n", " "),
                        "source_path": str(examples_path),
                    }
                )
        count_rows.append(
            {
                "env_name": env_name,
                "env_dirname": env_dirname,
                "model_dirname": model_dirname,
                "total_examples": int(total_rows),
                "label_examples": int(label_rows),
                "short_label_examples": int(short_rows),
            }
        )

    candidate_df = pd.DataFrame(rows)
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values(["env_name", "sentence_count", "example_id"]).reset_index(drop=True)
    count_df = pd.DataFrame(count_rows).sort_values("env_name").reset_index(drop=True)
    return candidate_df, lookup, count_df


def select_examples_round_robin(
    candidate_df: pd.DataFrame,
    *,
    env_names: list[str],
    num_examples: int,
    selection_seed: int,
    case_study_example_id: str | None,
) -> pd.DataFrame:
    if candidate_df.empty:
        raise ValueError("Candidate dataframe is empty.")

    rng = np.random.default_rng(int(selection_seed))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for env_name in env_names:
        env_rows = candidate_df.loc[candidate_df["env_name"].eq(env_name)].to_dict(orient="records")
        if env_rows:
            order = rng.permutation(len(env_rows))
            grouped[env_name] = [env_rows[idx] for idx in order]
        else:
            grouped[env_name] = []

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    if case_study_example_id:
        matches = candidate_df.loc[candidate_df["example_id"].eq(case_study_example_id)]
        if not matches.empty:
            row = matches.iloc[0].to_dict()
            selected.append(row)
            selected_keys.add(str(row["example_key"]))
            grouped[str(row["env_name"])] = [
                item for item in grouped[str(row["env_name"])] if str(item["example_key"]) != str(row["example_key"])
            ]

    available_total = int(candidate_df.shape[0])
    target_total = min(int(num_examples), available_total)
    while len(selected) < target_total and any(grouped[env_name] for env_name in env_names):
        for env_name in env_names:
            while grouped[env_name]:
                row = grouped[env_name].pop(0)
                example_key = str(row["example_key"])
                if example_key in selected_keys:
                    continue
                selected.append(row)
                selected_keys.add(example_key)
                break
            if len(selected) >= target_total:
                break

    selected_df = pd.DataFrame(selected)
    if selected_df.empty:
        raise ValueError("Selection produced zero examples.")
    return selected_df.sort_values(["env_name", "sentence_count", "example_id"]).reset_index(drop=True)


def prepare_example_runtime(
    example: dict[str, Any],
    *,
    slb_module: Any,
    prepare_messages_for_model_fn: Any,
    model_name: str,
    text_field: str,
    tokenizer: Any,
) -> dict[str, Any]:
    raw_text = extract_raw_text(example, text_field)
    if not raw_text:
        raise ValueError("Could not extract raw reasoning text.")
    sentences = split_sentence_spans(raw_text)
    if not sentences:
        raise ValueError("No sentence spans extracted for example.")

    prepared_messages = None
    prompt_text = None
    if example.get("messages"):
        try:
            prepared_messages = prepare_messages_for_model_fn(example["messages"], model_name=model_name)
            prompt_text = tokenizer.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            prompt_text = example.get("prompt")
    else:
        prompt_text = example.get("prompt")
    if not prompt_text:
        raise ValueError("Could not build prompt text.")

    game = slb_module._infer_game(example, "auto", prompt_text)
    context = slb_module._extract_eval_context(game, example, prompt_text)
    if context is None:
        raise ValueError("Could not infer evaluation context.")

    return {
        "raw_text": raw_text,
        "sentences": sentences,
        "prepared_messages": prepared_messages,
        "prompt_text": prompt_text,
        "game": game,
        "context": context,
    }


def compress_generation_outcomes(generations: list[dict[str, Any]]) -> list[int | None]:
    outcomes: list[int | None] = []
    for generation in generations:
        is_truthful = generation.get("is_truthful")
        if is_truthful is None:
            outcomes.append(None)
        else:
            outcomes.append(1 if bool(is_truthful) else 0)
    return outcomes


def localize_example_once(
    *,
    slb_module: Any,
    prepare_messages_for_model_fn: Any,
    llm: Any,
    tokenizer: Any,
    example_row: dict[str, Any],
    example: dict[str, Any],
    resolved_model_name: str,
    text_field: str,
    reference_sample_size: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    base_seed: int,
    save_full_generations: bool,
    disable_tqdm: bool,
) -> dict[str, Any]:
    runtime = prepare_example_runtime(
        example,
        slb_module=slb_module,
        prepare_messages_for_model_fn=prepare_messages_for_model_fn,
        model_name=resolved_model_name,
        text_field=text_field,
        tokenizer=tokenizer,
    )
    use_reasoning_parser = bool(slb_module._guess_reasoning_model(resolved_model_name))

    sentence_rows: list[dict[str, Any]] = []
    iterator = maybe_tqdm(
        list(enumerate(runtime["sentences"])),
        desc=f"Sentence prefixes:{example_row['env_name']}:{example_row['example_id']}",
        total=len(runtime["sentences"]),
        disable=disable_tqdm,
        leave=False,
    )
    for sentence_idx, sentence in iterator:
        prefix_text = runtime["raw_text"][: int(sentence["end"])]
        _, _, _, generations = slb_module.sample_actions_for_prefix(
            llm,
            tokenizer,
            resolved_model_name,
            runtime["game"],
            runtime["context"],
            runtime["prompt_text"],
            runtime["prepared_messages"],
            prefix_text,
            n_samples=int(reference_sample_size),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            max_new_tokens=int(max_new_tokens),
            base_seed=int(base_seed) + int(sentence_idx) + 1,
            use_reasoning_parser=bool(use_reasoning_parser),
        )
        outcomes = compress_generation_outcomes(generations)
        sentence_row = {
            "sentence_idx": int(sentence_idx),
            "sentence_number": int(sentence_idx) + 1,
            "sentence_text": str(sentence["text"]),
            "char_start": int(sentence["start"]),
            "char_end": int(sentence["end"]),
            "outcomes": outcomes,
        }
        if save_full_generations:
            sentence_row["generations"] = generations
        sentence_rows.append(sentence_row)

    return {
        "example_key": str(example_row["example_key"]),
        "example_id": str(example_row["example_id"]),
        "env_name": str(example_row["env_name"]),
        "env_dirname": str(example_row["env_dirname"]),
        "model_dirname": str(example_row["model_dirname"]),
        "meta_model_name": example.get("meta_model_name"),
        "resolved_model_name": resolved_model_name,
        "reference_sample_size": int(reference_sample_size),
        "use_reasoning_parser": bool(use_reasoning_parser),
        "sentence_count": int(len(sentence_rows)),
        "sentence_rows": sentence_rows,
    }


def build_analysis_tables(
    *,
    example_payloads: list[dict[str, Any]],
    sample_sizes: tuple[int, ...],
    reference_sample_size: int,
    resample_repeats: int,
    resample_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix_summary_rows: list[dict[str, Any]] = []
    prefix_repeat_rows: list[dict[str, Any]] = []
    example_repeat_rows: list[dict[str, Any]] = []

    sample_sizes = tuple(sorted(set(int(value) for value in sample_sizes)))
    if int(reference_sample_size) not in sample_sizes:
        sample_sizes = tuple(sorted(sample_sizes + (int(reference_sample_size),)))

    for example_idx, payload in enumerate(example_payloads):
        sentence_rate_by_n: dict[int, np.ndarray] = {}
        sentence_rows = list(payload["sentence_rows"])
        sentence_count = len(sentence_rows)
        if sentence_count == 0:
            continue

        reference_rates = np.empty(sentence_count, dtype=np.float32)

        for sentence_row in sentence_rows:
            sentence_idx = int(sentence_row["sentence_idx"])
            sentence_number = int(sentence_row["sentence_number"])
            outcomes_array = outcomes_to_float_array(list(sentence_row["outcomes"]))
            if outcomes_array.shape[0] < int(reference_sample_size):
                raise ValueError(
                    f"Example {payload['example_id']} sentence {sentence_idx} has only "
                    f"{outcomes_array.shape[0]} outcomes, expected at least {reference_sample_size}."
                )

            ref_stats = deception_stats_from_float_array(outcomes_array[: int(reference_sample_size)])
            reference_rates[sentence_idx] = float(ref_stats["deception_rate"])

            prefix_summary_rows.append(
                {
                    "example_key": str(payload["example_key"]),
                    "example_id": str(payload["example_id"]),
                    "env_name": str(payload["env_name"]),
                    "n_samples": int(reference_sample_size),
                    "sentence_idx": sentence_idx,
                    "sentence_number": sentence_number,
                    "sentence_text": str(sentence_row["sentence_text"]),
                    "reference_deception_rate": float(ref_stats["deception_rate"]),
                    "mean_deception_rate": float(ref_stats["deception_rate"]),
                    "q10_deception_rate": float(ref_stats["deception_rate"]),
                    "q90_deception_rate": float(ref_stats["deception_rate"]),
                    "mean_abs_diff_vs_reference": 0.0,
                    "p90_abs_diff_vs_reference": 0.0,
                    "prob_within_0_05": 1.0,
                    "prob_within_0_10": 1.0,
                    "num_valid_reference": int(ref_stats["num_valid"]),
                    "num_invalid_reference": int(ref_stats["num_invalid"]),
                    "resample_repeats": 1,
                }
            )

            for sample_size in sample_sizes:
                if int(sample_size) == int(reference_sample_size):
                    continue
                rng = np.random.default_rng(
                    int(resample_seed)
                    + (example_idx * 100_000)
                    + (sentence_idx * 1_000)
                    + int(sample_size)
                )
                subset_rates = np.empty(int(resample_repeats), dtype=np.float32)
                abs_diffs = np.empty(int(resample_repeats), dtype=np.float32)
                n_outcomes = int(reference_sample_size)
                for repeat_idx in range(int(resample_repeats)):
                    subset_idx = rng.choice(n_outcomes, size=int(sample_size), replace=False)
                    subset_stats = deception_stats_from_float_array(outcomes_array[subset_idx])
                    subset_rate = float(subset_stats["deception_rate"])
                    subset_rates[repeat_idx] = subset_rate
                    abs_diffs[repeat_idx] = abs(subset_rate - float(ref_stats["deception_rate"]))
                    prefix_repeat_rows.append(
                        {
                            "example_key": str(payload["example_key"]),
                            "example_id": str(payload["example_id"]),
                            "env_name": str(payload["env_name"]),
                            "n_samples": int(sample_size),
                            "sentence_idx": sentence_idx,
                            "sentence_number": sentence_number,
                            "repeat_idx": int(repeat_idx),
                            "subset_deception_rate": subset_rate,
                            "reference_deception_rate": float(ref_stats["deception_rate"]),
                            "abs_diff_vs_reference": float(abs_diffs[repeat_idx]),
                        }
                    )

                prefix_summary_rows.append(
                    {
                        "example_key": str(payload["example_key"]),
                        "example_id": str(payload["example_id"]),
                        "env_name": str(payload["env_name"]),
                        "n_samples": int(sample_size),
                        "sentence_idx": sentence_idx,
                        "sentence_number": sentence_number,
                        "sentence_text": str(sentence_row["sentence_text"]),
                        "reference_deception_rate": float(ref_stats["deception_rate"]),
                        "mean_deception_rate": float(np.mean(subset_rates, dtype=np.float32)),
                        "q10_deception_rate": float(np.quantile(subset_rates, 0.10)),
                        "q90_deception_rate": float(np.quantile(subset_rates, 0.90)),
                        "mean_abs_diff_vs_reference": float(np.mean(abs_diffs, dtype=np.float32)),
                        "p90_abs_diff_vs_reference": float(np.quantile(abs_diffs, 0.90)),
                        "prob_within_0_05": float(np.mean(abs_diffs <= 0.05, dtype=np.float32)),
                        "prob_within_0_10": float(np.mean(abs_diffs <= 0.10, dtype=np.float32)),
                        "num_valid_reference": int(ref_stats["num_valid"]),
                        "num_invalid_reference": int(ref_stats["num_invalid"]),
                        "resample_repeats": int(resample_repeats),
                    }
                )
                sentence_rate_by_n.setdefault(int(sample_size), np.empty((sentence_count, int(resample_repeats)), dtype=np.float32))
                sentence_rate_by_n[int(sample_size)][sentence_idx, :] = subset_rates

        reference_peak_idx = int(np.argmax(reference_rates))
        if sentence_count > 1:
            reference_jump_values = np.diff(reference_rates)
            reference_jump_idx = int(np.argmax(reference_jump_values) + 1)
            reference_jump_magnitude = float(np.max(reference_jump_values))
        else:
            reference_jump_idx = 0
            reference_jump_magnitude = 0.0

        example_repeat_rows.append(
            {
                "example_key": str(payload["example_key"]),
                "example_id": str(payload["example_id"]),
                "env_name": str(payload["env_name"]),
                "n_samples": int(reference_sample_size),
                "repeat_idx": 0,
                "reference_peak_sentence_idx": reference_peak_idx,
                "subset_peak_sentence_idx": reference_peak_idx,
                "peak_exact": 1.0,
                "peak_within_one": 1.0,
                "reference_jump_sentence_idx": reference_jump_idx,
                "reference_jump_magnitude": reference_jump_magnitude,
                "subset_jump_sentence_idx": reference_jump_idx,
                "jump_exact": 1.0,
                "jump_within_one": 1.0,
            }
        )

        for sample_size, rate_matrix in sentence_rate_by_n.items():
            for repeat_idx in range(rate_matrix.shape[1]):
                subset_rates = rate_matrix[:, repeat_idx]
                subset_peak_idx = int(np.argmax(subset_rates))
                if sentence_count > 1:
                    subset_jump_idx = int(np.argmax(np.diff(subset_rates)) + 1)
                else:
                    subset_jump_idx = 0
                example_repeat_rows.append(
                    {
                        "example_key": str(payload["example_key"]),
                        "example_id": str(payload["example_id"]),
                        "env_name": str(payload["env_name"]),
                        "n_samples": int(sample_size),
                        "repeat_idx": int(repeat_idx),
                        "reference_peak_sentence_idx": reference_peak_idx,
                        "subset_peak_sentence_idx": subset_peak_idx,
                        "peak_exact": float(subset_peak_idx == reference_peak_idx),
                        "peak_within_one": float(abs(subset_peak_idx - reference_peak_idx) <= 1),
                        "reference_jump_sentence_idx": reference_jump_idx,
                        "reference_jump_magnitude": reference_jump_magnitude,
                        "subset_jump_sentence_idx": subset_jump_idx,
                        "jump_exact": float(subset_jump_idx == reference_jump_idx),
                        "jump_within_one": float(abs(subset_jump_idx - reference_jump_idx) <= 1),
                    }
                )

    return (
        pd.DataFrame(prefix_summary_rows),
        pd.DataFrame(prefix_repeat_rows),
        pd.DataFrame(example_repeat_rows),
    )


def aggregate_budget_tables(
    *,
    prefix_summary_df: pd.DataFrame,
    prefix_repeat_df: pd.DataFrame,
    example_repeat_df: pd.DataFrame,
    sample_sizes: tuple[int, ...],
    reference_sample_size: int,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    jump_agreement_min_reference_magnitude: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix_budget_rows: list[dict[str, Any]] = []
    example_agreement_rows: list[dict[str, Any]] = []
    jump_reference_threshold = max(float(jump_agreement_min_reference_magnitude), 0.0)

    prefix_count = int(prefix_summary_df["example_key"].astype(str).str.cat(prefix_summary_df["sentence_idx"].astype(str), sep="::").nunique())
    example_count = int(prefix_summary_df["example_key"].nunique())

    for sample_size in sample_sizes:
        subset_prefix_summary = prefix_summary_df.loc[prefix_summary_df["n_samples"].eq(int(sample_size))].copy()
        subset_prefix_repeat = prefix_repeat_df.loc[prefix_repeat_df["n_samples"].eq(int(sample_size))].copy()
        subset_example_repeat = example_repeat_df.loc[example_repeat_df["n_samples"].eq(int(sample_size))].copy()
        subset_abs_diff = subset_prefix_repeat["abs_diff_vs_reference"].to_numpy(dtype=np.float32, copy=False)
        subset_num_valid = subset_prefix_summary["num_valid_reference"].to_numpy(dtype=np.float32, copy=False)

        mean_abs_diff, mean_abs_diff_lo, mean_abs_diff_hi = bootstrap_mean_ci(
            subset_prefix_summary["mean_abs_diff_vs_reference"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size)),
            repeats=int(bootstrap_repeats),
        )
        prob_005, prob_005_lo, prob_005_hi = bootstrap_mean_ci(
            subset_prefix_summary["prob_within_0_05"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size) + 1_000),
            repeats=int(bootstrap_repeats),
        )
        prob_010, prob_010_lo, prob_010_hi = bootstrap_mean_ci(
            subset_prefix_summary["prob_within_0_10"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size) + 2_000),
            repeats=int(bootstrap_repeats),
        )
        prefix_budget_rows.append(
            {
                "n_samples": int(sample_size),
                "reference_sample_size": int(reference_sample_size),
                "num_examples": example_count,
                "num_prefixes": prefix_count,
                "mean_abs_diff_vs_reference": mean_abs_diff,
                "mean_abs_diff_ci_low": mean_abs_diff_lo,
                "mean_abs_diff_ci_high": mean_abs_diff_hi,
                "median_abs_diff_vs_reference": (
                    float(np.median(subset_abs_diff))
                    if subset_abs_diff.size > 0
                    else 0.0
                ),
                "p90_abs_diff_vs_reference": (
                    float(np.quantile(subset_abs_diff, 0.90))
                    if subset_abs_diff.size > 0
                    else 0.0
                ),
                "frac_within_0_05": prob_005,
                "frac_within_0_05_ci_low": prob_005_lo,
                "frac_within_0_05_ci_high": prob_005_hi,
                "frac_within_0_10": prob_010,
                "frac_within_0_10_ci_low": prob_010_lo,
                "frac_within_0_10_ci_high": prob_010_hi,
                "mean_valid_reference": float(np.mean(subset_num_valid, dtype=np.float32)) if subset_num_valid.size > 0 else 0.0,
            }
        )

        per_example_agreement = (
            subset_example_repeat.groupby(["example_key", "n_samples"], as_index=False)
            .agg(
                mean_peak_exact=("peak_exact", "mean"),
                mean_peak_within_one=("peak_within_one", "mean"),
                mean_jump_exact=("jump_exact", "mean"),
                mean_jump_within_one=("jump_within_one", "mean"),
                reference_jump_magnitude=("reference_jump_magnitude", "first"),
            )
        )
        jump_eligible_mask = (
            per_example_agreement["reference_jump_magnitude"].to_numpy(dtype=np.float32, copy=False)
            >= jump_reference_threshold
        )
        jump_eligible_df = per_example_agreement.loc[jump_eligible_mask].copy()
        jump_eligible_count = int(jump_eligible_df["example_key"].nunique())
        per_example_count = int(per_example_agreement["example_key"].nunique())
        peak_exact, peak_exact_lo, peak_exact_hi = bootstrap_mean_ci(
            per_example_agreement["mean_peak_exact"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size) + 3_000),
            repeats=int(bootstrap_repeats),
        )
        peak_within, peak_within_lo, peak_within_hi = bootstrap_mean_ci(
            per_example_agreement["mean_peak_within_one"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size) + 4_000),
            repeats=int(bootstrap_repeats),
        )
        jump_exact, jump_exact_lo, jump_exact_hi = bootstrap_mean_ci(
            per_example_agreement["mean_jump_exact"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size) + 5_000),
            repeats=int(bootstrap_repeats),
        )
        jump_within, jump_within_lo, jump_within_hi = bootstrap_mean_ci(
            per_example_agreement["mean_jump_within_one"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size) + 6_000),
            repeats=int(bootstrap_repeats),
        )
        jump_exact_filtered, jump_exact_filtered_lo, jump_exact_filtered_hi = bootstrap_mean_ci(
            jump_eligible_df["mean_jump_exact"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size) + 7_000),
            repeats=int(bootstrap_repeats),
        )
        jump_within_filtered, jump_within_filtered_lo, jump_within_filtered_hi = bootstrap_mean_ci(
            jump_eligible_df["mean_jump_within_one"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(int(bootstrap_seed) + int(sample_size) + 8_000),
            repeats=int(bootstrap_repeats),
        )
        example_agreement_rows.append(
            {
                "n_samples": int(sample_size),
                "reference_sample_size": int(reference_sample_size),
                "num_examples": per_example_count,
                "peak_exact": peak_exact,
                "peak_exact_ci_low": peak_exact_lo,
                "peak_exact_ci_high": peak_exact_hi,
                "peak_within_one": peak_within,
                "peak_within_one_ci_low": peak_within_lo,
                "peak_within_one_ci_high": peak_within_hi,
                "jump_exact": jump_exact,
                "jump_exact_ci_low": jump_exact_lo,
                "jump_exact_ci_high": jump_exact_hi,
                "jump_within_one": jump_within,
                "jump_within_one_ci_low": jump_within_lo,
                "jump_within_one_ci_high": jump_within_hi,
                "jump_agreement_min_reference_magnitude": jump_reference_threshold,
                "jump_eligible_examples": jump_eligible_count,
                "jump_eligible_fraction": (
                    float(jump_eligible_count / per_example_count) if per_example_count > 0 else float("nan")
                ),
                "jump_exact_ref_jump_ge_threshold": jump_exact_filtered,
                "jump_exact_ref_jump_ge_threshold_ci_low": jump_exact_filtered_lo,
                "jump_exact_ref_jump_ge_threshold_ci_high": jump_exact_filtered_hi,
                "jump_within_one_ref_jump_ge_threshold": jump_within_filtered,
                "jump_within_one_ref_jump_ge_threshold_ci_low": jump_within_filtered_lo,
                "jump_within_one_ref_jump_ge_threshold_ci_high": jump_within_filtered_hi,
            }
        )

    prefix_budget_df = pd.DataFrame(prefix_budget_rows).sort_values("n_samples").reset_index(drop=True)
    example_agreement_df = pd.DataFrame(example_agreement_rows).sort_values("n_samples").reset_index(drop=True)
    paper_summary_df = prefix_budget_df.merge(
        example_agreement_df,
        on=["n_samples", "reference_sample_size", "num_examples"],
        how="left",
        validate="one_to_one",
    )
    return prefix_budget_df, example_agreement_df, paper_summary_df


def plot_error_vs_budget(prefix_budget_df: pd.DataFrame, *, out_path: Path) -> None:
    subset = prefix_budget_df.loc[prefix_budget_df["n_samples"].ne(prefix_budget_df["reference_sample_size"])].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = subset["n_samples"].to_numpy(dtype=int)
    y = subset["mean_abs_diff_vs_reference"].to_numpy(dtype=float)
    lo = subset["mean_abs_diff_ci_low"].to_numpy(dtype=float)
    hi = subset["mean_abs_diff_ci_high"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.2, color="#1f4e79")
    ax.fill_between(x, lo, hi, color="#1f4e79", alpha=0.18)
    ax.set_xlabel("Continuation samples per prefix")
    ax.set_ylabel("Mean absolute error vs 100-sample reference")
    ax.set_title("Sampling budget ablation")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_vs_budget(prefix_budget_df: pd.DataFrame, *, out_path: Path) -> None:
    subset = prefix_budget_df.loc[prefix_budget_df["n_samples"].ne(prefix_budget_df["reference_sample_size"])].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = subset["n_samples"].to_numpy(dtype=int)
    ax.plot(x, subset["frac_within_0_05"], marker="o", linewidth=2.0, label="|error| <= 0.05")
    ax.plot(x, subset["frac_within_0_10"], marker="o", linewidth=2.0, label="|error| <= 0.10")
    ax.set_xlabel("Continuation samples per prefix")
    ax.set_ylabel("Fraction of estimates within threshold")
    ax.set_title("Prefix-level stability vs budget")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_abs_diff_cdf(prefix_repeat_df: pd.DataFrame, *, reference_sample_size: int, out_path: Path) -> None:
    subset = prefix_repeat_df.copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for sample_size in sorted(subset["n_samples"].unique()):
        if int(sample_size) == int(reference_sample_size):
            continue
        values = np.sort(
            subset.loc[subset["n_samples"].eq(int(sample_size)), "abs_diff_vs_reference"].to_numpy(dtype=np.float32, copy=False)
        )
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1, dtype=np.float32) / float(values.size)
        ax.plot(values, y, linewidth=2.0, label=f"{int(sample_size)} samples")
    ax.set_xlabel(f"Absolute error vs {reference_sample_size}-sample reference")
    ax.set_ylabel("Cumulative fraction of resampled prefixes")
    ax.set_title("Distribution of localization error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_case_study_curves(
    prefix_summary_df: pd.DataFrame,
    *,
    example_key: str,
    reference_sample_size: int,
    out_path: Path,
) -> None:
    case_df = prefix_summary_df.loc[prefix_summary_df["example_key"].eq(example_key)].copy()
    if case_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.4), constrained_layout=True)
    sample_sizes = sorted(case_df["n_samples"].unique())
    for sample_size in sample_sizes:
        subset = case_df.loc[case_df["n_samples"].eq(int(sample_size))].sort_values("sentence_idx")
        x = subset["sentence_number"].to_numpy(dtype=int)
        y = subset["mean_deception_rate"].to_numpy(dtype=float)
        if int(sample_size) == int(reference_sample_size):
            ax.plot(x, y, color="black", linewidth=2.8, marker="o", label=f"{int(sample_size)} samples")
        else:
            lo = subset["q10_deception_rate"].to_numpy(dtype=float)
            hi = subset["q90_deception_rate"].to_numpy(dtype=float)
            ax.plot(x, y, linewidth=2.0, marker="o", label=f"{int(sample_size)} samples")
            ax.fill_between(x, lo, hi, alpha=0.14)
    title_example_id = str(case_df["example_id"].iloc[0])
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Estimated deception rate")
    ax.set_title(f"Case-study localization curve\n{title_example_id}")
    ax.set_xticks(sorted(case_df["sentence_number"].unique()))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Continuation budget")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_case_study_error(
    prefix_summary_df: pd.DataFrame,
    *,
    example_key: str,
    reference_sample_size: int,
    out_path: Path,
) -> None:
    case_df = prefix_summary_df.loc[prefix_summary_df["example_key"].eq(example_key)].copy()
    if case_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    for sample_size in sorted(case_df["n_samples"].unique()):
        if int(sample_size) == int(reference_sample_size):
            continue
        subset = case_df.loc[case_df["n_samples"].eq(int(sample_size))].sort_values("sentence_idx")
        ax.plot(
            subset["sentence_number"],
            subset["mean_abs_diff_vs_reference"],
            marker="o",
            linewidth=2.0,
            label=f"{int(sample_size)} vs {reference_sample_size}",
        )
    title_example_id = str(case_df["example_id"].iloc[0])
    ax.set_xlabel("Sentence index")
    ax.set_ylabel(f"Mean absolute error vs {reference_sample_size}")
    ax.set_title(f"Case-study error by sentence\n{title_example_id}")
    ax.set_xticks(sorted(case_df["sentence_number"].unique()))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Budget")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sentence_agreement(
    example_agreement_df: pd.DataFrame,
    *,
    reference_sample_size: int,
    jump_agreement_min_reference_magnitude: float,
    out_path: Path,
) -> None:
    subset = example_agreement_df.loc[example_agreement_df["n_samples"].ne(int(reference_sample_size))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    x = subset["n_samples"].to_numpy(dtype=int)

    axes[0].plot(x, subset["peak_exact"], marker="o", linewidth=2.0, label="Exact")
    axes[0].plot(x, subset["peak_within_one"], marker="o", linewidth=2.0, label="Within one sentence")
    axes[0].set_title("Peak-deception sentence agreement")
    axes[0].set_xlabel("Continuation samples per prefix")
    axes[0].set_ylabel("Agreement")
    axes[0].set_xticks(x)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    jump_threshold = max(float(jump_agreement_min_reference_magnitude), 0.0)
    if jump_threshold > 0.0 and "jump_within_one_ref_jump_ge_threshold" in subset.columns:
        axes[1].plot(x, subset["jump_exact_ref_jump_ge_threshold"], marker="o", linewidth=2.0, label="Exact")
        axes[1].plot(
            x,
            subset["jump_within_one_ref_jump_ge_threshold"],
            marker="o",
            linewidth=2.0,
            label="Within one sentence",
        )
        axes[1].set_title(f"Largest positive jump agreement (ref jump >= {jump_threshold:g})")
    else:
        axes[1].plot(x, subset["jump_exact"], marker="o", linewidth=2.0, label="Exact")
        axes[1].plot(x, subset["jump_within_one"], marker="o", linewidth=2.0, label="Within one sentence")
        axes[1].set_title("Largest positive jump agreement")
    axes[1].set_xlabel("Continuation samples per prefix")
    axes[1].set_ylabel("Agreement")
    axes[1].set_xticks(x)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_paper_summary_markdown(
    *,
    paper_summary_df: pd.DataFrame,
    selected_examples_df: pd.DataFrame,
    case_example_key: str,
    reference_sample_size: int,
    jump_agreement_min_reference_magnitude: float,
) -> str:
    row_50 = paper_summary_df.loc[paper_summary_df["n_samples"].eq(50)]
    case_row = selected_examples_df.loc[selected_examples_df["example_key"].eq(case_example_key)]
    lines: list[str] = []
    lines.append("# Localization Sampling Budget Summary")
    lines.append("")
    lines.append(f"- Examples analyzed: {int(selected_examples_df.shape[0])}")
    lines.append(f"- Reference budget: {int(reference_sample_size)} continuations per prefix")
    if not case_row.empty:
        lines.append(
            "- Case-study example: "
            f"{case_row.iloc[0]['env_name']} / {case_row.iloc[0]['example_id']} "
            f"({int(case_row.iloc[0]['sentence_count'])} sentences)"
        )
    if not row_50.empty:
        row = row_50.iloc[0]
        lines.append("")
        lines.append("## 50 vs 100")
        lines.append(
            "- Mean absolute error vs reference: "
            f"{row['mean_abs_diff_vs_reference']:.4f} "
            f"[{row['mean_abs_diff_ci_low']:.4f}, {row['mean_abs_diff_ci_high']:.4f}]"
        )
        lines.append(
            "- Fraction within 0.05: "
            f"{row['frac_within_0_05']:.3f} "
            f"[{row['frac_within_0_05_ci_low']:.3f}, {row['frac_within_0_05_ci_high']:.3f}]"
        )
        lines.append(
            "- Fraction within 0.10: "
            f"{row['frac_within_0_10']:.3f} "
            f"[{row['frac_within_0_10_ci_low']:.3f}, {row['frac_within_0_10_ci_high']:.3f}]"
        )
        lines.append(
            "- Peak-deception sentence agreement: "
            f"{row['peak_exact']:.3f} exact, {row['peak_within_one']:.3f} within one sentence"
        )
        jump_threshold = max(float(jump_agreement_min_reference_magnitude), 0.0)
        if jump_threshold > 0.0 and "jump_within_one_ref_jump_ge_threshold" in row.index:
            lines.append(
                "- Largest positive jump agreement "
                f"(reference jump >= {jump_threshold:g}; {int(row['jump_eligible_examples'])}/{int(row['num_examples'])} eligible examples): "
                f"{row['jump_exact_ref_jump_ge_threshold']:.3f} exact, "
                f"{row['jump_within_one_ref_jump_ge_threshold']:.3f} within one sentence"
            )
        else:
            lines.append(
                "- Largest positive jump agreement: "
                f"{row['jump_exact']:.3f} exact, {row['jump_within_one']:.3f} within one sentence"
            )
    return "\n".join(lines) + "\n"


def build_run_tag(model_dirname: str, num_examples: int, max_sentences: int) -> str:
    return f"{slugify(model_dirname)}__n{int(num_examples)}__shortlt{int(max_sentences)}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate localization sampling-budget ablation.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--model-dirname", type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--envs", nargs="*", default=list(ENV_SPECS.keys()))
    parser.add_argument("--text-field", type=str, default="action_reasoning")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--max-sentences", type=int, default=20)
    parser.add_argument("--sample-sizes", type=str, default="10,25,50,100")
    parser.add_argument("--resample-repeats", type=int, default=64)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--selection-seed", type=int, default=1234)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--resample-seed", type=int, default=2026)
    parser.add_argument("--jump-agreement-min-reference-magnitude", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="", help="Optional override for the HF/vLLM model identifier.")
    parser.add_argument("--case-study-example-id", type=str, default="")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(NOTEBOOK_ROOT / "datasetmain_localization_sampling_budget_ablation_outputs"),
    )
    parser.add_argument("--selection-only", action="store_true", default=False)
    parser.add_argument("--analysis-only", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--save-full-generations", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    args = parser.parse_args(argv)

    env_names = parse_envs(args.envs)
    sample_sizes = parse_int_list(args.sample_sizes)
    reference_sample_size = max(sample_sizes)
    if min(sample_sizes) <= 0:
        raise ValueError(f"--sample-sizes must be positive, got {sample_sizes}.")
    if int(reference_sample_size) < 2:
        raise ValueError(f"Reference sample size must be >= 2, got {reference_sample_size}.")
    if int(args.num_examples) < 1:
        raise ValueError("--num-examples must be >= 1.")

    dataset_root = Path(args.dataset_root)
    run_tag = args.run_tag.strip() or build_run_tag(args.model_dirname, args.num_examples, args.max_sentences)
    output_root = Path(args.output_root) / run_tag
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "per_example_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config_df = pd.DataFrame(
        [
            {"setting": "dataset_root", "value": str(dataset_root)},
            {"setting": "model_dirname", "value": args.model_dirname},
            {"setting": "envs", "value": ", ".join(env_names)},
            {"setting": "num_examples", "value": int(args.num_examples)},
            {"setting": "max_sentences", "value": int(args.max_sentences)},
            {"setting": "sample_sizes", "value": ", ".join(str(value) for value in sample_sizes)},
            {"setting": "reference_sample_size", "value": int(reference_sample_size)},
            {"setting": "resample_repeats", "value": int(args.resample_repeats)},
            {"setting": "bootstrap_repeats", "value": int(args.bootstrap_repeats)},
            {"setting": "selection_seed", "value": int(args.selection_seed)},
            {"setting": "base_seed", "value": int(args.base_seed)},
            {"setting": "resample_seed", "value": int(args.resample_seed)},
            {"setting": "jump_agreement_min_reference_magnitude", "value": float(args.jump_agreement_min_reference_magnitude)},
            {"setting": "selection_only", "value": bool(args.selection_only)},
            {"setting": "analysis_only", "value": bool(args.analysis_only)},
            {"setting": "output_root", "value": str(output_root)},
        ]
    )
    config_df.to_csv(output_root / "config.csv", index=False)

    candidate_df, candidate_lookup, candidate_count_df = build_candidate_pool(
        dataset_root,
        env_names=env_names,
        model_dirname=args.model_dirname,
        text_field=args.text_field,
        deceptive_only=True,
        max_sentences=int(args.max_sentences),
        disable_tqdm=bool(args.disable_tqdm),
    )
    if candidate_df.empty:
        raise ValueError("No short deceptive candidates found for the requested configuration.")
    candidate_df.to_csv(output_root / "candidate_examples.csv", index=False)
    candidate_count_df.to_csv(output_root / "candidate_counts_by_env.csv", index=False)

    selected_df = select_examples_round_robin(
        candidate_df,
        env_names=env_names,
        num_examples=int(args.num_examples),
        selection_seed=int(args.selection_seed),
        case_study_example_id=args.case_study_example_id.strip() or None,
    )
    selection_count_df = (
        selected_df.groupby("env_name", as_index=False)
        .agg(selected_examples=("example_id", "count"))
        .sort_values("env_name")
        .reset_index(drop=True)
    )
    selected_df.to_csv(output_root / "selected_examples.csv", index=False)
    selection_count_df.to_csv(output_root / "selection_counts_by_env.csv", index=False)

    case_study_example_key: str
    if args.case_study_example_id.strip():
        matches = selected_df.loc[selected_df["example_id"].eq(args.case_study_example_id.strip())]
        case_study_example_key = (
            str(matches.iloc[0]["example_key"])
            if not matches.empty
            else str(selected_df.sort_values(["sentence_count", "example_id"]).iloc[0]["example_key"])
        )
    else:
        case_study_example_key = str(selected_df.sort_values(["sentence_count", "example_id"]).iloc[0]["example_key"])

    if args.selection_only:
        print(f"Selection written to {output_root / 'selected_examples.csv'}")
        print(f"Selection counts written to {output_root / 'selection_counts_by_env.csv'}")
        return

    resolved_model_name = args.model_name.strip()
    if not resolved_model_name:
        first_nonempty = (
            selected_df["meta_model_name"]
            .dropna()
            .astype(str)
            .loc[lambda s: s.str.len() > 0]
        )
        if first_nonempty.empty:
            raise ValueError(
                "Could not infer a vLLM model identifier from selected examples. "
                "Pass --model-name explicitly."
            )
        resolved_model_name = str(first_nonempty.iloc[0])

    example_payloads: list[dict[str, Any]] = []
    missing_cache_keys: list[str] = []

    if not args.analysis_only:
        import torch
        from vllm import LLM

        slb_module = load_sentence_localization_batch()
        prepare_messages_for_model_fn = load_prepare_messages_for_model()

        visible_gpu_count = max(1, int(torch.cuda.device_count()))
        if int(args.tensor_parallel_size) > visible_gpu_count:
            raise ValueError(
                f"--tensor-parallel-size={args.tensor_parallel_size} exceeds visible GPU count={visible_gpu_count}."
            )

        llm = LLM(
            model=resolved_model_name,
            max_model_len=int(args.max_new_tokens),
            seed=1,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            tensor_parallel_size=int(args.tensor_parallel_size),
        )
        tokenizer = llm.get_tokenizer()

        iterator = maybe_tqdm(
            list(selected_df.to_dict(orient="records")),
            desc="Localize selected examples",
            total=len(selected_df),
            disable=bool(args.disable_tqdm),
        )
        for example_idx, example_row in enumerate(iterator):
            example_key = str(example_row["example_key"])
            cache_path = cache_dir / cache_filename_for_example(str(example_row["env_name"]), str(example_row["example_id"]))
            if cache_path.exists() and not args.overwrite:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                example_payloads.append(payload)
                continue

            example = candidate_lookup[example_key]
            payload = localize_example_once(
                slb_module=slb_module,
                prepare_messages_for_model_fn=prepare_messages_for_model_fn,
                llm=llm,
                tokenizer=tokenizer,
                example_row=example_row,
                example=example,
                resolved_model_name=resolved_model_name,
                text_field=args.text_field,
                reference_sample_size=int(reference_sample_size),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
                max_new_tokens=int(args.max_new_tokens),
                base_seed=int(args.base_seed) + (example_idx * 10_000),
                save_full_generations=bool(args.save_full_generations),
                disable_tqdm=bool(args.disable_tqdm),
            )
            cache_path.write_text(json.dumps(slb_module.to_json_safe(payload), indent=2), encoding="utf-8")
            example_payloads.append(payload)
    else:
        for example_row in selected_df.to_dict(orient="records"):
            cache_path = cache_dir / cache_filename_for_example(str(example_row["env_name"]), str(example_row["example_id"]))
            if not cache_path.exists():
                missing_cache_keys.append(str(example_row["example_key"]))
                continue
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            example_payloads.append(payload)
        if missing_cache_keys:
            missing_preview = "\n".join(missing_cache_keys[:20])
            raise FileNotFoundError(
                "analysis-only mode requires all per-example caches to exist. Missing caches for:\n"
                f"{missing_preview}"
            )

    prefix_summary_df, prefix_repeat_df, example_repeat_df = build_analysis_tables(
        example_payloads=example_payloads,
        sample_sizes=sample_sizes,
        reference_sample_size=int(reference_sample_size),
        resample_repeats=int(args.resample_repeats),
        resample_seed=int(args.resample_seed),
    )
    prefix_budget_df, example_agreement_df, paper_summary_df = aggregate_budget_tables(
        prefix_summary_df=prefix_summary_df,
        prefix_repeat_df=prefix_repeat_df,
        example_repeat_df=example_repeat_df,
        sample_sizes=sample_sizes,
        reference_sample_size=int(reference_sample_size),
        bootstrap_repeats=int(args.bootstrap_repeats),
        bootstrap_seed=int(args.resample_seed),
        jump_agreement_min_reference_magnitude=float(args.jump_agreement_min_reference_magnitude),
    )

    prefix_summary_df.to_csv(output_root / "prefix_summary.csv", index=False)
    prefix_repeat_df.to_csv(output_root / "prefix_repeat_samples.csv", index=False)
    example_repeat_df.to_csv(output_root / "example_repeat_agreement.csv", index=False)
    prefix_budget_df.to_csv(output_root / "budget_summary_prefix.csv", index=False)
    example_agreement_df.to_csv(output_root / "budget_summary_example_agreement.csv", index=False)
    paper_summary_df.to_csv(output_root / "paper_budget_summary.csv", index=False)

    plot_error_vs_budget(prefix_budget_df, out_path=figures_dir / "budget_mean_abs_error.png")
    plot_coverage_vs_budget(prefix_budget_df, out_path=figures_dir / "budget_coverage.png")
    plot_abs_diff_cdf(
        prefix_repeat_df,
        reference_sample_size=int(reference_sample_size),
        out_path=figures_dir / "budget_abs_error_cdf.png",
    )
    plot_case_study_curves(
        prefix_summary_df,
        example_key=case_study_example_key,
        reference_sample_size=int(reference_sample_size),
        out_path=figures_dir / "case_study_curves.png",
    )
    plot_case_study_error(
        prefix_summary_df,
        example_key=case_study_example_key,
        reference_sample_size=int(reference_sample_size),
        out_path=figures_dir / "case_study_error.png",
    )
    plot_sentence_agreement(
        example_agreement_df,
        reference_sample_size=int(reference_sample_size),
        jump_agreement_min_reference_magnitude=float(args.jump_agreement_min_reference_magnitude),
        out_path=figures_dir / "budget_sentence_agreement.png",
    )

    summary_md = build_paper_summary_markdown(
        paper_summary_df=paper_summary_df,
        selected_examples_df=selected_df,
        case_example_key=case_study_example_key,
        reference_sample_size=int(reference_sample_size),
        jump_agreement_min_reference_magnitude=float(args.jump_agreement_min_reference_magnitude),
    )
    (output_root / "paper_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Finished localization sampling-budget ablation for model_dirname={args.model_dirname}")
    print(f"Selected examples: {len(selected_df)}")
    print(f"Output root: {output_root}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
