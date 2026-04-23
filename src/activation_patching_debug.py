from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import activation_patching as ap
from sentence_pipeline import split_sentence_spans


DEFAULT_DEBUG_PAIR_COUNT = 12
DEFAULT_DEBUG_SAMPLE_COUNT = 4
DEFAULT_DEBUG_BATCH_SIZE = 4
DEFAULT_DEBUG_MAX_NEW_TOKENS = 2048
DEFAULT_DEBUG_LAYER_COUNT = 5
DEFAULT_DEBUG_PATCH_MODES = ("residual", "kv", "both")
DEFAULT_DEBUG_OUTPUT_ROOT = ap.ROOT_DIR / "Results" / "activation_patching_debug"
DEFAULT_COMMITMENT_PAIR_COUNT = 50
DEFAULT_COMMITMENT_IG_STEPS = 4
DEFAULT_COMMITMENT_FAITHFULNESS = 0.85
DEFAULT_COMMITMENT_RANDOM_EDGE_SAMPLES = 20
DEFAULT_COMMITMENT_TOKEN_BIN_COUNT = 8
DEFAULT_COMMITMENT_CURVE_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_COMMITMENT_OBJECTIVE_TOKEN_COUNT = 3
DEFAULT_COMMITMENT_OBJECTIVE_TOKEN_POSITION = "last"
DEFAULT_COMMITMENT_EXCLUDE_FINAL_LAYERS = 2
DEFAULT_COMMITMENT_PAIR_SEARCH_LIMIT = 400
DEFAULT_EXPERIMENT_MODE = "commitment_eap_ig"
TAG_ONLY_RE = re.compile(r"^\s*</?[^>]+>\s*$")


def parse_patch_modes(text: str) -> list[str]:
    if not str(text).strip():
        return list(DEFAULT_DEBUG_PATCH_MODES)
    allowed = {"residual", "kv", "both"}
    patch_modes: list[str] = []
    for raw_mode in str(text).split(","):
        patch_mode = raw_mode.strip().lower()
        if not patch_mode:
            continue
        if patch_mode not in allowed:
            raise ValueError(f"Unsupported patch mode {raw_mode!r}. Choose from residual, kv, both.")
        if patch_mode not in patch_modes:
            patch_modes.append(patch_mode)
    if not patch_modes:
        raise ValueError("At least one patch mode is required.")
    return patch_modes


def build_patch_conditions_with_modes(layer_candidates: list[int], *, patch_modes: Iterable[str]) -> list[dict[str, Any]]:
    if hasattr(ap, "build_single_layer_patch_conditions_with_modes"):
        return ap.build_single_layer_patch_conditions_with_modes(layer_candidates, patch_modes=patch_modes)

    mode_label_map = {
        "residual": "Residual",
        "kv": "K/V",
        "both": "Residual + K/V",
    }
    conditions: list[dict[str, Any]] = []
    for raw_patch_mode in patch_modes:
        patch_mode = str(raw_patch_mode).strip().lower()
        if patch_mode not in mode_label_map:
            raise ValueError(f"Unsupported patch_mode={raw_patch_mode!r}")
        mode_label = mode_label_map[patch_mode]
        for layer_idx in layer_candidates:
            layer_idx = int(layer_idx)
            conditions.append(
                {
                    "condition_name": f"denoising_layer_{layer_idx}__{patch_mode}",
                    "patch_label": f"{mode_label} | Denoising | Layer {layer_idx}",
                    "experiment": "denoising",
                    "target_prefix_role": "deceptive",
                    "donor_prefix_role": "truthful",
                    "patch_mode": patch_mode,
                    "layer_indices": (layer_idx,),
                }
            )
            conditions.append(
                {
                    "condition_name": f"noising_layer_{layer_idx}__{patch_mode}",
                    "patch_label": f"{mode_label} | Noising | Layer {layer_idx}",
                    "experiment": "noising",
                    "target_prefix_role": "truthful",
                    "donor_prefix_role": "deceptive",
                    "patch_mode": patch_mode,
                    "layer_indices": (layer_idx,),
                }
            )
    return conditions


def load_model_with_dtype_compat(
    model_name_or_path: str,
    *,
    common_kwargs: dict[str, Any],
    dtype_value: torch.dtype,
) -> Any:
    last_exc: Exception | None = None
    for dtype_key in ("dtype", "torch_dtype"):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **common_kwargs,
                **{dtype_key: dtype_value},
            )
        except TypeError as exc:
            last_exc = exc
            if "unexpected keyword argument" not in str(exc):
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unable to load model with either dtype or torch_dtype.")


def _supports_batched_patch_mode(patch_mode: str) -> bool:
    patch_mode = str(patch_mode).strip().lower()
    if patch_mode in {"none", "residual"}:
        return hasattr(ap, "run_generation_condition_batch_samples")
    batch_fn = getattr(ap, "generate_batch_with_sentence_patch", None)
    if batch_fn is None or not hasattr(ap, "run_generation_condition_batch_samples"):
        return False
    try:
        source = inspect.getsource(batch_fn)
    except (OSError, TypeError):
        return hasattr(ap, "build_single_layer_patch_conditions_with_modes")
    return "Batched generation only supports residual or unpatched conditions." not in source


def run_generation_condition_samples_compat(
    model: Any,
    tokenizer: Any,
    *,
    condition_name: str,
    target_text: str,
    target_prefix_boundary_text: str,
    patch_label: str | None,
    patch_mode: str,
    layer_indices: tuple[int, ...] | None,
    donor_source: dict[str, Any] | None,
    required_rank: int,
    max_model_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sample_indices: list[int],
    seed_start: int,
    patch_scope: str = "sentence_span",
    early_stop_on_valid_json: bool = False,
    early_stop_check_interval: int = 16,
    early_stop_min_new_tokens: int = 32,
) -> list[dict[str, Any]]:
    if not sample_indices:
        return []
    if _supports_batched_patch_mode(patch_mode):
        return ap.run_generation_condition_batch_samples(
            model,
            tokenizer,
            condition_name=condition_name,
            target_text=target_text,
            target_prefix_boundary_text=target_prefix_boundary_text,
            patch_label=patch_label,
            patch_mode=patch_mode,
            layer_indices=layer_indices,
            donor_source=donor_source,
            required_rank=required_rank,
            max_model_length=max_model_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            sample_indices=sample_indices,
            seed_start=seed_start,
            patch_scope=patch_scope,
            early_stop_on_valid_json=early_stop_on_valid_json,
            early_stop_check_interval=early_stop_check_interval,
            early_stop_min_new_tokens=early_stop_min_new_tokens,
        )

    rows: list[dict[str, Any]] = []
    for sample_idx in sample_indices:
        seed = int(seed_start) + int(sample_idx)
        row = ap.run_generation_condition(
            model,
            tokenizer,
            condition_name=condition_name,
            target_text=target_text,
            target_prefix_boundary_text=target_prefix_boundary_text,
            patch_label=patch_label,
            patch_mode=patch_mode,
            layer_indices=layer_indices,
            donor_source=donor_source,
            required_rank=required_rank,
            max_model_length=max_model_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            patch_scope=patch_scope,
            early_stop_on_valid_json=early_stop_on_valid_json,
            early_stop_check_interval=early_stop_check_interval,
            early_stop_min_new_tokens=early_stop_min_new_tokens,
        )
        row["sample_idx"] = int(sample_idx)
        rows.append(row)
    return rows


def _write_debug_delta_summaries(
    output_root: Path,
    stats: dict[tuple[str, str], dict[str, Any]],
) -> None:
    pair_rows = ap._pair_condition_summary_rows(stats)
    pair_df = pd.DataFrame(pair_rows)
    pair_delta_path = output_root / "pair_condition_delta_live.csv"
    pooled_delta_path = output_root / "condition_delta_live.csv"
    if pair_df.empty:
        pd.DataFrame().to_csv(pair_delta_path, index=False)
        pd.DataFrame().to_csv(pooled_delta_path, index=False)
        ap.write_jsonl(output_root / "pair_condition_delta_live.jsonl", [])
        ap.write_jsonl(output_root / "condition_delta_live.jsonl", [])
        return

    baseline_df = pair_df[pair_df["condition_name"].isin(["baseline_deceptive", "baseline_truthful"])].copy()
    baseline_df = baseline_df[
        [
            "pair_id",
            "condition_name",
            "deception_rate",
            "n_valid",
            "n_samples",
            "mean_new_tokens",
            "json_stop_rate",
        ]
    ].rename(
        columns={
            "condition_name": "baseline_condition_name",
            "deception_rate": "baseline_deception_rate",
            "n_valid": "baseline_n_valid",
            "n_samples": "baseline_n_samples",
            "mean_new_tokens": "baseline_mean_new_tokens",
            "json_stop_rate": "baseline_json_stop_rate",
        }
    )

    delta_df = pair_df[pair_df["experiment"].isin(["denoising", "noising"])].copy()
    if delta_df.empty:
        pd.DataFrame().to_csv(pair_delta_path, index=False)
        pd.DataFrame().to_csv(pooled_delta_path, index=False)
        ap.write_jsonl(output_root / "pair_condition_delta_live.jsonl", [])
        ap.write_jsonl(output_root / "condition_delta_live.jsonl", [])
        return

    delta_df["baseline_condition_name"] = np.where(
        delta_df["experiment"].eq("denoising"),
        "baseline_deceptive",
        "baseline_truthful",
    )
    delta_df = delta_df.merge(
        baseline_df,
        on=["pair_id", "baseline_condition_name"],
        how="left",
    )
    delta_df["condition_valid_rate"] = np.where(
        delta_df["n_samples"] > 0,
        delta_df["n_valid"] / delta_df["n_samples"],
        np.nan,
    )
    delta_df["baseline_valid_rate"] = np.where(
        delta_df["baseline_n_samples"] > 0,
        delta_df["baseline_n_valid"] / delta_df["baseline_n_samples"],
        np.nan,
    )
    delta_df["deception_rate_delta"] = delta_df["deception_rate"] - delta_df["baseline_deception_rate"]
    delta_df["valid_rate_delta"] = delta_df["condition_valid_rate"] - delta_df["baseline_valid_rate"]
    delta_df["mean_new_tokens_delta"] = delta_df["mean_new_tokens"] - delta_df["baseline_mean_new_tokens"]
    delta_df["json_stop_rate_delta"] = delta_df["json_stop_rate"] - delta_df["baseline_json_stop_rate"]
    delta_df["goal_direction"] = np.where(
        delta_df["experiment"].eq("denoising"),
        -1.0,
        1.0,
    )
    delta_df["goal_improvement"] = delta_df["deception_rate_delta"] * delta_df["goal_direction"]
    delta_df = delta_df.sort_values(
        ["experiment", "goal_improvement", "pair_index", "condition_name"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    delta_df.to_csv(pair_delta_path, index=False)
    ap.write_jsonl(output_root / "pair_condition_delta_live.jsonl", delta_df.to_dict(orient="records"))

    group_cols = [
        "condition_name",
        "patch_label",
        "experiment",
        "target_prefix_role",
        "donor_prefix_role",
        "patch_mode",
        "patch_scope",
        "layer_idx",
        "layer_indices",
        "baseline_condition_name",
    ]
    pooled_df = (
        delta_df.groupby(group_cols, dropna=False, sort=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_condition_deception_rate=("deception_rate", "mean"),
            mean_baseline_deception_rate=("baseline_deception_rate", "mean"),
            mean_deception_rate_delta=("deception_rate_delta", "mean"),
            median_deception_rate_delta=("deception_rate_delta", "median"),
            std_deception_rate_delta=("deception_rate_delta", lambda s: float(s.std(ddof=1)) if len(s) > 1 else float("nan")),
            mean_goal_improvement=("goal_improvement", "mean"),
            median_goal_improvement=("goal_improvement", "median"),
            mean_valid_rate_delta=("valid_rate_delta", "mean"),
            mean_new_tokens_delta=("mean_new_tokens_delta", "mean"),
            mean_json_stop_rate_delta=("json_stop_rate_delta", "mean"),
        )
        .reset_index()
    )
    pooled_df = pooled_df.sort_values(
        ["experiment", "mean_goal_improvement", "condition_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    pooled_df.to_csv(pooled_delta_path, index=False)
    ap.write_jsonl(output_root / "condition_delta_live.jsonl", pooled_df.to_dict(orient="records"))


def run_debug_patch_experiment(
    *,
    pairs_df: pd.DataFrame,
    output_root: Path,
    model_name_or_path: str = ap.DEFAULT_MODEL_NAME,
    max_model_length: int = 10000,
    max_new_tokens: int = DEFAULT_DEBUG_MAX_NEW_TOKENS,
    samples_per_condition: int = DEFAULT_DEBUG_SAMPLE_COUNT,
    batch_size: int = DEFAULT_DEBUG_BATCH_SIZE,
    temperature: float = 0.8,
    top_p: float = 0.95,
    base_seed: int = 17,
    cuda_device_name: str = "cuda:0",
    layer_candidates: list[int] | None = None,
    layer_count: int | None = DEFAULT_DEBUG_LAYER_COUNT,
    patch_modes: Iterable[str] = DEFAULT_DEBUG_PATCH_MODES,
    include_baselines: bool = True,
    early_stop_on_valid_json: bool = True,
    early_stop_check_interval: int = 16,
    early_stop_min_new_tokens: int = 32,
    resume: bool = True,
    disable_tqdm: bool = False,
) -> Path:
    if pairs_df.empty:
        raise ValueError("pairs_df is empty.")
    if int(samples_per_condition) <= 0:
        raise ValueError("samples_per_condition must be positive.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples_path = output_root / "samples.jsonl"
    completed_keys, stats = ap._load_completed_samples(samples_path) if resume else (set(), {})

    pairs_df = pairs_df.reset_index(drop=True).copy()
    if "pair_index" in pairs_df.columns:
        pairs_df = pairs_df.drop(columns=["pair_index"])
    pairs_df.insert(0, "pair_index", np.arange(len(pairs_df), dtype=int))
    pairs_df.to_csv(output_root / "matched_pairs.csv", index=False)
    ap.write_jsonl(output_root / "matched_pairs.jsonl", pairs_df.to_dict(orient="records"))

    ap.seed_everything(int(base_seed))
    cuda_device = ap.resolve_primary_cuda_device(cuda_device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(max_model_length)
    if hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs["model_max_length"] = int(max_model_length)

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": ap.single_gpu_device_map(cuda_device),
    }
    model = load_model_with_dtype_compat(
        model_name_or_path,
        common_kwargs=model_kwargs,
        dtype_value=torch.bfloat16,
    )
    model.eval()
    ap.assert_model_fully_on_cuda(model)

    model_context_limit = getattr(model.config, "max_position_embeddings", None)
    requested_total_tokens = int(max_model_length) + int(max_new_tokens)
    if model_context_limit is not None and requested_total_tokens > int(model_context_limit):
        raise ValueError(
            f"Requested max_model_length + max_new_tokens = {requested_total_tokens} exceeds "
            f"model max_position_embeddings = {int(model_context_limit)}."
        )

    layers, layer_path = ap.resolve_decoder_layers(model)
    n_layers = len(layers)
    if layer_candidates is None:
        if layer_count is not None and int(layer_count) > 0:
            layer_candidates = ap.build_evenly_spaced_layer_candidates(n_layers, int(layer_count))
        else:
            layer_candidates = ap.build_default_layer_candidates(n_layers)
    layer_candidates = sorted({int(layer_idx) for layer_idx in layer_candidates if 0 <= int(layer_idx) < int(n_layers)})
    patch_modes = parse_patch_modes(",".join(str(mode) for mode in patch_modes))
    patch_conditions = build_patch_conditions_with_modes(layer_candidates, patch_modes=patch_modes)
    all_conditions = (ap.build_baseline_conditions() if include_baselines else []) + patch_conditions
    capture_cache = any(str(condition["patch_mode"]) in {"kv", "both"} for condition in patch_conditions)

    run_config = {
        "mode": "matched_pair_last_token_patch_debug",
        "hypothesis": (
            "Residual-only prefix patching is weak for continuation generation because it changes "
            "the final prefix hidden state but leaves the cached K/V for future attention mostly "
            "unpatched. Sweep residual vs kv vs both at the last token to identify settings that "
            "actually move the continuation."
        ),
        "model_name_or_path": model_name_or_path,
        "environment": ap.DEFAULT_ENVIRONMENT,
        "model_tail": ap.DEFAULT_MODEL_TAIL,
        "n_pairs": int(len(pairs_df)),
        "max_model_length": int(max_model_length),
        "max_new_tokens": int(max_new_tokens),
        "samples_per_condition": int(samples_per_condition),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "base_seed": int(base_seed),
        "cuda_device": str(cuda_device),
        "model_context_limit": None if model_context_limit is None else int(model_context_limit),
        "requested_total_tokens": int(requested_total_tokens),
        "decoder_layer_path": layer_path,
        "n_layers": int(n_layers),
        "layer_candidates": [int(layer_idx) for layer_idx in layer_candidates],
        "layer_count": None if layer_count is None else int(layer_count),
        "patch_scope": "last_token",
        "patch_modes": [str(mode) for mode in patch_modes],
        "capture_cache": bool(capture_cache),
        "early_stop_on_valid_json": bool(early_stop_on_valid_json),
        "early_stop_check_interval": int(early_stop_check_interval),
        "early_stop_min_new_tokens": int(early_stop_min_new_tokens),
        "include_baselines": bool(include_baselines),
        "conditions": [
            {
                "condition_name": condition["condition_name"],
                "patch_label": condition["patch_label"],
                "experiment": condition["experiment"],
                "target_prefix_role": condition["target_prefix_role"],
                "donor_prefix_role": condition["donor_prefix_role"],
                "patch_mode": condition["patch_mode"],
                "layer_indices": [int(layer_idx) for layer_idx in condition["layer_indices"]],
            }
            for condition in all_conditions
        ],
        "parameter_devices": ap.parameter_device_summary(model),
        "resume": bool(resume),
    }
    ap.write_json(output_root / "run_config.json", run_config)

    token_debug_rows: list[dict[str, Any]] = []
    total_planned = len(pairs_df) * len(all_conditions) * int(samples_per_condition)
    remaining = 0
    for _, pair in pairs_df.iterrows():
        for condition in all_conditions:
            layer_indices = tuple(int(layer_idx) for layer_idx in condition["layer_indices"])
            layer_idx = layer_indices[0] if len(layer_indices) == 1 else None
            for sample_idx in range(int(samples_per_condition)):
                key = ap._planned_sample_key(
                    pair_id=str(pair["pair_id"]),
                    condition_name=str(condition["condition_name"]),
                    layer_idx=layer_idx,
                    sample_idx=sample_idx,
                )
                if key not in completed_keys:
                    remaining += 1

    print(f"Output root: {output_root}")
    print(f"Matched pairs: {len(pairs_df)}")
    print(f"Patch modes: {patch_modes}")
    print(f"Layer candidates: {layer_candidates}")
    print(
        "Workload: "
        f"{len(pairs_df)} pairs x {len(all_conditions)} conditions x {int(samples_per_condition)} samples "
        f"= {total_planned} generations ({remaining} remaining after resume)."
    )

    progress = None
    if not disable_tqdm and ap._tqdm is not None:
        progress = ap._tqdm(total=remaining, desc="Debug patch generations", leave=True)

    try:
        for pair_index, pair in pairs_df.iterrows():
            pair_dict = pair.to_dict()
            pair_texts = ap.resolve_pair_text_bundle(pair_dict)
            required_rank = int(pair_dict["required_rank"])

            token_debug_rows.extend(
                [
                    {
                        "pair_index": int(pair_index),
                        "pair_id": str(pair_dict["pair_id"]),
                        **ap.describe_text_for_model(
                            tokenizer,
                            "deceptive_prefix",
                            pair_texts["deceptive_model_input"],
                            max_model_length=int(max_model_length),
                        ),
                    },
                    {
                        "pair_index": int(pair_index),
                        "pair_id": str(pair_dict["pair_id"]),
                        **ap.describe_text_for_model(
                            tokenizer,
                            "truthful_prefix",
                            pair_texts["truthful_model_input"],
                            max_model_length=int(max_model_length),
                        ),
                    },
                ]
            )
            pd.DataFrame(token_debug_rows).to_csv(output_root / "token_debug_live.csv", index=False)

            truthful_source = ap.prepare_sentence_patch_source(
                model,
                tokenizer,
                donor_full_text=pair_texts["truthful_model_input"],
                donor_prefix_boundary_text=pair_texts["truthful_boundary_text"],
                max_model_length=int(max_model_length),
                patch_scope="last_token",
                capture_cache=bool(capture_cache),
            )
            deceptive_source = ap.prepare_sentence_patch_source(
                model,
                tokenizer,
                donor_full_text=pair_texts["deceptive_model_input"],
                donor_prefix_boundary_text=pair_texts["deceptive_boundary_text"],
                max_model_length=int(max_model_length),
                patch_scope="last_token",
                capture_cache=bool(capture_cache),
            )

            for condition_index, condition in enumerate(all_conditions):
                layer_indices = tuple(int(layer_idx) for layer_idx in condition["layer_indices"])
                layer_idx = layer_indices[0] if len(layer_indices) == 1 else None
                target_text, target_boundary_text, donor_source = ap._condition_target_and_donor(
                    condition,
                    pair_texts=pair_texts,
                    deceptive_source=deceptive_source,
                    truthful_source=truthful_source,
                )
                pending_sample_indices: list[int] = []
                for sample_idx in range(int(samples_per_condition)):
                    planned_key = ap._planned_sample_key(
                        pair_id=str(pair_dict["pair_id"]),
                        condition_name=str(condition["condition_name"]),
                        layer_idx=layer_idx,
                        sample_idx=sample_idx,
                    )
                    if planned_key not in completed_keys:
                        pending_sample_indices.append(int(sample_idx))

                seed_start = ap._sample_seed(int(base_seed), int(pair_index), int(condition_index), 0)
                for sample_chunk in ap.iter_chunks(pending_sample_indices, int(batch_size)):
                    if progress is not None:
                        progress.set_postfix_str(
                            f"pair={int(pair_index)} condition={condition['condition_name']} batch={len(sample_chunk)}"
                        )
                    batch_rows = run_generation_condition_samples_compat(
                        model,
                        tokenizer,
                        condition_name=str(condition["condition_name"]),
                        target_text=target_text,
                        target_prefix_boundary_text=target_boundary_text,
                        patch_label=str(condition["patch_label"]),
                        patch_mode=str(condition["patch_mode"]),
                        layer_indices=layer_indices,
                        donor_source=donor_source,
                        required_rank=required_rank,
                        max_model_length=int(max_model_length),
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        sample_indices=sample_chunk,
                        seed_start=seed_start,
                        patch_scope="last_token",
                        early_stop_on_valid_json=bool(early_stop_on_valid_json),
                        early_stop_check_interval=int(early_stop_check_interval),
                        early_stop_min_new_tokens=int(early_stop_min_new_tokens),
                    )
                    for row in batch_rows:
                        sample_idx = int(row["sample_idx"])
                        planned_key = ap._planned_sample_key(
                            pair_id=str(pair_dict["pair_id"]),
                            condition_name=str(condition["condition_name"]),
                            layer_idx=layer_idx,
                            sample_idx=sample_idx,
                        )
                        row.pop("target_text", None)
                        row.pop("target_prefix_boundary_text", None)
                        row.update(
                            {
                                "pair_index": int(pair_index),
                                "pair_id": str(pair_dict["pair_id"]),
                                "example_id": str(pair_dict["example_id"]),
                                "required_rank": required_rank,
                                "experiment": str(condition["experiment"]),
                                "target_prefix_role": str(condition["target_prefix_role"]),
                                "donor_prefix_role": condition.get("donor_prefix_role"),
                                "shared_context_deception_rate": float(pair_dict["shared_context_deception_rate"]),
                                "deceptive_prefix_deception_rate": float(pair_dict["deceptive_prefix_deception_rate"]),
                                "commitment_delta": float(pair_dict["commitment_delta"]),
                                "donor_generation_idx": int(pair_dict["donor_generation_idx"]),
                                "donor_clarity_score": float(pair_dict["donor_clarity_score"]),
                            }
                        )
                        ap.append_jsonl_row(samples_path, row)
                        completed_keys.add(planned_key)
                        ap._update_pair_condition_stats(stats, row)
                    if progress is not None:
                        progress.update(len(batch_rows))
                    ap._write_live_summaries(output_root, stats)
                    _write_debug_delta_summaries(output_root, stats)

                ap._write_live_summaries(output_root, stats)
                _write_debug_delta_summaries(output_root, stats)
    finally:
        if progress is not None:
            progress.close()

    ap._write_live_summaries(output_root, stats)
    _write_debug_delta_summaries(output_root, stats)
    for live_name, final_name in [
        ("pair_condition_summary_live.csv", "pair_condition_summary.csv"),
        ("condition_summary_live.csv", "condition_summary.csv"),
        ("pair_condition_delta_live.csv", "pair_condition_delta.csv"),
        ("condition_delta_live.csv", "condition_delta.csv"),
    ]:
        live_path = output_root / live_name
        if live_path.exists():
            shutil.copy2(live_path, output_root / final_name)
    print(f"Saved activation patching debug artifacts to {output_root}")
    return output_root


def _normalize_commitment_text(text: Any) -> str:
    return ap.normalize_sentence_for_compare(str(text or ""))


def _is_usable_commitment_text(text: Any) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    if TAG_ONLY_RE.fullmatch(clean):
        return False
    return True


def load_commitment_pairs(
    *,
    localization_dir: Path,
    pair_cache_path: Path,
    pair_count: int,
    pair_search_limit: int | None,
    refresh_cache: bool,
    min_commitment_delta: float,
    min_commitment_deception_rate: float,
    min_donor_clarity_score: float,
    disable_tqdm: bool,
) -> pd.DataFrame:
    pair_cache_path = Path(pair_cache_path).expanduser().resolve()
    localization_dir = Path(localization_dir).expanduser().resolve()

    requested_cache_count = max(int(pair_count), DEFAULT_COMMITMENT_PAIR_COUNT)
    search_limit = (
        int(pair_search_limit)
        if pair_search_limit is not None and int(pair_search_limit) > 0
        else max(int(requested_cache_count) * 4, DEFAULT_COMMITMENT_PAIR_SEARCH_LIMIT)
    )

    need_rebuild = bool(refresh_cache) or not pair_cache_path.exists()
    source_df = pd.DataFrame()
    if not need_rebuild:
        source_df = pd.DataFrame(ap.read_jsonl_rows(pair_cache_path))
        if source_df.empty or len(source_df) < int(pair_count) or len(source_df) < int(search_limit):
            need_rebuild = True

    if need_rebuild:
        source_df = ap.search_bs_activation_patch_examples(
            localization_dir,
            limit=int(search_limit),
            min_commitment_delta=float(min_commitment_delta),
            min_commitment_deception_rate=float(min_commitment_deception_rate),
            min_donor_clarity_score=float(min_donor_clarity_score),
            disable_tqdm=bool(disable_tqdm),
        )
        if source_df.empty:
            raise ValueError(f"No matched commitment pairs found in {localization_dir}")
        ap.write_jsonl(pair_cache_path, source_df.to_dict(orient="records"))
        source_df.to_csv(pair_cache_path.with_suffix(".csv"), index=False)
    if source_df.empty:
        raise ValueError(f"No matched pairs available in {pair_cache_path}")

    usable_rows: list[dict[str, Any]] = []
    for row in source_df.to_dict(orient="records"):
        prompt = str(row.get("prompt", ""))
        shared_context_text = str(row.get("shared_context_text", ""))
        deceptive_sentence = str(row.get("deceptive_commitment_sentence", ""))
        truthful_sentence = str(row.get("truthful_donor_sentence", ""))
        if not prompt or not shared_context_text:
            continue
        if not _is_usable_commitment_text(deceptive_sentence) or not _is_usable_commitment_text(truthful_sentence):
            continue
        if _normalize_commitment_text(deceptive_sentence) == _normalize_commitment_text(truthful_sentence):
            continue
        if not split_sentence_spans(deceptive_sentence) and not deceptive_sentence.strip():
            continue
        if not split_sentence_spans(truthful_sentence) and not truthful_sentence.strip():
            continue

        shared_prefix_text = prompt + shared_context_text
        deceptive_branch_text = prompt + ap.append_continuation(shared_context_text, deceptive_sentence)
        truthful_branch_text = prompt + ap.append_continuation(shared_context_text, truthful_sentence)
        usable_rows.append(
            {
                **row,
                "shared_prefix_text": shared_prefix_text,
                "deceptive_branch_text": deceptive_branch_text,
                "truthful_branch_text": truthful_branch_text,
                "deceptive_sentence_norm": _normalize_commitment_text(deceptive_sentence),
                "truthful_sentence_norm": _normalize_commitment_text(truthful_sentence),
            }
        )

    usable_df = pd.DataFrame(usable_rows)
    if usable_df.empty:
        raise ValueError("No usable commitment pairs remained after filtering.")
    if len(usable_df) < int(pair_count):
        raise ValueError(
            f"Only {len(usable_df)} usable commitment pairs remained after filtering, "
            f"but pair_count={int(pair_count)} was requested. "
            f"Try a larger --pair-search-limit (current effective limit: {int(search_limit)})."
        )
    usable_df = usable_df.head(int(pair_count)).reset_index(drop=True)
    usable_df.insert(0, "pair_index", np.arange(len(usable_df), dtype=int))
    return usable_df


def _encode_commitment_branch(
    tokenizer: Any,
    *,
    prefix_text: str,
    full_text: str,
    device: torch.device,
    max_model_length: int,
    objective_token_count: int | None,
    objective_token_position: str,
) -> dict[str, Any]:
    prefix_inputs = ap.encode_text_for_model(tokenizer, prefix_text, max_input_tokens=max_model_length)
    full_inputs = ap.encode_text_for_model(tokenizer, full_text, max_input_tokens=max_model_length)
    prefix_ids = prefix_inputs["input_ids"][0]
    full_ids = full_inputs["input_ids"][0]
    prefix_len = int(prefix_ids.shape[0])
    total_len = int(full_ids.shape[0])
    sentence_len = total_len - prefix_len
    if sentence_len <= 0:
        raise ValueError("Commitment branch must add at least one token after the shared prefix.")
    if prefix_len > total_len or not torch.equal(full_ids[:prefix_len], prefix_ids):
        raise ValueError("Full branch tokenization does not begin with the shared prefix tokenization.")
    scored_token_count = int(sentence_len)
    if objective_token_count is not None and int(objective_token_count) > 0:
        scored_token_count = min(int(sentence_len), int(objective_token_count))
    objective_token_position = str(objective_token_position).strip().lower()
    if objective_token_position not in {"first", "last"}:
        raise ValueError(
            f"Unsupported objective_token_position={objective_token_position!r}; expected 'first' or 'last'."
        )
    if objective_token_position == "first":
        score_start_pos = int(prefix_len)
        score_stop_pos = int(prefix_len) + int(scored_token_count)
    else:
        score_stop_pos = int(total_len)
        score_start_pos = int(total_len) - int(scored_token_count)
    device_inputs = {key: value.to(device) for key, value in full_inputs.items()}
    return {
        "input_ids": device_inputs["input_ids"],
        "attention_mask": device_inputs["attention_mask"],
        "prefix_len": int(prefix_len),
        "total_len": int(total_len),
        "sentence_len": int(sentence_len),
        "score_start_pos": int(score_start_pos),
        "score_stop_pos": int(score_stop_pos),
        "scored_token_count": int(scored_token_count),
        "objective_token_position": str(objective_token_position),
    }


def _sentence_logprob_total_tensor(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start_pos: int,
    stop_pos: int,
) -> torch.Tensor:
    start_pos = int(start_pos)
    stop_pos = int(stop_pos)
    if start_pos <= 0:
        raise ValueError("Sentence scoring expects at least one prefix token before the commitment sentence.")
    if stop_pos <= start_pos:
        raise ValueError(f"Invalid scoring span: start={start_pos}, stop={stop_pos}")
    shifted_logits = logits[:, start_pos - 1 : stop_pos - 1, :].float()
    target_ids = input_ids[:, start_pos:stop_pos]
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()


def _capture_sentence_hidden_states(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    score_start_pos: int,
    score_stop_pos: int,
    prefix_len: int,
    total_len: int,
    scored_token_count: int,
    layer_indices: list[int],
) -> dict[str, Any]:
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    sentence_total_logprob = float(
        _sentence_logprob_total_tensor(
            outputs.logits,
            input_ids,
            score_start_pos,
            score_stop_pos,
        ).item()
    )
    sentence_len = int(total_len) - int(prefix_len)
    hidden_by_layer = {
        int(layer_idx): outputs.hidden_states[int(layer_idx) + 1][0, int(prefix_len) : int(total_len), :].detach()
        for layer_idx in layer_indices
    }
    return {
        "sentence_total_logprob": sentence_total_logprob,
        "sentence_avg_logprob": sentence_total_logprob / float(int(scored_token_count)),
        "sentence_hidden_by_layer": hidden_by_layer,
        "scored_token_count": int(scored_token_count),
        "sentence_len": int(sentence_len),
    }


def _build_token_bin_assignments(sentence_len: int, token_bin_count: int) -> np.ndarray:
    sentence_len = int(sentence_len)
    token_bin_count = int(token_bin_count)
    if sentence_len <= 0:
        raise ValueError("sentence_len must be positive.")
    if token_bin_count <= 0:
        raise ValueError("token_bin_count must be positive.")
    assignments = np.floor(np.arange(sentence_len, dtype=float) * float(token_bin_count) / float(sentence_len)).astype(int)
    return np.clip(assignments, 0, token_bin_count - 1)


def _map_target_offsets_to_donor_offsets(target_len: int, donor_len: int) -> np.ndarray:
    target_len = int(target_len)
    donor_len = int(donor_len)
    if target_len <= 0 or donor_len <= 0:
        raise ValueError("Both target_len and donor_len must be positive.")
    if target_len == 1 or donor_len == 1:
        return np.zeros(target_len, dtype=int)
    scale = float(donor_len - 1) / float(target_len - 1)
    mapped = np.rint(np.arange(target_len, dtype=float) * scale).astype(int)
    return np.clip(mapped, 0, donor_len - 1)


def _build_site_layout(
    *,
    donor_hidden_by_layer: dict[int, torch.Tensor],
    layer_indices: list[int],
    prefix_len: int,
    target_len: int,
    donor_len: int,
    token_bin_count: int,
) -> dict[int, dict[int, dict[str, Any]]]:
    target_bin_assignments = _build_token_bin_assignments(int(target_len), int(token_bin_count))
    donor_offset_map = _map_target_offsets_to_donor_offsets(int(target_len), int(donor_len))
    layout: dict[int, dict[int, dict[str, Any]]] = {}

    for layer_idx in layer_indices:
        donor_hidden = donor_hidden_by_layer[int(layer_idx)]
        layer_layout: dict[int, dict[str, Any]] = {}
        for token_bin in range(int(token_bin_count)):
            rel_idx_np = np.nonzero(target_bin_assignments == int(token_bin))[0]
            rel_idx_tensor = torch.tensor(rel_idx_np, dtype=torch.long, device=donor_hidden.device)
            donor_rel_tensor = torch.tensor(donor_offset_map[rel_idx_np], dtype=torch.long, device=donor_hidden.device)
            layer_layout[int(token_bin)] = {
                "rel_positions": rel_idx_tensor,
                "abs_positions": rel_idx_tensor + int(prefix_len),
                "donor_hidden": donor_hidden.index_select(0, donor_rel_tensor).detach(),
                "token_count": int(rel_idx_tensor.numel()),
            }
        layout[int(layer_idx)] = layer_layout
    return layout


def _score_target_with_site_masks(
    model: Any,
    layers: Any,
    *,
    branch_inputs: dict[str, Any],
    layer_indices: list[int],
    site_layout: dict[int, dict[int, dict[str, Any]]],
    mask_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    handles = []

    for local_layer_pos, layer_idx in enumerate(layer_indices):
        layer_layout = site_layout[int(layer_idx)]

        def hook(
            module: Any,
            inputs: Any,
            output: Any,
            *,
            local_layer_pos: int = int(local_layer_pos),
            layer_layout: dict[int, dict[str, Any]] = layer_layout,
        ) -> Any:
            hidden = ap.hidden_from_output(output)
            patched = hidden.clone()
            for token_bin, info in layer_layout.items():
                abs_positions = info["abs_positions"]
                if int(abs_positions.numel()) == 0:
                    continue
                donor_hidden = info["donor_hidden"].to(device=hidden.device, dtype=hidden.dtype)
                scale = mask_tensor[int(local_layer_pos), int(token_bin)].to(device=hidden.device, dtype=hidden.dtype)
                current_hidden = patched[:, abs_positions, :]
                patched[:, abs_positions, :] = current_hidden + scale * (donor_hidden.unsqueeze(0) - current_hidden)
            return ap.replace_hidden_in_output(output, patched)

        handles.append(layers[int(layer_idx)].register_forward_hook(hook))

    try:
        outputs = model(
            input_ids=branch_inputs["input_ids"],
            attention_mask=branch_inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        total_logprob = _sentence_logprob_total_tensor(
            outputs.logits,
            branch_inputs["input_ids"],
            branch_inputs["score_start_pos"],
            branch_inputs["score_stop_pos"],
        )
        avg_logprob = total_logprob / float(branch_inputs["scored_token_count"])
        return total_logprob, avg_logprob
    finally:
        for handle in handles:
            handle.remove()


def _make_mask_tensor(
    *,
    layer_indices: list[int],
    token_bin_count: int,
    device: torch.device,
    active_sites: Iterable[tuple[int, int]] | None = None,
    fill_value: float = 0.0,
) -> torch.Tensor:
    mask = torch.full(
        (len(layer_indices), int(token_bin_count)),
        float(fill_value),
        dtype=torch.float32,
        device=device,
    )
    if active_sites is None:
        return mask
    layer_to_local_idx = {int(layer_idx): local_idx for local_idx, layer_idx in enumerate(layer_indices)}
    for raw_layer_idx, raw_token_bin in active_sites:
        layer_idx = int(raw_layer_idx)
        token_bin = int(raw_token_bin)
        local_idx = layer_to_local_idx.get(layer_idx)
        if local_idx is None or token_bin < 0 or token_bin >= int(token_bin_count):
            continue
        mask[int(local_idx), int(token_bin)] = 1.0
    return mask


def _compute_commitment_ig_attributions(
    model: Any,
    layers: Any,
    *,
    branch_inputs: dict[str, Any],
    layer_indices: list[int],
    site_layout: dict[int, dict[int, dict[str, Any]]],
    token_bin_count: int,
    ig_steps: int,
) -> dict[str, Any]:
    device = branch_inputs["input_ids"].device
    total_grad = torch.zeros((len(layer_indices), int(token_bin_count)), dtype=torch.float32, device=device)
    step_rows: list[dict[str, Any]] = []

    for step_idx, alpha in enumerate(torch.linspace(0.0, 1.0, int(ig_steps) + 1, device=device)[1:], start=1):
        model.zero_grad(set_to_none=True)
        mask = torch.full(
            (len(layer_indices), int(token_bin_count)),
            float(alpha.item()),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        total_logprob, avg_logprob = _score_target_with_site_masks(
            model,
            layers,
            branch_inputs=branch_inputs,
            layer_indices=layer_indices,
            site_layout=site_layout,
            mask_tensor=mask,
        )
        objective = -total_logprob
        objective.backward()
        if mask.grad is None:
            raise RuntimeError("Mask gradient was None during commitment IG computation.")
        total_grad += mask.grad.detach()
        step_rows.append(
            {
                "ig_step": int(step_idx),
                "alpha": float(alpha.item()),
                "patched_target_total_logprob": float(total_logprob.detach().item()),
                "patched_target_avg_logprob": float(avg_logprob.detach().item()),
                "objective_value": float(objective.detach().item()),
            }
        )

    attributions = (total_grad / float(int(ig_steps))).detach().cpu().numpy()
    return {
        "attributions": attributions,
        "step_rows": step_rows,
    }


def _prepare_commitment_pair_bundle(
    pair_row: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    layer_indices: list[int],
    token_bin_count: int,
    max_model_length: int,
    objective_token_count: int | None,
    objective_token_position: str,
) -> dict[str, Any]:
    model_device = ap.resolve_model_device(model)
    prefix_text = str(pair_row["shared_prefix_text"])

    deceptive_inputs = _encode_commitment_branch(
        tokenizer,
        prefix_text=prefix_text,
        full_text=str(pair_row["deceptive_branch_text"]),
        device=model_device,
        max_model_length=int(max_model_length),
        objective_token_count=objective_token_count,
        objective_token_position=objective_token_position,
    )
    truthful_inputs = _encode_commitment_branch(
        tokenizer,
        prefix_text=prefix_text,
        full_text=str(pair_row["truthful_branch_text"]),
        device=model_device,
        max_model_length=int(max_model_length),
        objective_token_count=objective_token_count,
        objective_token_position=objective_token_position,
    )
    if int(deceptive_inputs["prefix_len"]) != int(truthful_inputs["prefix_len"]):
        raise ValueError("Shared prefix token length differs across deceptive/truthful branches.")

    deceptive_capture = _capture_sentence_hidden_states(
        model,
        input_ids=deceptive_inputs["input_ids"],
        attention_mask=deceptive_inputs["attention_mask"],
        score_start_pos=int(deceptive_inputs["score_start_pos"]),
        score_stop_pos=int(deceptive_inputs["score_stop_pos"]),
        prefix_len=int(deceptive_inputs["prefix_len"]),
        total_len=int(deceptive_inputs["total_len"]),
        scored_token_count=int(deceptive_inputs["scored_token_count"]),
        layer_indices=layer_indices,
    )
    truthful_capture = _capture_sentence_hidden_states(
        model,
        input_ids=truthful_inputs["input_ids"],
        attention_mask=truthful_inputs["attention_mask"],
        score_start_pos=int(truthful_inputs["score_start_pos"]),
        score_stop_pos=int(truthful_inputs["score_stop_pos"]),
        prefix_len=int(truthful_inputs["prefix_len"]),
        total_len=int(truthful_inputs["total_len"]),
        scored_token_count=int(truthful_inputs["scored_token_count"]),
        layer_indices=layer_indices,
    )

    deceptive_to_truthful_layout = _build_site_layout(
        donor_hidden_by_layer=truthful_capture["sentence_hidden_by_layer"],
        layer_indices=layer_indices,
        prefix_len=int(deceptive_inputs["prefix_len"]),
        target_len=int(deceptive_inputs["sentence_len"]),
        donor_len=int(truthful_inputs["sentence_len"]),
        token_bin_count=int(token_bin_count),
    )
    truthful_to_deceptive_layout = _build_site_layout(
        donor_hidden_by_layer=deceptive_capture["sentence_hidden_by_layer"],
        layer_indices=layer_indices,
        prefix_len=int(truthful_inputs["prefix_len"]),
        target_len=int(truthful_inputs["sentence_len"]),
        donor_len=int(deceptive_inputs["sentence_len"]),
        token_bin_count=int(token_bin_count),
    )

    baseline_margin_total = float(
        deceptive_capture["sentence_total_logprob"] - truthful_capture["sentence_total_logprob"]
    )
    baseline_margin_avg = float(
        deceptive_capture["sentence_avg_logprob"] - truthful_capture["sentence_avg_logprob"]
    )

    return {
        "pair_id": str(pair_row["pair_id"]),
        "pair_index": int(pair_row["pair_index"]),
        "example_id": str(pair_row["example_id"]),
        "required_rank": pair_row.get("required_rank"),
        "shared_prefix_text": prefix_text,
        "baseline_margin_total_logprob": baseline_margin_total,
        "baseline_margin_avg_logprob": baseline_margin_avg,
        "objective_token_count": None if objective_token_count is None else int(objective_token_count),
        "objective_token_position": str(objective_token_position),
        "deceptive_branch": {
            "branch_label": "deceptive",
            "sentence_text": str(pair_row["deceptive_commitment_sentence"]),
            "branch_text": str(pair_row["deceptive_branch_text"]),
            "inputs": deceptive_inputs,
            "capture": deceptive_capture,
        },
        "truthful_branch": {
            "branch_label": "truthful",
            "sentence_text": str(pair_row["truthful_donor_sentence"]),
            "branch_text": str(pair_row["truthful_branch_text"]),
            "inputs": truthful_inputs,
            "capture": truthful_capture,
        },
        "directions": {
            "deceptive_to_truthful": {
                "direction": "deceptive_to_truthful",
                "target_role": "deceptive",
                "donor_role": "truthful",
                "target_sentence_text": str(pair_row["deceptive_commitment_sentence"]),
                "donor_sentence_text": str(pair_row["truthful_donor_sentence"]),
                "branch_inputs": deceptive_inputs,
                "baseline_target_total_logprob": float(deceptive_capture["sentence_total_logprob"]),
                "baseline_target_avg_logprob": float(deceptive_capture["sentence_avg_logprob"]),
                "baseline_other_total_logprob": float(truthful_capture["sentence_total_logprob"]),
                "baseline_other_avg_logprob": float(truthful_capture["sentence_avg_logprob"]),
                "target_scored_token_count": int(deceptive_capture["scored_token_count"]),
                "other_scored_token_count": int(truthful_capture["scored_token_count"]),
                "site_layout": deceptive_to_truthful_layout,
            },
            "truthful_to_deceptive": {
                "direction": "truthful_to_deceptive",
                "target_role": "truthful",
                "donor_role": "deceptive",
                "target_sentence_text": str(pair_row["truthful_donor_sentence"]),
                "donor_sentence_text": str(pair_row["deceptive_commitment_sentence"]),
                "branch_inputs": truthful_inputs,
                "baseline_target_total_logprob": float(truthful_capture["sentence_total_logprob"]),
                "baseline_target_avg_logprob": float(truthful_capture["sentence_avg_logprob"]),
                "baseline_other_total_logprob": float(deceptive_capture["sentence_total_logprob"]),
                "baseline_other_avg_logprob": float(deceptive_capture["sentence_avg_logprob"]),
                "target_scored_token_count": int(truthful_capture["scored_token_count"]),
                "other_scored_token_count": int(deceptive_capture["scored_token_count"]),
                "site_layout": truthful_to_deceptive_layout,
            },
        },
    }


def _append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_commitment_discovery_state(
    discovery_path: Path,
    attribution_path: Path,
) -> tuple[set[tuple[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    discovery_rows = ap.read_jsonl_rows(discovery_path) if discovery_path.exists() else []
    attribution_rows = ap.read_jsonl_rows(attribution_path) if attribution_path.exists() else []
    completed = {
        (str(row["pair_id"]), str(row["direction"]))
        for row in discovery_rows
        if row.get("pair_id") is not None and row.get("direction") is not None
    }
    return completed, discovery_rows, attribution_rows


def _summarize_site_rankings(attribution_rows: list[dict[str, Any]]) -> pd.DataFrame:
    attr_df = pd.DataFrame(attribution_rows)
    if attr_df.empty:
        return pd.DataFrame()
    ranking_df = (
        attr_df.groupby(
            ["direction", "target_role", "donor_role", "layer_idx", "token_bin"],
            dropna=False,
            sort=False,
        )
        .agg(
            n_examples=("pair_id", "nunique"),
            mean_attribution=("attribution", "mean"),
            median_attribution=("attribution", "median"),
            mean_abs_attribution=("abs_attribution", "mean"),
            mean_score_drop=("score_drop_full_patch", "mean"),
        )
        .reset_index()
    )
    ranking_df["token_bin_start_frac"] = ranking_df["token_bin"].astype(float) / float(DEFAULT_COMMITMENT_TOKEN_BIN_COUNT)
    ranking_df["token_bin_end_frac"] = (ranking_df["token_bin"].astype(float) + 1.0) / float(
        DEFAULT_COMMITMENT_TOKEN_BIN_COUNT
    )
    ranking_df = ranking_df.sort_values(
        ["direction", "mean_attribution", "mean_abs_attribution", "layer_idx", "token_bin"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    ranking_df["site_rank"] = ranking_df.groupby("direction").cumcount() + 1
    return ranking_df


def _build_curve_sizes(total_sites: int, curve_sizes_text: str) -> list[int]:
    total_sites = int(total_sites)
    if total_sites <= 0:
        return []
    requested: list[int] = []
    if curve_sizes_text.strip():
        for piece in curve_sizes_text.split(","):
            value = piece.strip()
            if not value:
                continue
            requested.append(int(value))
    else:
        requested.extend(int(value) for value in DEFAULT_COMMITMENT_CURVE_SIZES)
    requested.append(total_sites)
    requested = sorted({value for value in requested if 0 < int(value) <= total_sites})
    if not requested:
        requested = [total_sites]
    return requested


def _direction_site_ranking_map(ranking_df: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    if ranking_df.empty:
        return {}
    site_map: dict[str, list[tuple[int, int]]] = {}
    for direction, group in ranking_df.groupby("direction", sort=False):
        site_map[str(direction)] = [
            (int(row.layer_idx), int(row.token_bin))
            for row in group.itertuples(index=False)
        ]
    return site_map


def _write_commitment_discovery_live_outputs(
    output_root: Path,
    discovery_rows: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    *,
    token_bin_count: int,
) -> pd.DataFrame:
    discovery_df = pd.DataFrame(discovery_rows)
    discovery_df.to_csv(output_root / "discovery_records_live.csv", index=False)
    ranking_df = pd.DataFrame()
    if attribution_rows:
        attr_df = pd.DataFrame(attribution_rows)
        attr_df.to_csv(output_root / "site_attributions_live.csv", index=False)
        ranking_df = _summarize_site_rankings(attribution_rows)
        if not ranking_df.empty:
            ranking_df["token_bin_start_frac"] = ranking_df["token_bin"].astype(float) / float(token_bin_count)
            ranking_df["token_bin_end_frac"] = (ranking_df["token_bin"].astype(float) + 1.0) / float(token_bin_count)
        ranking_df.to_csv(output_root / "site_ranking_live.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_root / "site_attributions_live.csv", index=False)
        pd.DataFrame().to_csv(output_root / "site_ranking_live.csv", index=False)
    return ranking_df


def _mean_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return float("nan")
    return float(series.std(ddof=1))


def _choose_circuit_sizes(
    curve_rows: list[dict[str, Any]],
    *,
    faithfulness_threshold: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    curve_df = pd.DataFrame(curve_rows)
    if curve_df.empty:
        return curve_df, {}
    summary_df = (
        curve_df.groupby(["direction", "target_role", "donor_role", "circuit_size"], dropna=False, sort=False)
        .agg(
            n_examples=("pair_id", "nunique"),
            mean_score_drop=("score_drop", "mean"),
            mean_faithfulness=("faithfulness", "mean"),
            median_faithfulness=("faithfulness", "median"),
            std_faithfulness=("faithfulness", _mean_std),
        )
        .reset_index()
        .sort_values(["direction", "circuit_size"], ascending=[True, True])
        .reset_index(drop=True)
    )
    chosen_sizes: dict[str, int] = {}
    for direction, group in summary_df.groupby("direction", sort=False):
        feasible = group[group["mean_faithfulness"] >= float(faithfulness_threshold)]
        if not feasible.empty:
            chosen_sizes[str(direction)] = int(feasible.iloc[0]["circuit_size"])
        else:
            chosen_sizes[str(direction)] = int(group["circuit_size"].max())
    return summary_df, chosen_sizes


def _write_commitment_evaluation_live_outputs(
    output_root: Path,
    curve_rows: list[dict[str, Any]],
    verification_pair_rows: list[dict[str, Any]],
) -> None:
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(output_root / "faithfulness_curve_pairs_live.csv", index=False)
    verification_df = pd.DataFrame(verification_pair_rows)
    verification_df.to_csv(output_root / "verification_pair_live.csv", index=False)


def run_commitment_eap_ig_experiment(
    *,
    pairs_df: pd.DataFrame,
    output_root: Path,
    model_name_or_path: str,
    max_model_length: int,
    cuda_device_name: str,
    layer_candidates: list[int] | None,
    layer_count: int | None,
    exclude_final_layers: int,
    ig_steps: int,
    token_bin_count: int,
    objective_token_count: int | None,
    objective_token_position: str,
    faithfulness_threshold: float,
    random_edge_samples: int,
    curve_sizes_text: str,
    base_seed: int,
    resume: bool,
    disable_tqdm: bool,
) -> Path:
    if pairs_df.empty:
        raise ValueError("pairs_df is empty.")
    if int(ig_steps) <= 0:
        raise ValueError("ig_steps must be positive.")
    if int(token_bin_count) <= 0:
        raise ValueError("token_bin_count must be positive.")
    if int(random_edge_samples) <= 0:
        raise ValueError("random_edge_samples must be positive.")
    if int(exclude_final_layers) < 0:
        raise ValueError("exclude_final_layers must be non-negative.")
    objective_token_position = str(objective_token_position).strip().lower()
    if objective_token_position not in {"first", "last"}:
        raise ValueError("--objective-token-position must be 'first' or 'last'.")

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    discovery_path = output_root / "discovery_records.jsonl"
    attribution_path = output_root / "site_attributions.jsonl"
    if not resume:
        for path in [
            discovery_path,
            attribution_path,
            output_root / "ig_steps.jsonl",
            output_root / "verification_samples.jsonl",
        ]:
            if path.exists():
                path.unlink()
    discovery_completed, discovery_rows, attribution_rows = (
        _load_commitment_discovery_state(discovery_path, attribution_path) if resume else (set(), [], [])
    )

    pairs_df = pairs_df.reset_index(drop=True).copy()
    if "pair_index" in pairs_df.columns:
        pairs_df = pairs_df.drop(columns=["pair_index"])
    pairs_df.insert(0, "pair_index", np.arange(len(pairs_df), dtype=int))
    pairs_df.to_csv(output_root / "commitment_pairs.csv", index=False)
    ap.write_jsonl(output_root / "commitment_pairs.jsonl", pairs_df.to_dict(orient="records"))

    ap.seed_everything(int(base_seed))
    rng = np.random.default_rng(int(base_seed))
    cuda_device = ap.resolve_primary_cuda_device(cuda_device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(max_model_length)
    if hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs["model_max_length"] = int(max_model_length)

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": ap.single_gpu_device_map(cuda_device),
    }
    model = load_model_with_dtype_compat(
        model_name_or_path,
        common_kwargs=model_kwargs,
        dtype_value=torch.bfloat16,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ap.assert_model_fully_on_cuda(model)

    layers, layer_path = ap.resolve_decoder_layers(model)
    n_layers = len(layers)
    if layer_candidates is None:
        if layer_count is None or int(layer_count) <= 0:
            layer_candidates = list(range(n_layers))
        else:
            layer_candidates = ap.build_evenly_spaced_layer_candidates(n_layers, int(layer_count))
    layer_candidates = sorted({int(layer_idx) for layer_idx in layer_candidates if 0 <= int(layer_idx) < int(n_layers)})
    if int(exclude_final_layers) > 0:
        min_excluded_layer = int(n_layers) - int(exclude_final_layers)
        layer_candidates = [int(layer_idx) for layer_idx in layer_candidates if int(layer_idx) < int(min_excluded_layer)]
    if not layer_candidates:
        raise ValueError("No valid layer candidates were selected.")

    total_sites = len(layer_candidates) * int(token_bin_count)
    curve_sizes = _build_curve_sizes(total_sites, curve_sizes_text)
    run_config = {
        "mode": "commitment_eap_ig",
        "hypothesis": (
            "The commitment decision is maintained by a sparse set of residual-stream intervention sites "
            "inside the commitment sentence. We compare matched deceptive vs truthful commitment continuations "
            "from the same shared prefix and rank layer/bin sites by integrated gradients."
        ),
        "objective": "logp(s_k^target | y_1:k-1)",
        "objective_token_count": None if objective_token_count is None else int(objective_token_count),
        "objective_token_position": str(objective_token_position),
        "margin_reference": "logp(s_k^D | y_1:k-1) - logp(s_k^H | y_1:k-1)",
        "model_name_or_path": model_name_or_path,
        "environment": ap.DEFAULT_ENVIRONMENT,
        "model_tail": ap.DEFAULT_MODEL_TAIL,
        "n_pairs": int(len(pairs_df)),
        "max_model_length": int(max_model_length),
        "ig_steps": int(ig_steps),
        "token_bin_count": int(token_bin_count),
        "faithfulness_threshold": float(faithfulness_threshold),
        "random_edge_samples": int(random_edge_samples),
        "curve_sizes": [int(value) for value in curve_sizes],
        "cuda_device": str(cuda_device),
        "decoder_layer_path": layer_path,
        "n_layers": int(n_layers),
        "layer_candidates": [int(layer_idx) for layer_idx in layer_candidates],
        "layer_count": None if layer_count is None else int(layer_count),
        "exclude_final_layers": int(exclude_final_layers),
        "excluded_layer_indices": [layer_idx for layer_idx in range(max(0, int(n_layers) - int(exclude_final_layers)), int(n_layers))],
        "parameter_devices": ap.parameter_device_summary(model),
        "resume": bool(resume),
    }
    ap.write_json(output_root / "run_config.json", run_config)

    total_directions = int(len(pairs_df)) * 2
    remaining_directions = 0
    for row in pairs_df.to_dict(orient="records"):
        for direction_name in ("deceptive_to_truthful", "truthful_to_deceptive"):
            if (str(row["pair_id"]), str(direction_name)) not in discovery_completed:
                remaining_directions += 1

    print(f"Output root: {output_root}")
    print(f"Commitment pairs: {len(pairs_df)}")
    print(f"Layer candidates: {layer_candidates}")
    print(f"Token bins: {int(token_bin_count)}")
    print(
        "Discovery workload: "
        f"{len(pairs_df)} pairs x 2 directions = {total_directions} direction-runs "
        f"({remaining_directions} remaining after resume)."
    )

    progress = None
    if not disable_tqdm and ap._tqdm is not None:
        progress = ap._tqdm(total=remaining_directions, desc="Commitment EAP-IG discovery", leave=True)

    try:
        for pair_row in pairs_df.to_dict(orient="records"):
            pending_directions = [
                direction_name
                for direction_name in ("deceptive_to_truthful", "truthful_to_deceptive")
                if (str(pair_row["pair_id"]), str(direction_name)) not in discovery_completed
            ]
            if not pending_directions:
                continue
            pair_bundle = _prepare_commitment_pair_bundle(
                pair_row,
                model=model,
                tokenizer=tokenizer,
                layer_indices=layer_candidates,
                token_bin_count=int(token_bin_count),
                max_model_length=int(max_model_length),
                objective_token_count=objective_token_count,
                objective_token_position=objective_token_position,
            )
            for direction_name in pending_directions:
                direction_bundle = pair_bundle["directions"][str(direction_name)]
                discovery_key = (str(pair_bundle["pair_id"]), str(direction_name))
                if progress is not None:
                    progress.set_postfix_str(f"pair={pair_bundle['pair_index']} direction={direction_name}")

                ig_result = _compute_commitment_ig_attributions(
                    model,
                    layers,
                    branch_inputs=direction_bundle["branch_inputs"],
                    layer_indices=layer_candidates,
                    site_layout=direction_bundle["site_layout"],
                    token_bin_count=int(token_bin_count),
                    ig_steps=int(ig_steps),
                )
                full_mask = _make_mask_tensor(
                    layer_indices=layer_candidates,
                    token_bin_count=int(token_bin_count),
                    device=direction_bundle["branch_inputs"]["input_ids"].device,
                    fill_value=1.0,
                )
                with torch.inference_mode():
                    full_patch_total, full_patch_avg = _score_target_with_site_masks(
                        model,
                        layers,
                        branch_inputs=direction_bundle["branch_inputs"],
                        layer_indices=layer_candidates,
                        site_layout=direction_bundle["site_layout"],
                        mask_tensor=full_mask,
                    )

                baseline_target_total = float(direction_bundle["baseline_target_total_logprob"])
                baseline_target_avg = float(direction_bundle["baseline_target_avg_logprob"])
                full_patch_total_value = float(full_patch_total.item())
                full_patch_avg_value = float(full_patch_avg.item())
                score_drop_full_patch = baseline_target_total - full_patch_total_value
                baseline_margin_total = float(pair_bundle["baseline_margin_total_logprob"])
                patched_margin_total = (
                    full_patch_total_value - float(direction_bundle["baseline_other_total_logprob"])
                    if str(direction_name) == "deceptive_to_truthful"
                    else float(direction_bundle["baseline_other_total_logprob"]) - full_patch_total_value
                )
                ig_attributions = np.asarray(ig_result["attributions"], dtype=float)
                attr_sum = float(np.nansum(ig_attributions))
                completeness_gap = float(attr_sum - score_drop_full_patch)

                discovery_row = {
                    "pair_index": int(pair_bundle["pair_index"]),
                    "pair_id": str(pair_bundle["pair_id"]),
                    "example_id": str(pair_bundle["example_id"]),
                    "required_rank": pair_bundle["required_rank"],
                    "direction": str(direction_name),
                    "target_role": str(direction_bundle["target_role"]),
                    "donor_role": str(direction_bundle["donor_role"]),
                    "baseline_margin_total_logprob": baseline_margin_total,
                    "baseline_margin_avg_logprob": float(pair_bundle["baseline_margin_avg_logprob"]),
                    "objective_token_count": None if objective_token_count is None else int(objective_token_count),
                    "objective_token_position": str(objective_token_position),
                    "baseline_target_total_logprob": baseline_target_total,
                    "baseline_target_avg_logprob": baseline_target_avg,
                    "baseline_other_total_logprob": float(direction_bundle["baseline_other_total_logprob"]),
                    "baseline_other_avg_logprob": float(direction_bundle["baseline_other_avg_logprob"]),
                    "target_scored_token_count": int(direction_bundle["target_scored_token_count"]),
                    "other_scored_token_count": int(direction_bundle["other_scored_token_count"]),
                    "full_patch_target_total_logprob": full_patch_total_value,
                    "full_patch_target_avg_logprob": full_patch_avg_value,
                    "score_drop_full_patch": score_drop_full_patch,
                    "patched_margin_total_logprob": patched_margin_total,
                    "margin_shift": float(patched_margin_total - baseline_margin_total),
                    "attribution_sum": attr_sum,
                    "completeness_gap": completeness_gap,
                    "target_sentence_token_count": int(direction_bundle["branch_inputs"]["sentence_len"]),
                    "donor_sentence_token_count": int(
                        pair_bundle["truthful_branch"]["inputs"]["sentence_len"]
                        if str(direction_name) == "deceptive_to_truthful"
                        else pair_bundle["deceptive_branch"]["inputs"]["sentence_len"]
                    ),
                }
                discovery_rows.append(discovery_row)
                ap.append_jsonl_row(discovery_path, discovery_row)

                direction_attr_rows: list[dict[str, Any]] = []
                for local_layer_pos, layer_idx in enumerate(layer_candidates):
                    layer_layout = direction_bundle["site_layout"][int(layer_idx)]
                    for token_bin in range(int(token_bin_count)):
                        info = layer_layout[int(token_bin)]
                        direction_attr_rows.append(
                            {
                                "pair_index": int(pair_bundle["pair_index"]),
                                "pair_id": str(pair_bundle["pair_id"]),
                                "example_id": str(pair_bundle["example_id"]),
                                "direction": str(direction_name),
                                "target_role": str(direction_bundle["target_role"]),
                                "donor_role": str(direction_bundle["donor_role"]),
                                "layer_idx": int(layer_idx),
                                "token_bin": int(token_bin),
                                "token_bin_start_frac": float(token_bin) / float(token_bin_count),
                                "token_bin_end_frac": float(token_bin + 1) / float(token_bin_count),
                                "token_count_in_bin": int(info["token_count"]),
                                "attribution": float(ig_attributions[int(local_layer_pos), int(token_bin)]),
                                "abs_attribution": float(abs(ig_attributions[int(local_layer_pos), int(token_bin)])),
                                "score_drop_full_patch": float(score_drop_full_patch),
                            }
                        )
                attribution_rows.extend(direction_attr_rows)
                _append_jsonl_rows(attribution_path, direction_attr_rows)
                step_rows = [
                    {
                        "pair_index": int(pair_bundle["pair_index"]),
                        "pair_id": str(pair_bundle["pair_id"]),
                        "direction": str(direction_name),
                        **step_row,
                    }
                    for step_row in ig_result["step_rows"]
                ]
                _append_jsonl_rows(output_root / "ig_steps.jsonl", step_rows)

                _write_commitment_discovery_live_outputs(
                    output_root,
                    discovery_rows,
                    attribution_rows,
                    token_bin_count=int(token_bin_count),
                )
                discovery_completed.add(discovery_key)
                if progress is not None:
                    progress.update(1)
                torch.cuda.empty_cache()
    finally:
        if progress is not None:
            progress.close()

    ranking_df = _write_commitment_discovery_live_outputs(
        output_root,
        discovery_rows,
        attribution_rows,
        token_bin_count=int(token_bin_count),
    )
    ranking_df.to_csv(output_root / "site_ranking.csv", index=False)
    pd.DataFrame(discovery_rows).to_csv(output_root / "discovery_records.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(output_root / "site_attributions.csv", index=False)

    if ranking_df.empty:
        raise RuntimeError("No site attributions were computed; cannot continue to circuit evaluation.")

    direction_site_map = _direction_site_ranking_map(ranking_df)
    curve_rows: list[dict[str, Any]] = []
    print(f"Faithfulness curve sizes: {curve_sizes}")
    eval_progress = None
    if not disable_tqdm and ap._tqdm is not None:
        eval_progress = ap._tqdm(total=int(len(pairs_df)) * 2, desc="Faithfulness evaluation", leave=True)

    try:
        for pair_row in pairs_df.to_dict(orient="records"):
            pair_bundle = _prepare_commitment_pair_bundle(
                pair_row,
                model=model,
                tokenizer=tokenizer,
                layer_indices=layer_candidates,
                token_bin_count=int(token_bin_count),
                max_model_length=int(max_model_length),
                objective_token_count=objective_token_count,
                objective_token_position=objective_token_position,
            )
            for direction_name, direction_bundle in pair_bundle["directions"].items():
                ranked_sites = direction_site_map.get(str(direction_name), [])
                if not ranked_sites:
                    continue
                if eval_progress is not None:
                    eval_progress.set_postfix_str(f"pair={pair_bundle['pair_index']} direction={direction_name}")
                baseline_target_total = float(direction_bundle["baseline_target_total_logprob"])
                discovery_match = next(
                    (
                        row
                        for row in discovery_rows
                        if str(row["pair_id"]) == str(pair_bundle["pair_id"]) and str(row["direction"]) == str(direction_name)
                    ),
                    None,
                )
                full_patch_total_value = (
                    float(discovery_match["full_patch_target_total_logprob"])
                    if discovery_match is not None
                    else float("nan")
                )
                full_patch_drop = baseline_target_total - float(full_patch_total_value)
                for circuit_size in curve_sizes:
                    active_sites = ranked_sites[: int(circuit_size)]
                    mask = _make_mask_tensor(
                        layer_indices=layer_candidates,
                        token_bin_count=int(token_bin_count),
                        device=direction_bundle["branch_inputs"]["input_ids"].device,
                        active_sites=active_sites,
                    )
                    with torch.inference_mode():
                        patched_total, patched_avg = _score_target_with_site_masks(
                            model,
                            layers,
                            branch_inputs=direction_bundle["branch_inputs"],
                            layer_indices=layer_candidates,
                            site_layout=direction_bundle["site_layout"],
                            mask_tensor=mask,
                        )
                    patched_total_value = float(patched_total.item())
                    score_drop = baseline_target_total - patched_total_value
                    faithfulness = float("nan")
                    if math.isfinite(full_patch_drop) and abs(full_patch_drop) > 1e-8:
                        faithfulness = score_drop / full_patch_drop
                    curve_rows.append(
                        {
                            "pair_index": int(pair_bundle["pair_index"]),
                            "pair_id": str(pair_bundle["pair_id"]),
                            "example_id": str(pair_bundle["example_id"]),
                            "direction": str(direction_name),
                            "target_role": str(direction_bundle["target_role"]),
                            "donor_role": str(direction_bundle["donor_role"]),
                            "objective_token_count": None if objective_token_count is None else int(objective_token_count),
                            "objective_token_position": str(objective_token_position),
                            "circuit_size": int(circuit_size),
                            "patched_target_total_logprob": patched_total_value,
                            "patched_target_avg_logprob": float(patched_avg.item()),
                            "score_drop": score_drop,
                            "full_patch_score_drop": full_patch_drop,
                            "faithfulness": faithfulness,
                        }
                    )
                if eval_progress is not None:
                    eval_progress.update(1)
                torch.cuda.empty_cache()
    finally:
        if eval_progress is not None:
            eval_progress.close()

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(output_root / "faithfulness_curve_pairs.csv", index=False)
    curve_summary_df, chosen_sizes = _choose_circuit_sizes(
        curve_rows,
        faithfulness_threshold=float(faithfulness_threshold),
    )
    curve_summary_df.to_csv(output_root / "faithfulness_curve_summary.csv", index=False)

    chosen_circuits: dict[str, Any] = {}
    for direction_name, chosen_size in chosen_sizes.items():
        ranked_sites = direction_site_map.get(str(direction_name), [])
        chosen_circuits[str(direction_name)] = {
            "direction": str(direction_name),
            "chosen_circuit_size": int(chosen_size),
            "faithfulness_threshold": float(faithfulness_threshold),
            "sites": [
                {"layer_idx": int(layer_idx), "token_bin": int(token_bin)}
                for layer_idx, token_bin in ranked_sites[: int(chosen_size)]
            ],
        }
    ap.write_json(output_root / "chosen_circuits.json", chosen_circuits)

    verification_pair_rows: list[dict[str, Any]] = []
    verification_sample_rows: list[dict[str, Any]] = []
    verification_samples_path = output_root / "verification_samples.jsonl"
    if verification_samples_path.exists():
        verification_samples_path.unlink()
    verify_progress = None
    if not disable_tqdm and ap._tqdm is not None:
        verify_progress = ap._tqdm(total=int(len(pairs_df)) * 2, desc="Random-edge verification", leave=True)

    curve_score_lookup = {
        (str(row["pair_id"]), str(row["direction"]), int(row["circuit_size"])): row
        for row in curve_rows
    }

    try:
        for pair_row in pairs_df.to_dict(orient="records"):
            pair_bundle = _prepare_commitment_pair_bundle(
                pair_row,
                model=model,
                tokenizer=tokenizer,
                layer_indices=layer_candidates,
                token_bin_count=int(token_bin_count),
                max_model_length=int(max_model_length),
                objective_token_count=objective_token_count,
                objective_token_position=objective_token_position,
            )
            for direction_name, direction_bundle in pair_bundle["directions"].items():
                ranked_sites = direction_site_map.get(str(direction_name), [])
                chosen_size = int(chosen_sizes.get(str(direction_name), 0))
                if chosen_size <= 0 or not ranked_sites:
                    continue
                if verify_progress is not None:
                    verify_progress.set_postfix_str(f"pair={pair_bundle['pair_index']} direction={direction_name}")
                chosen_curve_row = curve_score_lookup.get((str(pair_bundle["pair_id"]), str(direction_name), int(chosen_size)))
                if chosen_curve_row is None:
                    continue
                baseline_target_total = float(direction_bundle["baseline_target_total_logprob"])
                chosen_score_drop = float(chosen_curve_row["score_drop"])
                universe_sites = ranked_sites.copy()

                random_score_drops: list[float] = []
                for sample_idx in range(int(random_edge_samples)):
                    sampled_indices = rng.choice(len(universe_sites), size=int(chosen_size), replace=False)
                    sampled_sites = [universe_sites[int(idx)] for idx in np.asarray(sampled_indices).tolist()]
                    mask = _make_mask_tensor(
                        layer_indices=layer_candidates,
                        token_bin_count=int(token_bin_count),
                        device=direction_bundle["branch_inputs"]["input_ids"].device,
                        active_sites=sampled_sites,
                    )
                    with torch.inference_mode():
                        random_total, random_avg = _score_target_with_site_masks(
                            model,
                            layers,
                            branch_inputs=direction_bundle["branch_inputs"],
                            layer_indices=layer_candidates,
                            site_layout=direction_bundle["site_layout"],
                            mask_tensor=mask,
                        )
                    random_total_value = float(random_total.item())
                    random_score_drop = baseline_target_total - random_total_value
                    random_score_drops.append(random_score_drop)
                    verification_sample_rows.append(
                        {
                            "pair_index": int(pair_bundle["pair_index"]),
                            "pair_id": str(pair_bundle["pair_id"]),
                            "direction": str(direction_name),
                            "sample_idx": int(sample_idx),
                            "circuit_size": int(chosen_size),
                            "random_patched_target_total_logprob": random_total_value,
                            "random_patched_target_avg_logprob": float(random_avg.item()),
                            "random_score_drop": random_score_drop,
                        }
                    )
                random_mean_drop = float(np.mean(random_score_drops)) if random_score_drops else float("nan")
                verification_pair_rows.append(
                    {
                        "pair_index": int(pair_bundle["pair_index"]),
                        "pair_id": str(pair_bundle["pair_id"]),
                        "example_id": str(pair_bundle["example_id"]),
                        "direction": str(direction_name),
                        "target_role": str(direction_bundle["target_role"]),
                        "donor_role": str(direction_bundle["donor_role"]),
                        "chosen_circuit_size": int(chosen_size),
                        "chosen_score_drop": chosen_score_drop,
                        "random_mean_score_drop": random_mean_drop,
                        "delta_vs_random_mean": float(chosen_score_drop - random_mean_drop),
                        "beat_random_rate": float(
                            np.mean([chosen_score_drop > value for value in random_score_drops]) if random_score_drops else float("nan")
                        ),
                    }
                )
                _write_commitment_evaluation_live_outputs(output_root, curve_rows, verification_pair_rows)
                _append_jsonl_rows(verification_samples_path, verification_sample_rows[-int(random_edge_samples) :])
                if verify_progress is not None:
                    verify_progress.update(1)
                torch.cuda.empty_cache()
    finally:
        if verify_progress is not None:
            verify_progress.close()

    verification_pair_df = pd.DataFrame(verification_pair_rows)
    verification_pair_df.to_csv(output_root / "verification_pair.csv", index=False)
    verification_summary_df = (
        verification_pair_df.groupby(["direction", "target_role", "donor_role", "chosen_circuit_size"], dropna=False, sort=False)
        .agg(
            n_examples=("pair_id", "nunique"),
            mean_chosen_score_drop=("chosen_score_drop", "mean"),
            mean_random_score_drop=("random_mean_score_drop", "mean"),
            mean_delta_vs_random=("delta_vs_random_mean", "mean"),
            mean_beat_random_rate=("beat_random_rate", "mean"),
        )
        .reset_index()
        .sort_values(["direction", "chosen_circuit_size"], ascending=[True, True])
        .reset_index(drop=True)
    )
    verification_summary_df.to_csv(output_root / "verification_summary.csv", index=False)
    _write_commitment_evaluation_live_outputs(output_root, curve_rows, verification_pair_rows)

    for live_name, final_name in [
        ("discovery_records_live.csv", "discovery_records.csv"),
        ("site_attributions_live.csv", "site_attributions.csv"),
        ("site_ranking_live.csv", "site_ranking.csv"),
        ("faithfulness_curve_pairs_live.csv", "faithfulness_curve_pairs.csv"),
        ("verification_pair_live.csv", "verification_pair.csv"),
    ]:
        live_path = output_root / live_name
        if live_path.exists():
            shutil.copy2(live_path, output_root / final_name)

    print(f"Saved commitment EAP-IG artifacts to {output_root}")
    return output_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Activation patching debug utilities for BS/Qwen7B. "
            "Default mode runs a commitment-decision EAP-IG discovery pass over matched "
            "deceptive/truthful commitment sentences from the same shared prefix."
        )
    )
    parser.add_argument(
        "--experiment-mode",
        type=str,
        default=DEFAULT_EXPERIMENT_MODE,
        choices=["commitment_eap_ig", "patch_debug"],
        help="Choose the commitment EAP-IG experiment or the older generation patch debug sweep.",
    )
    parser.add_argument(
        "--localization-dir",
        type=str,
        default=str(ap.DEFAULT_LOCALIZATION_DIR),
        help="Qwen-7B BS localization directory.",
    )
    parser.add_argument("--pair-cache-path", type=str, default=str(ap.DEFAULT_PAIR_CACHE_PATH))
    parser.add_argument("--refresh-pair-cache", action="store_true", default=False)
    parser.add_argument("--cache-only", action="store_true", default=False)
    parser.add_argument(
        "--pair-search-limit",
        type=int,
        default=0,
        help="Commitment mode only. When positive, mine up to this many candidate matched pairs before filtering/selecting.",
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        default=0,
        help="Mode-dependent default: 50 for commitment EAP-IG, 12 for patch_debug.",
    )
    parser.add_argument("--min-commitment-delta", type=float, default=0.0)
    parser.add_argument("--min-commitment-deception-rate", type=float, default=0.0)
    parser.add_argument("--min-donor-clarity-score", type=float, default=float("-inf"))
    parser.add_argument("--model-name-or-path", type=str, default=ap.DEFAULT_MODEL_NAME)
    parser.add_argument("--max-model-length", type=int, default=10000)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_DEBUG_MAX_NEW_TOKENS)
    parser.add_argument("--rate-sample-count", type=int, default=DEFAULT_DEBUG_SAMPLE_COUNT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_DEBUG_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--base-seed", type=int, default=17)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument(
        "--patch-modes",
        type=str,
        default=",".join(DEFAULT_DEBUG_PATCH_MODES),
        help="Comma-separated subset of residual,kv,both.",
    )
    parser.add_argument(
        "--layer-candidates",
        type=str,
        default="",
        help="Comma-separated layer list. Overrides --layer-count.",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=-1,
        help="Mode-dependent default: all layers for commitment EAP-IG, 5 evenly-spaced layers for patch_debug.",
    )
    parser.add_argument("--ig-steps", type=int, default=DEFAULT_COMMITMENT_IG_STEPS)
    parser.add_argument("--token-bin-count", type=int, default=DEFAULT_COMMITMENT_TOKEN_BIN_COUNT)
    parser.add_argument(
        "--objective-token-count",
        type=int,
        default=DEFAULT_COMMITMENT_OBJECTIVE_TOKEN_COUNT,
        help="Commitment mode only. Score only N commitment tokens; set <=0 to score the full sentence.",
    )
    parser.add_argument(
        "--objective-token-position",
        type=str,
        default=DEFAULT_COMMITMENT_OBJECTIVE_TOKEN_POSITION,
        choices=["first", "last"],
        help="Commitment mode only. Score the first or last N commitment tokens.",
    )
    parser.add_argument(
        "--exclude-final-layers",
        type=int,
        default=DEFAULT_COMMITMENT_EXCLUDE_FINAL_LAYERS,
        help="Commitment mode only. Exclude this many final decoder layers from discovery.",
    )
    parser.add_argument("--faithfulness-threshold", type=float, default=DEFAULT_COMMITMENT_FAITHFULNESS)
    parser.add_argument("--random-edge-samples", type=int, default=DEFAULT_COMMITMENT_RANDOM_EDGE_SAMPLES)
    parser.add_argument(
        "--curve-sizes",
        type=str,
        default=",".join(str(value) for value in DEFAULT_COMMITMENT_CURVE_SIZES),
        help="Comma-separated circuit sizes for the faithfulness curve. The full site count is always added.",
    )
    parser.add_argument("--no-early-stop-on-valid-json", action="store_true", default=False)
    parser.add_argument("--early-stop-check-interval", type=int, default=16)
    parser.add_argument("--early-stop-min-new-tokens", type=int, default=32)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_DEBUG_OUTPUT_ROOT))
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--no-baselines", action="store_true", default=False)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument(
        "--plot-only-run-dir",
        type=str,
        default="",
        help="Skip generation and rebuild live summaries from samples.jsonl in this run directory.",
    )
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    args = parser.parse_args(argv)

    experiment_mode = str(args.experiment_mode)
    pair_count = (
        int(args.pair_count)
        if int(args.pair_count) > 0
        else (DEFAULT_COMMITMENT_PAIR_COUNT if experiment_mode == "commitment_eap_ig" else DEFAULT_DEBUG_PAIR_COUNT)
    )
    if int(pair_count) <= 0:
        raise ValueError("--pair-count must be positive.")
    if int(args.max_model_length) <= 0:
        raise ValueError("--max-model-length must be positive.")
    if experiment_mode == "patch_debug" and int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if experiment_mode == "patch_debug" and int(args.rate_sample_count) <= 0:
        raise ValueError("--rate-sample-count must be positive.")
    if experiment_mode == "patch_debug" and int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if experiment_mode == "patch_debug" and int(args.early_stop_check_interval) <= 0:
        raise ValueError("--early-stop-check-interval must be positive.")
    if experiment_mode == "patch_debug" and int(args.early_stop_min_new_tokens) < 0:
        raise ValueError("--early-stop-min-new-tokens must be non-negative.")
    if experiment_mode == "commitment_eap_ig" and int(args.ig_steps) <= 0:
        raise ValueError("--ig-steps must be positive.")
    if experiment_mode == "commitment_eap_ig" and int(args.token_bin_count) <= 0:
        raise ValueError("--token-bin-count must be positive.")
    if experiment_mode == "commitment_eap_ig" and int(args.exclude_final_layers) < 0:
        raise ValueError("--exclude-final-layers must be non-negative.")
    if experiment_mode == "commitment_eap_ig" and int(args.random_edge_samples) <= 0:
        raise ValueError("--random-edge-samples must be positive.")
    if experiment_mode == "commitment_eap_ig" and int(args.pair_search_limit) < 0:
        raise ValueError("--pair-search-limit must be non-negative.")

    if experiment_mode == "patch_debug" and args.plot_only_run_dir.strip():
        run_dir = Path(args.plot_only_run_dir).expanduser().resolve()
        samples_path = run_dir / "samples.jsonl"
        if not samples_path.exists():
            raise FileNotFoundError(samples_path)
        _, stats = ap._load_completed_samples(samples_path)
        ap._write_live_summaries(run_dir, stats)
        _write_debug_delta_summaries(run_dir, stats)
        for live_name, final_name in [
            ("pair_condition_summary_live.csv", "pair_condition_summary.csv"),
            ("condition_summary_live.csv", "condition_summary.csv"),
            ("pair_condition_delta_live.csv", "pair_condition_delta.csv"),
            ("condition_delta_live.csv", "condition_delta.csv"),
        ]:
            live_path = run_dir / live_name
            if live_path.exists():
                shutil.copy2(live_path, run_dir / final_name)
        print(f"Rebuilt live summaries from {samples_path}")
        return

    localization_dir = Path(args.localization_dir).expanduser().resolve()
    pair_cache_path = Path(args.pair_cache_path).expanduser().resolve()
    if experiment_mode == "commitment_eap_ig":
        pairs_df = load_commitment_pairs(
            localization_dir=localization_dir,
            pair_cache_path=pair_cache_path,
            pair_count=int(pair_count),
            pair_search_limit=None if int(args.pair_search_limit) <= 0 else int(args.pair_search_limit),
            refresh_cache=bool(args.refresh_pair_cache),
            min_commitment_delta=float(args.min_commitment_delta),
            min_commitment_deception_rate=float(args.min_commitment_deception_rate),
            min_donor_clarity_score=float(args.min_donor_clarity_score),
            disable_tqdm=bool(args.disable_tqdm),
        )
        print(f"Loaded {len(pairs_df)} commitment pairs from {pair_cache_path}")
    else:
        pairs_df = ap.load_or_build_bs_activation_patch_pair_cache(
            localization_dir,
            pair_cache_path=pair_cache_path,
            pair_count=int(pair_count),
            refresh_cache=bool(args.refresh_pair_cache),
            min_commitment_delta=float(args.min_commitment_delta),
            min_commitment_deception_rate=float(args.min_commitment_deception_rate),
            min_donor_clarity_score=float(args.min_donor_clarity_score),
            disable_tqdm=bool(args.disable_tqdm),
        )
        print(f"Loaded {len(pairs_df)} matched pairs from {pair_cache_path}")
    if args.cache_only:
        print("Cache-only mode complete.")
        return

    layer_candidates = ap.parse_layer_candidates(args.layer_candidates)
    layer_count = None
    if layer_candidates is None:
        if experiment_mode == "patch_debug":
            layer_count = DEFAULT_DEBUG_LAYER_COUNT if int(args.layer_count) < 0 else int(args.layer_count)
        else:
            layer_count = None if int(args.layer_count) <= 0 else int(args.layer_count)
    if layer_candidates is not None:
        layer_tag = f"layers_{'_'.join(str(layer_idx) for layer_idx in layer_candidates)}"
    else:
        layer_tag = "layers_all" if layer_count is None or int(layer_count) <= 0 else f"layers{int(layer_count)}even"

    if experiment_mode == "commitment_eap_ig":
        objective_token_count = None if int(args.objective_token_count) <= 0 else int(args.objective_token_count)
        objective_token_position = str(args.objective_token_position).strip().lower()
        exclude_tag = f"exclast{int(args.exclude_final_layers)}"
        objective_tag = (
            "objfull"
            if objective_token_count is None
            else f"obj{objective_token_position}{int(objective_token_count)}"
        )
        run_tag = args.run_tag.strip() or (
            f"{ap.DEFAULT_ENVIRONMENT}_{ap.slugify(ap.DEFAULT_MODEL_TAIL)}_commitment_eapig_"
            f"pairs{int(pair_count)}_{layer_tag}_{exclude_tag}_{objective_tag}_"
            f"bins{int(args.token_bin_count)}_ig{int(args.ig_steps)}_"
            f"faith{int(round(float(args.faithfulness_threshold) * 100.0))}_seed{int(args.base_seed)}"
        )
        output_root = Path(args.output_root).expanduser().resolve() / run_tag
        run_commitment_eap_ig_experiment(
            pairs_df=pairs_df,
            output_root=output_root,
            model_name_or_path=str(args.model_name_or_path),
            max_model_length=int(args.max_model_length),
            cuda_device_name=str(args.cuda_device),
            layer_candidates=layer_candidates,
            layer_count=layer_count,
            exclude_final_layers=int(args.exclude_final_layers),
            ig_steps=int(args.ig_steps),
            token_bin_count=int(args.token_bin_count),
            objective_token_count=objective_token_count,
            objective_token_position=objective_token_position,
            faithfulness_threshold=float(args.faithfulness_threshold),
            random_edge_samples=int(args.random_edge_samples),
            curve_sizes_text=str(args.curve_sizes),
            base_seed=int(args.base_seed),
            resume=not bool(args.no_resume),
            disable_tqdm=bool(args.disable_tqdm),
        )
        return

    patch_modes = parse_patch_modes(args.patch_modes)
    stop_tag = "jsonstop" if not bool(args.no_early_stop_on_valid_json) else "nojsonstop"
    mode_tag = "modes_" + "_".join(patch_modes)
    run_tag = args.run_tag.strip() or (
        f"{ap.DEFAULT_ENVIRONMENT}_{ap.slugify(ap.DEFAULT_MODEL_TAIL)}_debug_matched{int(pair_count)}_"
        f"{mode_tag}_{layer_tag}_n{int(args.rate_sample_count)}_maxnew{int(args.max_new_tokens)}_"
        f"batch{int(args.batch_size)}_{stop_tag}_seed{int(args.base_seed)}"
    )
    output_root = Path(args.output_root).expanduser().resolve() / run_tag
    run_debug_patch_experiment(
        pairs_df=pairs_df,
        output_root=output_root,
        model_name_or_path=str(args.model_name_or_path),
        max_model_length=int(args.max_model_length),
        max_new_tokens=int(args.max_new_tokens),
        samples_per_condition=int(args.rate_sample_count),
        batch_size=int(args.batch_size),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        base_seed=int(args.base_seed),
        cuda_device_name=str(args.cuda_device),
        layer_candidates=layer_candidates,
        layer_count=layer_count,
        patch_modes=patch_modes,
        include_baselines=not bool(args.no_baselines),
        early_stop_on_valid_json=not bool(args.no_early_stop_on_valid_json),
        early_stop_check_interval=int(args.early_stop_check_interval),
        early_stop_min_new_tokens=int(args.early_stop_min_new_tokens),
        resume=not bool(args.no_resume),
        disable_tqdm=bool(args.disable_tqdm),
    )


if __name__ == "__main__":
    main()
