from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import activation_patching as ap


DEFAULT_DEBUG_PAIR_COUNT = 12
DEFAULT_DEBUG_SAMPLE_COUNT = 4
DEFAULT_DEBUG_BATCH_SIZE = 4
DEFAULT_DEBUG_MAX_NEW_TOKENS = 2048
DEFAULT_DEBUG_LAYER_COUNT = 5
DEFAULT_DEBUG_PATCH_MODES = ("residual", "kv", "both")
DEFAULT_DEBUG_OUTPUT_ROOT = ap.ROOT_DIR / "Results" / "activation_patching_debug"


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast activation patching debug sweep for BS/Qwen7B. "
            "Compares residual vs kv vs both when patching only the final token "
            "of the commitment sentence, and writes paired live deltas versus the right baseline."
        )
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
    parser.add_argument("--pair-count", type=int, default=DEFAULT_DEBUG_PAIR_COUNT)
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
        default=DEFAULT_DEBUG_LAYER_COUNT,
        help="Use this many evenly-spaced layers unless --layer-candidates is set.",
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

    if int(args.pair_count) <= 0:
        raise ValueError("--pair-count must be positive.")
    if int(args.max_model_length) <= 0:
        raise ValueError("--max-model-length must be positive.")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if int(args.rate_sample_count) <= 0:
        raise ValueError("--rate-sample-count must be positive.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.early_stop_check_interval) <= 0:
        raise ValueError("--early-stop-check-interval must be positive.")
    if int(args.early_stop_min_new_tokens) < 0:
        raise ValueError("--early-stop-min-new-tokens must be non-negative.")

    if args.plot_only_run_dir.strip():
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
    pairs_df = ap.load_or_build_bs_activation_patch_pair_cache(
        localization_dir,
        pair_cache_path=pair_cache_path,
        pair_count=int(args.pair_count),
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

    patch_modes = parse_patch_modes(args.patch_modes)
    layer_candidates = ap.parse_layer_candidates(args.layer_candidates)
    if layer_candidates is not None:
        layer_tag = f"layers_{'_'.join(str(layer_idx) for layer_idx in layer_candidates)}"
    else:
        layer_tag = f"layers{int(args.layer_count)}even"
    stop_tag = "jsonstop" if not bool(args.no_early_stop_on_valid_json) else "nojsonstop"
    mode_tag = "modes_" + "_".join(patch_modes)
    run_tag = args.run_tag.strip() or (
        f"{ap.DEFAULT_ENVIRONMENT}_{ap.slugify(ap.DEFAULT_MODEL_TAIL)}_debug_matched{int(args.pair_count)}_"
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
        layer_count=int(args.layer_count) if layer_candidates is None and int(args.layer_count) > 0 else None,
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
