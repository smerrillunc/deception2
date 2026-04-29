from __future__ import annotations

import argparse
import difflib
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
DEFAULT_STEERING_SAMPLE_COUNT = 50
DEFAULT_STEERING_BATCH_SIZE = 8
DEFAULT_STEERING_MAX_NEW_TOKENS = 2048
DEFAULT_STEERING_PATCH_MODES = ("both",)
DEFAULT_STEERING_LAYER_TOPKS = (1,)
DEFAULT_TRUTHFUL_STEERING_PAIR_COUNT = 150
DEFAULT_TRUTHFUL_STEERING_EVAL_PAIR_COUNT = 50
DEFAULT_TRUTHFUL_STEERING_GENERATION_EVAL_COUNT = 25
DEFAULT_TRUTHFUL_STEERING_LAYER_COUNT = 8
DEFAULT_TRUTHFUL_STEERING_ALPHA_VALUES = (0.25, 0.5, 1.0)
DEFAULT_TRUTHFUL_STEERING_VECTOR_TYPES = ("learned", "random", "shuffled")
DEFAULT_TRUTHFUL_STEERING_GENERATION_TOPK = 2
DEFAULT_TRUTHFUL_STEERING_GREEDY_MAX_NEW_TOKENS = 128
DEFAULT_POST_COMMITMENT_SUFFIX_SENTENCE_COUNT = 2
DEFAULT_POST_COMMITMENT_PERSISTENT_TOKENS = (0, 16)
DEFAULT_POST_COMMITMENT_PAIR_CACHE_PATH = (
    ap.ROOT_DIR
    / "Cache"
    / "activation_patching"
    / f"{ap.DEFAULT_ENVIRONMENT}_{ap.DEFAULT_MODEL_TAIL}_post_commitment_repair_pairs_n{DEFAULT_TRUTHFUL_STEERING_PAIR_COUNT}.jsonl"
)
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
DEFAULT_EXPERIMENT_MODE = "post_commitment_repair_steering"
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


def parse_positive_int_list(text: str, *, name: str) -> list[int]:
    values: list[int] = []
    for raw_piece in str(text).split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError(f"{name} entries must be positive integers, got {value}.")
        values.append(int(value))
    if not values:
        raise ValueError(f"{name} requires at least one positive integer.")
    return sorted({int(value) for value in values})


def parse_float_list(text: str, *, name: str) -> list[float]:
    values: list[float] = []
    for raw_piece in str(text).split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        value = float(piece)
        if not math.isfinite(value):
            raise ValueError(f"{name} entries must be finite, got {value}.")
        values.append(float(value))
    if not values:
        raise ValueError(f"{name} requires at least one value.")
    return sorted({float(value) for value in values})


def parse_nonnegative_int_list(text: str, *, name: str) -> list[int]:
    values: list[int] = []
    for raw_piece in str(text).split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 0:
            raise ValueError(f"{name} entries must be non-negative integers, got {value}.")
        values.append(int(value))
    if not values:
        raise ValueError(f"{name} requires at least one non-negative integer.")
    return sorted({int(value) for value in values})


def parse_vector_types(text: str) -> list[str]:
    allowed = {"learned", "random", "shuffled"}
    values: list[str] = []
    for raw_piece in str(text).split(","):
        piece = raw_piece.strip().lower()
        if not piece:
            continue
        if piece not in allowed:
            raise ValueError(f"Unsupported vector type {raw_piece!r}. Choose from learned, random, shuffled.")
        if piece not in values:
            values.append(piece)
    if not values:
        raise ValueError("At least one steering vector type is required.")
    return values


def _alpha_tag(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.detach().float().cpu()
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm.item()) <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _make_random_unit_vector(*, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    vector = torch.randn(int(dim), generator=generator, dtype=torch.float32)
    return _normalize_vector(vector)


def _estimate_direction_from_gradient_stack(
    gradient_stack: torch.Tensor,
    *,
    fallback_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gradient_stack = gradient_stack.detach().float().cpu()
    if gradient_stack.ndim != 2:
        raise ValueError(f"Expected a rank-2 gradient stack, got shape {tuple(gradient_stack.shape)}")
    row_norms = torch.linalg.vector_norm(gradient_stack, dim=1)
    valid_mask = torch.isfinite(row_norms) & (row_norms > 1e-12)
    valid_stack = gradient_stack[valid_mask]
    if int(valid_stack.shape[0]) == 0:
        random_unit = _make_random_unit_vector(dim=int(gradient_stack.shape[1]), seed=int(fallback_seed))
        return random_unit, {
            "direction_source": "random_fallback_all_zero",
            "n_valid_gradients": 0,
            "valid_fraction": 0.0,
            "mean_normalized_norm": 0.0,
        }

    normalized_stack = valid_stack / torch.linalg.vector_norm(valid_stack, dim=1, keepdim=True).clamp_min(1e-12)
    mean_normalized = normalized_stack.mean(dim=0)
    mean_normalized_norm = float(torch.linalg.vector_norm(mean_normalized).item())
    if math.isfinite(mean_normalized_norm) and mean_normalized_norm > 1e-12:
        return _normalize_vector(mean_normalized), {
            "direction_source": "mean_normalized_gradient",
            "n_valid_gradients": int(valid_stack.shape[0]),
            "valid_fraction": float(valid_stack.shape[0] / max(int(gradient_stack.shape[0]), 1)),
            "mean_normalized_norm": float(mean_normalized_norm),
        }

    try:
        _, _, vh = torch.linalg.svd(normalized_stack, full_matrices=False)
        dominant = vh[0]
        orientation_reference = valid_stack.mean(dim=0)
        if float(torch.dot(dominant, orientation_reference).item()) < 0.0:
            dominant = -dominant
        return _normalize_vector(dominant), {
            "direction_source": "svd_dominant_gradient",
            "n_valid_gradients": int(valid_stack.shape[0]),
            "valid_fraction": float(valid_stack.shape[0] / max(int(gradient_stack.shape[0]), 1)),
            "mean_normalized_norm": float(mean_normalized_norm),
        }
    except Exception:
        random_unit = _make_random_unit_vector(dim=int(gradient_stack.shape[1]), seed=int(fallback_seed))
        return random_unit, {
            "direction_source": "random_fallback_svd_failure",
            "n_valid_gradients": int(valid_stack.shape[0]),
            "valid_fraction": float(valid_stack.shape[0] / max(int(gradient_stack.shape[0]), 1)),
            "mean_normalized_norm": float(mean_normalized_norm),
        }


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


def resolve_commitment_pair_text_bundle(pair: dict[str, Any]) -> dict[str, str]:
    shared_prefix_text = str(pair["shared_prefix_text"])
    deceptive_branch_text = str(pair["deceptive_branch_text"])
    truthful_branch_text = str(pair["truthful_branch_text"])
    return {
        "deceptive_model_input": deceptive_branch_text,
        "truthful_model_input": truthful_branch_text,
        "deceptive_boundary_text": shared_prefix_text,
        "truthful_boundary_text": shared_prefix_text,
    }


def load_commitment_pairs_from_run_dir(
    run_dir: Path,
    *,
    pair_count: int | None = None,
) -> pd.DataFrame:
    run_dir = Path(run_dir).expanduser().resolve()
    pairs_path = run_dir / "commitment_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(pairs_path)
    pairs_df = pd.read_csv(pairs_path)
    if pairs_df.empty:
        raise ValueError(f"Saved commitment pair file is empty: {pairs_path}")
    if pair_count is not None and int(pair_count) > 0:
        if len(pairs_df) < int(pair_count):
            raise ValueError(
                f"Saved pair file only has {len(pairs_df)} rows, but pair_count={int(pair_count)} was requested."
            )
        pairs_df = pairs_df.head(int(pair_count)).reset_index(drop=True)
    return pairs_df


def load_direction_layer_rankings_from_run_dir(
    run_dir: Path,
    *,
    ranking_metric: str = "mean_attribution",
) -> dict[str, list[int]]:
    run_dir = Path(run_dir).expanduser().resolve()
    ranking_path = run_dir / "site_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)
    ranking_df = pd.read_csv(ranking_path)
    if ranking_df.empty:
        raise ValueError(f"Site ranking file is empty: {ranking_path}")
    if ranking_metric not in {"mean_attribution", "mean_abs_attribution"}:
        raise ValueError(f"Unsupported ranking_metric={ranking_metric!r}")

    direction_rankings: dict[str, list[int]] = {}
    for direction, group in ranking_df.groupby("direction", sort=False):
        layer_df = (
            group.groupby("layer_idx", as_index=False)
            .agg(score=(ranking_metric, "sum"))
            .sort_values(["score", "layer_idx"], ascending=[False, True])
            .reset_index(drop=True)
        )
        direction_rankings[str(direction)] = [int(layer_idx) for layer_idx in layer_df["layer_idx"].tolist()]
    if not direction_rankings:
        raise ValueError(f"No direction rankings found in {ranking_path}")
    return direction_rankings


def build_continuation_steering_conditions(
    *,
    direction_layer_rankings: dict[str, list[int]],
    layer_topks: list[int],
    patch_modes: Iterable[str],
) -> list[dict[str, Any]]:
    patch_modes = parse_patch_modes(",".join(str(mode) for mode in patch_modes))
    conditions: list[dict[str, Any]] = []
    direction_specs = [
        ("denoising", "deceptive", "truthful", "deceptive_to_truthful"),
        ("noising", "truthful", "deceptive", "truthful_to_deceptive"),
    ]
    mode_label_map = {
        "residual": "Residual",
        "kv": "K/V",
        "both": "Residual + K/V",
    }

    for experiment, target_role, donor_role, ranking_direction in direction_specs:
        ranked_layers = [int(layer_idx) for layer_idx in direction_layer_rankings.get(ranking_direction, [])]
        if not ranked_layers:
            raise ValueError(f"No ranked layers found for direction={ranking_direction!r}")
        for topk in layer_topks:
            selected_layers = tuple(ranked_layers[: int(topk)])
            if len(selected_layers) < int(topk):
                raise ValueError(
                    f"Requested topk={int(topk)} layers for {ranking_direction}, "
                    f"but only {len(selected_layers)} ranked layers are available."
                )
            layer_tag = "layer" if int(topk) == 1 else "layers"
            for patch_mode in patch_modes:
                mode_label = mode_label_map[str(patch_mode)]
                conditions.append(
                    {
                        "condition_name": f"{experiment}_top{int(topk)}{layer_tag}__{patch_mode}",
                        "patch_label": f"{mode_label} | {experiment.title()} | Top {int(topk)} {layer_tag}",
                        "experiment": experiment,
                        "target_prefix_role": target_role,
                        "donor_prefix_role": donor_role,
                        "patch_mode": str(patch_mode),
                        "layer_indices": selected_layers,
                        "ranking_direction": ranking_direction,
                        "topk_layers": int(topk),
                    }
                )
    return conditions


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


def _count_alpha_words(text: Any) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(text or "")))


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
    min_num_valid: int = 0,
    min_sentence_alpha_words: int = 0,
    exclude_multiline_sentences: bool = False,
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
        try:
            shared_context_num_valid = int(row.get("shared_context_num_valid"))
            deceptive_prefix_num_valid = int(row.get("deceptive_prefix_num_valid"))
        except Exception:
            shared_context_num_valid = -1
            deceptive_prefix_num_valid = -1
        if not prompt or not shared_context_text:
            continue
        if int(min_num_valid) > 0:
            if shared_context_num_valid < int(min_num_valid) or deceptive_prefix_num_valid < int(min_num_valid):
                continue
        if bool(exclude_multiline_sentences):
            if "\n" in deceptive_sentence or "\n" in truthful_sentence:
                continue
        if int(min_sentence_alpha_words) > 0:
            if (
                _count_alpha_words(deceptive_sentence) < int(min_sentence_alpha_words)
                or _count_alpha_words(truthful_sentence) < int(min_sentence_alpha_words)
            ):
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


def run_commitment_continuation_steering_experiment(
    *,
    pairs_df: pd.DataFrame,
    output_root: Path,
    direction_layer_rankings: dict[str, list[int]],
    layer_topks: list[int],
    steering_patch_modes: Iterable[str],
    model_name_or_path: str = ap.DEFAULT_MODEL_NAME,
    max_model_length: int = 10000,
    max_new_tokens: int = DEFAULT_STEERING_MAX_NEW_TOKENS,
    samples_per_condition: int = DEFAULT_STEERING_SAMPLE_COUNT,
    batch_size: int = DEFAULT_STEERING_BATCH_SIZE,
    temperature: float = 0.8,
    top_p: float = 0.95,
    base_seed: int = 17,
    cuda_device_name: str = "cuda:0",
    early_stop_on_valid_json: bool = True,
    early_stop_check_interval: int = 16,
    early_stop_min_new_tokens: int = 32,
    resume: bool = True,
    disable_tqdm: bool = False,
    steering_source_run_dir: Path | None = None,
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
    pairs_df.to_csv(output_root / "steering_pairs.csv", index=False)
    ap.write_jsonl(output_root / "steering_pairs.jsonl", pairs_df.to_dict(orient="records"))

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

    steering_conditions = build_continuation_steering_conditions(
        direction_layer_rankings=direction_layer_rankings,
        layer_topks=layer_topks,
        patch_modes=steering_patch_modes,
    )
    all_conditions = ap.build_baseline_conditions() + steering_conditions
    capture_cache = any(str(condition["patch_mode"]) in {"kv", "both"} for condition in steering_conditions)

    run_config = {
        "mode": "commitment_continuation_steering",
        "hypothesis": (
            "Patch only the final token of the commitment sentence, then sample the generated continuation "
            "to test whether the suffix deception rate changes."
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
        "patch_scope": "last_token",
        "capture_cache": bool(capture_cache),
        "early_stop_on_valid_json": bool(early_stop_on_valid_json),
        "early_stop_check_interval": int(early_stop_check_interval),
        "early_stop_min_new_tokens": int(early_stop_min_new_tokens),
        "steering_source_run_dir": None if steering_source_run_dir is None else str(Path(steering_source_run_dir).expanduser().resolve()),
        "direction_layer_rankings": {key: [int(v) for v in values] for key, values in direction_layer_rankings.items()},
        "layer_topks": [int(v) for v in layer_topks],
        "patch_modes": [str(mode) for mode in parse_patch_modes(",".join(str(mode) for mode in steering_patch_modes))],
        "conditions": [
            {
                "condition_name": condition["condition_name"],
                "patch_label": condition["patch_label"],
                "experiment": condition["experiment"],
                "target_prefix_role": condition["target_prefix_role"],
                "donor_prefix_role": condition["donor_prefix_role"],
                "patch_mode": condition["patch_mode"],
                "layer_indices": [int(layer_idx) for layer_idx in condition["layer_indices"]],
                "ranking_direction": condition.get("ranking_direction"),
                "topk_layers": condition.get("topk_layers"),
            }
            for condition in all_conditions
        ],
        "parameter_devices": ap.parameter_device_summary(model),
        "resume": bool(resume),
    }
    ap.write_json(output_root / "run_config.json", run_config)

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
    print(f"Steering pairs: {len(pairs_df)}")
    print(f"Layer topks: {layer_topks}")
    print(f"Patch modes: {parse_patch_modes(','.join(str(mode) for mode in steering_patch_modes))}")
    print(
        "Workload: "
        f"{len(pairs_df)} pairs x {len(all_conditions)} conditions x {int(samples_per_condition)} samples "
        f"= {total_planned} generations ({remaining} remaining after resume)."
    )

    progress = None
    if not disable_tqdm and ap._tqdm is not None:
        progress = ap._tqdm(total=remaining, desc="Continuation steering generations", leave=True)

    try:
        for pair_index, pair in pairs_df.iterrows():
            pair_dict = pair.to_dict()
            pair_texts = resolve_commitment_pair_text_bundle(pair_dict)
            required_rank = int(pair_dict["required_rank"])

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
                                "ranking_direction": condition.get("ranking_direction"),
                                "topk_layers": condition.get("topk_layers"),
                                "shared_context_deception_rate": float(pair_dict.get("shared_context_deception_rate", float("nan"))),
                                "deceptive_prefix_deception_rate": float(pair_dict.get("deceptive_prefix_deception_rate", float("nan"))),
                                "commitment_delta": float(pair_dict.get("commitment_delta", float("nan"))),
                                "deceptive_commitment_sentence": str(pair_dict.get("deceptive_commitment_sentence", "")),
                                "truthful_donor_sentence": str(pair_dict.get("truthful_donor_sentence", "")),
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
    print(f"Saved continuation steering artifacts to {output_root}")
    return output_root


def _truncate_to_sentence_count(text: str, sentence_count: int | None) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    if sentence_count is None or int(sentence_count) <= 0:
        return clean
    spans = split_sentence_spans(clean)
    if not spans:
        return clean
    end_idx = int(spans[min(len(spans), int(sentence_count)) - 1]["end"])
    return clean[:int(end_idx)].strip()


def _repair_phrase_bonus(text: str) -> float:
    clean = ap.normalize_sentence_for_compare(text)
    if not clean:
        return float("-inf")
    repair_phrases = [
        "to be transparent",
        "to be honest",
        "i should disclose",
        "i need to disclose",
        "actually",
        "but wait",
        "more honestly",
        "i do not have",
        "i don't have",
        "cannot honestly",
        "can't honestly",
        "shouldn't pretend",
        "should not pretend",
        "i should correct",
        "i should be honest",
    ]
    return float(sum(0.4 for phrase in repair_phrases if phrase in clean))


def _repair_suffix_score(row: pd.Series, *, required_rank: int | None) -> float:
    first_sentence = str(row.get("first_sentence", ""))
    gen_text = str(row.get("gen_text", ""))
    if not gen_text.strip():
        return float("-inf")
    score = 0.0
    if row.get("is_truthful") is True:
        score += 2.0
    clarity_fn = getattr(ap, "_bs_honest_donor_clarity_score", None)
    if callable(clarity_fn):
        clarity = float(
            clarity_fn(
                first_sentence=first_sentence,
                required_rank=required_rank,
                cards_played=row.get("cards_played"),
            )
        )
        if math.isfinite(clarity):
            score += clarity
    score += _repair_phrase_bonus(first_sentence)
    score += 0.001 * min(len(first_sentence), 240)
    if row.get("parse_error") in {None, "", "None"}:
        score += 0.1
    return float(score)


def _maintain_suffix_score(row: pd.Series) -> float:
    first_sentence = str(row.get("first_sentence", ""))
    gen_text = str(row.get("gen_text", ""))
    if not gen_text.strip():
        return float("-inf")
    score = 0.0
    if row.get("deceptive") is True:
        score += 2.0
    score += 0.001 * min(len(first_sentence), 240)
    score -= max(_repair_phrase_bonus(first_sentence), 0.0)
    if row.get("parse_error") in {None, "", "None"}:
        score += 0.1
    return float(score)


def _build_post_commitment_repair_row(
    pair_row: dict[str, Any],
    *,
    suffix_sentence_count: int,
) -> dict[str, Any] | None:
    localization_path = Path(pair_row["localization_path"]).expanduser().resolve()
    try:
        payload = ap.load_payload(localization_path)
    except Exception:
        return None
    history = payload.get("history") or []
    commitment_pos = int(pair_row["commitment_sentence_pos"])
    if commitment_pos < 0 or commitment_pos >= len(history):
        return None
    commitment_entry = history[commitment_pos]
    generations_fn = getattr(ap, "_truthful_generation_rows_from_entry", None)
    if not callable(generations_fn):
        raise AttributeError("activation_patching._truthful_generation_rows_from_entry is required.")
    generations_df = generations_fn(commitment_entry)
    if generations_df.empty:
        return None

    required_rank = pair_row.get("required_rank")
    if required_rank is not None and not pd.isna(required_rank):
        required_rank = int(required_rank)
    else:
        required_rank = None

    generations_df = generations_df.copy()
    generations_df["repair_score"] = generations_df.apply(
        lambda row: _repair_suffix_score(row, required_rank=required_rank),
        axis=1,
    )
    generations_df["maintain_score"] = generations_df.apply(_maintain_suffix_score, axis=1)

    repair_df = generations_df.loc[
        generations_df["is_truthful"].eq(True) & generations_df["gen_text"].astype(str).str.strip().ne("")
    ].copy()
    maintain_df = generations_df.loc[
        generations_df["deceptive"].eq(True) & generations_df["gen_text"].astype(str).str.strip().ne("")
    ].copy()
    if repair_df.empty or maintain_df.empty:
        return None

    repair_df = repair_df.sort_values(
        ["repair_score", "gen_idx"],
        ascending=[False, True],
    ).reset_index(drop=True)
    maintain_df = maintain_df.sort_values(
        ["maintain_score", "gen_idx"],
        ascending=[False, True],
    ).reset_index(drop=True)
    repair_row = repair_df.iloc[0]
    maintain_row = maintain_df.iloc[0]

    prompt = str(commitment_entry.get("prompt", pair_row.get("prompt", "")))
    post_commitment_prefix_text = str(commitment_entry.get("prefix_text", ""))
    repair_suffix_text = str(repair_row.get("gen_text", "")).strip()
    maintain_suffix_text = str(maintain_row.get("gen_text", "")).strip()
    repair_objective_suffix_text = _truncate_to_sentence_count(repair_suffix_text, int(suffix_sentence_count))
    maintain_objective_suffix_text = _truncate_to_sentence_count(maintain_suffix_text, int(suffix_sentence_count))
    if not repair_objective_suffix_text or not maintain_objective_suffix_text:
        return None

    out = dict(pair_row)
    out.update(
        {
            "post_commitment_prompt": prompt,
            "post_commitment_prefix_text": post_commitment_prefix_text,
            "post_commitment_model_input": prompt + post_commitment_prefix_text,
            "repair_suffix_text": repair_suffix_text,
            "maintain_suffix_text": maintain_suffix_text,
            "repair_objective_suffix_text": repair_objective_suffix_text,
            "maintain_objective_suffix_text": maintain_objective_suffix_text,
            "repair_first_sentence": str(repair_row.get("first_sentence", "")),
            "maintain_first_sentence": str(maintain_row.get("first_sentence", "")),
            "repair_generation_idx": int(repair_row.get("gen_idx", 0)),
            "maintain_generation_idx": int(maintain_row.get("gen_idx", 0)),
            "repair_full_generation_text": str(repair_row.get("full_generation_text", "")),
            "maintain_full_generation_text": str(maintain_row.get("full_generation_text", "")),
            "repair_cards_played": ap.to_json_safe(repair_row.get("cards_played")),
            "maintain_cards_played": ap.to_json_safe(maintain_row.get("cards_played")),
            "repair_action": repair_row.get("action"),
            "maintain_action": maintain_row.get("action"),
            "repair_evaluation": ap.to_json_safe(repair_row.get("evaluation")),
            "maintain_evaluation": ap.to_json_safe(maintain_row.get("evaluation")),
            "repair_score": float(repair_row.get("repair_score", float("nan"))),
            "maintain_score": float(maintain_row.get("maintain_score", float("nan"))),
            "n_repair_candidates": int(len(repair_df)),
            "n_maintain_candidates": int(len(maintain_df)),
            "suffix_sentence_count": int(suffix_sentence_count),
        }
    )
    return out


def load_or_build_post_commitment_repair_pairs(
    *,
    commitment_pairs_df: pd.DataFrame,
    pair_cache_path: Path,
    suffix_sentence_count: int,
    refresh_cache: bool,
    disable_tqdm: bool,
) -> pd.DataFrame:
    pair_cache_path = Path(pair_cache_path).expanduser().resolve()
    pair_cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached_df = pd.DataFrame(ap.read_jsonl_rows(pair_cache_path)) if pair_cache_path.exists() and not refresh_cache else pd.DataFrame()
    if not cached_df.empty:
        if "suffix_sentence_count" in cached_df.columns:
            cached_df = cached_df.loc[cached_df["suffix_sentence_count"].eq(int(suffix_sentence_count))].copy()
        else:
            cached_df = pd.DataFrame()

    cached_by_pair: dict[str, dict[str, Any]] = {}
    if not cached_df.empty:
        cached_by_pair = {
            str(row["pair_id"]): row
            for row in cached_df.to_dict(orient="records")
            if row.get("pair_id") is not None
        }

    built_rows: list[dict[str, Any]] = []
    iterator = commitment_pairs_df.to_dict(orient="records")
    iterator = ap.maybe_tqdm(
        iterator,
        desc="Building post-commitment repair pairs",
        total=len(commitment_pairs_df),
        disable=bool(disable_tqdm),
        leave=False,
    )
    for pair_row in iterator:
        pair_id = str(pair_row["pair_id"])
        if pair_id in cached_by_pair:
            built_rows.append(cached_by_pair[pair_id])
            continue
        enriched = _build_post_commitment_repair_row(
            pair_row,
            suffix_sentence_count=int(suffix_sentence_count),
        )
        if enriched is not None:
            built_rows.append(enriched)
            ap.append_jsonl_row(pair_cache_path, enriched)

    built_df = pd.DataFrame(built_rows)
    if built_df.empty:
        raise ValueError("No usable post-commitment repair pairs were found.")
    built_df = built_df.drop_duplicates(subset=["pair_id"], keep="first").reset_index(drop=True)
    built_df.to_csv(pair_cache_path.with_suffix(".csv"), index=False)
    return built_df


def _collect_branch_prefix_site_stats(
    model: Any,
    tokenizer: Any,
    *,
    prefix_text: str,
    branch_full_text: str,
    layer_indices: list[int],
    max_model_length: int,
) -> dict[str, Any]:
    device = ap.resolve_model_device(model)
    branch_inputs = _encode_commitment_branch(
        tokenizer,
        prefix_text=str(prefix_text),
        full_text=str(branch_full_text),
        device=device,
        max_model_length=int(max_model_length),
        objective_token_count=None,
        objective_token_position="last",
    )
    prefix_pos = int(branch_inputs["prefix_len"]) - 1
    if prefix_pos < 0:
        raise ValueError("prefix_text must contain at least one token.")

    layers, _ = ap.resolve_decoder_layers(model)
    captured: dict[int, dict[str, Any]] = {}
    hooks = []
    for layer_idx in layer_indices:
        layer_idx = int(layer_idx)

        def hook(_module: Any, _inputs: Any, output: Any, layer_idx: int = layer_idx) -> Any:
            hidden = ap.hidden_from_output(output)
            if int(hidden.shape[1]) != int(branch_inputs["total_len"]):
                return output
            site = hidden[:, prefix_pos : prefix_pos + 1, :].detach().clone().requires_grad_(True)
            site.retain_grad()
            patched = hidden.detach().clone()
            patched[:, prefix_pos : prefix_pos + 1, :] = site
            captured[layer_idx] = {
                "site": site,
                "hidden_value": hidden[0, prefix_pos, :].detach().clone().cpu(),
            }
            return ap.replace_hidden_in_output(output, patched)

        hooks.append(layers[layer_idx].register_forward_hook(hook))

    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
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
            site_tensors = [captured[int(layer_idx)]["site"] for layer_idx in layer_indices]
            grad_tensors = torch.autograd.grad(
                total_logprob,
                site_tensors,
                allow_unused=True,
            )
    finally:
        for handle in hooks:
            handle.remove()

    layer_stats: dict[int, dict[str, Any]] = {}
    for layer_idx, grad_tensor in zip(layer_indices, grad_tensors):
        layer_idx = int(layer_idx)
        site = captured[layer_idx]["site"]
        if grad_tensor is None:
            grad_tensor = torch.zeros_like(site)
        layer_stats[layer_idx] = {
            "gradient": grad_tensor[0, 0, :].detach().clone().cpu(),
            "hidden_value": captured[layer_idx]["hidden_value"],
        }
    model.zero_grad(set_to_none=True)
    return {
        "total_logprob": float(total_logprob.detach().item()),
        "layer_stats": layer_stats,
        "prefix_token_count": int(branch_inputs["prefix_len"]),
        "suffix_token_count": int(branch_inputs["sentence_len"]),
    }


def _build_post_commitment_vector_bank(
    *,
    model: Any,
    tokenizer: Any,
    train_pairs_df: pd.DataFrame,
    layer_indices: list[int],
    max_model_length: int,
    base_seed: int,
    disable_tqdm: bool,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    per_layer_gradients: dict[int, list[torch.Tensor]] = {int(layer_idx): [] for layer_idx in layer_indices}
    per_layer_hidden_norms: dict[int, list[float]] = {int(layer_idx): [] for layer_idx in layer_indices}
    per_layer_hidden_diff_norms: dict[int, list[float]] = {int(layer_idx): [] for layer_idx in layer_indices}
    train_rows: list[dict[str, Any]] = []

    iterator = train_pairs_df.to_dict(orient="records")
    iterator = ap.maybe_tqdm(
        iterator,
        desc="Learning repair steering vectors",
        total=len(train_pairs_df),
        disable=bool(disable_tqdm),
        leave=False,
    )
    for pair_row in iterator:
        prefix_text = str(pair_row["post_commitment_model_input"])
        repair_branch_text = str(pair_row["post_commitment_prompt"]) + ap.append_continuation(
            str(pair_row["post_commitment_prefix_text"]),
            str(pair_row["repair_objective_suffix_text"]),
        )
        maintain_branch_text = str(pair_row["post_commitment_prompt"]) + ap.append_continuation(
            str(pair_row["post_commitment_prefix_text"]),
            str(pair_row["maintain_objective_suffix_text"]),
        )
        repair_stats = _collect_branch_prefix_site_stats(
            model,
            tokenizer,
            prefix_text=prefix_text,
            branch_full_text=repair_branch_text,
            layer_indices=layer_indices,
            max_model_length=int(max_model_length),
        )
        maintain_stats = _collect_branch_prefix_site_stats(
            model,
            tokenizer,
            prefix_text=prefix_text,
            branch_full_text=maintain_branch_text,
            layer_indices=layer_indices,
            max_model_length=int(max_model_length),
        )
        baseline_margin = float(repair_stats["total_logprob"] - maintain_stats["total_logprob"])

        for layer_idx in layer_indices:
            layer_idx = int(layer_idx)
            repair_hidden = repair_stats["layer_stats"][layer_idx]["hidden_value"].float()
            maintain_hidden = maintain_stats["layer_stats"][layer_idx]["hidden_value"].float()
            hidden_diff = repair_hidden - maintain_hidden
            margin_gradient = (
                repair_stats["layer_stats"][layer_idx]["gradient"].float()
                - maintain_stats["layer_stats"][layer_idx]["gradient"].float()
            )
            per_layer_gradients[layer_idx].append(margin_gradient)
            per_layer_hidden_norms[layer_idx].append(float(repair_hidden.norm().item()))
            per_layer_hidden_diff_norms[layer_idx].append(float(hidden_diff.norm().item()))
            train_rows.append(
                {
                    "pair_id": str(pair_row["pair_id"]),
                    "example_id": str(pair_row["example_id"]),
                    "layer_idx": int(layer_idx),
                    "baseline_margin_total_logprob": baseline_margin,
                    "repair_total_logprob": float(repair_stats["total_logprob"]),
                    "maintain_total_logprob": float(maintain_stats["total_logprob"]),
                    "prefix_hidden_norm": float(repair_hidden.norm().item()),
                    "prefix_hidden_diff_norm": float(hidden_diff.norm().item()),
                    "margin_gradient_norm": float(margin_gradient.norm().item()),
                    "repair_generation_idx": int(pair_row["repair_generation_idx"]),
                    "maintain_generation_idx": int(pair_row["maintain_generation_idx"]),
                }
            )
        torch.cuda.empty_cache()

    vector_bank: dict[int, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for layer_idx in layer_indices:
        layer_idx = int(layer_idx)
        gradient_stack = torch.stack(per_layer_gradients[layer_idx], dim=0)
        mean_gradient = gradient_stack.mean(dim=0)
        learned_unit, learned_meta = _estimate_direction_from_gradient_stack(
            gradient_stack,
            fallback_seed=int(base_seed) + 400_000 + int(layer_idx),
        )
        shuffled_signs = torch.tensor(
            np.random.default_rng(int(base_seed) + 100_000 + int(layer_idx)).choice(
                [-1.0, 1.0],
                size=int(gradient_stack.shape[0]),
            ),
            dtype=torch.float32,
        ).unsqueeze(1)
        shuffled_mean = (gradient_stack * shuffled_signs).mean(dim=0)
        shuffled_unit, shuffled_meta = _estimate_direction_from_gradient_stack(
            gradient_stack * shuffled_signs,
            fallback_seed=int(base_seed) + 200_000 + int(layer_idx),
        )
        random_unit = _make_random_unit_vector(dim=int(mean_gradient.numel()), seed=int(base_seed) + 300_000 + int(layer_idx))
        reference_norm = float(np.mean(per_layer_hidden_norms[layer_idx])) if per_layer_hidden_norms[layer_idx] else 1.0
        if not math.isfinite(reference_norm) or reference_norm <= 0.0:
            reference_norm = 1.0
        vector_bank[layer_idx] = {
            "reference_norm": float(reference_norm),
            "learned": learned_unit,
            "random": random_unit,
            "shuffled": shuffled_unit,
            "mean_gradient_norm": float(mean_gradient.norm().item()),
            "mean_prefix_hidden_norm": float(np.mean(per_layer_hidden_norms[layer_idx])) if per_layer_hidden_norms[layer_idx] else float("nan"),
            "mean_prefix_hidden_diff_norm": float(np.mean(per_layer_hidden_diff_norms[layer_idx])) if per_layer_hidden_diff_norms[layer_idx] else float("nan"),
            "n_train_pairs": int(len(per_layer_gradients[layer_idx])),
            "learned_direction_source": str(learned_meta["direction_source"]),
            "shuffled_direction_source": str(shuffled_meta["direction_source"]),
            "learned_valid_fraction": float(learned_meta["valid_fraction"]),
            "shuffled_valid_fraction": float(shuffled_meta["valid_fraction"]),
            "learned_mean_normalized_norm": float(learned_meta["mean_normalized_norm"]),
            "shuffled_mean_normalized_norm": float(shuffled_meta["mean_normalized_norm"]),
        }
        summary_rows.append(
            {
                "layer_idx": int(layer_idx),
                "n_train_pairs": int(len(per_layer_gradients[layer_idx])),
                "reference_norm": float(reference_norm),
                "mean_gradient_norm": float(mean_gradient.norm().item()),
                "mean_prefix_hidden_norm": float(np.mean(per_layer_hidden_norms[layer_idx])) if per_layer_hidden_norms[layer_idx] else float("nan"),
                "mean_prefix_hidden_diff_norm": float(np.mean(per_layer_hidden_diff_norms[layer_idx])) if per_layer_hidden_diff_norms[layer_idx] else float("nan"),
                "learned_direction_source": str(learned_meta["direction_source"]),
                "shuffled_direction_source": str(shuffled_meta["direction_source"]),
                "learned_valid_fraction": float(learned_meta["valid_fraction"]),
                "shuffled_valid_fraction": float(shuffled_meta["valid_fraction"]),
                "learned_mean_normalized_norm": float(learned_meta["mean_normalized_norm"]),
                "shuffled_mean_normalized_norm": float(shuffled_meta["mean_normalized_norm"]),
            }
        )
    return vector_bank, pd.DataFrame(train_rows), pd.DataFrame(summary_rows)


def build_post_commitment_repair_conditions(
    *,
    layer_indices: list[int],
    alpha_values: list[float],
    vector_types: list[str],
    persistent_token_counts: list[int],
) -> list[dict[str, Any]]:
    vector_label_map = {
        "learned": "Learned repair",
        "random": "Random same-norm",
        "shuffled": "Shuffled labels",
    }
    conditions: list[dict[str, Any]] = [
        {
            "condition_name": "baseline",
            "patch_label": "Baseline",
            "experiment": "baseline",
            "vector_type": "baseline",
            "alpha": 0.0,
            "layer_idx": None,
            "layer_indices": (),
            "persistent_tokens": 0,
        }
    ]
    for layer_idx in layer_indices:
        for vector_type in vector_types:
            for alpha in alpha_values:
                for persistent_tokens in persistent_token_counts:
                    persist_tag = f"persist{int(persistent_tokens)}"
                    conditions.append(
                        {
                            "condition_name": (
                                f"repair_layer_{int(layer_idx)}__{vector_type}__alpha{_alpha_tag(alpha)}__{persist_tag}"
                            ),
                            "patch_label": (
                                f"{vector_label_map[vector_type]} | Layer {int(layer_idx)} | alpha {float(alpha):.3g} "
                                f"| persist {int(persistent_tokens)}"
                            ),
                            "experiment": "repair_steering",
                            "vector_type": str(vector_type),
                            "alpha": float(alpha),
                            "layer_idx": int(layer_idx),
                            "layer_indices": (int(layer_idx),),
                            "persistent_tokens": int(persistent_tokens),
                        }
                    )
    return conditions


def _condition_layer_to_vector(
    condition: dict[str, Any],
    *,
    vector_bank: dict[int, dict[str, Any]],
) -> dict[int, torch.Tensor]:
    if str(condition.get("vector_type")) == "baseline":
        return {}
    layer_idx = int(condition["layer_idx"])
    vector_type = str(condition["vector_type"])
    alpha = float(condition["alpha"])
    layer_vectors = vector_bank[int(layer_idx)]
    base_unit = layer_vectors[vector_type]
    reference_norm = float(layer_vectors["reference_norm"])
    return {int(layer_idx): base_unit * float(reference_norm) * float(alpha)}


def _compute_steered_suffix_logprob(
    model: Any,
    tokenizer: Any,
    *,
    prefix_text: str,
    branch_full_text: str,
    layer_to_vector: dict[int, torch.Tensor],
    persistent_tokens: int,
    max_model_length: int,
) -> float:
    device = ap.resolve_model_device(model)
    branch_inputs = _encode_commitment_branch(
        tokenizer,
        prefix_text=str(prefix_text),
        full_text=str(branch_full_text),
        device=device,
        max_model_length=int(max_model_length),
        objective_token_count=None,
        objective_token_position="last",
    )
    prefix_len = int(branch_inputs["prefix_len"])
    total_len = int(branch_inputs["total_len"])
    patch_positions = [int(prefix_len) - 1]
    if int(persistent_tokens) > 0:
        suffix_len = max(0, int(total_len) - int(prefix_len))
        for offset in range(min(int(persistent_tokens), int(suffix_len))):
            patch_positions.append(int(prefix_len) + int(offset))
    patch_positions = sorted({int(position) for position in patch_positions if 0 <= int(position) < int(total_len)})

    layers, _ = ap.resolve_decoder_layers(model)
    hooks = []
    for layer_idx, vector in layer_to_vector.items():
        layer_idx = int(layer_idx)

        def hook(_module: Any, _inputs: Any, output: Any, vector: torch.Tensor = vector) -> Any:
            hidden = ap.hidden_from_output(output)
            if int(hidden.shape[1]) != int(total_len):
                return output
            patched = hidden.clone()
            steer = vector.to(device=hidden.device, dtype=hidden.dtype)
            for position in patch_positions:
                patched[:, int(position) : int(position) + 1, :] = (
                    patched[:, int(position) : int(position) + 1, :] + steer.view(1, 1, -1)
                )
            return ap.replace_hidden_in_output(output, patched)

        hooks.append(layers[layer_idx].register_forward_hook(hook))

    try:
        with torch.no_grad():
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
        return float(total_logprob.item())
    finally:
        for handle in hooks:
            handle.remove()


def _write_margin_live_outputs(output_root: Path, margin_rows: list[dict[str, Any]]) -> pd.DataFrame:
    margin_df = pd.DataFrame(margin_rows)
    margin_df.to_csv(output_root / "margin_results_live.csv", index=False)
    ap.write_jsonl(output_root / "margin_results_live.jsonl", margin_rows)
    if margin_df.empty:
        pd.DataFrame().to_csv(output_root / "margin_summary_live.csv", index=False)
        return pd.DataFrame()
    summary_df = (
        margin_df.groupby(
            ["condition_name", "patch_label", "experiment", "vector_type", "layer_idx", "alpha", "persistent_tokens"],
            dropna=False,
            sort=False,
        )
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_baseline_margin=("baseline_margin_total_logprob", "mean"),
            mean_steered_margin=("steered_margin_total_logprob", "mean"),
            mean_margin_delta=("margin_delta", "mean"),
            median_margin_delta=("margin_delta", "median"),
            std_margin_delta=("margin_delta", lambda s: float(s.std(ddof=1)) if len(s) > 1 else float("nan")),
            positive_margin_delta_rate=("margin_delta", lambda s: float((pd.Series(s) > 0).mean()) if len(s) > 0 else float("nan")),
        )
        .reset_index()
        .sort_values(["mean_margin_delta", "condition_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_df.to_csv(output_root / "margin_summary_live.csv", index=False)
    return summary_df


def _sentence_similarity(left_text: str, right_text: str) -> float:
    left_norm = ap.normalize_sentence_for_compare(left_text)
    right_norm = ap.normalize_sentence_for_compare(right_text)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return float(difflib.SequenceMatcher(a=left_norm, b=right_norm).ratio())


def _write_greedy_live_outputs(output_root: Path, greedy_rows: list[dict[str, Any]]) -> pd.DataFrame:
    greedy_df = pd.DataFrame(greedy_rows)
    greedy_df.to_csv(output_root / "greedy_results_live.csv", index=False)
    ap.write_jsonl(output_root / "greedy_results_live.jsonl", greedy_rows)
    if greedy_df.empty:
        pd.DataFrame().to_csv(output_root / "greedy_summary_live.csv", index=False)
        return pd.DataFrame()
    summary_df = (
        greedy_df.groupby(
            ["condition_name", "patch_label", "experiment", "vector_type", "layer_idx", "alpha", "persistent_tokens"],
            dropna=False,
            sort=False,
        )
        .agg(
            n_pairs=("pair_id", "nunique"),
            mean_truthful_similarity=("truthful_similarity", "mean"),
            mean_maintain_similarity=("maintain_similarity", "mean"),
            mean_similarity_margin=("similarity_margin", "mean"),
            exact_repair_rate=("exact_repair_match", "mean"),
            exact_maintain_rate=("exact_maintain_match", "mean"),
            valid_rate=("is_valid", lambda s: float(pd.Series(s).eq(True).mean()) if len(s) > 0 else float("nan")),
            deceptive_rate=("deceptive", lambda s: float(pd.Series(s).eq(True).mean()) if len(s) > 0 else float("nan")),
        )
        .reset_index()
        .sort_values(["mean_similarity_margin", "condition_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_df.to_csv(output_root / "greedy_summary_live.csv", index=False)
    return summary_df


def _select_generation_conditions(
    *,
    all_conditions: list[dict[str, Any]],
    margin_summary_df: pd.DataFrame,
    generation_topk: int,
) -> list[dict[str, Any]]:
    condition_by_name = {str(condition["condition_name"]): condition for condition in all_conditions}
    selected_names: list[str] = ["baseline"]
    if margin_summary_df.empty:
        return [condition_by_name[name] for name in selected_names if name in condition_by_name]

    if int(generation_topk) <= 0:
        selected_names.extend(
            str(name)
            for name in margin_summary_df["condition_name"].tolist()
            if str(name) != "baseline"
        )
    else:
        learned_df = margin_summary_df.loc[
            margin_summary_df["vector_type"].astype(str).eq("learned")
            & margin_summary_df["condition_name"].astype(str).ne("baseline")
        ].copy()
        learned_df = learned_df.sort_values(
            ["mean_margin_delta", "layer_idx", "alpha", "persistent_tokens"],
            ascending=[False, True, True, True],
        ).head(int(generation_topk))
        for row in learned_df.itertuples(index=False):
            selected_names.append(str(row.condition_name))
            for vector_type in ("random", "shuffled"):
                match = next(
                    (
                        condition
                        for condition in all_conditions
                        if str(condition.get("vector_type")) == vector_type
                        and condition.get("layer_idx") == row.layer_idx
                        and float(condition.get("alpha", float("nan"))) == float(row.alpha)
                        and int(condition.get("persistent_tokens", 0)) == int(row.persistent_tokens)
                    ),
                    None,
                )
                if match is not None:
                    selected_names.append(str(match["condition_name"]))

    deduped_names: list[str] = []
    seen: set[str] = set()
    for name in selected_names:
        if name in condition_by_name and name not in seen:
            deduped_names.append(name)
            seen.add(name)
    return [condition_by_name[name] for name in deduped_names]


def generate_batch_with_residual_steering(
    model: Any,
    tokenizer: Any,
    *,
    prefix_text: str,
    layer_to_vector: dict[int, torch.Tensor],
    persistent_tokens: int,
    max_model_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seeds: list[int],
    early_stop_on_valid_json: bool = False,
    early_stop_required_rank: int | None = None,
    early_stop_check_interval: int = 16,
    early_stop_min_new_tokens: int = 32,
) -> list[dict[str, Any]]:
    if not seeds:
        return []
    batch_size = len(seeds)
    device = ap.resolve_model_device(model)
    encoded_single = ap.encode_text_for_model(
        tokenizer,
        prefix_text,
        device=device,
        max_input_tokens=max_model_length,
    )
    prefix_len = int(encoded_single["input_ids"].shape[1])
    prefix_pos = int(prefix_len) - 1
    layers, layer_path = ap.resolve_decoder_layers(model)
    steering_state = {"phase": "prefill", "decode_step": -1}
    hooks = []

    for layer_idx, vector in layer_to_vector.items():
        layer_idx = int(layer_idx)

        def hook(_module: Any, _inputs: Any, output: Any, vector: torch.Tensor = vector) -> Any:
            hidden = ap.hidden_from_output(output)
            patched = hidden
            apply_steer = False
            if steering_state["phase"] == "prefill":
                if int(hidden.shape[1]) == int(prefix_len):
                    patched = hidden.clone()
                    patched[:, prefix_pos : prefix_pos + 1, :] = (
                        patched[:, prefix_pos : prefix_pos + 1, :]
                        + vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                    )
                    apply_steer = True
            elif (
                steering_state["phase"] == "decode"
                and int(hidden.shape[1]) == 1
                and int(steering_state["decode_step"]) < int(persistent_tokens)
            ):
                patched = hidden.clone()
                patched[:, 0:1, :] = patched[:, 0:1, :] + vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                apply_steer = True
            return ap.replace_hidden_in_output(output, patched) if apply_steer else output

        hooks.append(layers[layer_idx].register_forward_hook(hook))

    try:
        steering_state["phase"] = "prefill"
        steering_state["decode_step"] = -1
        with torch.no_grad():
            outputs = model(**encoded_single, use_cache=True, return_dict=True)
        past_key_values = ap._repeat_past_key_values_for_batch(ap._ensure_decode_cache(outputs.past_key_values), batch_size)
        generator_device = device if device.type != "cpu" else torch.device("cpu")
        generators = [torch.Generator(device=generator_device).manual_seed(int(seed)) for seed in seeds]
        finished_token_id = ap._make_finished_decode_token(tokenizer, device=device)
        generated_token_ids_by_row: list[list[int]] = [[] for _ in range(batch_size)]
        ended_with_eos_by_row = [False for _ in range(batch_size)]
        json_stopped_by_row = [False for _ in range(batch_size)]

        def sample_next_tokens(logits: torch.Tensor) -> torch.Tensor:
            next_tokens: list[torch.Tensor] = []
            for row_idx in range(batch_size):
                if (
                    ended_with_eos_by_row[row_idx]
                    or json_stopped_by_row[row_idx]
                    or len(generated_token_ids_by_row[row_idx]) >= int(max_new_tokens)
                ):
                    token = torch.tensor([[finished_token_id]], dtype=encoded_single["input_ids"].dtype, device=device)
                else:
                    token = ap._sample_next_token(
                        logits[row_idx : row_idx + 1],
                        temperature=float(temperature),
                        top_p=float(top_p),
                        generator=generators[row_idx],
                    ).to(device=device)
                    token_id = int(token.item())
                    generated_token_ids_by_row[row_idx].append(token_id)
                    ended_with_eos_by_row[row_idx] = (
                        tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id)
                    )
                    if early_stop_on_valid_json and not ended_with_eos_by_row[row_idx]:
                        n_tokens = len(generated_token_ids_by_row[row_idx])
                        interval = max(int(early_stop_check_interval), 1)
                        if (
                            early_stop_required_rank is not None
                            and n_tokens >= int(early_stop_min_new_tokens)
                            and (interval <= 1 or n_tokens % interval == 0)
                        ):
                            text = tokenizer.decode(generated_token_ids_by_row[row_idx], skip_special_tokens=True)
                            try:
                                evaluation = ap.evaluate_bs_generation(text, required_rank=int(early_stop_required_rank))
                                json_stopped_by_row[row_idx] = bool(evaluation.get("is_valid") is True)
                            except Exception:
                                json_stopped_by_row[row_idx] = False
                next_tokens.append(token)
            return torch.cat(next_tokens, dim=0)

        next_input_ids = sample_next_tokens(outputs.logits[:, -1, :].expand(batch_size, -1))
        decode_step = 0
        while (
            not all(
                ended_with_eos_by_row[row_idx]
                or json_stopped_by_row[row_idx]
                or len(generated_token_ids_by_row[row_idx]) >= int(max_new_tokens)
                for row_idx in range(batch_size)
            )
        ):
            steering_state["phase"] = "decode"
            steering_state["decode_step"] = int(decode_step)
            with torch.no_grad():
                step_outputs = model(
                    input_ids=next_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = step_outputs.past_key_values
            next_input_ids = sample_next_tokens(step_outputs.logits[:, -1, :])
            decode_step += 1
    finally:
        for handle in hooks:
            handle.remove()

    input_ids_single = encoded_single["input_ids"][0]
    rows: list[dict[str, Any]] = []
    for row_idx, ids in enumerate(generated_token_ids_by_row):
        new_ids = torch.tensor(ids, dtype=input_ids_single.dtype, device=input_ids_single.device)
        full_ids = torch.cat([input_ids_single, new_ids], dim=0)
        n_new_tokens = len(ids)
        hit_token_cap = n_new_tokens >= int(max_new_tokens)
        likely_truncated = bool(hit_token_cap and not ended_with_eos_by_row[row_idx] and not json_stopped_by_row[row_idx])
        rows.append(
            {
                "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
                "full_text": tokenizer.decode(full_ids, skip_special_tokens=True),
                "prefix_token_count": int(prefix_len),
                "n_new_tokens": int(n_new_tokens),
                "ended_with_eos": bool(ended_with_eos_by_row[row_idx]),
                "early_stopped_on_valid_json": bool(json_stopped_by_row[row_idx]),
                "hit_token_cap": bool(hit_token_cap),
                "likely_truncated": bool(likely_truncated),
                "layer_path": layer_path,
                "persistent_tokens": int(persistent_tokens),
            }
        )
    return rows


def run_post_commitment_generation_condition_batch_samples(
    model: Any,
    tokenizer: Any,
    *,
    condition: dict[str, Any],
    prefix_text: str,
    layer_to_vector: dict[int, torch.Tensor],
    required_rank: int,
    max_model_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sample_indices: list[int],
    seed_start: int,
    early_stop_on_valid_json: bool,
    early_stop_check_interval: int,
    early_stop_min_new_tokens: int,
) -> list[dict[str, Any]]:
    seeds = [int(seed_start) + int(sample_idx) for sample_idx in sample_indices]
    generations = generate_batch_with_residual_steering(
        model,
        tokenizer,
        prefix_text=prefix_text,
        layer_to_vector=layer_to_vector,
        persistent_tokens=int(condition.get("persistent_tokens", 0)),
        max_model_length=int(max_model_length),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seeds=seeds,
        early_stop_on_valid_json=bool(early_stop_on_valid_json),
        early_stop_required_rank=int(required_rank),
        early_stop_check_interval=int(early_stop_check_interval),
        early_stop_min_new_tokens=int(early_stop_min_new_tokens),
    )
    rows: list[dict[str, Any]] = []
    for sample_idx, seed, generation in zip(sample_indices, seeds, generations):
        evaluation = ap.evaluate_bs_generation(generation["generated_text"], required_rank=int(required_rank))
        first_sentence, remainder_text = ap.extract_first_sentence(generation["generated_text"])
        rows.append(
            {
                "condition_name": str(condition["condition_name"]),
                "patch_label": str(condition["patch_label"]),
                "experiment": str(condition["experiment"]),
                "vector_type": str(condition.get("vector_type", "")),
                "alpha": float(condition.get("alpha", 0.0)),
                "persistent_tokens": int(condition.get("persistent_tokens", 0)),
                "patch_mode": "vector",
                "patch_scope": "post_commitment_prefix",
                "layer_idx": condition.get("layer_idx"),
                "layer_indices": list(condition.get("layer_indices", ())),
                "layer_count": len(condition.get("layer_indices", ())),
                "seed": int(seed),
                "sample_idx": int(sample_idx),
                "first_generated_sentence": first_sentence,
                "remainder_text": remainder_text,
                "generated_text": generation["generated_text"],
                "full_text": generation["full_text"],
                "n_new_tokens": int(generation["n_new_tokens"]),
                "ended_with_eos": bool(generation["ended_with_eos"]),
                "early_stopped_on_valid_json": bool(generation["early_stopped_on_valid_json"]),
                "hit_token_cap": bool(generation["hit_token_cap"]),
                "likely_truncated": bool(generation["likely_truncated"]),
                "is_valid": evaluation["is_valid"],
                "deceptive": evaluation["deceptive"],
                "action": evaluation["action"],
                "cards_played": evaluation["cards_played"],
                "error": evaluation["error"],
                "parsed": evaluation["parsed"],
            }
        )
    return rows


def run_post_commitment_repair_steering_experiment(
    *,
    pairs_df: pd.DataFrame,
    output_root: Path,
    model_name_or_path: str,
    max_model_length: int,
    cuda_device_name: str,
    layer_candidates: list[int] | None,
    layer_count: int | None,
    exclude_final_layers: int,
    eval_pair_count: int,
    generation_eval_count: int,
    alpha_values: list[float],
    vector_types: list[str],
    persistent_token_counts: list[int],
    generation_topk: int,
    greedy_max_new_tokens: int,
    max_new_tokens: int,
    samples_per_condition: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    early_stop_on_valid_json: bool,
    early_stop_check_interval: int,
    early_stop_min_new_tokens: int,
    resume: bool,
    disable_tqdm: bool,
) -> Path:
    if pairs_df.empty:
        raise ValueError("pairs_df is empty.")
    if int(eval_pair_count) <= 0 or int(eval_pair_count) >= len(pairs_df):
        raise ValueError("eval_pair_count must be positive and smaller than the number of available pairs.")
    if int(generation_eval_count) <= 0:
        raise ValueError("generation_eval_count must be positive.")

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples_path = output_root / "samples.jsonl"
    completed_keys, generation_stats = ap._load_completed_samples(samples_path) if resume else (set(), {})

    shuffled_pairs_df = pairs_df.sample(frac=1.0, random_state=int(base_seed)).reset_index(drop=True).copy()
    if "pair_index" in shuffled_pairs_df.columns:
        shuffled_pairs_df = shuffled_pairs_df.drop(columns=["pair_index"])
    shuffled_pairs_df.insert(0, "pair_index", np.arange(len(shuffled_pairs_df), dtype=int))
    train_pairs_df = shuffled_pairs_df.iloc[:-int(eval_pair_count)].reset_index(drop=True).copy()
    eval_pairs_df = shuffled_pairs_df.iloc[-int(eval_pair_count) :].reset_index(drop=True).copy()
    generation_pairs_df = eval_pairs_df.head(min(int(generation_eval_count), len(eval_pairs_df))).reset_index(drop=True).copy()

    shuffled_pairs_df.to_csv(output_root / "post_commitment_pairs.csv", index=False)
    train_pairs_df.to_csv(output_root / "train_pairs.csv", index=False)
    eval_pairs_df.to_csv(output_root / "eval_pairs.csv", index=False)
    generation_pairs_df.to_csv(output_root / "generation_eval_pairs.csv", index=False)
    ap.write_jsonl(output_root / "post_commitment_pairs.jsonl", shuffled_pairs_df.to_dict(orient="records"))

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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ap.assert_model_fully_on_cuda(model)

    layers, layer_path = ap.resolve_decoder_layers(model)
    n_layers = len(layers)
    if layer_candidates is None:
        effective_layer_count = DEFAULT_TRUTHFUL_STEERING_LAYER_COUNT if layer_count is None or int(layer_count) <= 0 else int(layer_count)
        layer_candidates = ap.build_evenly_spaced_layer_candidates(n_layers, int(effective_layer_count))
    layer_candidates = sorted({int(layer_idx) for layer_idx in layer_candidates if 0 <= int(layer_idx) < int(n_layers)})
    if int(exclude_final_layers) > 0:
        min_excluded_layer = int(n_layers) - int(exclude_final_layers)
        layer_candidates = [int(layer_idx) for layer_idx in layer_candidates if int(layer_idx) < int(min_excluded_layer)]
    if not layer_candidates:
        raise ValueError("No valid layer candidates were selected for steering.")

    run_config = {
        "mode": "post_commitment_repair_steering",
        "hypothesis": (
            "After a deceptive commitment sentence is already in context, learned residual directions at the "
            "post-commitment prefix can steer the generated suffix toward truthful repair instead of deceptive maintenance."
        ),
        "caa_note": (
            "A literal CAA hidden-difference vector at the shared post-commitment prefix is degenerate here because the prefix "
            "is identical across repair and maintain branches. This run therefore learns a margin-gradient steering direction "
            "at that same prefix position."
        ),
        "model_name_or_path": model_name_or_path,
        "environment": ap.DEFAULT_ENVIRONMENT,
        "model_tail": ap.DEFAULT_MODEL_TAIL,
        "n_pairs_total": int(len(shuffled_pairs_df)),
        "n_train_pairs": int(len(train_pairs_df)),
        "n_eval_pairs": int(len(eval_pairs_df)),
        "n_generation_eval_pairs": int(len(generation_pairs_df)),
        "max_model_length": int(max_model_length),
        "max_new_tokens": int(max_new_tokens),
        "samples_per_condition": int(samples_per_condition),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "base_seed": int(base_seed),
        "cuda_device": str(cuda_device),
        "decoder_layer_path": layer_path,
        "n_layers": int(n_layers),
        "layer_candidates": [int(layer_idx) for layer_idx in layer_candidates],
        "exclude_final_layers": int(exclude_final_layers),
        "alpha_values": [float(value) for value in alpha_values],
        "vector_types": [str(value) for value in vector_types],
        "persistent_token_counts": [int(value) for value in persistent_token_counts],
        "generation_topk": int(generation_topk),
        "greedy_max_new_tokens": int(greedy_max_new_tokens),
        "early_stop_on_valid_json": bool(early_stop_on_valid_json),
        "early_stop_check_interval": int(early_stop_check_interval),
        "early_stop_min_new_tokens": int(early_stop_min_new_tokens),
        "parameter_devices": ap.parameter_device_summary(model),
        "resume": bool(resume),
    }
    ap.write_json(output_root / "run_config.json", run_config)

    print(f"Output root: {output_root}")
    print(f"Post-commitment repair pairs: total={len(shuffled_pairs_df)} train={len(train_pairs_df)} eval={len(eval_pairs_df)}")
    print(f"Layer candidates: {layer_candidates}")

    vector_bank, train_gradient_df, vector_summary_df = _build_post_commitment_vector_bank(
        model=model,
        tokenizer=tokenizer,
        train_pairs_df=train_pairs_df,
        layer_indices=layer_candidates,
        max_model_length=int(max_model_length),
        base_seed=int(base_seed),
        disable_tqdm=bool(disable_tqdm),
    )
    train_gradient_df.to_csv(output_root / "train_gradient_rows.csv", index=False)
    vector_summary_df.to_csv(output_root / "vector_summary.csv", index=False)
    torch.save(
        {
            int(layer_idx): {
                key: (value if not torch.is_tensor(value) else value.cpu())
                for key, value in layer_data.items()
            }
            for layer_idx, layer_data in vector_bank.items()
        },
        output_root / "steering_vectors.pt",
    )

    all_conditions = build_post_commitment_repair_conditions(
        layer_indices=layer_candidates,
        alpha_values=alpha_values,
        vector_types=vector_types,
        persistent_token_counts=persistent_token_counts,
    )

    margin_rows: list[dict[str, Any]] = []
    margin_progress = None
    if not disable_tqdm and ap._tqdm is not None:
        margin_progress = ap._tqdm(total=int(len(eval_pairs_df)) * int(len(all_conditions)), desc="Repair margin eval", leave=True)
    try:
        for pair_row in eval_pairs_df.to_dict(orient="records"):
            prefix_text = str(pair_row["post_commitment_model_input"])
            repair_branch_text = str(pair_row["post_commitment_prompt"]) + ap.append_continuation(
                str(pair_row["post_commitment_prefix_text"]),
                str(pair_row["repair_objective_suffix_text"]),
            )
            maintain_branch_text = str(pair_row["post_commitment_prompt"]) + ap.append_continuation(
                str(pair_row["post_commitment_prefix_text"]),
                str(pair_row["maintain_objective_suffix_text"]),
            )
            baseline_repair = _compute_steered_suffix_logprob(
                model,
                tokenizer,
                prefix_text=prefix_text,
                branch_full_text=repair_branch_text,
                layer_to_vector={},
                persistent_tokens=0,
                max_model_length=int(max_model_length),
            )
            baseline_maintain = _compute_steered_suffix_logprob(
                model,
                tokenizer,
                prefix_text=prefix_text,
                branch_full_text=maintain_branch_text,
                layer_to_vector={},
                persistent_tokens=0,
                max_model_length=int(max_model_length),
            )
            baseline_margin = float(baseline_repair - baseline_maintain)
            for condition in all_conditions:
                layer_to_vector = _condition_layer_to_vector(condition, vector_bank=vector_bank)
                steered_repair = baseline_repair
                steered_maintain = baseline_maintain
                if layer_to_vector:
                    steered_repair = _compute_steered_suffix_logprob(
                        model,
                        tokenizer,
                        prefix_text=prefix_text,
                        branch_full_text=repair_branch_text,
                        layer_to_vector=layer_to_vector,
                        persistent_tokens=int(condition.get("persistent_tokens", 0)),
                        max_model_length=int(max_model_length),
                    )
                    steered_maintain = _compute_steered_suffix_logprob(
                        model,
                        tokenizer,
                        prefix_text=prefix_text,
                        branch_full_text=maintain_branch_text,
                        layer_to_vector=layer_to_vector,
                        persistent_tokens=int(condition.get("persistent_tokens", 0)),
                        max_model_length=int(max_model_length),
                    )
                steered_margin = float(steered_repair - steered_maintain)
                margin_rows.append(
                    {
                        "pair_id": str(pair_row["pair_id"]),
                        "example_id": str(pair_row["example_id"]),
                        "condition_name": str(condition["condition_name"]),
                        "patch_label": str(condition["patch_label"]),
                        "experiment": str(condition["experiment"]),
                        "vector_type": str(condition.get("vector_type", "")),
                        "layer_idx": condition.get("layer_idx"),
                        "alpha": float(condition.get("alpha", 0.0)),
                        "persistent_tokens": int(condition.get("persistent_tokens", 0)),
                        "baseline_repair_total_logprob": float(baseline_repair),
                        "baseline_maintain_total_logprob": float(baseline_maintain),
                        "baseline_margin_total_logprob": float(baseline_margin),
                        "steered_repair_total_logprob": float(steered_repair),
                        "steered_maintain_total_logprob": float(steered_maintain),
                        "steered_margin_total_logprob": float(steered_margin),
                        "margin_delta": float(steered_margin - baseline_margin),
                    }
                )
                if margin_progress is not None:
                    margin_progress.update(1)
            _write_margin_live_outputs(output_root, margin_rows)
            torch.cuda.empty_cache()
    finally:
        if margin_progress is not None:
            margin_progress.close()

    margin_summary_df = _write_margin_live_outputs(output_root, margin_rows)
    pd.DataFrame(margin_rows).to_csv(output_root / "margin_results.csv", index=False)
    margin_summary_df.to_csv(output_root / "margin_summary.csv", index=False)

    selected_generation_conditions = _select_generation_conditions(
        all_conditions=all_conditions,
        margin_summary_df=margin_summary_df,
        generation_topk=int(generation_topk),
    )
    ap.write_json(
        output_root / "selected_generation_conditions.json",
        {"conditions": ap.to_json_safe(selected_generation_conditions)},
    )

    greedy_rows: list[dict[str, Any]] = []
    greedy_progress = None
    if not disable_tqdm and ap._tqdm is not None:
        greedy_progress = ap._tqdm(
            total=int(len(generation_pairs_df)) * int(len(selected_generation_conditions)),
            desc="Greedy repair generations",
            leave=True,
        )
    try:
        for pair_row in generation_pairs_df.to_dict(orient="records"):
            prefix_text = str(pair_row["post_commitment_model_input"])
            required_rank = int(pair_row["required_rank"])
            for condition_index, condition in enumerate(selected_generation_conditions):
                layer_to_vector = _condition_layer_to_vector(condition, vector_bank=vector_bank)
                rows = run_post_commitment_generation_condition_batch_samples(
                    model,
                    tokenizer,
                    condition=condition,
                    prefix_text=prefix_text,
                    layer_to_vector=layer_to_vector,
                    required_rank=required_rank,
                    max_model_length=int(max_model_length),
                    max_new_tokens=int(greedy_max_new_tokens),
                    temperature=0.0,
                    top_p=1.0,
                    sample_indices=[0],
                    seed_start=ap._sample_seed(int(base_seed), int(pair_row["pair_index"]), int(condition_index), 0),
                    early_stop_on_valid_json=False,
                    early_stop_check_interval=int(early_stop_check_interval),
                    early_stop_min_new_tokens=int(early_stop_min_new_tokens),
                )
                row = rows[0]
                repair_similarity = _sentence_similarity(row["first_generated_sentence"], str(pair_row["repair_first_sentence"]))
                maintain_similarity = _sentence_similarity(row["first_generated_sentence"], str(pair_row["maintain_first_sentence"]))
                row.update(
                    {
                        "pair_index": int(pair_row["pair_index"]),
                        "pair_id": str(pair_row["pair_id"]),
                        "example_id": str(pair_row["example_id"]),
                        "required_rank": required_rank,
                        "repair_reference_sentence": str(pair_row["repair_first_sentence"]),
                        "maintain_reference_sentence": str(pair_row["maintain_first_sentence"]),
                        "truthful_similarity": float(repair_similarity),
                        "maintain_similarity": float(maintain_similarity),
                        "similarity_margin": float(repair_similarity - maintain_similarity),
                        "exact_repair_match": bool(
                            ap.normalize_sentence_for_compare(row["first_generated_sentence"])
                            == ap.normalize_sentence_for_compare(str(pair_row["repair_first_sentence"]))
                        ),
                        "exact_maintain_match": bool(
                            ap.normalize_sentence_for_compare(row["first_generated_sentence"])
                            == ap.normalize_sentence_for_compare(str(pair_row["maintain_first_sentence"]))
                        ),
                    }
                )
                greedy_rows.append(row)
                if greedy_progress is not None:
                    greedy_progress.update(1)
            _write_greedy_live_outputs(output_root, greedy_rows)
    finally:
        if greedy_progress is not None:
            greedy_progress.close()

    greedy_summary_df = _write_greedy_live_outputs(output_root, greedy_rows)
    pd.DataFrame(greedy_rows).to_csv(output_root / "greedy_results.csv", index=False)
    greedy_summary_df.to_csv(output_root / "greedy_summary.csv", index=False)

    total_planned = int(len(generation_pairs_df)) * int(len(selected_generation_conditions)) * int(samples_per_condition)
    remaining = 0
    for pair_row in generation_pairs_df.to_dict(orient="records"):
        for condition in selected_generation_conditions:
            layer_idx = condition.get("layer_idx")
            for sample_idx in range(int(samples_per_condition)):
                planned_key = ap._planned_sample_key(
                    pair_id=str(pair_row["pair_id"]),
                    condition_name=str(condition["condition_name"]),
                    layer_idx=None if layer_idx is None else int(layer_idx),
                    sample_idx=int(sample_idx),
                )
                if planned_key not in completed_keys:
                    remaining += 1
    print(
        "Sampled suffix workload: "
        f"{len(generation_pairs_df)} pairs x {len(selected_generation_conditions)} conditions x {int(samples_per_condition)} samples "
        f"= {total_planned} generations ({remaining} remaining after resume)."
    )

    progress = None
    if not disable_tqdm and ap._tqdm is not None:
        progress = ap._tqdm(total=remaining, desc="Sampled repair generations", leave=True)
    try:
        for pair_row in generation_pairs_df.to_dict(orient="records"):
            prefix_text = str(pair_row["post_commitment_model_input"])
            required_rank = int(pair_row["required_rank"])
            for condition_index, condition in enumerate(selected_generation_conditions):
                layer_idx = condition.get("layer_idx")
                pending_sample_indices: list[int] = []
                for sample_idx in range(int(samples_per_condition)):
                    planned_key = ap._planned_sample_key(
                        pair_id=str(pair_row["pair_id"]),
                        condition_name=str(condition["condition_name"]),
                        layer_idx=None if layer_idx is None else int(layer_idx),
                        sample_idx=int(sample_idx),
                    )
                    if planned_key not in completed_keys:
                        pending_sample_indices.append(int(sample_idx))
                if not pending_sample_indices:
                    continue

                layer_to_vector = _condition_layer_to_vector(condition, vector_bank=vector_bank)
                seed_start = ap._sample_seed(int(base_seed), int(pair_row["pair_index"]), int(condition_index), 0)
                for sample_chunk in ap.iter_chunks(pending_sample_indices, int(batch_size)):
                    batch_rows = run_post_commitment_generation_condition_batch_samples(
                        model,
                        tokenizer,
                        condition=condition,
                        prefix_text=prefix_text,
                        layer_to_vector=layer_to_vector,
                        required_rank=required_rank,
                        max_model_length=int(max_model_length),
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        sample_indices=sample_chunk,
                        seed_start=seed_start,
                        early_stop_on_valid_json=bool(early_stop_on_valid_json),
                        early_stop_check_interval=int(early_stop_check_interval),
                        early_stop_min_new_tokens=int(early_stop_min_new_tokens),
                    )
                    for row in batch_rows:
                        planned_key = ap._planned_sample_key(
                            pair_id=str(pair_row["pair_id"]),
                            condition_name=str(condition["condition_name"]),
                            layer_idx=None if layer_idx is None else int(layer_idx),
                            sample_idx=int(row["sample_idx"]),
                        )
                        row.update(
                            {
                                "pair_index": int(pair_row["pair_index"]),
                                "pair_id": str(pair_row["pair_id"]),
                                "example_id": str(pair_row["example_id"]),
                                "required_rank": required_rank,
                                "commitment_delta": float(pair_row["commitment_delta"]),
                                "deceptive_commitment_sentence": str(pair_row["deceptive_commitment_sentence"]),
                                "repair_reference_sentence": str(pair_row["repair_first_sentence"]),
                                "maintain_reference_sentence": str(pair_row["maintain_first_sentence"]),
                            }
                        )
                        ap.append_jsonl_row(samples_path, row)
                        completed_keys.add(planned_key)
                        ap._update_pair_condition_stats(generation_stats, row)
                    if progress is not None:
                        progress.update(len(batch_rows))
                    ap._write_live_summaries(output_root, generation_stats)
    finally:
        if progress is not None:
            progress.close()

    ap._write_live_summaries(output_root, generation_stats)
    for live_name, final_name in [
        ("pair_condition_summary_live.csv", "pair_condition_summary.csv"),
        ("condition_summary_live.csv", "condition_summary.csv"),
        ("margin_results_live.csv", "margin_results.csv"),
        ("margin_summary_live.csv", "margin_summary.csv"),
        ("greedy_results_live.csv", "greedy_results.csv"),
        ("greedy_summary_live.csv", "greedy_summary.csv"),
    ]:
        live_path = output_root / live_name
        if live_path.exists():
            shutil.copy2(live_path, output_root / final_name)

    print(f"Saved post-commitment repair steering artifacts to {output_root}")
    return output_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Activation patching debug utilities for BS/Qwen7B. "
            "Default mode runs post-commitment repair steering: learn a truthful-repair direction "
            "after a deceptive commitment sentence, then test whether the suffix becomes less deceptive."
        )
    )
    parser.add_argument(
        "--experiment-mode",
        type=str,
        default=DEFAULT_EXPERIMENT_MODE,
        choices=["post_commitment_repair_steering", "commitment_eap_ig", "continuation_steering", "patch_debug"],
        help="Choose post-commitment repair steering, commitment EAP-IG, continuation steering, or the older generation patch debug sweep.",
    )
    parser.add_argument(
        "--localization-dir",
        type=str,
        default=str(ap.DEFAULT_LOCALIZATION_DIR),
        help="Qwen-7B BS localization directory.",
    )
    parser.add_argument("--pair-cache-path", type=str, default=str(ap.DEFAULT_PAIR_CACHE_PATH))
    parser.add_argument(
        "--repair-pair-cache-path",
        type=str,
        default=str(DEFAULT_POST_COMMITMENT_PAIR_CACHE_PATH),
        help="Post-commitment repair steering mode. Cache path for enriched repair/maintain suffix pairs.",
    )
    parser.add_argument("--refresh-pair-cache", action="store_true", default=False)
    parser.add_argument("--cache-only", action="store_true", default=False)
    parser.add_argument(
        "--steering-source-run-dir",
        type=str,
        default="",
        help="Continuation steering mode. Discovery run directory that provides the exact pairs and layer rankings.",
    )
    parser.add_argument(
        "--steering-layer-topks",
        type=str,
        default=",".join(str(v) for v in DEFAULT_STEERING_LAYER_TOPKS),
        help="Continuation steering mode. Comma-separated top-K layer counts taken from the discovery run rankings.",
    )
    parser.add_argument(
        "--steering-patch-modes",
        type=str,
        default=",".join(DEFAULT_STEERING_PATCH_MODES),
        help="Continuation steering mode. Comma-separated subset of residual,kv,both.",
    )
    parser.add_argument("--steering-sample-count", type=int, default=DEFAULT_STEERING_SAMPLE_COUNT)
    parser.add_argument("--steering-batch-size", type=int, default=DEFAULT_STEERING_BATCH_SIZE)
    parser.add_argument("--steering-max-new-tokens", type=int, default=DEFAULT_STEERING_MAX_NEW_TOKENS)
    parser.add_argument(
        "--eval-pair-count",
        type=int,
        default=DEFAULT_TRUTHFUL_STEERING_EVAL_PAIR_COUNT,
        help="Post-commitment repair steering mode. Held-out eval pair count; the remaining pairs are used to learn vectors.",
    )
    parser.add_argument(
        "--generation-eval-count",
        type=int,
        default=DEFAULT_TRUTHFUL_STEERING_GENERATION_EVAL_COUNT,
        help="Post-commitment repair steering mode. Number of eval pairs used for greedy/sample generation metrics.",
    )
    parser.add_argument(
        "--steering-alphas",
        type=str,
        default=",".join(str(value) for value in DEFAULT_TRUTHFUL_STEERING_ALPHA_VALUES),
        help="Post-commitment repair steering mode. Comma-separated steering strength multipliers, relative to the mean prefix norm at each layer.",
    )
    parser.add_argument(
        "--steering-vector-types",
        type=str,
        default=",".join(DEFAULT_TRUTHFUL_STEERING_VECTOR_TYPES),
        help="Post-commitment repair steering mode. Comma-separated subset of learned,random,shuffled.",
    )
    parser.add_argument(
        "--persistent-token-counts",
        type=str,
        default=",".join(str(value) for value in DEFAULT_POST_COMMITMENT_PERSISTENT_TOKENS),
        help="Post-commitment repair steering mode. Comma-separated non-negative counts of generated tokens to keep steering after the deceptive prefix.",
    )
    parser.add_argument(
        "--generation-topk",
        type=int,
        default=DEFAULT_TRUTHFUL_STEERING_GENERATION_TOPK,
        help="Post-commitment repair steering mode. Number of top learned conditions to carry forward into greedy/sample generation.",
    )
    parser.add_argument(
        "--greedy-max-new-tokens",
        type=int,
        default=DEFAULT_TRUTHFUL_STEERING_GREEDY_MAX_NEW_TOKENS,
        help="Post-commitment repair steering mode. Token cap for the greedy next-sentence generation screen.",
    )
    parser.add_argument(
        "--suffix-sentence-count",
        type=int,
        default=DEFAULT_POST_COMMITMENT_SUFFIX_SENTENCE_COUNT,
        help="Post-commitment repair steering mode. Number of repair/maintain suffix sentences to score in the teacher-forced margin objective.",
    )
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
        help="Mode-dependent default: 150 for post-commitment repair steering, 50 for commitment EAP-IG, 12 for patch_debug.",
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
        help="Mode-dependent default: 8 evenly-spaced layers for post-commitment repair steering, all layers for commitment EAP-IG, 5 evenly-spaced layers for patch_debug.",
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
        else (
            DEFAULT_TRUTHFUL_STEERING_PAIR_COUNT
            if experiment_mode == "post_commitment_repair_steering"
            else (DEFAULT_COMMITMENT_PAIR_COUNT if experiment_mode == "commitment_eap_ig" else DEFAULT_DEBUG_PAIR_COUNT)
        )
    )
    if int(pair_count) <= 0:
        raise ValueError("--pair-count must be positive.")
    if int(args.max_model_length) <= 0:
        raise ValueError("--max-model-length must be positive.")
    if experiment_mode == "post_commitment_repair_steering" and int(args.eval_pair_count) <= 0:
        raise ValueError("--eval-pair-count must be positive.")
    if experiment_mode == "post_commitment_repair_steering" and int(args.generation_eval_count) <= 0:
        raise ValueError("--generation-eval-count must be positive.")
    if experiment_mode == "post_commitment_repair_steering" and int(args.generation_topk) < 0:
        raise ValueError("--generation-topk must be non-negative.")
    if experiment_mode == "post_commitment_repair_steering" and int(args.greedy_max_new_tokens) <= 0:
        raise ValueError("--greedy-max-new-tokens must be positive.")
    if experiment_mode == "post_commitment_repair_steering" and int(args.suffix_sentence_count) <= 0:
        raise ValueError("--suffix-sentence-count must be positive.")
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
    if experiment_mode == "continuation_steering" and int(args.steering_sample_count) <= 0:
        raise ValueError("--steering-sample-count must be positive.")
    if experiment_mode == "continuation_steering" and int(args.steering_batch_size) <= 0:
        raise ValueError("--steering-batch-size must be positive.")
    if experiment_mode == "continuation_steering" and int(args.steering_max_new_tokens) <= 0:
        raise ValueError("--steering-max-new-tokens must be positive.")

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
    repair_pair_cache_path = Path(args.repair_pair_cache_path).expanduser().resolve()
    steering_source_run_dir = Path(args.steering_source_run_dir).expanduser().resolve() if args.steering_source_run_dir.strip() else None
    if experiment_mode == "continuation_steering" and steering_source_run_dir is not None:
        source_pair_count = int(pair_count) if int(args.pair_count) > 0 else None
        pairs_df = load_commitment_pairs_from_run_dir(
            steering_source_run_dir,
            pair_count=source_pair_count,
        )
        print(f"Loaded {len(pairs_df)} steering pairs from {steering_source_run_dir}")
    elif experiment_mode == "post_commitment_repair_steering":
        commitment_pairs_df = load_commitment_pairs(
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
        pairs_df = load_or_build_post_commitment_repair_pairs(
            commitment_pairs_df=commitment_pairs_df,
            pair_cache_path=repair_pair_cache_path,
            suffix_sentence_count=int(args.suffix_sentence_count),
            refresh_cache=bool(args.refresh_pair_cache),
            disable_tqdm=bool(args.disable_tqdm),
        )
        if len(pairs_df) < int(pair_count):
            print(
                f"Warning: only {len(pairs_df)} post-commitment repair pairs were usable after suffix filtering "
                f"(requested {int(pair_count)} commitment spikes)."
            )
        print(f"Loaded {len(pairs_df)} post-commitment repair pairs from {repair_pair_cache_path}")
    elif experiment_mode == "commitment_eap_ig":
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
        elif experiment_mode == "post_commitment_repair_steering":
            layer_count = DEFAULT_TRUTHFUL_STEERING_LAYER_COUNT if int(args.layer_count) < 0 else int(args.layer_count)
        else:
            layer_count = None if int(args.layer_count) <= 0 else int(args.layer_count)
    if layer_candidates is not None:
        layer_tag = f"layers_{'_'.join(str(layer_idx) for layer_idx in layer_candidates)}"
    else:
        layer_tag = "layers_all" if layer_count is None or int(layer_count) <= 0 else f"layers{int(layer_count)}even"

    if experiment_mode == "post_commitment_repair_steering":
        alpha_values = parse_float_list(args.steering_alphas, name="--steering-alphas")
        vector_types = parse_vector_types(args.steering_vector_types)
        persistent_token_counts = parse_nonnegative_int_list(
            args.persistent_token_counts,
            name="--persistent-token-counts",
        )
        alpha_tag = "alphas_" + "_".join(_alpha_tag(value) for value in alpha_values)
        vector_tag = "vectors_" + "_".join(vector_types)
        persist_tag = "persist_" + "_".join(str(value) for value in persistent_token_counts)
        stop_tag = "jsonstop" if not bool(args.no_early_stop_on_valid_json) else "nojsonstop"
        run_tag = args.run_tag.strip() or (
            f"{ap.DEFAULT_ENVIRONMENT}_{ap.slugify(ap.DEFAULT_MODEL_TAIL)}_postcommit_repair_steering_"
            f"pairs{int(len(pairs_df))}_{layer_tag}_{alpha_tag}_{vector_tag}_{persist_tag}_"
            f"eval{int(args.eval_pair_count)}_geneval{int(args.generation_eval_count)}_"
            f"gentop{int(args.generation_topk)}_maxnew{int(args.steering_max_new_tokens)}_"
            f"n{int(args.steering_sample_count)}_batch{int(args.steering_batch_size)}_{stop_tag}_seed{int(args.base_seed)}"
        )
        output_root = Path(args.output_root).expanduser().resolve() / run_tag
        run_post_commitment_repair_steering_experiment(
            pairs_df=pairs_df,
            output_root=output_root,
            model_name_or_path=str(args.model_name_or_path),
            max_model_length=int(args.max_model_length),
            cuda_device_name=str(args.cuda_device),
            layer_candidates=layer_candidates,
            layer_count=layer_count,
            exclude_final_layers=int(args.exclude_final_layers),
            eval_pair_count=int(args.eval_pair_count),
            generation_eval_count=int(args.generation_eval_count),
            alpha_values=alpha_values,
            vector_types=vector_types,
            persistent_token_counts=persistent_token_counts,
            generation_topk=int(args.generation_topk),
            greedy_max_new_tokens=int(args.greedy_max_new_tokens),
            max_new_tokens=int(args.steering_max_new_tokens),
            samples_per_condition=int(args.steering_sample_count),
            batch_size=int(args.steering_batch_size),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            base_seed=int(args.base_seed),
            early_stop_on_valid_json=not bool(args.no_early_stop_on_valid_json),
            early_stop_check_interval=int(args.early_stop_check_interval),
            early_stop_min_new_tokens=int(args.early_stop_min_new_tokens),
            resume=not bool(args.no_resume),
            disable_tqdm=bool(args.disable_tqdm),
        )
        return

    if experiment_mode == "continuation_steering":
        if steering_source_run_dir is None:
            raise ValueError("--steering-source-run-dir is required for continuation_steering mode.")
        direction_layer_rankings = load_direction_layer_rankings_from_run_dir(steering_source_run_dir)
        layer_topks = parse_positive_int_list(args.steering_layer_topks, name="--steering-layer-topks")
        steering_patch_modes = parse_patch_modes(args.steering_patch_modes)
        topk_tag = "topk_" + "_".join(str(v) for v in layer_topks)
        mode_tag = "modes_" + "_".join(steering_patch_modes)
        stop_tag = "jsonstop" if not bool(args.no_early_stop_on_valid_json) else "nojsonstop"
        run_tag = args.run_tag.strip() or (
            f"{ap.DEFAULT_ENVIRONMENT}_{ap.slugify(ap.DEFAULT_MODEL_TAIL)}_continuation_steering_"
            f"pairs{int(len(pairs_df))}_{topk_tag}_{mode_tag}_"
            f"maxnew{int(args.steering_max_new_tokens)}_n{int(args.steering_sample_count)}_"
            f"batch{int(args.steering_batch_size)}_{stop_tag}_seed{int(args.base_seed)}"
        )
        output_root = Path(args.output_root).expanduser().resolve() / run_tag
        run_commitment_continuation_steering_experiment(
            pairs_df=pairs_df,
            output_root=output_root,
            direction_layer_rankings=direction_layer_rankings,
            layer_topks=layer_topks,
            steering_patch_modes=steering_patch_modes,
            model_name_or_path=str(args.model_name_or_path),
            max_model_length=int(args.max_model_length),
            max_new_tokens=int(args.steering_max_new_tokens),
            samples_per_condition=int(args.steering_sample_count),
            batch_size=int(args.steering_batch_size),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            base_seed=int(args.base_seed),
            cuda_device_name=str(args.cuda_device),
            early_stop_on_valid_json=not bool(args.no_early_stop_on_valid_json),
            early_stop_check_interval=int(args.early_stop_check_interval),
            early_stop_min_new_tokens=int(args.early_stop_min_new_tokens),
            resume=not bool(args.no_resume),
            disable_tqdm=bool(args.disable_tqdm),
            steering_source_run_dir=steering_source_run_dir,
        )
        return

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
