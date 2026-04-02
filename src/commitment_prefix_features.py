#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from attention_features import (
    DatasetPaths,
    ExampleValidationError,
    StreamingParquetWriter,
    add_span_match_columns,
    align_localized_sentences_to_tokens,
    build_localized_sentence_df,
    cleanup_tensors,
    infer_model_id,
    iter_localization_paths,
    maybe_raise_invalid_example,
    maybe_raise_runtime_error,
    resolve_device,
    resolve_dtype,
)


DEFAULT_ATTN_IMPLEMENTATION = "eager"
DEFAULT_OUTPUT_NAME = "commitment_prefix_features.parquet"
DEFAULT_WRITE_EVERY_EXAMPLES = 32
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_RECENT_WINDOW_TOKENS = 128
DEFAULT_FEATURE_SET = "core"
DEFAULT_NUM_LAYER_BLOCKS = 4
EPS = 1e-6

GEOMETRY_METRIC_NAMES = (
    "geom_end_vs_sentence_cos",
    "geom_end_vs_prefix_cos",
    "geom_end_vs_recent_cos",
    "geom_sentence_vs_prefix_cos",
    "geom_sentence_vs_recent_cos",
    "geom_recent_vs_prefix_cos",
    "geom_end_norm_vs_sentence",
    "geom_sentence_norm_vs_prefix",
    "geom_recent_norm_vs_prefix",
    "geom_sentence_token_cohesion",
    "geom_sentence_effective_rank",
    "geom_sentence_pc1_explained",
    "geom_recent_token_cohesion",
    "geom_recent_effective_rank",
    "geom_recent_pc1_explained",
)

DYNAMICS_METRIC_NAMES = (
    "dyn_end_step",
    "dyn_prefix_step",
    "dyn_end_accel",
    "dyn_prefix_accel",
    "dyn_end_update_align",
    "dyn_prefix_update_align",
    "dyn_end_straightness",
    "dyn_prefix_straightness",
    "dyn_end_step_share",
    "dyn_prefix_step_share",
    "dyn_end_vs_running_mean_cos",
    "dyn_prefix_vs_running_mean_cos",
    "dyn_end_vs_first_cos",
    "dyn_prefix_vs_first_cos",
)

BASE_METRIC_NAMES = GEOMETRY_METRIC_NAMES + DYNAMICS_METRIC_NAMES

CORE_BASE_METRIC_NAMES = (
    "geom_end_vs_prefix_cos",
    "geom_recent_vs_prefix_cos",
    "geom_sentence_token_cohesion",
    "geom_recent_token_cohesion",
    "geom_sentence_effective_rank",
    "geom_recent_effective_rank",
    "dyn_prefix_step",
    "dyn_prefix_accel",
    "dyn_prefix_update_align",
    "dyn_prefix_straightness",
    "dyn_prefix_vs_running_mean_cos",
    "dyn_prefix_vs_first_cos",
)

FULL_TRANSITION_SUFFIXES = ("delta", "devrun", "logratio_prev", "slope3", "min_gap", "max_gap")
CORE_TRANSITION_SUFFIXES = ("delta",)

METADATA_COLUMNS = [
    "example_id",
    "sentence_idx",
    "sentence_text",
    "deception_rate",
    "num_truthful",
    "num_valid",
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
    "start_token",
    "end_token",
    "token_count",
    "context_token_count",
    "prompt_token_count",
    "raw_text_context_token_count",
    "available_token_count",
    "prior_all_token_count",
    "previous_sentence_token_count",
    "recent_token_count",
    "early_token_count",
]

STRING_COLUMNS = ["example_id", "sentence_text"]
FLOAT_COLUMNS = ["deception_rate"]
INT_COLUMNS = [
    "sentence_idx",
    "num_truthful",
    "num_valid",
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
    "start_token",
    "end_token",
    "token_count",
    "context_token_count",
    "prompt_token_count",
    "raw_text_context_token_count",
    "available_token_count",
    "prior_all_token_count",
    "previous_sentence_token_count",
    "recent_token_count",
    "early_token_count",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cross-environment commitment-juncture features from hidden-state prefix geometry "
            "and write them as a sentence-level parquet dataset."
        )
    )
    parser.add_argument(
        "input_path",
        type=str,
        help=(
            "Dataset directory like "
            "/playpen-ssd/smerrill/deception2/Dataset/AdvisorAudit/DeepSeek-R1-Distill-Qwen-7B "
            "or its localization subdirectory."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Parquet output path. Defaults to <dataset_dir>/{DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model id. Defaults to the value inferred from examples.jsonl.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, or cuda:<idx>.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=DEFAULT_ATTN_IMPLEMENTATION,
        help="Attention implementation passed to from_pretrained.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to the tokenizer, config, and model loaders.",
    )
    parser.add_argument(
        "--recent-window-tokens",
        type=int,
        default=DEFAULT_RECENT_WINDOW_TOKENS,
        help="Number of trailing prefix tokens to treat as the recent prefix window.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=("core", "full"),
        default=DEFAULT_FEATURE_SET,
        help=(
            "core = compact OOD-safe set (12 metrics x layer blocks with raw/delta/pct); "
            "full = emit the full per-layer exploratory feature set."
        ),
    )
    parser.add_argument(
        "--num-layer-blocks",
        type=int,
        default=DEFAULT_NUM_LAYER_BLOCKS,
        help="Number of layer blocks to pool over when --feature-set core is used.",
    )
    parser.add_argument(
        "--state-cache-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for per-example compressed .npz caches containing end/sentence/prefix/recent "
            "states. Useful for later analysis, but not needed for the main OOD feature run."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap on the number of localization JSON files to process.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-based shard index to process after sorting localization files.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards to split the sorted localization files across.",
    )
    parser.add_argument(
        "--write-every-examples",
        type=int,
        default=DEFAULT_WRITE_EVERY_EXAMPLES,
        help="Flush buffered example feature frames to parquet after this many examples.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print a progress update every N processed files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail immediately on invalid examples instead of skipping them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing parquet output.",
    )
    return parser.parse_args(argv)


def resolve_dataset_paths(input_path: str | Path, output_path: Optional[str | Path]) -> DatasetPaths:
    root = Path(input_path).expanduser().resolve()
    if root.name == "localization":
        dataset_dir = root.parent
        localization_dir = root
    else:
        dataset_dir = root
        localization_dir = dataset_dir / "localization"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not localization_dir.is_dir():
        raise FileNotFoundError(f"Localization directory does not exist: {localization_dir}")

    output = Path(output_path).expanduser().resolve() if output_path else dataset_dir / DEFAULT_OUTPUT_NAME
    return DatasetPaths(
        dataset_dir=dataset_dir,
        localization_dir=localization_dir,
        output_path=output,
        examples_path=dataset_dir / "examples.jsonl",
    )


def build_layer_blocks(num_layers: int, num_layer_blocks: int) -> list[tuple[str, np.ndarray]]:
    num_layers = int(num_layers)
    num_layer_blocks = max(1, min(int(num_layer_blocks), num_layers))
    layer_indices = np.arange(num_layers, dtype=int)
    split_blocks = [block for block in np.array_split(layer_indices, num_layer_blocks) if block.size > 0]
    return [(f"b{block_idx}", block.astype(int, copy=False)) for block_idx, block in enumerate(split_blocks)]


def build_base_feature_columns(*, num_layers: int, feature_set: str, num_layer_blocks: int) -> list[str]:
    if feature_set == "core":
        return [
            f"{metric_name}_{block_name}"
            for block_name, _ in build_layer_blocks(num_layers, num_layer_blocks)
            for metric_name in CORE_BASE_METRIC_NAMES
        ]
    return [f"{metric_name}_l{layer_idx}" for layer_idx in range(int(num_layers)) for metric_name in BASE_METRIC_NAMES]


def build_change_feature_columns(*, num_layers: int, feature_set: str, num_layer_blocks: int) -> list[str]:
    columns: list[str] = []
    if feature_set == "core":
        base_columns = build_base_feature_columns(num_layers=num_layers, feature_set=feature_set, num_layer_blocks=num_layer_blocks)
        for column in base_columns:
            for suffix in CORE_TRANSITION_SUFFIXES:
                columns.append(f"{suffix}_{column}")
        return columns

    for layer_idx in range(int(num_layers)):
        for metric_name in BASE_METRIC_NAMES:
            for suffix in FULL_TRANSITION_SUFFIXES:
                columns.append(f"{suffix}_{metric_name}_l{layer_idx}")
    return columns


def build_normalized_feature_columns(*, num_layers: int, feature_set: str, num_layer_blocks: int) -> list[str]:
    columns: list[str] = []
    if feature_set == "core":
        for column in build_base_feature_columns(num_layers=num_layers, feature_set=feature_set, num_layer_blocks=num_layer_blocks):
            columns.append(f"pct_{column}")
        return columns

    for layer_idx in range(int(num_layers)):
        for metric_name in BASE_METRIC_NAMES:
            columns.extend([
                f"z_{metric_name}_l{layer_idx}",
                f"pct_{metric_name}_l{layer_idx}",
            ])
    return columns


def build_feature_columns(*, num_layers: int, feature_set: str, num_layer_blocks: int) -> list[str]:
    return (
        build_base_feature_columns(num_layers=num_layers, feature_set=feature_set, num_layer_blocks=num_layer_blocks)
        + build_change_feature_columns(num_layers=num_layers, feature_set=feature_set, num_layer_blocks=num_layer_blocks)
        + build_normalized_feature_columns(num_layers=num_layers, feature_set=feature_set, num_layer_blocks=num_layer_blocks)
    )


def build_empty_feature_frame(*, num_layers: int, feature_set: str, num_layer_blocks: int) -> pd.DataFrame:
    return pd.DataFrame(columns=METADATA_COLUMNS + build_feature_columns(num_layers=num_layers, feature_set=feature_set, num_layer_blocks=num_layer_blocks))


def coerce_feature_frame_columns(feature_df: pd.DataFrame, *, ordered_columns: Sequence[str]) -> pd.DataFrame:
    df = feature_df.copy().reindex(columns=list(ordered_columns)).copy()
    for column in STRING_COLUMNS:
        df[column] = df[column].astype("string")
    for column in INT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    feature_columns = [column for column in df.columns if column not in STRING_COLUMNS and column not in INT_COLUMNS]
    for column in FLOAT_COLUMNS + [column for column in feature_columns if column not in FLOAT_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return df


def flush_feature_buffer(writer: StreamingParquetWriter, buffer: list[pd.DataFrame], *, ordered_columns: Sequence[str]) -> int:
    if not buffer:
        return 0
    chunk_df = pd.concat(buffer, ignore_index=True)
    buffer.clear()
    chunk_df = coerce_feature_frame_columns(chunk_df, ordered_columns=ordered_columns)
    writer.write(chunk_df)
    return len(chunk_df)


def _softsign_scalar(value: float) -> float:
    value = float(value)
    return value / (1.0 + abs(value))


def _safe_mean_pool(hidden_slice: torch.Tensor) -> Optional[torch.Tensor]:
    if hidden_slice.ndim != 2 or hidden_slice.shape[0] <= 0:
        return None
    return hidden_slice.mean(dim=0)


def _cosine_similarity_unit_interval(vec_a: Optional[torch.Tensor], vec_b: Optional[torch.Tensor]) -> float:
    if vec_a is None or vec_b is None:
        return float("nan")
    norm_a = float(torch.linalg.vector_norm(vec_a).item())
    norm_b = float(torch.linalg.vector_norm(vec_b).item())
    if norm_a <= EPS or norm_b <= EPS:
        return float("nan")
    cosine = float(torch.dot(vec_a, vec_b).item() / (norm_a * norm_b))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.clip(0.5 * (cosine + 1.0), 0.0, 1.0))


def _bounded_ratio_scalar(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    total = abs(numerator) + abs(denominator)
    if total <= EPS:
        return 0.5
    return float(np.clip(abs(numerator) / total, 0.0, 1.0))


def _bounded_norm_change(vec_a: Optional[torch.Tensor], vec_b: Optional[torch.Tensor]) -> float:
    if vec_a is None or vec_b is None:
        return float("nan")
    diff = float(torch.linalg.vector_norm(vec_a - vec_b).item())
    denom = float(torch.linalg.vector_norm(vec_a).item() + torch.linalg.vector_norm(vec_b).item())
    if denom <= EPS:
        return 0.0
    return float(np.clip(diff / denom, 0.0, 1.0))


def _token_norm_summary(hidden_slice: torch.Tensor) -> tuple[float, float]:
    if hidden_slice.ndim != 2 or hidden_slice.shape[0] <= 0:
        return float("nan"), float("nan")
    norms = torch.linalg.vector_norm(hidden_slice, ord=2, dim=1)
    return float(norms.mean().item()), float(norms.std(unbiased=False).item())


def _mean_pairwise_cosine_cohesion(hidden_slice: torch.Tensor) -> float:
    if hidden_slice.ndim != 2 or hidden_slice.shape[0] <= 0:
        return float("nan")
    if hidden_slice.shape[0] == 1:
        return 1.0
    norms = torch.linalg.vector_norm(hidden_slice, ord=2, dim=1)
    valid = norms > EPS
    if int(valid.sum().item()) <= 1:
        return 1.0
    unit = hidden_slice[valid] / norms[valid].unsqueeze(1)
    sim = torch.clamp(unit @ unit.T, min=-1.0, max=1.0)
    off_diag_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    if not bool(off_diag_mask.any()):
        return 1.0
    mean_cos = float(sim[off_diag_mask].mean().item())
    return float(np.clip(0.5 * (mean_cos + 1.0), 0.0, 1.0))


def _effective_rank_and_pc1_explained(hidden_slice: torch.Tensor) -> tuple[float, float]:
    if hidden_slice.ndim != 2 or hidden_slice.shape[0] <= 0:
        return float("nan"), float("nan")
    if hidden_slice.shape[0] == 1:
        return 0.0, 1.0
    centered = hidden_slice - hidden_slice.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    if singular_values.numel() <= 0:
        return float("nan"), float("nan")
    energy = singular_values.square()
    total_energy = float(energy.sum().item())
    if total_energy <= EPS:
        return 0.0, 1.0
    support_k = int(energy.shape[0])
    if support_k <= 1:
        return 0.0, 1.0
    participation_ratio = float(total_energy ** 2 / energy.square().sum().item())
    effective_rank = float(np.clip((participation_ratio - 1.0) / (support_k - 1.0), 0.0, 1.0))
    pc1_share = float(energy[0].item() / total_energy)
    pc1_explained = float(np.clip((pc1_share - (1.0 / support_k)) / (1.0 - (1.0 / support_k)), 0.0, 1.0))
    return effective_rank, pc1_explained


def add_trace_region_columns(aligned_sentence_df: pd.DataFrame, *, recent_window_tokens: int) -> pd.DataFrame:
    df = aligned_sentence_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    previous_sentence_start_tokens: list[Optional[int]] = []
    previous_sentence_end_tokens: list[Optional[int]] = []
    previous_sentence_token_counts: list[int] = []
    recent_token_counts: list[int] = []
    early_token_counts: list[int] = []
    prior_all_token_counts: list[int] = []
    available_token_counts: list[int] = []

    for row_idx, row in enumerate(df.itertuples()):
        if row.start_token is None or pd.isna(row.start_token):
            previous_sentence_start_tokens.append(None)
            previous_sentence_end_tokens.append(None)
            previous_sentence_token_counts.append(0)
            recent_token_counts.append(0)
            early_token_counts.append(0)
            prior_all_token_counts.append(0)
            available_token_counts.append(0)
            continue

        start_token = int(row.start_token)
        end_token = int(row.end_token)
        prior_all_token_counts.append(start_token)
        available_token_counts.append(end_token + 1)

        if row_idx == 0:
            previous_sentence_start_tokens.append(None)
            previous_sentence_end_tokens.append(None)
            previous_sentence_token_counts.append(0)
        else:
            prev_row = df.iloc[row_idx - 1]
            prev_start = int(prev_row["start_token"])
            prev_end = int(prev_row["end_token"])
            previous_sentence_start_tokens.append(prev_start)
            previous_sentence_end_tokens.append(prev_end)
            previous_sentence_token_counts.append(prev_end - prev_start + 1)

        recent_start = max(0, end_token + 1 - int(recent_window_tokens))
        recent_token_counts.append(end_token + 1 - recent_start)
        early_token_counts.append(recent_start)

    df["previous_sentence_start_token"] = previous_sentence_start_tokens
    df["previous_sentence_end_token"] = previous_sentence_end_tokens
    df["previous_sentence_token_count"] = previous_sentence_token_counts
    df["recent_token_count"] = recent_token_counts
    df["early_token_count"] = early_token_counts
    df["prior_all_token_count"] = prior_all_token_counts
    df["available_token_count"] = available_token_counts
    return df


def _compute_state_cache(hidden_states: Sequence[torch.Tensor], aligned_sentence_df: pd.DataFrame, *, recent_window_tokens: int) -> dict[str, np.ndarray]:
    num_sentences = len(aligned_sentence_df)
    num_layers = len(hidden_states)
    hidden_size = int(hidden_states[0].shape[-1])

    state_end = np.full((num_sentences, num_layers, hidden_size), np.nan, dtype=np.float32)
    state_sentence_mean = np.full_like(state_end, np.nan)
    state_prefix_mean = np.full_like(state_end, np.nan)
    state_recent_mean = np.full_like(state_end, np.nan)

    scalar_cache = {
        "sentence_token_norm_mean": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "sentence_token_norm_std": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "prefix_token_norm_mean": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "prefix_token_norm_std": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "recent_token_norm_mean": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "recent_token_norm_std": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "sentence_token_cohesion": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "sentence_effective_rank": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "sentence_pc1_explained": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "recent_token_cohesion": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "recent_effective_rank": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
        "recent_pc1_explained": np.full((num_sentences, num_layers), np.nan, dtype=np.float32),
    }

    for row_idx, row in enumerate(aligned_sentence_df.itertuples()):
        end_token = int(row.end_token)
        recent_start = max(0, end_token + 1 - int(recent_window_tokens))
        q_idx = torch.tensor(row.token_indices, device=hidden_states[0].device, dtype=torch.long)

        for layer_idx, layer_hidden in enumerate(hidden_states):
            hidden = layer_hidden[0].to(dtype=torch.float32)
            sentence_hidden = hidden[q_idx]
            prefix_hidden = hidden[: end_token + 1]
            recent_hidden = hidden[recent_start : end_token + 1]

            sentence_mean = _safe_mean_pool(sentence_hidden)
            prefix_mean = _safe_mean_pool(prefix_hidden)
            recent_mean = _safe_mean_pool(recent_hidden)
            end_state = hidden[end_token] if end_token < hidden.shape[0] else None

            if sentence_mean is not None:
                state_sentence_mean[row_idx, layer_idx] = sentence_mean.detach().cpu().numpy().astype(np.float32, copy=False)
            if prefix_mean is not None:
                state_prefix_mean[row_idx, layer_idx] = prefix_mean.detach().cpu().numpy().astype(np.float32, copy=False)
            if recent_mean is not None:
                state_recent_mean[row_idx, layer_idx] = recent_mean.detach().cpu().numpy().astype(np.float32, copy=False)
            if end_state is not None:
                state_end[row_idx, layer_idx] = end_state.detach().cpu().numpy().astype(np.float32, copy=False)

            sent_norm_mean, sent_norm_std = _token_norm_summary(sentence_hidden)
            pref_norm_mean, pref_norm_std = _token_norm_summary(prefix_hidden)
            recent_norm_mean, recent_norm_std = _token_norm_summary(recent_hidden)
            scalar_cache["sentence_token_norm_mean"][row_idx, layer_idx] = sent_norm_mean
            scalar_cache["sentence_token_norm_std"][row_idx, layer_idx] = sent_norm_std
            scalar_cache["prefix_token_norm_mean"][row_idx, layer_idx] = pref_norm_mean
            scalar_cache["prefix_token_norm_std"][row_idx, layer_idx] = pref_norm_std
            scalar_cache["recent_token_norm_mean"][row_idx, layer_idx] = recent_norm_mean
            scalar_cache["recent_token_norm_std"][row_idx, layer_idx] = recent_norm_std
            scalar_cache["sentence_token_cohesion"][row_idx, layer_idx] = _mean_pairwise_cosine_cohesion(sentence_hidden)
            eff_rank, pc1 = _effective_rank_and_pc1_explained(sentence_hidden)
            scalar_cache["sentence_effective_rank"][row_idx, layer_idx] = eff_rank
            scalar_cache["sentence_pc1_explained"][row_idx, layer_idx] = pc1
            scalar_cache["recent_token_cohesion"][row_idx, layer_idx] = _mean_pairwise_cosine_cohesion(recent_hidden)
            eff_rank_recent, pc1_recent = _effective_rank_and_pc1_explained(recent_hidden)
            scalar_cache["recent_effective_rank"][row_idx, layer_idx] = eff_rank_recent
            scalar_cache["recent_pc1_explained"][row_idx, layer_idx] = pc1_recent

    return {
        "state_end": state_end,
        "state_sentence_mean": state_sentence_mean,
        "state_prefix_mean": state_prefix_mean,
        "state_recent_mean": state_recent_mean,
        **scalar_cache,
    }


def _vec_from_cache(array: np.ndarray, sentence_idx: int, layer_idx: int) -> Optional[torch.Tensor]:
    vec = array[sentence_idx, layer_idx]
    if not np.isfinite(vec).any():
        return None
    return torch.from_numpy(vec.astype(np.float32, copy=False))


def _build_base_metric_arrays(state_cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state_end = state_cache["state_end"]
    state_sentence_mean = state_cache["state_sentence_mean"]
    state_prefix_mean = state_cache["state_prefix_mean"]
    state_recent_mean = state_cache["state_recent_mean"]
    num_sentences, num_layers, _ = state_end.shape

    metrics = {metric_name: np.full((num_sentences, num_layers), np.nan, dtype=np.float32) for metric_name in BASE_METRIC_NAMES}

    cum_path_end = np.zeros((num_layers,), dtype=np.float32)
    cum_path_prefix = np.zeros((num_layers,), dtype=np.float32)

    for layer_idx in range(num_layers):
        first_end = _vec_from_cache(state_end, 0, layer_idx)
        first_prefix = _vec_from_cache(state_prefix_mean, 0, layer_idx)
        prev_end_delta: Optional[torch.Tensor] = None
        prev_prefix_delta: Optional[torch.Tensor] = None

        for row_idx in range(num_sentences):
            end_vec = _vec_from_cache(state_end, row_idx, layer_idx)
            sent_vec = _vec_from_cache(state_sentence_mean, row_idx, layer_idx)
            prefix_vec = _vec_from_cache(state_prefix_mean, row_idx, layer_idx)
            recent_vec = _vec_from_cache(state_recent_mean, row_idx, layer_idx)

            metrics["geom_end_vs_sentence_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(end_vec, sent_vec)
            metrics["geom_end_vs_prefix_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(end_vec, prefix_vec)
            metrics["geom_end_vs_recent_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(end_vec, recent_vec)
            metrics["geom_sentence_vs_prefix_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(sent_vec, prefix_vec)
            metrics["geom_sentence_vs_recent_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(sent_vec, recent_vec)
            metrics["geom_recent_vs_prefix_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(recent_vec, prefix_vec)

            end_norm = float(torch.linalg.vector_norm(end_vec).item()) if end_vec is not None else float("nan")
            sent_norm = float(state_cache["sentence_token_norm_mean"][row_idx, layer_idx])
            prefix_norm = float(state_cache["prefix_token_norm_mean"][row_idx, layer_idx])
            recent_norm = float(state_cache["recent_token_norm_mean"][row_idx, layer_idx])
            metrics["geom_end_norm_vs_sentence"][row_idx, layer_idx] = _bounded_ratio_scalar(end_norm, sent_norm)
            metrics["geom_sentence_norm_vs_prefix"][row_idx, layer_idx] = _bounded_ratio_scalar(sent_norm, prefix_norm)
            metrics["geom_recent_norm_vs_prefix"][row_idx, layer_idx] = _bounded_ratio_scalar(recent_norm, prefix_norm)
            metrics["geom_sentence_token_cohesion"][row_idx, layer_idx] = state_cache["sentence_token_cohesion"][row_idx, layer_idx]
            metrics["geom_sentence_effective_rank"][row_idx, layer_idx] = state_cache["sentence_effective_rank"][row_idx, layer_idx]
            metrics["geom_sentence_pc1_explained"][row_idx, layer_idx] = state_cache["sentence_pc1_explained"][row_idx, layer_idx]
            metrics["geom_recent_token_cohesion"][row_idx, layer_idx] = state_cache["recent_token_cohesion"][row_idx, layer_idx]
            metrics["geom_recent_effective_rank"][row_idx, layer_idx] = state_cache["recent_effective_rank"][row_idx, layer_idx]
            metrics["geom_recent_pc1_explained"][row_idx, layer_idx] = state_cache["recent_pc1_explained"][row_idx, layer_idx]

            if row_idx == 0:
                continue

            prev_end = _vec_from_cache(state_end, row_idx - 1, layer_idx)
            prev_prefix = _vec_from_cache(state_prefix_mean, row_idx - 1, layer_idx)
            running_end = None
            running_prefix = None
            if row_idx > 0:
                running_end_arr = state_end[:row_idx, layer_idx]
                running_prefix_arr = state_prefix_mean[:row_idx, layer_idx]
                if np.isfinite(running_end_arr).all():
                    running_end = torch.from_numpy(running_end_arr.mean(axis=0).astype(np.float32, copy=False))
                if np.isfinite(running_prefix_arr).all():
                    running_prefix = torch.from_numpy(running_prefix_arr.mean(axis=0).astype(np.float32, copy=False))

            step_end = _bounded_norm_change(end_vec, prev_end)
            step_prefix = _bounded_norm_change(prefix_vec, prev_prefix)
            metrics["dyn_end_step"][row_idx, layer_idx] = step_end
            metrics["dyn_prefix_step"][row_idx, layer_idx] = step_prefix

            end_delta = end_vec - prev_end if end_vec is not None and prev_end is not None else None
            prefix_delta = prefix_vec - prev_prefix if prefix_vec is not None and prev_prefix is not None else None

            metrics["dyn_end_accel"][row_idx, layer_idx] = _bounded_norm_change(end_delta, prev_end_delta)
            metrics["dyn_prefix_accel"][row_idx, layer_idx] = _bounded_norm_change(prefix_delta, prev_prefix_delta)
            metrics["dyn_end_update_align"][row_idx, layer_idx] = _cosine_similarity_unit_interval(end_delta, prev_end_delta)
            metrics["dyn_prefix_update_align"][row_idx, layer_idx] = _cosine_similarity_unit_interval(prefix_delta, prev_prefix_delta)
            metrics["dyn_end_vs_running_mean_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(end_vec, running_end)
            metrics["dyn_prefix_vs_running_mean_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(prefix_vec, running_prefix)
            metrics["dyn_end_vs_first_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(end_vec, first_end)
            metrics["dyn_prefix_vs_first_cos"][row_idx, layer_idx] = _cosine_similarity_unit_interval(prefix_vec, first_prefix)

            if np.isfinite(step_end):
                cum_path_end[layer_idx] += float(step_end)
            if np.isfinite(step_prefix):
                cum_path_prefix[layer_idx] += float(step_prefix)

            net_end = _bounded_norm_change(end_vec, first_end)
            net_prefix = _bounded_norm_change(prefix_vec, first_prefix)
            metrics["dyn_end_straightness"][row_idx, layer_idx] = (
                float(np.clip(net_end / max(float(cum_path_end[layer_idx]), EPS), 0.0, 1.0)) if np.isfinite(net_end) else float("nan")
            )
            metrics["dyn_prefix_straightness"][row_idx, layer_idx] = (
                float(np.clip(net_prefix / max(float(cum_path_prefix[layer_idx]), EPS), 0.0, 1.0)) if np.isfinite(net_prefix) else float("nan")
            )
            metrics["dyn_end_step_share"][row_idx, layer_idx] = (
                float(np.clip(step_end / max(float(cum_path_end[layer_idx]), EPS), 0.0, 1.0)) if np.isfinite(step_end) else float("nan")
            )
            metrics["dyn_prefix_step_share"][row_idx, layer_idx] = (
                float(np.clip(step_prefix / max(float(cum_path_prefix[layer_idx]), EPS), 0.0, 1.0)) if np.isfinite(step_prefix) else float("nan")
            )

            prev_end_delta = end_delta
            prev_prefix_delta = prefix_delta

    return metrics


def _aggregate_metric_arrays_to_blocks(
    metric_arrays: dict[str, np.ndarray],
    *,
    num_layers: int,
    num_layer_blocks: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    block_defs = build_layer_blocks(num_layers, num_layer_blocks)
    first_metric = metric_arrays[next(iter(metric_arrays.keys()))]
    num_sentences = int(first_metric.shape[0])
    aggregated = {
        metric_name: np.full((num_sentences, len(block_defs)), np.nan, dtype=np.float32)
        for metric_name in CORE_BASE_METRIC_NAMES
    }
    block_names: list[str] = []

    for block_idx, (block_name, layer_block) in enumerate(block_defs):
        block_names.append(block_name)
        for metric_name in CORE_BASE_METRIC_NAMES:
            block_values = metric_arrays[metric_name][:, layer_block]
            with np.errstate(invalid="ignore"):
                aggregated[metric_name][:, block_idx] = np.nanmean(block_values, axis=1).astype(np.float32, copy=False)
            no_valid = ~np.isfinite(block_values).any(axis=1)
            aggregated[metric_name][no_valid, block_idx] = np.nan

    return aggregated, block_names


def _build_metadata_row(*, example_id: str, sentence_row: Any, prompt_token_count: int) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "sentence_idx": int(sentence_row.sentence_idx),
        "sentence_text": sentence_row.sentence_text,
        "deception_rate": float(sentence_row.deception_rate),
        "num_truthful": sentence_row.num_truthful,
        "num_valid": sentence_row.num_valid,
        "raw_start": int(sentence_row.raw_start),
        "raw_end": int(sentence_row.raw_end),
        "full_start": int(sentence_row.full_start),
        "full_end": int(sentence_row.full_end),
        "start_token": int(sentence_row.start_token),
        "end_token": int(sentence_row.end_token),
        "token_count": int(sentence_row.token_count),
        "context_token_count": int(sentence_row.context_token_count),
        "prompt_token_count": int(prompt_token_count),
        "raw_text_context_token_count": max(0, int(sentence_row.start_token) - int(prompt_token_count)),
        "available_token_count": int(sentence_row.available_token_count),
        "prior_all_token_count": int(sentence_row.prior_all_token_count),
        "previous_sentence_token_count": int(sentence_row.previous_sentence_token_count),
        "recent_token_count": int(sentence_row.recent_token_count),
        "early_token_count": int(sentence_row.early_token_count),
    }


def _bounded_relative_change(current: pd.Series, reference: pd.Series) -> pd.Series:
    current_values = pd.to_numeric(current, errors="coerce").astype(float)
    reference_values = pd.to_numeric(reference, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=current_values.index, dtype=float)
    valid = current_values.notna() & reference_values.notna()
    if valid.any():
        numerator = current_values.loc[valid] - reference_values.loc[valid]
        denominator = current_values.loc[valid].abs() + reference_values.loc[valid].abs() + EPS
        out.loc[valid] = numerator / denominator
    return out.clip(-1.0, 1.0)


def _causal_rolling_mean(current: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(current, errors="coerce").astype(float)
    return values.shift(1).rolling(window=window, min_periods=1).mean()


def add_transition_features(feature_df: pd.DataFrame, *, base_feature_columns: Sequence[str], transition_suffixes: Sequence[str]) -> pd.DataFrame:
    df = feature_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    new_columns: dict[str, pd.Series] = {}

    for column in base_feature_columns:
        current = pd.to_numeric(df[column], errors="coerce").astype(float)
        prev = current.shift(1)
        prev_running_mean = current.expanding(min_periods=1).mean().shift(1)
        prev_running_min = current.cummin().shift(1)
        prev_running_max = current.cummax().shift(1)
        prev_rolling_mean = _causal_rolling_mean(current, window=3)

        for suffix in transition_suffixes:
            if suffix == "delta":
                new_columns[f"delta_{column}"] = _bounded_relative_change(current, prev)
            elif suffix == "devrun":
                new_columns[f"devrun_{column}"] = _bounded_relative_change(current, prev_running_mean)
            elif suffix == "logratio_prev":
                new_columns[f"logratio_prev_{column}"] = _bounded_relative_change(current, prev)
            elif suffix == "slope3":
                new_columns[f"slope3_{column}"] = _bounded_relative_change(current, prev_rolling_mean)
            elif suffix == "min_gap":
                new_columns[f"min_gap_{column}"] = _bounded_relative_change(current, prev_running_min)
            elif suffix == "max_gap":
                new_columns[f"max_gap_{column}"] = _bounded_relative_change(current, prev_running_max)
            else:
                raise ValueError(f"Unsupported transition suffix: {suffix}")

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def _causal_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    for row_idx in range(values.shape[0]):
        current = values[row_idx]
        if not np.isfinite(current):
            continue
        previous = values[:row_idx]
        previous = previous[np.isfinite(previous)]
        if previous.size == 0:
            continue
        raw_z = float((current - previous.mean()) / (previous.std(ddof=0) + EPS))
        out[row_idx] = _softsign_scalar(raw_z)
    return out


def _causal_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    for row_idx in range(values.shape[0]):
        current = values[row_idx]
        if not np.isfinite(current):
            continue
        previous = values[:row_idx]
        previous = previous[np.isfinite(previous)]
        if previous.size == 0:
            continue
        out[row_idx] = float(np.mean(previous <= current))
    return out


def add_within_trace_normalization(
    feature_df: pd.DataFrame,
    *,
    base_feature_columns: Sequence[str],
    include_z: bool,
    include_pct: bool,
) -> pd.DataFrame:
    df = feature_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    new_columns: dict[str, pd.Series] = {}

    for column in base_feature_columns:
        values = pd.to_numeric(df[column], errors="coerce").astype(float).to_numpy(dtype=float)
        if include_z:
            new_columns[f"z_{column}"] = pd.Series(_causal_zscore(values), index=df.index)
        if include_pct:
            new_columns[f"pct_{column}"] = pd.Series(_causal_percentile(values), index=df.index)

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def compute_commitment_prefix_features(
    hidden_states: Sequence[torch.Tensor],
    aligned_sentence_df: pd.DataFrame,
    *,
    example_id: str,
    prompt_token_count: int,
    recent_window_tokens: int,
    feature_set: str,
    num_layer_blocks: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    working_df = aligned_sentence_df.copy()
    required_region_columns = {
        "previous_sentence_start_token",
        "previous_sentence_end_token",
        "previous_sentence_token_count",
        "recent_token_count",
        "early_token_count",
        "prior_all_token_count",
        "available_token_count",
    }
    if not required_region_columns.issubset(working_df.columns):
        working_df = add_trace_region_columns(working_df, recent_window_tokens=recent_window_tokens)

    state_cache = _compute_state_cache(hidden_states, working_df, recent_window_tokens=recent_window_tokens)
    metric_arrays = _build_base_metric_arrays(state_cache)
    num_layers = len(hidden_states)

    rows: list[dict[str, Any]] = []
    if feature_set == "core":
        aggregated_arrays, block_names = _aggregate_metric_arrays_to_blocks(
            metric_arrays,
            num_layers=num_layers,
            num_layer_blocks=num_layer_blocks,
        )
        base_feature_columns = [f"{metric_name}_{block_name}" for block_name in block_names for metric_name in CORE_BASE_METRIC_NAMES]
        transition_suffixes = CORE_TRANSITION_SUFFIXES
        include_z = False
        include_pct = True

        for row_idx, row in enumerate(working_df.itertuples()):
            record = _build_metadata_row(example_id=example_id, sentence_row=row, prompt_token_count=prompt_token_count)
            for block_idx, block_name in enumerate(block_names):
                for metric_name in CORE_BASE_METRIC_NAMES:
                    record[f"{metric_name}_{block_name}"] = float(aggregated_arrays[metric_name][row_idx, block_idx])
            rows.append(record)
    else:
        base_feature_columns = [f"{metric_name}_l{layer_idx}" for layer_idx in range(num_layers) for metric_name in BASE_METRIC_NAMES]
        transition_suffixes = FULL_TRANSITION_SUFFIXES
        include_z = True
        include_pct = True

        for row_idx, row in enumerate(working_df.itertuples()):
            record = _build_metadata_row(example_id=example_id, sentence_row=row, prompt_token_count=prompt_token_count)
            for layer_idx in range(num_layers):
                for metric_name in BASE_METRIC_NAMES:
                    record[f"{metric_name}_l{layer_idx}"] = float(metric_arrays[metric_name][row_idx, layer_idx])
            rows.append(record)

    if not rows:
        return build_empty_feature_frame(num_layers=num_layers, feature_set=feature_set, num_layer_blocks=num_layer_blocks), state_cache

    feature_df = pd.DataFrame(rows)
    feature_df = add_transition_features(feature_df, base_feature_columns=base_feature_columns, transition_suffixes=transition_suffixes)
    feature_df = add_within_trace_normalization(
        feature_df,
        base_feature_columns=base_feature_columns,
        include_z=include_z,
        include_pct=include_pct,
    )
    return feature_df, state_cache


def maybe_write_state_cache(*, state_cache_dir: Optional[Path], example_id: str, feature_df: pd.DataFrame, state_cache: dict[str, np.ndarray]) -> None:
    if state_cache_dir is None:
        return
    state_cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = state_cache_dir / f"{example_id}.npz"
    np.savez_compressed(
        out_path,
        example_id=np.asarray([example_id], dtype=object),
        sentence_idx=feature_df["sentence_idx"].to_numpy(dtype=np.int64, copy=False),
        deception_rate=feature_df["deception_rate"].to_numpy(dtype=np.float32, copy=False),
        state_end=state_cache["state_end"],
        state_sentence_mean=state_cache["state_sentence_mean"],
        state_prefix_mean=state_cache["state_prefix_mean"],
        state_recent_mean=state_cache["state_recent_mean"],
    )


def extract_example_feature_df(
    *,
    example: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    recent_window_tokens: int,
    num_layers: int,
    state_cache_dir: Optional[Path],
    feature_set: str,
    num_layer_blocks: int,
) -> pd.DataFrame:
    example_id = example.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ExampleValidationError("missing_example_id", "Localization example is missing example_id.")

    full_text = example.get("raw_text")
    if not isinstance(full_text, str) or not full_text:
        raise ExampleValidationError("missing_raw_text", f"{example_id} is missing raw_text.")

    localized_sentence_df = build_localized_sentence_df(example)
    if localized_sentence_df.empty:
        raise ExampleValidationError("empty_history", f"{example_id} has no localized history entries.")

    localized_sentence_df = add_span_match_columns(full_text, localized_sentence_df)
    if not localized_sentence_df["span_matches"].all():
        bad_count = int((~localized_sentence_df["span_matches"]).sum())
        raise ExampleValidationError(
            "span_mismatch",
            f"{example_id} has {bad_count} localized sentence spans that do not match raw_text.",
        )

    tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids_list = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]
    if not input_ids_list:
        raise ExampleValidationError("no_tokens", f"{example_id} tokenized to zero tokens.")

    aligned_sentence_df = align_localized_sentences_to_tokens(offsets, localized_sentence_df)
    if not (aligned_sentence_df["token_count"] > 0).all():
        bad_count = int((aligned_sentence_df["token_count"] == 0).sum())
        raise ExampleValidationError(
            "unmapped_sentence",
            f"{example_id} has {bad_count} localized sentences that failed to map to tokens.",
        )

    aligned_sentence_df = add_trace_region_columns(aligned_sentence_df, recent_window_tokens=recent_window_tokens)
    modeling_sentence_df = aligned_sentence_df.loc[aligned_sentence_df["start_token"].fillna(0).astype(int) > 0].copy()
    if modeling_sentence_df.empty:
        return build_empty_feature_frame(num_layers=num_layers, feature_set=feature_set, num_layer_blocks=num_layer_blocks)

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states[1:] if getattr(outputs, "hidden_states", None) is not None else None
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        feature_df, state_cache = compute_commitment_prefix_features(
            hidden_states,
            modeling_sentence_df,
            example_id=example_id,
            prompt_token_count=0,
            recent_window_tokens=recent_window_tokens,
            feature_set=feature_set,
            num_layer_blocks=num_layer_blocks,
        )
        maybe_write_state_cache(
            state_cache_dir=state_cache_dir,
            example_id=example_id,
            feature_df=feature_df,
            state_cache=state_cache,
        )
    finally:
        if "outputs" in locals():
            del outputs
        if "hidden_states" in locals():
            del hidden_states
        if "state_cache" in locals():
            del state_cache
        del input_ids
        cleanup_tensors()

    return feature_df


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset_paths = resolve_dataset_paths(args.input_path, args.output)
    model_id = infer_model_id(dataset_paths, args.model_id)
    device, gpu_df = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    write_every_examples = max(1, int(args.write_every_examples))
    state_cache_dir = Path(args.state_cache_dir).expanduser().resolve() if args.state_cache_dir else None

    all_localization_paths = iter_localization_paths(dataset_paths.localization_dir, max_examples=int(args.max_examples))
    if not all_localization_paths:
        raise FileNotFoundError(f"No localization JSON files found in {dataset_paths.localization_dir}")
    localization_paths = iter_localization_paths(
        dataset_paths.localization_dir,
        max_examples=int(args.max_examples),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
    )
    if not localization_paths:
        print(f"Dataset dir: {dataset_paths.dataset_dir}")
        print(f"Localization dir: {dataset_paths.localization_dir}")
        print(f"Output parquet: {dataset_paths.output_path}")
        print(f"Shard: {int(args.shard_id) + 1}/{int(args.num_shards)}")
        print(f"Localization files before sharding: {len(all_localization_paths)}")
        print("Localization files to process on this shard: 0")
        print("No localization files assigned to this shard. Exiting without writing output.")
        return

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    num_layers = int(config.num_hidden_layers)
    num_layer_blocks = max(1, min(int(args.num_layer_blocks), num_layers))
    ordered_columns = METADATA_COLUMNS + build_feature_columns(
        num_layers=num_layers,
        feature_set=args.feature_set,
        num_layer_blocks=num_layer_blocks,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("This script requires a fast tokenizer because it uses offset mappings.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    writer = StreamingParquetWriter(dataset_paths.output_path, overwrite=args.overwrite)
    buffered_frames: list[pd.DataFrame] = []
    skip_counts: dict[str, int] = {}
    processed = 0
    successful = 0

    base_feature_count = len(build_base_feature_columns(num_layers=num_layers, feature_set=args.feature_set, num_layer_blocks=num_layer_blocks))
    change_feature_count = len(build_change_feature_columns(num_layers=num_layers, feature_set=args.feature_set, num_layer_blocks=num_layer_blocks))
    normalized_feature_count = len(build_normalized_feature_columns(num_layers=num_layers, feature_set=args.feature_set, num_layer_blocks=num_layer_blocks))

    print(f"Dataset dir: {dataset_paths.dataset_dir}")
    print(f"Localization dir: {dataset_paths.localization_dir}")
    print(f"Output parquet: {dataset_paths.output_path}")
    print(f"Model id: {model_id}")
    print(f"Device: {device}")
    print(f"Model dtype: {model_dtype}")
    print(f"Layers: {num_layers}")
    print(f"Feature set: {args.feature_set}")
    if args.feature_set == "core":
        block_defs = build_layer_blocks(num_layers, num_layer_blocks)
        block_desc = ", ".join(f"{name}=[{int(block[0])}-{int(block[-1])}]" for name, block in block_defs)
        print(f"Layer blocks ({len(block_defs)}): {block_desc}")
    print(f"Shard: {int(args.shard_id) + 1}/{int(args.num_shards)}")
    print(
        "Feature columns: "
        f"{base_feature_count} base + {change_feature_count} transition + {normalized_feature_count} within-trace "
        f"= {len(ordered_columns) - len(METADATA_COLUMNS)}"
    )
    print(f"Recent window tokens: {int(args.recent_window_tokens)}")
    if state_cache_dir is not None:
        print(f"State cache dir: {state_cache_dir}")
    if not gpu_df.empty:
        print("Visible GPUs:")
        print(gpu_df.to_string(index=False))
    print(f"Localization files before sharding: {len(all_localization_paths)}")
    print(f"Localization files to process on this shard: {len(localization_paths)}")

    try:
        for path in localization_paths:
            processed += 1
            try:
                example = json.loads(path.read_text(encoding="utf-8"))
                feature_df = extract_example_feature_df(
                    example=example,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    recent_window_tokens=int(args.recent_window_tokens),
                    num_layers=num_layers,
                    state_cache_dir=state_cache_dir,
                    feature_set=args.feature_set,
                    num_layer_blocks=num_layer_blocks,
                )
            except json.JSONDecodeError as exc:
                skip_counts["invalid_json"] = skip_counts.get("invalid_json", 0) + 1
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
            except ExampleValidationError as exc:
                skip_counts[exc.reason] = skip_counts.get(exc.reason, 0) + 1
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                skip_counts["malformed_example"] = skip_counts.get("malformed_example", 0) + 1
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    skip_counts["oom"] = skip_counts.get("oom", 0) + 1
                    cleanup_tensors()
                    maybe_raise_runtime_error(args, path, exc)
                    feature_df = None
                else:
                    raise

            if feature_df is not None and not feature_df.empty:
                buffered_frames.append(feature_df)
                successful += 1
            elif feature_df is not None and feature_df.empty:
                skip_counts["no_prior_reasoning_context"] = skip_counts.get("no_prior_reasoning_context", 0) + 1

            if len(buffered_frames) >= write_every_examples:
                flush_feature_buffer(writer, buffered_frames, ordered_columns=ordered_columns)

            if int(args.progress_every) > 0 and processed % int(args.progress_every) == 0:
                buffered_row_count = sum(len(df) for df in buffered_frames)
                print(
                    f"Processed {processed}/{len(localization_paths)} files | successful={successful} | "
                    f"skipped={sum(skip_counts.values())} | rows_buffered_or_written={writer.rows_written + buffered_row_count}"
                )

        flush_feature_buffer(writer, buffered_frames, ordered_columns=ordered_columns)
        if writer.rows_written == 0:
            writer.write(
                coerce_feature_frame_columns(
                    build_empty_feature_frame(num_layers=num_layers, feature_set=args.feature_set, num_layer_blocks=num_layer_blocks),
                    ordered_columns=ordered_columns,
                )
            )
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        del model
        cleanup_tensors()

    print(f"Wrote commitment-prefix features to: {dataset_paths.output_path}")
    print(f"Processed files: {processed}")
    print(f"Examples with output rows: {successful}")
    print(f"Total parquet rows: {writer.rows_written}")
    if skip_counts:
        print("Skipped examples by reason:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
