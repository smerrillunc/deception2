#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    build_localized_sentence_df,
    cleanup_tensors,
    infer_model_id,
    iter_localization_paths,
    maybe_raise_invalid_example,
    maybe_raise_runtime_error,
    resolve_device,
    resolve_dtype,
    tokenize_and_align_localized_sentences,
)


DEFAULT_ATTN_IMPLEMENTATION = "eager"
DEFAULT_OUTPUT_NAME = "attention_features2.parquet"
DEFAULT_WRITE_EVERY_EXAMPLES = 32
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_RECENT_WINDOW_TOKENS = 64
DEFAULT_SLOPE_WINDOW = 3
EPS = 1e-6

GROUNDING_METRIC_NAMES = (
    "g_prior_vs_self",
    "g_prev_vs_self",
    "g_recent_vs_self",
    "g_early_vs_recent",
    "g_prev_share_of_prior",
)

CONCENTRATION_METRIC_NAMES = (
    "entropy_full",
    "entropy_prior",
    "entropy_self",
    "top1_full",
    "top1_prior",
    "top1_self",
    "top5_full",
    "top5_prior",
    "top5_self",
    "top10_full",
    "top10_prior",
    "top10_self",
    "herfindahl_full",
    "herfindahl_prior",
    "herfindahl_self",
    "effective_support_full",
    "effective_support_prior",
    "effective_support_self",
)

BASE_METRIC_NAMES = GROUNDING_METRIC_NAMES + CONCENTRATION_METRIC_NAMES
LAYER_AGG_NAMES = ("mean", "std", "min", "max")
CHANGE_TARGET_METRIC_NAMES = (
    "g_prior_vs_self",
    "g_prev_vs_self",
    "g_early_vs_recent",
    "entropy_prior",
    "top5_full",
    "herfindahl_full",
)
CHANGE_TARGET_AGG_NAMES = ("mean", "std")
WITHIN_TRACE_NORM_METRIC_NAMES = BASE_METRIC_NAMES
WITHIN_TRACE_NORM_AGG_NAMES = LAYER_AGG_NAMES

ACTIVATION_GROUNDING_METRIC_NAMES = (
    "act_cos_prev",
    "act_cos_prior",
    "act_cos_recent",
)

ACTIVATION_CONCENTRATION_METRIC_NAMES = (
    "act_norm_mean_vs_prior",
    "act_norm_std_vs_prior",
    "act_token_cos_cohesion",
    "act_effective_rank",
    "act_pc1_explained",
)

ACTIVATION_METRIC_NAMES = ACTIVATION_GROUNDING_METRIC_NAMES + ACTIVATION_CONCENTRATION_METRIC_NAMES
ACTIVATION_CHANGE_TARGET_METRIC_NAMES = (
    "act_cos_prev",
    "act_cos_prior",
    "act_norm_mean_vs_prior",
    "act_token_cos_cohesion",
    "act_effective_rank",
)
ACTIVATION_CHANGE_TARGET_AGG_NAMES = ("mean",)

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

STRING_COLUMNS = [
    "example_id",
    "sentence_text",
]

FLOAT_COLUMNS = [
    "deception_rate",
]

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
            "Build OOD-oriented reasoning-trace attention and activation features from localization JSON files "
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
        help="Parquet output path. Defaults to <dataset_dir>/attention_features2.parquet.",
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
        help="Number of trailing prior tokens to treat as the recent context window.",
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


def build_base_feature_columns(num_layers: int) -> list[str]:
    columns: list[str] = []
    for layer_idx in range(int(num_layers)):
        for metric_name in BASE_METRIC_NAMES:
            for agg_name in LAYER_AGG_NAMES:
                columns.append(f"{metric_name}_{agg_name}_l{layer_idx}")
        for metric_name in ACTIVATION_METRIC_NAMES:
            columns.append(f"{metric_name}_mean_l{layer_idx}")
    return columns


def build_change_feature_columns(num_layers: int) -> list[str]:
    columns: list[str] = []
    for layer_idx in range(int(num_layers)):
        for metric_name in CHANGE_TARGET_METRIC_NAMES:
            for agg_name in CHANGE_TARGET_AGG_NAMES:
                columns.extend(
                    [
                        f"delta_{metric_name}_{agg_name}_l{layer_idx}",
                        f"devrun_{metric_name}_{agg_name}_l{layer_idx}",
                        f"logratio_prev_{metric_name}_{agg_name}_l{layer_idx}",
                        f"slope3_{metric_name}_{agg_name}_l{layer_idx}",
                        f"min_gap_{metric_name}_{agg_name}_l{layer_idx}",
                        f"max_gap_{metric_name}_{agg_name}_l{layer_idx}",
                    ]
                )
        for metric_name in ACTIVATION_CHANGE_TARGET_METRIC_NAMES:
            for agg_name in ACTIVATION_CHANGE_TARGET_AGG_NAMES:
                columns.extend(
                    [
                        f"delta_{metric_name}_{agg_name}_l{layer_idx}",
                        f"devrun_{metric_name}_{agg_name}_l{layer_idx}",
                        f"logratio_prev_{metric_name}_{agg_name}_l{layer_idx}",
                        f"slope3_{metric_name}_{agg_name}_l{layer_idx}",
                        f"min_gap_{metric_name}_{agg_name}_l{layer_idx}",
                        f"max_gap_{metric_name}_{agg_name}_l{layer_idx}",
                    ]
                )
    return columns


def build_normalized_feature_columns(num_layers: int) -> list[str]:
    columns: list[str] = []
    for layer_idx in range(int(num_layers)):
        for metric_name in WITHIN_TRACE_NORM_METRIC_NAMES:
            for agg_name in WITHIN_TRACE_NORM_AGG_NAMES:
                columns.extend(
                    [
                        f"z_{metric_name}_{agg_name}_l{layer_idx}",
                        f"pct_{metric_name}_{agg_name}_l{layer_idx}",
                    ]
                )
    return columns


def build_feature_columns(num_layers: int) -> list[str]:
    return (
        build_base_feature_columns(num_layers)
        + build_change_feature_columns(num_layers)
        + build_normalized_feature_columns(num_layers)
    )


def build_empty_feature_frame(num_layers: int) -> pd.DataFrame:
    return pd.DataFrame(columns=METADATA_COLUMNS + build_feature_columns(num_layers))


def coerce_feature_frame_columns(
    feature_df: pd.DataFrame,
    *,
    ordered_columns: Sequence[str],
) -> pd.DataFrame:
    # Reindex once so pandas materializes missing columns in a single block
    # instead of fragmenting the frame via repeated column inserts.
    df = feature_df.copy().reindex(columns=list(ordered_columns)).copy()

    for column in STRING_COLUMNS:
        df[column] = df[column].astype("string")
    for column in INT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    feature_columns = [column for column in df.columns if column not in STRING_COLUMNS and column not in INT_COLUMNS]
    for column in FLOAT_COLUMNS + [column for column in feature_columns if column not in FLOAT_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return df


def flush_feature_buffer(
    writer: StreamingParquetWriter,
    buffer: list[pd.DataFrame],
    *,
    ordered_columns: Sequence[str],
) -> int:
    if not buffer:
        return 0
    chunk_df = pd.concat(buffer, ignore_index=True)
    buffer.clear()
    chunk_df = coerce_feature_frame_columns(chunk_df, ordered_columns=ordered_columns)
    writer.write(chunk_df)
    return len(chunk_df)


def _empty_slice(reference: torch.Tensor) -> torch.Tensor:
    return reference[:, :0]


def _normalize_slice(slice_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = slice_tensor.sum(dim=1)
    probs = torch.zeros_like(slice_tensor)
    valid = mass > 0
    if valid.any():
        probs[valid] = slice_tensor[valid] / mass[valid].unsqueeze(1).clamp_min(EPS)
    return probs, mass


def _softsign_scalar(value: float) -> float:
    value = float(value)
    return value / (1.0 + abs(value))


def _per_token_mass(region_mass: torch.Tensor, width: int) -> torch.Tensor:
    out = torch.full_like(region_mass, float("nan"), dtype=torch.float32)
    if width <= 0:
        return out
    out[:] = torch.clamp(region_mass / float(width), min=0.0, max=1.0)
    return out


def _normalized_entropy(slice_tensor: torch.Tensor) -> torch.Tensor:
    width = int(slice_tensor.shape[1])
    out = torch.full((slice_tensor.shape[0],), float("nan"), device=slice_tensor.device, dtype=torch.float32)
    probs, mass = _normalize_slice(slice_tensor)
    valid = mass > 0
    if not valid.any():
        return out
    if width <= 1:
        out[valid] = 0.0
        return out
    probs_valid = probs[valid]
    entropy = -(probs_valid * probs_valid.clamp_min(EPS).log()).sum(dim=1)
    out[valid] = torch.clamp(entropy / float(math.log(width)), min=0.0, max=1.0)
    return out


def _topk_mass(slice_tensor: torch.Tensor, k: int) -> torch.Tensor:
    width = int(slice_tensor.shape[1])
    out = torch.full((slice_tensor.shape[0],), float("nan"), device=slice_tensor.device, dtype=torch.float32)
    probs, mass = _normalize_slice(slice_tensor)
    valid = mass > 0
    if not valid.any():
        return out
    k = max(1, min(int(k), width))
    out[valid] = torch.clamp(torch.topk(probs[valid], k=k, dim=1).values.sum(dim=1), min=0.0, max=1.0)
    return out


def _herfindahl(slice_tensor: torch.Tensor) -> torch.Tensor:
    width = int(slice_tensor.shape[1])
    out = torch.full((slice_tensor.shape[0],), float("nan"), device=slice_tensor.device, dtype=torch.float32)
    probs, mass = _normalize_slice(slice_tensor)
    valid = mass > 0
    if not valid.any():
        return out
    if width <= 1:
        out[valid] = 1.0
        return out
    raw_herfindahl = (probs[valid] ** 2).sum(dim=1)
    uniform_baseline = 1.0 / float(width)
    out[valid] = torch.clamp(
        (raw_herfindahl - uniform_baseline) / (1.0 - uniform_baseline),
        min=0.0,
        max=1.0,
    )
    return out


def _effective_support(slice_tensor: torch.Tensor) -> torch.Tensor:
    width = int(slice_tensor.shape[1])
    out = torch.full((slice_tensor.shape[0],), float("nan"), device=slice_tensor.device, dtype=torch.float32)
    probs, mass = _normalize_slice(slice_tensor)
    valid = mass > 0
    if not valid.any():
        return out
    if width <= 1:
        out[valid] = 1.0
        return out
    raw_herfindahl = (probs[valid] ** 2).sum(dim=1)
    raw_effective_support = 1.0 / raw_herfindahl.clamp_min(EPS)
    out[valid] = torch.clamp(
        (raw_effective_support - 1.0) / (float(width) - 1.0),
        min=0.0,
        max=1.0,
    )
    return out


def _ratio_from_per_token_mass(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(numerator, float("nan"), dtype=torch.float32)
    total = numerator + denominator
    valid = total > 0
    out[valid] = torch.clamp(numerator[valid] / total[valid], min=0.0, max=1.0)
    return out


def _bounded_ratio_scalar(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    total = numerator + denominator
    if total <= EPS:
        return 0.5
    return float(np.clip(numerator / total, 0.0, 1.0))


def _safe_slice_hidden(layer_hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    start = int(max(0, start))
    end = int(max(start, end))
    return layer_hidden[start:end]


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


def add_attention_region_columns(
    aligned_sentence_df: pd.DataFrame,
    *,
    recent_window_tokens: int,
) -> pd.DataFrame:
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

        recent_start = max(0, start_token - int(recent_window_tokens))
        recent_token_counts.append(start_token - recent_start)
        early_token_counts.append(recent_start)

    df["previous_sentence_start_token"] = previous_sentence_start_tokens
    df["previous_sentence_end_token"] = previous_sentence_end_tokens
    df["previous_sentence_token_count"] = previous_sentence_token_counts
    df["recent_token_count"] = recent_token_counts
    df["early_token_count"] = early_token_counts
    df["prior_all_token_count"] = prior_all_token_counts
    df["available_token_count"] = available_token_counts
    return df


def compute_sentence_metric_tensors(
    attentions: Sequence[torch.Tensor],
    sentence_row: Any,
    *,
    recent_window_tokens: int,
) -> dict[str, np.ndarray]:
    q_idx = torch.tensor(sentence_row.token_indices, device=attentions[0].device, dtype=torch.long)
    start_token = int(sentence_row.start_token)
    end_token = int(sentence_row.end_token)
    recent_start = max(0, start_token - int(recent_window_tokens))
    prev_start = None if sentence_row.previous_sentence_start_token is None or pd.isna(sentence_row.previous_sentence_start_token) else int(sentence_row.previous_sentence_start_token)
    prev_end = None if sentence_row.previous_sentence_end_token is None or pd.isna(sentence_row.previous_sentence_end_token) else int(sentence_row.previous_sentence_end_token)

    num_layers = len(attentions)
    num_heads = int(attentions[0].shape[1])
    metric_tensors = {
        metric_name: torch.full((num_layers, num_heads), float("nan"), device=attentions[0].device, dtype=torch.float32)
        for metric_name in BASE_METRIC_NAMES
    }

    for layer_idx, layer_attn in enumerate(attentions):
        layer = layer_attn[0].to(dtype=torch.float32)
        avg_attn = layer[:, q_idx, :].mean(dim=1)

        full_slice = avg_attn[:, : end_token + 1]
        prior_slice = avg_attn[:, :start_token]
        self_slice = avg_attn[:, start_token : end_token + 1]
        prev_slice = avg_attn[:, prev_start : prev_end + 1] if prev_start is not None and prev_end is not None else _empty_slice(avg_attn)
        recent_slice = avg_attn[:, recent_start:start_token]
        early_slice = avg_attn[:, :recent_start]

        prior_mass = prior_slice.sum(dim=1)
        prev_mass = prev_slice.sum(dim=1)
        recent_mass = recent_slice.sum(dim=1)
        early_mass = early_slice.sum(dim=1)
        self_mass = self_slice.sum(dim=1)

        m_prior = _per_token_mass(prior_mass, int(prior_slice.shape[1]))
        m_prev = _per_token_mass(prev_mass, int(prev_slice.shape[1]))
        m_recent = _per_token_mass(recent_mass, int(recent_slice.shape[1]))
        m_early = _per_token_mass(early_mass, int(early_slice.shape[1]))
        m_self = _per_token_mass(self_mass, int(self_slice.shape[1]))

        metric_tensors["g_prior_vs_self"][layer_idx] = _ratio_from_per_token_mass(m_prior, m_self)
        metric_tensors["g_prev_vs_self"][layer_idx] = _ratio_from_per_token_mass(m_prev, m_self)
        metric_tensors["g_recent_vs_self"][layer_idx] = _ratio_from_per_token_mass(m_recent, m_self)
        metric_tensors["g_early_vs_recent"][layer_idx] = _ratio_from_per_token_mass(m_early, m_recent)

        prev_share = torch.clamp(prev_mass / (prior_mass + EPS), min=0.0, max=1.0)
        metric_tensors["g_prev_share_of_prior"][layer_idx] = prev_share

        metric_tensors["entropy_full"][layer_idx] = _normalized_entropy(full_slice)
        metric_tensors["entropy_prior"][layer_idx] = _normalized_entropy(prior_slice)
        metric_tensors["entropy_self"][layer_idx] = _normalized_entropy(self_slice)
        metric_tensors["top1_full"][layer_idx] = _topk_mass(full_slice, 1)
        metric_tensors["top1_prior"][layer_idx] = _topk_mass(prior_slice, 1)
        metric_tensors["top1_self"][layer_idx] = _topk_mass(self_slice, 1)
        metric_tensors["top5_full"][layer_idx] = _topk_mass(full_slice, 5)
        metric_tensors["top5_prior"][layer_idx] = _topk_mass(prior_slice, 5)
        metric_tensors["top5_self"][layer_idx] = _topk_mass(self_slice, 5)
        metric_tensors["top10_full"][layer_idx] = _topk_mass(full_slice, 10)
        metric_tensors["top10_prior"][layer_idx] = _topk_mass(prior_slice, 10)
        metric_tensors["top10_self"][layer_idx] = _topk_mass(self_slice, 10)
        metric_tensors["herfindahl_full"][layer_idx] = _herfindahl(full_slice)
        metric_tensors["herfindahl_prior"][layer_idx] = _herfindahl(prior_slice)
        metric_tensors["herfindahl_self"][layer_idx] = _herfindahl(self_slice)
        metric_tensors["effective_support_full"][layer_idx] = _effective_support(full_slice)
        metric_tensors["effective_support_prior"][layer_idx] = _effective_support(prior_slice)
        metric_tensors["effective_support_self"][layer_idx] = _effective_support(self_slice)

    return {metric_name: tensor.detach().cpu().numpy() for metric_name, tensor in metric_tensors.items()}


def compute_sentence_activation_metrics(
    hidden_states: Sequence[torch.Tensor],
    sentence_row: Any,
    *,
    recent_window_tokens: int,
) -> dict[str, np.ndarray]:
    q_idx = torch.tensor(sentence_row.token_indices, device=hidden_states[0].device, dtype=torch.long)
    start_token = int(sentence_row.start_token)
    recent_start = max(0, start_token - int(recent_window_tokens))
    prev_start = None if sentence_row.previous_sentence_start_token is None or pd.isna(sentence_row.previous_sentence_start_token) else int(sentence_row.previous_sentence_start_token)
    prev_end = None if sentence_row.previous_sentence_end_token is None or pd.isna(sentence_row.previous_sentence_end_token) else int(sentence_row.previous_sentence_end_token)

    metric_values = {
        metric_name: np.full(len(hidden_states), np.nan, dtype=float)
        for metric_name in ACTIVATION_METRIC_NAMES
    }

    for layer_idx, layer_hidden in enumerate(hidden_states):
        hidden = layer_hidden[0].to(dtype=torch.float32)

        sentence_hidden = hidden[q_idx]
        prior_hidden = _safe_slice_hidden(hidden, 0, start_token)
        recent_hidden = _safe_slice_hidden(hidden, recent_start, start_token)
        prev_hidden = (
            _safe_slice_hidden(hidden, prev_start, prev_end + 1)
            if prev_start is not None and prev_end is not None
            else hidden[:0]
        )

        sentence_pool = _safe_mean_pool(sentence_hidden)
        prior_pool = _safe_mean_pool(prior_hidden)
        recent_pool = _safe_mean_pool(recent_hidden)
        prev_pool = _safe_mean_pool(prev_hidden)

        metric_values["act_cos_prev"][layer_idx] = _cosine_similarity_unit_interval(sentence_pool, prev_pool)
        metric_values["act_cos_prior"][layer_idx] = _cosine_similarity_unit_interval(sentence_pool, prior_pool)
        metric_values["act_cos_recent"][layer_idx] = _cosine_similarity_unit_interval(sentence_pool, recent_pool)

        sentence_norm_mean, sentence_norm_std = _token_norm_summary(sentence_hidden)
        prior_norm_mean, prior_norm_std = _token_norm_summary(prior_hidden)
        metric_values["act_norm_mean_vs_prior"][layer_idx] = _bounded_ratio_scalar(sentence_norm_mean, prior_norm_mean)
        metric_values["act_norm_std_vs_prior"][layer_idx] = _bounded_ratio_scalar(sentence_norm_std, prior_norm_std)
        metric_values["act_token_cos_cohesion"][layer_idx] = _mean_pairwise_cosine_cohesion(sentence_hidden)
        effective_rank, pc1_explained = _effective_rank_and_pc1_explained(sentence_hidden)
        metric_values["act_effective_rank"][layer_idx] = effective_rank
        metric_values["act_pc1_explained"][layer_idx] = pc1_explained

    return metric_values


def _summarize_head_values(head_values: np.ndarray) -> dict[str, float]:
    valid = np.asarray(head_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return {agg_name: float("nan") for agg_name in LAYER_AGG_NAMES}
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def build_layer_summary_df(metric_tensors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_name, metric_tensor in metric_tensors.items():
        for layer_idx, layer_values in enumerate(np.asarray(metric_tensor, dtype=float)):
            summary = _summarize_head_values(layer_values)
            rows.append(
                {
                    "metric": metric_name,
                    "layer_idx": int(layer_idx),
                    "head_mean": summary["mean"],
                    "head_std": summary["std"],
                    "head_min": summary["min"],
                    "head_max": summary["max"],
                }
            )
    return pd.DataFrame(rows)


def build_base_feature_record(
    *,
    example_id: str,
    sentence_row: Any,
    metric_tensors: dict[str, np.ndarray],
    activation_metric_values: Optional[dict[str, np.ndarray]],
    prompt_token_count: int,
) -> dict[str, Any]:
    feature_row: dict[str, Any] = {
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

    for metric_name, metric_tensor in metric_tensors.items():
        for layer_idx, layer_values in enumerate(np.asarray(metric_tensor, dtype=float)):
            summary = _summarize_head_values(layer_values)
            for agg_name, value in summary.items():
                feature_row[f"{metric_name}_{agg_name}_l{layer_idx}"] = value

    if activation_metric_values:
        for metric_name, metric_values in activation_metric_values.items():
            for layer_idx, value in enumerate(np.asarray(metric_values, dtype=float)):
                feature_row[f"{metric_name}_mean_l{layer_idx}"] = float(value) if np.isfinite(value) else float("nan")

    return feature_row


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


def add_transition_features(
    base_feature_df: pd.DataFrame,
    *,
    num_layers: int,
) -> pd.DataFrame:
    df = base_feature_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    new_columns: dict[str, pd.Series] = {}

    for layer_idx in range(int(num_layers)):
        for metric_name in CHANGE_TARGET_METRIC_NAMES:
            for agg_name in CHANGE_TARGET_AGG_NAMES:
                column = f"{metric_name}_{agg_name}_l{layer_idx}"
                current = pd.to_numeric(df[column], errors="coerce").astype(float)
                prev = current.shift(1)
                prev_running_mean = current.expanding(min_periods=1).mean().shift(1)
                prev_running_min = current.cummin().shift(1)
                prev_running_max = current.cummax().shift(1)
                bounded_prev_change = _bounded_relative_change(current, prev)

                new_columns[f"delta_{metric_name}_{agg_name}_l{layer_idx}"] = (current - prev)/(abs(current)+abs(prev))
                new_columns[f"devrun_{metric_name}_{agg_name}_l{layer_idx}"] = (current - prev_running_mean)/(abs(current) + abs(prev_running_mean))
                # Keep the historical column name, but emit a bounded relative change for portability.
                new_columns[f"logratio_prev_{metric_name}_{agg_name}_l{layer_idx}"] = bounded_prev_change
                new_columns[f"min_gap_{metric_name}_{agg_name}_l{layer_idx}"] = (current - prev_running_min)/(abs(current) + abs(prev_running_min))
                new_columns[f"max_gap_{metric_name}_{agg_name}_l{layer_idx}"] = (current - prev_running_max)/(abs(current) + abs(prev_running_max))

        for metric_name in ACTIVATION_CHANGE_TARGET_METRIC_NAMES:
            for agg_name in ACTIVATION_CHANGE_TARGET_AGG_NAMES:
                column = f"{metric_name}_{agg_name}_l{layer_idx}"
                current = pd.to_numeric(df[column], errors="coerce").astype(float)
                prev = current.shift(1)
                prev_running_mean = current.expanding(min_periods=1).mean().shift(1)
                prev_running_min = current.cummin().shift(1)
                prev_running_max = current.cummax().shift(1)
                bounded_prev_change = _bounded_relative_change(current, prev)

                new_columns[f"delta_{metric_name}_{agg_name}_l{layer_idx}"] = (current - prev).clip(-1.0, 1.0)
                new_columns[f"devrun_{metric_name}_{agg_name}_l{layer_idx}"] = (current - prev_running_mean).clip(-1.0, 1.0)
                new_columns[f"logratio_prev_{metric_name}_{agg_name}_l{layer_idx}"] = bounded_prev_change
                new_columns[f"min_gap_{metric_name}_{agg_name}_l{layer_idx}"] = (current - prev_running_min).clip(-1.0, 1.0)
                new_columns[f"max_gap_{metric_name}_{agg_name}_l{layer_idx}"] = (current - prev_running_max).clip(-1.0, 1.0)

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
    num_layers: int,
) -> pd.DataFrame:
    df = feature_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    new_columns: dict[str, pd.Series] = {}

    for layer_idx in range(int(num_layers)):
        for metric_name in WITHIN_TRACE_NORM_METRIC_NAMES:
            for agg_name in WITHIN_TRACE_NORM_AGG_NAMES:
                column = f"{metric_name}_{agg_name}_l{layer_idx}"
                values = pd.to_numeric(df[column], errors="coerce").astype(float).to_numpy(dtype=float)
                new_columns[f"z_{metric_name}_{agg_name}_l{layer_idx}"] = pd.Series(_causal_zscore(values), index=df.index)
                new_columns[f"pct_{metric_name}_{agg_name}_l{layer_idx}"] = pd.Series(_causal_percentile(values), index=df.index)

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def compute_attention_features(
    attentions: Sequence[torch.Tensor],
    hidden_states: Optional[Sequence[torch.Tensor]],
    aligned_sentence_df: pd.DataFrame,
    *,
    example_id: str,
    prompt_token_count: int,
    recent_window_tokens: int,
) -> pd.DataFrame:
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
        working_df = add_attention_region_columns(
            working_df,
            recent_window_tokens=recent_window_tokens,
        )

    base_records: list[dict[str, Any]] = []
    for row in working_df.itertuples():
        metric_tensors = compute_sentence_metric_tensors(
            attentions,
            row,
            recent_window_tokens=recent_window_tokens,
        )
        activation_metric_values = (
            compute_sentence_activation_metrics(
                hidden_states,
                row,
                recent_window_tokens=recent_window_tokens,
            )
            if hidden_states is not None
            else None
        )
        base_records.append(
            build_base_feature_record(
                example_id=example_id,
                sentence_row=row,
                metric_tensors=metric_tensors,
                activation_metric_values=activation_metric_values,
                prompt_token_count=prompt_token_count,
            )
        )

    if not base_records:
        return build_empty_feature_frame(num_layers=len(attentions))

    feature_df = pd.DataFrame(base_records)
    feature_df = add_transition_features(feature_df, num_layers=len(attentions))
    feature_df = add_within_trace_normalization(feature_df, num_layers=len(attentions))
    return feature_df


def extract_example_feature_df(
    *,
    example: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    recent_window_tokens: int,
    num_layers: int,
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

    token_alignment = tokenize_and_align_localized_sentences(
        tokenizer=tokenizer,
        full_text=full_text,
        sentence_df=localized_sentence_df,
        raw_text_start_char=0,
    )
    input_ids_list = token_alignment.input_ids
    if not input_ids_list:
        raise ExampleValidationError("no_tokens", f"{example_id} tokenized to zero tokens.")

    aligned_sentence_df = token_alignment.aligned_sentence_df
    if not (aligned_sentence_df["token_count"] > 0).all():
        bad_count = int((aligned_sentence_df["token_count"] == 0).sum())
        raise ExampleValidationError(
            "unmapped_sentence",
            f"{example_id} has {bad_count} localized sentences that failed to map to tokens.",
        )

    aligned_sentence_df = add_attention_region_columns(
        aligned_sentence_df,
        recent_window_tokens=recent_window_tokens,
    )
    modeling_sentence_df = aligned_sentence_df.loc[aligned_sentence_df["start_token"].fillna(0).astype(int) > 0].copy()
    if modeling_sentence_df.empty:
        return build_empty_feature_frame(num_layers=num_layers)

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True, output_hidden_states=True, use_cache=False)
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states[1:] if getattr(outputs, "hidden_states", None) is not None else None
        feature_df = compute_attention_features(
            attentions,
            hidden_states,
            modeling_sentence_df,
            example_id=example_id,
            prompt_token_count=0,
            recent_window_tokens=recent_window_tokens,
        )
    finally:
        if "outputs" in locals():
            del outputs
        if "attentions" in locals():
            del attentions
        if "hidden_states" in locals():
            del hidden_states
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

    all_localization_paths = iter_localization_paths(
        dataset_paths.localization_dir,
        max_examples=int(args.max_examples),
    )
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
    num_heads = int(config.num_attention_heads)
    ordered_columns = METADATA_COLUMNS + build_feature_columns(num_layers)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
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

    attention_base_feature_count = int(num_layers) * len(BASE_METRIC_NAMES) * len(LAYER_AGG_NAMES)
    activation_base_feature_count = int(num_layers) * len(ACTIVATION_METRIC_NAMES)
    attention_change_feature_count = int(num_layers) * len(CHANGE_TARGET_METRIC_NAMES) * len(CHANGE_TARGET_AGG_NAMES) * 6
    activation_change_feature_count = int(num_layers) * len(ACTIVATION_CHANGE_TARGET_METRIC_NAMES) * len(ACTIVATION_CHANGE_TARGET_AGG_NAMES) * 6
    base_feature_count = len(build_base_feature_columns(num_layers))
    change_feature_count = len(build_change_feature_columns(num_layers))
    normalized_feature_count = len(build_normalized_feature_columns(num_layers))

    print(f"Dataset dir: {dataset_paths.dataset_dir}")
    print(f"Localization dir: {dataset_paths.localization_dir}")
    print(f"Output parquet: {dataset_paths.output_path}")
    print(f"Model id: {model_id}")
    print(f"Device: {device}")
    print(f"Model dtype: {model_dtype}")
    print(f"Layers: {num_layers} | Heads: {num_heads}")
    print(f"Shard: {int(args.shard_id) + 1}/{int(args.num_shards)}")
    print(
        "Feature columns: "
        f"{attention_base_feature_count} attention-base + "
        f"{activation_base_feature_count} activation-base + "
        f"{attention_change_feature_count} attention-change + "
        f"{activation_change_feature_count} activation-change + "
        f"{normalized_feature_count} within-trace = {len(ordered_columns) - len(METADATA_COLUMNS)}"
    )
    print(f"Recent window tokens: {int(args.recent_window_tokens)}")
    print(f"Slope window: {DEFAULT_SLOPE_WINDOW} | Ratio epsilon: {EPS}")
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
                flush_feature_buffer(
                    writer,
                    buffered_frames,
                    ordered_columns=ordered_columns,
                )

            if int(args.progress_every) > 0 and processed % int(args.progress_every) == 0:
                buffered_row_count = sum(len(df) for df in buffered_frames)
                print(
                    f"Processed {processed}/{len(localization_paths)} files | "
                    f"successful={successful} | skipped={sum(skip_counts.values())} | "
                    f"rows_buffered_or_written={writer.rows_written + buffered_row_count}"
                )

        flush_feature_buffer(
            writer,
            buffered_frames,
            ordered_columns=ordered_columns,
        )

        if writer.rows_written == 0:
            writer.write(
                coerce_feature_frame_columns(
                    build_empty_feature_frame(num_layers=num_layers),
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

    print(f"Wrote attention/activation features to: {dataset_paths.output_path}")
    print(f"Processed files: {processed}")
    print(f"Examples with output rows: {successful}")
    print(f"Total parquet rows: {writer.rows_written}")
    if skip_counts:
        print("Skipped examples by reason:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
