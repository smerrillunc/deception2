#!/usr/bin/env python3
"""
Sentence-prefix baselines for predicting

    y_i = p(deceptive continuation | prefix up to sentence i)

The baselines are intentionally different:
1. text-only verifier:
   a black-box text baseline that only sees the rendered full prefix text
2. uncertainty-only:
   an output-confidence baseline built from token-level uncertainty features
3. hidden-state recent-sentence:
   a gray-box latent-state baseline built from recent sentence embeddings

Important design choices:
- All uncertainty and hidden-state features are computed from a forward pass on the
  FULL model context: prompt/instructions + reasoning text up to the current point.
- For the hidden-state baseline, we still run the model on the full context, but we
  only extract representations for the most recent K sentences. This keeps the
  baseline focused on local commitment dynamics while staying faithful to the causal
  context used by the model.
- Uncertainty matrices have variable length because prefixes contain different
  numbers of tokens. This is handled in two standard ways:
  * pooled baseline: summarize the n x d token matrix into a fixed vector
  * sequence baseline: pad + mask the variable-length matrix and train a GRU
- K=5 recent sentences is the default because it is usually large enough to capture
  local reasoning dynamics without making the fixed-vector representation explode.
  The code keeps K configurable so 3 / 5 / 8 sentence comparisons are easy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, classification_report, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None

from attention_features import (
    DatasetPaths,
    ExampleValidationError,
    StreamingParquetWriter,
    add_span_match_columns,
    build_localized_sentence_df,
    cleanup_tensors,
    count_tokens_before_char_boundary,
    infer_model_id,
    iter_localization_paths,
    resolve_device,
    resolve_dtype,
    tokenize_and_align_localized_sentences,
)


DEFAULT_CACHE_DIRNAME = "prefix_target_baseline_cache"
DEFAULT_DATASET_FRAME_NAME = "prefix_target_dataset.parquet"
DEFAULT_EXAMPLE_CACHE_DIRNAME = "example_forward_caches"
DEFAULT_TEXT_EMBED_DIRNAME = "text_embedding_cache"
DEFAULT_SPLIT_FILENAME = "group_splits.parquet"
DEFAULT_TRAIN_OUTPUT_DIRNAME = "baseline_runs"
DEFAULT_TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SENTENCE_EMBED_CACHE_DTYPE = np.float16
DEFAULT_RECENT_SENTENCE_COUNT = 5
DEFAULT_RECENT_POOL_TOKENS = DEFAULT_RECENT_SENTENCE_COUNT
DEFAULT_BATCH_SIZE = 8
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_WRITE_EVERY_EXAMPLES = 16
DEFAULT_SEQUENCE_BATCH_SIZE = 64
DEFAULT_SEQUENCE_HIDDEN_DIM = 64
DEFAULT_SEQUENCE_NUM_EPOCHS = 12
DEFAULT_SEQUENCE_PATIENCE = 3

UNCERTAINTY_FEATURE_NAMES = (
    "token_negative_log_prob",
    "token_entropy",
    "token_top1_top2_margin",
    "token_surprisal_delta",
    "token_realized_rank_log_norm",
)

DATASET_METADATA_COLUMNS = [
    "dataset",
    "model_name",
    "example_id",
    "trace_id",
    "localization_path",
    "cache_path",
    "sentence_idx",
    "sentence_text",
    "prefix_text",
    "full_prefix_text",
    "prompt",
    "target_value",
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


@dataclass(frozen=True)
class PrefixBaselinePaths:
    dataset_paths: DatasetPaths
    cache_dir: Path
    dataset_frame_path: Path
    example_cache_dir: Path
    split_path: Path
    text_embedding_cache_dir: Path
    train_output_dir: Path


@dataclass(frozen=True)
class GroupSplitConfig:
    train_size: float
    val_size: float
    test_size: float
    seed: int
    group_col: str = "trace_id"


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    pearson: float
    spearman: float
    auroc_at_threshold: float
    average_precision_at_threshold: float


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    average_precision: float


@dataclass(frozen=True)
class BaselineResult:
    baseline_name: str
    config: dict[str, Any]
    metrics_by_split: pd.DataFrame
    predictions_df: pd.DataFrame
    output_dir: Path


def maybe_tqdm(iterable: Iterable[Any], *, desc: str, total: Optional[int] = None, disable: bool = False):
    if disable or _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total)


def slugify_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text)).strip("_")


def example_id_to_filename(example_id: str) -> str:
    return slugify_token(example_id.replace("/", "__")) + ".npz"


def resolve_prefix_baseline_paths(
    input_path: str | Path,
    *,
    cache_dir: Optional[str | Path] = None,
) -> PrefixBaselinePaths:
    root = Path(input_path).expanduser().resolve()
    if root.is_dir() and (root / DEFAULT_DATASET_FRAME_NAME).exists():
        inferred_cache_dir = root
        dataset_dir = inferred_cache_dir.parent
        localization_dir = dataset_dir / "localization"
    elif root.is_file() and root.name == DEFAULT_DATASET_FRAME_NAME:
        inferred_cache_dir = root.parent
        dataset_dir = inferred_cache_dir.parent
        localization_dir = dataset_dir / "localization"
    elif root.name == "localization":
        dataset_dir = root.parent
        localization_dir = root
        inferred_cache_dir = dataset_dir / DEFAULT_CACHE_DIRNAME
    else:
        dataset_dir = root
        localization_dir = dataset_dir / "localization"
        inferred_cache_dir = dataset_dir / DEFAULT_CACHE_DIRNAME

    dataset_paths = DatasetPaths(
        dataset_dir=dataset_dir,
        localization_dir=localization_dir,
        output_path=dataset_dir / "unused.parquet",
        examples_path=dataset_dir / "examples.jsonl",
    )
    if not dataset_paths.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_paths.dataset_dir}")
    if not dataset_paths.localization_dir.exists() and cache_dir is None and not inferred_cache_dir.exists():
        raise FileNotFoundError(f"Localization directory does not exist: {dataset_paths.localization_dir}")

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else inferred_cache_dir
    )
    return PrefixBaselinePaths(
        dataset_paths=dataset_paths,
        cache_dir=cache_root,
        dataset_frame_path=cache_root / DEFAULT_DATASET_FRAME_NAME,
        example_cache_dir=cache_root / DEFAULT_EXAMPLE_CACHE_DIRNAME,
        split_path=cache_root / DEFAULT_SPLIT_FILENAME,
        text_embedding_cache_dir=cache_root / DEFAULT_TEXT_EMBED_DIRNAME,
        train_output_dir=cache_root / DEFAULT_TRAIN_OUTPUT_DIRNAME,
    )


def infer_dataset_name(dataset_dir: Path) -> str:
    return str(dataset_dir.parent.name)


def render_full_prefix_context(
    *,
    tokenizer: Any,
    prompt: str,
    prompt_messages: Optional[list[dict[str, Any]]],
    prefix_text: str,
) -> str:
    if not isinstance(prompt_messages, list) or not prompt_messages:
        return f"{prompt}{prefix_text}"

    if not prefix_text:
        try:
            return tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            )
        except Exception:
            return str(prompt)

    messages = list(prompt_messages) + [{"role": "assistant", "content": prefix_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=True,
        )
    except (TypeError, ValueError):
        try:
            base_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            )
            return f"{base_prompt}{prefix_text}"
        except Exception:
            return f"{prompt}{prefix_text}"


def locate_raw_text_span(full_context: str, raw_text: str) -> tuple[int, int]:
    if full_context.endswith(raw_text):
        start = len(full_context) - len(raw_text)
        return start, len(full_context)

    first_idx = full_context.find(raw_text)
    if first_idx >= 0:
        return first_idx, first_idx + len(raw_text)

    raise ExampleValidationError(
        "raw_text_span_not_found",
        "Could not locate raw_text inside the rendered full context.",
    )


def count_prompt_tokens(offset_mapping: Sequence[Sequence[int]], raw_text_start_char: int) -> int:
    return count_tokens_before_char_boundary(offset_mapping, raw_text_start_char)


def shift_localized_sentence_spans(sentence_df: pd.DataFrame, *, raw_text_start_char: int) -> pd.DataFrame:
    out = sentence_df.copy()
    out["full_start"] = pd.to_numeric(out["raw_start"], errors="coerce").astype(int) + int(raw_text_start_char)
    out["full_end"] = pd.to_numeric(out["raw_end"], errors="coerce").astype(int) + int(raw_text_start_char)
    return out


def _safe_mean_pool(hidden_slice: torch.Tensor) -> Optional[torch.Tensor]:
    if hidden_slice.ndim != 2 or hidden_slice.shape[0] <= 0:
        return None
    return hidden_slice.mean(dim=0)


def compute_sentence_representation_cache(
    final_hidden: torch.Tensor,
    aligned_sentence_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    num_sentences = len(aligned_sentence_df)
    hidden_size = int(final_hidden.shape[-1])
    sentence_end = np.full((num_sentences, hidden_size), np.nan, dtype=np.float32)
    sentence_mean = np.full((num_sentences, hidden_size), np.nan, dtype=np.float32)

    for row_idx, row in enumerate(aligned_sentence_df.itertuples()):
        end_token = int(row.end_token)
        token_indices = list(row.token_indices)
        if 0 <= end_token < final_hidden.shape[0]:
            sentence_end[row_idx] = final_hidden[end_token].detach().cpu().numpy().astype(np.float32, copy=False)
        if token_indices:
            token_index_tensor = torch.tensor(token_indices, device=final_hidden.device, dtype=torch.long)
            pooled = _safe_mean_pool(final_hidden[token_index_tensor])
            if pooled is not None:
                sentence_mean[row_idx] = pooled.detach().cpu().numpy().astype(np.float32, copy=False)

    return sentence_end, sentence_mean


def compute_token_uncertainty_features(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> np.ndarray:
    """
    Build one compact uncertainty-feature vector per token position.

    The first token has no previous-token predictive distribution, so its feature
    row is left as NaN and downstream pooled models use NaN-aware statistics.
    Sequence models later replace NaNs with zeros after padding.
    """
    seq_len = int(input_ids.shape[0])
    out = np.full((seq_len, len(UNCERTAINTY_FEATURE_NAMES)), np.nan, dtype=np.float32)
    if seq_len <= 1:
        return out

    step_logits = logits[:-1].to(dtype=torch.float32)
    target_ids = input_ids[1:]

    log_probs = torch.log_softmax(step_logits, dim=-1)
    probs = torch.softmax(step_logits, dim=-1)
    realized_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    negative_log_prob = -realized_log_probs
    entropy = -(probs * log_probs).sum(dim=-1)
    top2_logits = torch.topk(step_logits, k=2, dim=-1).values
    top12_margin = top2_logits[:, 0] - top2_logits[:, 1]
    realized_logits = step_logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    realized_rank = torch.ones_like(realized_logits, dtype=torch.int64)
    vocab_chunk = 8192
    for start_idx in range(0, int(step_logits.shape[-1]), vocab_chunk):
        chunk = step_logits[:, start_idx : start_idx + vocab_chunk]
        realized_rank = realized_rank + (chunk > realized_logits.unsqueeze(-1)).sum(dim=-1).to(dtype=torch.int64)
    rank_log_norm = torch.log1p(realized_rank.to(dtype=torch.float32)) / math.log(float(step_logits.shape[-1]) + 1.0)

    surprisal_delta = torch.full_like(negative_log_prob, float("nan"))
    if negative_log_prob.shape[0] > 1:
        surprisal_delta[1:] = negative_log_prob[1:] - negative_log_prob[:-1]

    out[1:, 0] = negative_log_prob.detach().cpu().numpy().astype(np.float32, copy=False)
    out[1:, 1] = entropy.detach().cpu().numpy().astype(np.float32, copy=False)
    out[1:, 2] = top12_margin.detach().cpu().numpy().astype(np.float32, copy=False)
    out[1:, 3] = surprisal_delta.detach().cpu().numpy().astype(np.float32, copy=False)
    out[1:, 4] = rank_log_norm.detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def build_example_dataset_rows(
    *,
    dataset_name: str,
    model_name: str,
    localization_path: Path,
    example_id: str,
    prompt: str,
    prompt_messages: Optional[list[dict[str, Any]]],
    aligned_sentence_df: pd.DataFrame,
    prompt_token_count: int,
    tokenizer: Any,
    cache_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aligned_lookup = {
        int(row.sentence_idx): row
        for row in aligned_sentence_df.itertuples()
    }

    for sentence_row in aligned_sentence_df.itertuples():
        prefix_text = str(getattr(sentence_row, "sentence_text"))
        # The localization history already stores prefix_text, but sentence_text-only rows can
        # appear in derived tables; fall back to the full raw prefix implied by full_end when needed.
        full_prefix_text = render_full_prefix_context(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_messages=prompt_messages,
            prefix_text=str(getattr(sentence_row, "prefix_text", prefix_text)),
        )
        aligned = aligned_lookup[int(sentence_row.sentence_idx)]
        rows.append(
            {
                "dataset": dataset_name,
                "model_name": model_name,
                "example_id": example_id,
                "trace_id": example_id,
                "localization_path": str(localization_path),
                "cache_path": str(cache_path),
                "sentence_idx": int(aligned.sentence_idx),
                "sentence_text": str(aligned.sentence_text),
                "prefix_text": str(getattr(sentence_row, "prefix_text", aligned.sentence_text)),
                "full_prefix_text": full_prefix_text,
                "prompt": str(prompt),
                "target_value": float(aligned.deception_rate),
                "deception_rate": float(aligned.deception_rate),
                "num_truthful": int(aligned.num_truthful) if pd.notna(aligned.num_truthful) else pd.NA,
                "num_valid": int(aligned.num_valid) if pd.notna(aligned.num_valid) else pd.NA,
                "raw_start": int(aligned.raw_start),
                "raw_end": int(aligned.raw_end),
                "full_start": int(aligned.full_start),
                "full_end": int(aligned.full_end),
                "start_token": int(aligned.start_token),
                "end_token": int(aligned.end_token),
                "token_count": int(aligned.token_count),
                "context_token_count": int(aligned.context_token_count),
                "prompt_token_count": int(prompt_token_count),
                "raw_text_context_token_count": max(0, int(aligned.start_token) - int(prompt_token_count)),
                "available_token_count": int(aligned.available_token_count),
                "prior_all_token_count": int(aligned.prior_all_token_count),
                "previous_sentence_token_count": int(aligned.previous_sentence_token_count),
                "recent_token_count": int(aligned.recent_token_count),
                "early_token_count": int(aligned.early_token_count),
            }
        )
    return pd.DataFrame(rows, columns=DATASET_METADATA_COLUMNS)


def build_aligned_sentence_frame_for_full_context(
    *,
    example: dict[str, Any],
    tokenizer: Any,
    recent_window_tokens: int,
) -> tuple[pd.DataFrame, str, int, int]:
    example_id = example.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ExampleValidationError("missing_example_id", "Localization example is missing example_id.")

    raw_text = example.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        raise ExampleValidationError("missing_raw_text", f"{example_id} is missing raw_text.")

    prompt = str(example.get("prompt") or "")
    prompt_messages = example.get("prompt_messages")
    full_context = render_full_prefix_context(
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_messages=prompt_messages if isinstance(prompt_messages, list) else None,
        prefix_text=raw_text,
    )
    raw_text_start, _ = locate_raw_text_span(full_context, raw_text)

    localized_sentence_df = build_localized_sentence_df(example)
    if localized_sentence_df.empty:
        raise ExampleValidationError("empty_history", f"{example_id} has no localized history entries.")

    localized_sentence_df = add_span_match_columns(raw_text, localized_sentence_df)
    if not localized_sentence_df["span_matches"].all():
        bad_count = int((~localized_sentence_df["span_matches"]).sum())
        raise ExampleValidationError(
            "span_mismatch",
            f"{example_id} has {bad_count} localized sentence spans that do not match raw_text.",
        )

    history_prefix_lookup = {
        int(hist["sentence_idx_inclusive"]): str(hist.get("prefix_text", ""))
        for hist in sorted(example.get("history", []), key=lambda item: int(item["sentence_idx_inclusive"]))
    }
    shifted_sentence_df = shift_localized_sentence_spans(localized_sentence_df, raw_text_start_char=raw_text_start)
    token_alignment = tokenize_and_align_localized_sentences(
        tokenizer=tokenizer,
        full_text=full_context,
        sentence_df=shifted_sentence_df,
        raw_text_start_char=raw_text_start,
    )
    input_ids_list = token_alignment.input_ids
    if not input_ids_list:
        raise ExampleValidationError("no_tokens", f"{example_id} tokenized to zero tokens.")

    aligned_sentence_df = token_alignment.aligned_sentence_df
    aligned_sentence_df["prefix_text"] = aligned_sentence_df["sentence_idx"].map(history_prefix_lookup)
    if not (aligned_sentence_df["token_count"] > 0).all():
        bad_count = int((aligned_sentence_df["token_count"] == 0).sum())
        raise ExampleValidationError(
            "unmapped_sentence",
            f"{example_id} has {bad_count} localized sentences that failed to map to tokens.",
        )

    from commitment_prefix_features import add_trace_region_columns

    aligned_sentence_df = add_trace_region_columns(
        aligned_sentence_df,
        recent_window_tokens=recent_window_tokens,
    )
    prompt_token_count = int(token_alignment.prompt_token_count)
    return aligned_sentence_df, full_context, prompt_token_count, len(input_ids_list)


def write_example_cache(
    *,
    example_cache_dir: Path,
    example_id: str,
    sentence_idx: np.ndarray,
    start_token: np.ndarray,
    end_token: np.ndarray,
    target_value: np.ndarray,
    prompt_token_count: int,
    token_uncertainty: np.ndarray,
    sentence_end_embeddings: np.ndarray,
    sentence_mean_embeddings: np.ndarray,
) -> Path:
    example_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = example_cache_dir / example_id_to_filename(example_id)
    np.savez_compressed(
        cache_path,
        example_id=np.asarray([example_id], dtype=object),
        sentence_idx=np.asarray(sentence_idx, dtype=np.int64),
        start_token=np.asarray(start_token, dtype=np.int64),
        end_token=np.asarray(end_token, dtype=np.int64),
        target_value=np.asarray(target_value, dtype=np.float32),
        prompt_token_count=np.asarray([prompt_token_count], dtype=np.int64),
        token_uncertainty=np.asarray(token_uncertainty, dtype=np.float32),
        sentence_end_embeddings=np.asarray(
            sentence_end_embeddings,
            dtype=DEFAULT_SENTENCE_EMBED_CACHE_DTYPE,
        ),
        sentence_mean_embeddings=np.asarray(
            sentence_mean_embeddings,
            dtype=DEFAULT_SENTENCE_EMBED_CACHE_DTYPE,
        ),
    )
    return cache_path


def load_example_cache(cache_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(cache_path), allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def extract_example_forward_cache(
    *,
    example: dict[str, Any],
    dataset_name: str,
    model_name: str,
    localization_path: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    recent_window_tokens: int,
    example_cache_dir: Path,
    max_seq_len: int = 10000,
) -> pd.DataFrame:
    example_id = str(example["example_id"])
    prompt = str(example.get("prompt") or "")
    prompt_messages = example.get("prompt_messages")

    aligned_sentence_df, full_context, prompt_token_count, _ = build_aligned_sentence_frame_for_full_context(
        example=example,
        tokenizer=tokenizer,
        recent_window_tokens=recent_window_tokens,
    )

    input_ids_list = tokenizer(full_context, add_special_tokens=False)["input_ids"]
    if len(input_ids_list) > int(max_seq_len):
        input_ids_list = input_ids_list[:int(max_seq_len)]
        # Filter aligned_sentence_df to only include sentences that fit within the truncated sequence
        aligned_sentence_df = aligned_sentence_df[aligned_sentence_df["end_token"] < int(max_seq_len)].reset_index(drop=True)
        if aligned_sentence_df.empty:
            raise ExampleValidationError(
                "seq_truncated_all_sentences",
                f"{example_id} has no sentences within max_seq_len={max_seq_len}.",
            )
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        logits = outputs.logits[0].to(dtype=torch.float32)
        final_hidden = outputs.hidden_states[-1][0].to(dtype=torch.float32)
        token_uncertainty = compute_token_uncertainty_features(logits, input_ids[0])
        sentence_end, sentence_mean = compute_sentence_representation_cache(final_hidden, aligned_sentence_df)
    finally:
        if "outputs" in locals():
            del outputs
        if "logits" in locals():
            del logits
        if "final_hidden" in locals():
            del final_hidden
        del input_ids
        cleanup_tensors()

    cache_path = write_example_cache(
        example_cache_dir=example_cache_dir,
        example_id=example_id,
        sentence_idx=aligned_sentence_df["sentence_idx"].to_numpy(dtype=np.int64, copy=False),
        start_token=aligned_sentence_df["start_token"].to_numpy(dtype=np.int64, copy=False),
        end_token=aligned_sentence_df["end_token"].to_numpy(dtype=np.int64, copy=False),
        target_value=aligned_sentence_df["deception_rate"].to_numpy(dtype=np.float32, copy=False),
        prompt_token_count=prompt_token_count,
        token_uncertainty=token_uncertainty,
        sentence_end_embeddings=sentence_end,
        sentence_mean_embeddings=sentence_mean,
    )

    return build_example_dataset_rows(
        dataset_name=dataset_name,
        model_name=model_name,
        localization_path=localization_path,
        example_id=example_id,
        prompt=prompt,
        prompt_messages=prompt_messages if isinstance(prompt_messages, list) else None,
        aligned_sentence_df=aligned_sentence_df,
        prompt_token_count=prompt_token_count,
        tokenizer=tokenizer,
        cache_path=cache_path,
    )


def coerce_dataset_frame_columns(dataset_df: pd.DataFrame) -> pd.DataFrame:
    df = dataset_df.copy()
    for column_name in DATASET_METADATA_COLUMNS:
        if column_name not in df.columns:
            df[column_name] = pd.NA
    df = df.loc[:, DATASET_METADATA_COLUMNS]

    string_columns = {
        "dataset",
        "model_name",
        "example_id",
        "trace_id",
        "localization_path",
        "cache_path",
        "sentence_text",
        "prefix_text",
        "full_prefix_text",
        "prompt",
    }
    float_columns = {"target_value", "deception_rate"}
    int_columns = set(DATASET_METADATA_COLUMNS) - string_columns - float_columns

    for column_name in sorted(string_columns):
        df[column_name] = df[column_name].astype("string")
    for column_name in sorted(float_columns):
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("float64")
    for column_name in sorted(int_columns):
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("Int64")
    return df


def flush_dataset_buffer(
    writer: StreamingParquetWriter,
    buffer: list[pd.DataFrame],
) -> int:
    if not buffer:
        return 0
    chunk_df = pd.concat(buffer, ignore_index=True)
    buffer.clear()
    chunk_df = coerce_dataset_frame_columns(chunk_df)
    writer.write(chunk_df)
    return len(chunk_df)


def dataset_row_fingerprint(dataset_df: pd.DataFrame) -> str:
    key_df = dataset_df.loc[:, ["example_id", "sentence_idx"]].copy()
    hashed = pd.util.hash_pandas_object(key_df, index=False).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha1(hashed.tobytes()).hexdigest()
    return digest


def build_or_load_group_splits(
    dataset_df: pd.DataFrame,
    split_path: Path,
    *,
    config: GroupSplitConfig,
    overwrite: bool = False,
) -> pd.DataFrame:
    if split_path.exists() and not overwrite:
        split_df = pd.read_parquet(split_path)
        return split_df

    group_values = dataset_df[config.group_col].astype(str).to_numpy(copy=False)
    groups = dataset_df[config.group_col].astype(str)
    unique_groups = groups.drop_duplicates().to_numpy(dtype=object, copy=False)

    if len(unique_groups) < 3:
        raise ValueError("Need at least 3 unique groups to build train/val/test grouped splits.")

    total_fraction = float(config.train_size + config.val_size + config.test_size)
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(
            f"train_size + val_size + test_size must sum to 1.0, got {total_fraction:.6f}"
        )

    temp_size = float(config.val_size + config.test_size)
    if not 0.0 < temp_size < 1.0:
        raise ValueError("val_size + test_size must be in (0, 1).")

    unique_group_df = pd.DataFrame({config.group_col: unique_groups})
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=temp_size, random_state=int(config.seed))
    train_group_idx, temp_group_idx = next(gss_outer.split(unique_group_df, groups=unique_group_df[config.group_col]))
    train_groups = set(unique_group_df.iloc[train_group_idx][config.group_col].astype(str))
    temp_groups = unique_group_df.iloc[temp_group_idx][config.group_col].astype(str).reset_index(drop=True)

    relative_test_size = float(config.test_size / temp_size)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=int(config.seed) + 1)
    val_group_idx, test_group_idx = next(gss_inner.split(temp_groups.to_frame(name=config.group_col), groups=temp_groups))
    val_groups = set(temp_groups.iloc[val_group_idx].astype(str))
    test_groups = set(temp_groups.iloc[test_group_idx].astype(str))

    split_labels: list[str] = []
    for group_name in group_values:
        if group_name in train_groups:
            split_labels.append("train")
        elif group_name in val_groups:
            split_labels.append("val")
        elif group_name in test_groups:
            split_labels.append("test")
        else:
            raise RuntimeError(f"Group {group_name!r} did not land in any split.")

    split_df = dataset_df.loc[:, ["example_id", "sentence_idx", config.group_col]].copy()
    split_df["split"] = split_labels
    split_df.to_parquet(split_path, index=False)
    return split_df


def merge_dataset_with_splits(dataset_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    merged = dataset_df.merge(
        split_df.loc[:, ["example_id", "sentence_idx", "split"]],
        on=["example_id", "sentence_idx"],
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Some dataset rows are missing split assignments.")
    return merged


def pooled_stat_vector(
    token_matrix: np.ndarray,
    *,
    recent_matrix: np.ndarray,
) -> np.ndarray:
    token_matrix = np.asarray(token_matrix, dtype=np.float32)
    recent_matrix = np.asarray(recent_matrix, dtype=np.float32)
    if token_matrix.ndim != 2:
        raise ValueError(f"Expected token_matrix to be 2D, got shape={token_matrix.shape}")
    if recent_matrix.ndim != 2:
        raise ValueError(f"Expected recent_matrix to be 2D, got shape={recent_matrix.shape}")

    with np.errstate(invalid="ignore"):
        full_mean = np.nanmean(token_matrix, axis=0)
        full_std = np.nanstd(token_matrix, axis=0)
        full_max = np.nanmax(token_matrix, axis=0)
        recent_mean = np.nanmean(recent_matrix, axis=0)
        recent_std = np.nanstd(recent_matrix, axis=0)
        recent_max = np.nanmax(recent_matrix, axis=0)

    return np.concatenate([full_mean, full_std, full_max, recent_mean, recent_std, recent_max]).astype(np.float32)


def build_uncertainty_pooled_features(
    dataset_df: pd.DataFrame,
    *,
    recent_window_tokens: int,
) -> np.ndarray:
    out_rows: list[np.ndarray] = []
    for example_id, example_rows in dataset_df.groupby("example_id", sort=False):
        cache = load_example_cache(example_rows["cache_path"].iloc[0])
        token_uncertainty = np.asarray(cache["token_uncertainty"], dtype=np.float32)
        row_lookup = {
            int(sentence_idx): pos
            for pos, sentence_idx in enumerate(cache["sentence_idx"].astype(np.int64))
        }

        for row in example_rows.itertuples():
            cache_row_idx = row_lookup[int(row.sentence_idx)]
            end_token = int(cache["end_token"][cache_row_idx])
            prefix_token_matrix = token_uncertainty[: end_token + 1]
            if int(recent_window_tokens) > 0:
                start_sentence_idx = max(0, cache_row_idx - int(recent_window_tokens) + 1)
                recent_start_token = int(cache["start_token"][start_sentence_idx])
                recent_matrix = token_uncertainty[recent_start_token : end_token + 1]
            else:
                recent_matrix = prefix_token_matrix
            out_rows.append(pooled_stat_vector(prefix_token_matrix, recent_matrix=recent_matrix))

    return np.stack(out_rows, axis=0).astype(np.float32)


def build_uncertainty_sequence_examples(
    dataset_df: pd.DataFrame,
    *,
    max_tokens: int = 0,
) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    for _, example_rows in dataset_df.groupby("example_id", sort=False):
        cache = load_example_cache(example_rows["cache_path"].iloc[0])
        token_uncertainty = np.asarray(cache["token_uncertainty"], dtype=np.float32)
        row_lookup = {
            int(sentence_idx): pos
            for pos, sentence_idx in enumerate(cache["sentence_idx"].astype(np.int64))
        }
        for row in example_rows.itertuples():
            cache_row_idx = row_lookup[int(row.sentence_idx)]
            end_token = int(cache["end_token"][cache_row_idx])
            seq = token_uncertainty[: end_token + 1]
            if int(max_tokens) > 0 and seq.shape[0] > int(max_tokens):
                seq = seq[-int(max_tokens):]
            sequences.append(seq.astype(np.float32, copy=False))
    return sequences


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-6, a_max=None)
    return matrix / norms


def build_recent_sentence_fixed_features(
    dataset_df: pd.DataFrame,
    *,
    k_recent_sentences: int,
    sentence_representation: str,
    l2_normalize: str,
) -> np.ndarray:
    out_rows: list[np.ndarray] = []
    cache_key = {
        "sentence_end": "sentence_end_embeddings",
        "sentence_mean": "sentence_mean_embeddings",
    }[sentence_representation]

    for _, example_rows in dataset_df.groupby("example_id", sort=False):
        cache = load_example_cache(example_rows["cache_path"].iloc[0])
        sentence_embeddings = np.asarray(cache[cache_key], dtype=np.float32)
        row_lookup = {
            int(sentence_idx): pos
            for pos, sentence_idx in enumerate(cache["sentence_idx"].astype(np.int64))
        }
        hidden_size = int(sentence_embeddings.shape[-1])

        for row in example_rows.itertuples():
            cache_row_idx = row_lookup[int(row.sentence_idx)]
            start_idx = max(0, cache_row_idx - int(k_recent_sentences) + 1)
            seq = sentence_embeddings[start_idx : cache_row_idx + 1]
            if l2_normalize == "sentence":
                seq = l2_normalize_rows(seq)

            padded = np.zeros((int(k_recent_sentences), hidden_size), dtype=np.float32)
            padded[-seq.shape[0] :] = seq
            flat = padded.reshape(-1)
            if l2_normalize == "vector":
                denom = max(float(np.linalg.norm(flat)), 1e-6)
                flat = flat / denom
            out_rows.append(flat.astype(np.float32, copy=False))

    return np.stack(out_rows, axis=0).astype(np.float32)


def build_recent_sentence_sequence_examples(
    dataset_df: pd.DataFrame,
    *,
    k_recent_sentences: int,
    sentence_representation: str,
    l2_normalize: str,
) -> list[np.ndarray]:
    sequences: list[np.ndarray] = []
    cache_key = {
        "sentence_end": "sentence_end_embeddings",
        "sentence_mean": "sentence_mean_embeddings",
    }[sentence_representation]

    for _, example_rows in dataset_df.groupby("example_id", sort=False):
        cache = load_example_cache(example_rows["cache_path"].iloc[0])
        sentence_embeddings = np.asarray(cache[cache_key], dtype=np.float32)
        row_lookup = {
            int(sentence_idx): pos
            for pos, sentence_idx in enumerate(cache["sentence_idx"].astype(np.int64))
        }

        for row in example_rows.itertuples():
            cache_row_idx = row_lookup[int(row.sentence_idx)]
            start_idx = max(0, cache_row_idx - int(k_recent_sentences) + 1)
            seq = sentence_embeddings[start_idx : cache_row_idx + 1]
            if l2_normalize == "sentence":
                seq = l2_normalize_rows(seq)
            sequences.append(seq.astype(np.float32, copy=False))

    return sequences


def apply_dense_pca(
    x_train: np.ndarray,
    x_other: Sequence[np.ndarray],
    *,
    pca_dim: int,
) -> tuple[np.ndarray, list[np.ndarray], PCA]:
    pca = PCA(n_components=int(pca_dim), random_state=0)
    x_train_pca = pca.fit_transform(x_train)
    transformed_other = [pca.transform(arr) for arr in x_other]
    return x_train_pca.astype(np.float32), [arr.astype(np.float32) for arr in transformed_other], pca


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    binary_threshold: Optional[float] = None,
) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))

    auroc = float("nan")
    avg_precision = float("nan")
    if binary_threshold is not None:
        y_binary = (y_true > float(binary_threshold)).astype(np.int64)
        if np.unique(y_binary).size == 2:
            auroc = float(roc_auc_score(y_binary, y_pred))
            avg_precision = float(average_precision_score(y_binary, y_pred))

    return RegressionMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        pearson=pearson,
        spearman=spearman,
        auroc_at_threshold=auroc,
        average_precision_at_threshold=avg_precision,
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float32)
    y_pred = (y_pred_proba > 0.5).astype(np.int64)

    accuracy = float((y_pred == y_true).mean())
    precision = float(average_precision_score(y_true, y_pred_proba))
    recall = float(roc_auc_score(y_true, y_pred_proba))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auroc = float(roc_auc_score(y_true, y_pred_proba))
    avg_precision = float(average_precision_score(y_true, y_pred_proba))

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc,
        average_precision=avg_precision,
    )


def fit_ridge_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: dict[str, np.ndarray],
    y_eval: dict[str, np.ndarray],
    *,
    use_standard_scaler: bool = True,
    alpha: float = 1.0,
    binary_threshold: Optional[float] = None,
) -> tuple[Pipeline, pd.DataFrame, dict[str, np.ndarray]]:
    scaler_step: StandardScaler | str = StandardScaler() if use_standard_scaler else "passthrough"
    model = Pipeline(
        [
            ("scaler", scaler_step),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(x_train, y_train)

    metrics_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for split_name, x_split in x_eval.items():
        y_pred = model.predict(x_split).astype(np.float32)
        predictions[split_name] = y_pred
        metrics = compute_regression_metrics(y_eval[split_name], y_pred, binary_threshold=binary_threshold)
        metrics_rows.append(
            {
                "split": split_name,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "pearson": metrics.pearson,
                "spearman": metrics.spearman,
                "auroc_at_threshold": metrics.auroc_at_threshold,
                "average_precision_at_threshold": metrics.average_precision_at_threshold,
            }
        )

    return model, pd.DataFrame(metrics_rows), predictions


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: dict[str, np.ndarray],
    y_eval: dict[str, np.ndarray],
    *,
    use_standard_scaler: bool = True,
    max_iter: int = 1000,
) -> tuple[Pipeline, pd.DataFrame, dict[str, np.ndarray]]:
    scaler_step: StandardScaler | str = StandardScaler() if use_standard_scaler else "passthrough"
    model = Pipeline(
        [
            ("scaler", scaler_step),
            ("logistic", LogisticRegression(random_state=42, max_iter=int(max_iter))),
        ]
    )
    model.fit(x_train, y_train)

    metrics_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for split_name, x_split in x_eval.items():
        y_pred_proba = model.predict_proba(x_split)[:, 1].astype(np.float32)
        predictions[split_name] = y_pred_proba
        metrics_class = compute_classification_metrics(y_eval[split_name], y_pred_proba)
        metrics_reg = compute_regression_metrics(y_eval[split_name], y_pred_proba, binary_threshold=0.5)
        metrics_rows.append(
            {
                "split": split_name,
                **metrics_class.__dict__,
                **metrics_reg.__dict__,
            }
        )

    return model, pd.DataFrame(metrics_rows), predictions


class SequenceRegressionDataset(Dataset):
    def __init__(self, sequences: Sequence[np.ndarray], targets: np.ndarray) -> None:
        self.sequences = [np.asarray(seq, dtype=np.float32) for seq in sequences]
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        return self.sequences[idx], self.targets[idx]


def collate_padded_sequences(batch: Sequence[tuple[np.ndarray, np.float32]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, targets = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    feature_dim = int(sequences[0].shape[1]) if sequences else 0
    max_len = int(max(lengths).item()) if len(lengths) else 0

    padded = torch.zeros((len(sequences), max_len, feature_dim), dtype=torch.float32)
    for row_idx, seq in enumerate(sequences):
        seq_tensor = torch.from_numpy(np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0))
        padded[row_idx, : seq_tensor.shape[0]] = seq_tensor
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return padded, lengths, target_tensor


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        final_hidden = hidden[-1]
        return self.head(final_hidden).squeeze(-1)


def fit_gru_regression(
    *,
    train_sequences: Sequence[np.ndarray],
    train_targets: np.ndarray,
    val_sequences: Sequence[np.ndarray],
    val_targets: np.ndarray,
    test_sequences: Sequence[np.ndarray],
    test_targets: np.ndarray,
    device: str,
    hidden_dim: int,
    batch_size: int,
    num_epochs: int,
    patience: int,
    learning_rate: float = 1e-3,
    binary_threshold: Optional[float] = None,
    show_progress: bool = False,
    progress_desc: str = "GRU epochs",
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    input_dim = int(train_sequences[0].shape[1])
    model = GRURegressor(input_dim=input_dim, hidden_dim=int(hidden_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        SequenceRegressionDataset(train_sequences, train_targets),
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate_padded_sequences,
    )
    eval_sets = {
        "train": (list(train_sequences), np.asarray(train_targets, dtype=np.float32)),
        "val": (list(val_sequences), np.asarray(val_targets, dtype=np.float32)),
        "test": (list(test_sequences), np.asarray(test_targets, dtype=np.float32)),
    }

    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val_rmse = float("inf")
    stale_epochs = 0

    epoch_iter = maybe_tqdm(
        range(int(num_epochs)),
        desc=progress_desc,
        total=int(num_epochs),
        disable=not show_progress,
    )
    for _epoch in epoch_iter:
        model.train()
        for padded, lengths, targets in train_loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(padded, lengths)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = predict_sequence_regressor(model, val_sequences, device=device, batch_size=batch_size)
        val_metrics = compute_regression_metrics(val_targets, val_predictions, binary_threshold=binary_threshold)
        if val_metrics.rmse < best_val_rmse:
            best_val_rmse = val_metrics.rmse
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    predictions: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, Any]] = []
    for split_name, (seqs, y_true) in eval_sets.items():
        y_pred = predict_sequence_regressor(model, seqs, device=device, batch_size=batch_size)
        predictions[split_name] = y_pred
        metrics = compute_regression_metrics(y_true, y_pred, binary_threshold=binary_threshold)
        metrics_rows.append(
            {
                "split": split_name,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "pearson": metrics.pearson,
                "spearman": metrics.spearman,
                "auroc_at_threshold": metrics.auroc_at_threshold,
                "average_precision_at_threshold": metrics.average_precision_at_threshold,
            }
        )

    return (
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_dim": input_dim,
            "hidden_dim": int(hidden_dim),
        },
        pd.DataFrame(metrics_rows),
        predictions,
    )


def predict_sequence_regressor(
    model: GRURegressor,
    sequences: Sequence[np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        SequenceRegressionDataset(sequences, np.zeros(len(sequences), dtype=np.float32)),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_padded_sequences,
    )
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for padded, lengths, _targets in loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            pred = model(padded, lengths).detach().cpu().numpy().astype(np.float32, copy=False)
            preds.append(pred)
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)


class HFTextEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model.to(device)
        self.model.eval()

    def encode(self, texts: Sequence[str], *, batch_size: int, show_progress: bool = False) -> np.ndarray:
        batches: list[np.ndarray] = []
        batch_starts = range(0, len(texts), int(batch_size))
        batch_starts = maybe_tqdm(
            batch_starts,
            desc="Text embeddings",
            total=len(range(0, len(texts), int(batch_size))),
            disable=not show_progress,
        )
        for start_idx in batch_starts:
            batch_texts = list(texts[start_idx : start_idx + int(batch_size)])
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                hidden = outputs.last_hidden_state.to(dtype=torch.float32)
                attn_mask = batch["attention_mask"].unsqueeze(-1).to(dtype=torch.float32)
                pooled = (hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp_min(1e-6)
                pooled = F.normalize(pooled, p=2, dim=1)
            batches.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
            cleanup_tensors()
        return np.concatenate(batches, axis=0) if batches else np.zeros((0, 0), dtype=np.float32)


def compute_or_load_text_embeddings(
    dataset_df: pd.DataFrame,
    *,
    cache_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    show_progress: bool = False,
    trust_remote_code: bool = False,
    overwrite: bool = False,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_slug = slugify_token(model_name)
    fingerprint = dataset_row_fingerprint(dataset_df)
    embedding_path = cache_dir / f"{model_slug}__{fingerprint}.npy"
    meta_path = cache_dir / f"{model_slug}__{fingerprint}.json"

    if embedding_path.exists() and meta_path.exists() and not overwrite:
        return np.load(embedding_path)

    embedder = HFTextEmbedder(
        model_name=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    embeddings = embedder.encode(
        dataset_df["full_prefix_text"].astype(str).tolist(),
        batch_size=int(batch_size),
        show_progress=show_progress,
    )
    np.save(embedding_path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "fingerprint": fingerprint,
                "num_rows": int(len(dataset_df)),
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings


def split_masks(dataset_df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        split_name: (dataset_df["split"].to_numpy(copy=False) == split_name)
        for split_name in ("train", "val", "test")
    }


def save_baseline_outputs(
    *,
    output_dir: Path,
    baseline_name: str,
    config: dict[str, Any],
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    sklearn_model: Any = None,
    pca_model: Any = None,
    torch_bundle: Optional[dict[str, Any]] = None,
    vectorizer: Any = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_parquet(output_dir / "predictions.parquet", index=False)
    if sklearn_model is not None:
        dump(sklearn_model, output_dir / "model.joblib")
    if pca_model is not None:
        dump(pca_model, output_dir / "pca.joblib")
    if vectorizer is not None:
        dump(vectorizer, output_dir / "vectorizer.joblib")
    if torch_bundle is not None:
        torch.save(torch_bundle, output_dir / "model.pt")


def attach_predictions(
    dataset_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_name, y_pred in predictions.items():
        split_df = dataset_df.loc[dataset_df["split"] == split_name, ["dataset", "example_id", "sentence_idx", "target_value", "split"]].copy()
        split_df["prediction"] = y_pred
        rows.append(split_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["dataset", "example_id", "sentence_idx", "target_value", "split", "prediction"]
    )


def run_text_embedding_ridge_baseline(
    dataset_df: pd.DataFrame,
    *,
    output_dir: Path,
    embedding_cache_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    binary_threshold: Optional[float],
    show_progress: bool = False,
    trust_remote_code: bool = False,
    overwrite_embedding_cache: bool = False,
) -> BaselineResult:
    embeddings = compute_or_load_text_embeddings(
        dataset_df,
        cache_dir=embedding_cache_dir,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
        trust_remote_code=trust_remote_code,
        overwrite=overwrite_embedding_cache,
    )
    masks = split_masks(dataset_df)
    model, metrics_df, predictions = fit_ridge_regression(
        x_train=embeddings[masks["train"]],
        y_train=dataset_df.loc[masks["train"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        x_eval={split: embeddings[mask] for split, mask in masks.items()},
        y_eval={
            split: dataset_df.loc[mask, "target_value"].to_numpy(dtype=np.float32, copy=False)
            for split, mask in masks.items()
        },
        use_standard_scaler=True,
        binary_threshold=binary_threshold,
    )
    predictions_df = attach_predictions(dataset_df, predictions)
    config = {
        "baseline_name": "text_embedding_ridge",
        "text_embedding_model": model_name,
        "batch_size": int(batch_size),
        "binary_threshold": binary_threshold,
    }
    save_baseline_outputs(
        output_dir=output_dir,
        baseline_name="text_embedding_ridge",
        config=config,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        sklearn_model=model,
    )
    return BaselineResult("text_embedding_ridge", config, metrics_df, predictions_df, output_dir)


def run_text_cbow_ridge_baseline(
    dataset_df: pd.DataFrame,
    *,
    output_dir: Path,
    max_features: int,
    ngram_range: tuple[int, int],
    binary_threshold: Optional[float],
    text_column: str = "full_prefix_text",
    baseline_name: str = "text_cbow_ridge",
) -> BaselineResult:
    if text_column not in dataset_df.columns:
        raise ValueError(f"CBOW baseline requested missing text column: {text_column}")

    masks = split_masks(dataset_df)
    vectorizer = CountVectorizer(
        max_features=int(max_features),
        ngram_range=tuple(int(x) for x in ngram_range),
        lowercase=True,
    )
    x_train = vectorizer.fit_transform(dataset_df.loc[masks["train"], text_column].astype(str))
    x_val = vectorizer.transform(dataset_df.loc[masks["val"], text_column].astype(str))
    x_test = vectorizer.transform(dataset_df.loc[masks["test"], text_column].astype(str))
    model, metrics_df, predictions = fit_ridge_regression(
        x_train=x_train,
        y_train=dataset_df.loc[masks["train"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        x_eval={"train": x_train, "val": x_val, "test": x_test},
        y_eval={
            "train": dataset_df.loc[masks["train"], "target_value"].to_numpy(dtype=np.float32, copy=False),
            "val": dataset_df.loc[masks["val"], "target_value"].to_numpy(dtype=np.float32, copy=False),
            "test": dataset_df.loc[masks["test"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        },
        use_standard_scaler=False,
        binary_threshold=binary_threshold,
    )
    predictions_df = attach_predictions(dataset_df, predictions)
    config = {
        "baseline_name": str(baseline_name),
        "text_column": str(text_column),
        "max_features": int(max_features),
        "ngram_range": list(ngram_range),
        "binary_threshold": binary_threshold,
    }
    save_baseline_outputs(
        output_dir=output_dir,
        baseline_name=str(baseline_name),
        config=config,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        sklearn_model=model,
        vectorizer=vectorizer,
    )
    return BaselineResult(str(baseline_name), config, metrics_df, predictions_df, output_dir)


def run_uncertainty_pooled_ridge_baseline(
    dataset_df: pd.DataFrame,
    *,
    output_dir: Path,
    recent_window_tokens: int,
    binary_threshold: Optional[float],
    pca_dim: int = 0,
) -> BaselineResult:
    features = build_uncertainty_pooled_features(dataset_df, recent_window_tokens=recent_window_tokens)
    masks = split_masks(dataset_df)
    if pca_dim > 0:
        x_train_pca, x_eval_pca, pca_model = apply_dense_pca(
            x_train=features[masks["train"]],
            x_other=[features[masks["val"]], features[masks["test"]]],
            pca_dim=pca_dim,
        )
        features = np.concatenate([x_train_pca] + x_eval_pca, axis=0)
    model, metrics_df, predictions = fit_ridge_regression(
        x_train=features[masks["train"]],
        y_train=dataset_df.loc[masks["train"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        x_eval={split: features[mask] for split, mask in masks.items()},
        y_eval={
            split: dataset_df.loc[mask, "target_value"].to_numpy(dtype=np.float32, copy=False)
            for split, mask in masks.items()
        },
        use_standard_scaler=True,
        binary_threshold=binary_threshold,
    )
    predictions_df = attach_predictions(dataset_df, predictions)
    config = {
        "baseline_name": "uncertainty_pooled_ridge",
        "recent_window_tokens": int(recent_window_tokens),
        "uncertainty_features": list(UNCERTAINTY_FEATURE_NAMES),
        "binary_threshold": binary_threshold,
        "pca_dim": pca_dim,
    }
    save_baseline_outputs(
        output_dir=output_dir,
        baseline_name="uncertainty_pooled_ridge",
        config=config,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        sklearn_model=model,
        pca_model=pca_model if pca_dim > 0 else None,
    )
    return BaselineResult("uncertainty_pooled_ridge", config, metrics_df, predictions_df, output_dir)


def run_uncertainty_gru_baseline(
    dataset_df: pd.DataFrame,
    *,
    output_dir: Path,
    device: str,
    max_tokens: int,
    hidden_dim: int,
    batch_size: int,
    num_epochs: int,
    patience: int,
    binary_threshold: Optional[float],
    show_progress: bool = False,
) -> BaselineResult:
    sequences = build_uncertainty_sequence_examples(dataset_df, max_tokens=max_tokens)
    masks = split_masks(dataset_df)
    train_sequences = [seq for seq, keep in zip(sequences, masks["train"]) if keep]
    val_sequences = [seq for seq, keep in zip(sequences, masks["val"]) if keep]
    test_sequences = [seq for seq, keep in zip(sequences, masks["test"]) if keep]
    torch_bundle, metrics_df, predictions = fit_gru_regression(
        train_sequences=train_sequences,
        train_targets=dataset_df.loc[masks["train"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        val_sequences=val_sequences,
        val_targets=dataset_df.loc[masks["val"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        test_sequences=test_sequences,
        test_targets=dataset_df.loc[masks["test"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        device=device,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=patience,
        binary_threshold=binary_threshold,
        show_progress=show_progress,
        progress_desc="Uncertainty GRU",
    )
    predictions_df = attach_predictions(dataset_df, predictions)
    config = {
        "baseline_name": "uncertainty_gru",
        "max_tokens": int(max_tokens),
        "hidden_dim": int(hidden_dim),
        "batch_size": int(batch_size),
        "num_epochs": int(num_epochs),
        "patience": int(patience),
        "binary_threshold": binary_threshold,
        "uncertainty_features": list(UNCERTAINTY_FEATURE_NAMES),
    }
    save_baseline_outputs(
        output_dir=output_dir,
        baseline_name="uncertainty_gru",
        config=config,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        torch_bundle=torch_bundle,
    )
    return BaselineResult("uncertainty_gru", config, metrics_df, predictions_df, output_dir)


def run_hidden_recent_fixed_ridge_baseline(
    dataset_df: pd.DataFrame,
    *,
    output_dir: Path,
    k_recent_sentences: int,
    sentence_representation: str,
    l2_normalize: str,
    pca_dim: int,
    binary_threshold: Optional[float],
) -> BaselineResult:
    features = build_recent_sentence_fixed_features(
        dataset_df,
        k_recent_sentences=k_recent_sentences,
        sentence_representation=sentence_representation,
        l2_normalize=l2_normalize,
    )
    masks = split_masks(dataset_df)
    train_features = features[masks["train"]]
    eval_feature_map = {split: features[mask] for split, mask in masks.items()}
    pca_model = None
    if int(pca_dim) > 0 and int(pca_dim) < train_features.shape[1]:
        train_features, other_arrays, pca_model = apply_dense_pca(
            train_features,
            [eval_feature_map["val"], eval_feature_map["test"]],
            pca_dim=int(pca_dim),
        )
        eval_feature_map = {
            "train": train_features,
            "val": other_arrays[0],
            "test": other_arrays[1],
        }

    model, metrics_df, predictions = fit_ridge_regression(
        x_train=eval_feature_map["train"],
        y_train=dataset_df.loc[masks["train"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        x_eval=eval_feature_map,
        y_eval={
            split: dataset_df.loc[mask, "target_value"].to_numpy(dtype=np.float32, copy=False)
            for split, mask in masks.items()
        },
        use_standard_scaler=True,
        binary_threshold=binary_threshold,
    )
    predictions_df = attach_predictions(dataset_df, predictions)
    config = {
        "baseline_name": "hidden_recent_fixed_ridge",
        "k_recent_sentences": int(k_recent_sentences),
        "sentence_representation": sentence_representation,
        "l2_normalize": l2_normalize,
        "pca_dim": int(pca_dim),
        "binary_threshold": binary_threshold,
    }
    save_baseline_outputs(
        output_dir=output_dir,
        baseline_name="hidden_recent_fixed_ridge",
        config=config,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        sklearn_model=model,
        pca_model=pca_model,
    )
    return BaselineResult("hidden_recent_fixed_ridge", config, metrics_df, predictions_df, output_dir)


def run_hidden_recent_gru_baseline(
    dataset_df: pd.DataFrame,
    *,
    output_dir: Path,
    device: str,
    k_recent_sentences: int,
    sentence_representation: str,
    l2_normalize: str,
    hidden_dim: int,
    batch_size: int,
    num_epochs: int,
    patience: int,
    binary_threshold: Optional[float],
    show_progress: bool = False,
) -> BaselineResult:
    sequences = build_recent_sentence_sequence_examples(
        dataset_df,
        k_recent_sentences=k_recent_sentences,
        sentence_representation=sentence_representation,
        l2_normalize=l2_normalize,
    )
    masks = split_masks(dataset_df)
    train_sequences = [seq for seq, keep in zip(sequences, masks["train"]) if keep]
    val_sequences = [seq for seq, keep in zip(sequences, masks["val"]) if keep]
    test_sequences = [seq for seq, keep in zip(sequences, masks["test"]) if keep]
    torch_bundle, metrics_df, predictions = fit_gru_regression(
        train_sequences=train_sequences,
        train_targets=dataset_df.loc[masks["train"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        val_sequences=val_sequences,
        val_targets=dataset_df.loc[masks["val"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        test_sequences=test_sequences,
        test_targets=dataset_df.loc[masks["test"], "target_value"].to_numpy(dtype=np.float32, copy=False),
        device=device,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=patience,
        binary_threshold=binary_threshold,
        show_progress=show_progress,
        progress_desc="Hidden recent GRU",
    )
    predictions_df = attach_predictions(dataset_df, predictions)
    config = {
        "baseline_name": "hidden_recent_gru",
        "k_recent_sentences": int(k_recent_sentences),
        "sentence_representation": sentence_representation,
        "l2_normalize": l2_normalize,
        "hidden_dim": int(hidden_dim),
        "batch_size": int(batch_size),
        "num_epochs": int(num_epochs),
        "patience": int(patience),
        "binary_threshold": binary_threshold,
    }
    save_baseline_outputs(
        output_dir=output_dir,
        baseline_name="hidden_recent_gru",
        config=config,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        torch_bundle=torch_bundle,
    )
    return BaselineResult("hidden_recent_gru", config, metrics_df, predictions_df, output_dir)


def run_uncertainty_pooled_logistic_baseline(
    dataset_df: pd.DataFrame,
    *,
    output_dir: Path,
    recent_window_tokens: int,
    pca_dim: int = 0,
) -> BaselineResult:
    features = build_uncertainty_pooled_features(dataset_df, recent_window_tokens=recent_window_tokens)
    masks = split_masks(dataset_df)
    if pca_dim > 0:
        x_train_pca, x_eval_pca, pca_model = apply_dense_pca(
            x_train=features[masks["train"]],
            x_other=[features[masks["val"]], features[masks["test"]]],
            pca_dim=pca_dim,
        )
        features = np.concatenate([x_train_pca] + x_eval_pca, axis=0)
    
    # Create binary labels: deception_rate > 0.5 = 1, else 0
    y_train = (dataset_df.loc[masks["train"], "deception_rate"] > 0.5).astype(int).values
    y_val = (dataset_df.loc[masks["val"], "deception_rate"] > 0.5).astype(int).values
    y_test = (dataset_df.loc[masks["test"], "deception_rate"] > 0.5).astype(int).values
    
    model, metrics_df, predictions = fit_logistic_regression(
        x_train=features[masks["train"]],
        y_train=y_train,
        x_eval={split: features[mask] for split, mask in masks.items()},
        y_eval={"train": y_train, "val": y_val, "test": y_test},
        use_standard_scaler=True,  # Normalizes features
    )
    predictions_df = attach_predictions(dataset_df, predictions)
    config = {
        "baseline_name": "uncertainty_pooled_logistic",
        "recent_window_tokens": int(recent_window_tokens),
        "pca_dim": pca_dim,
    }
    save_baseline_outputs(
        output_dir=output_dir,
        baseline_name="uncertainty_pooled_logistic",
        config=config,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        sklearn_model=model,
        pca_model=pca_model if pca_dim > 0 else None,
    )
    return BaselineResult("uncertainty_pooled_logistic", config, metrics_df, predictions_df, output_dir)
