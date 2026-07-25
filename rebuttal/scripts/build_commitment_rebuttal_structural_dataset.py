#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from commitment_rebuttal_lib import (
    DEFAULT_LABEL_KINDS,
    DEFAULT_MIN_VALID,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_NAME,
    DEFAULT_TAU_VALUES,
    BundleSpec,
    ENV_DISPLAY_BY_NAME,
    MODEL_DISPLAY_BY_BUNDLE,
    MODEL_ID_BY_BUNDLE,
    attention_features,
    bundle_dir_for_kind,
    compute_commitment_indices,
    infer_example_label,
    iter_localization_paths,
    label_column_name,
    load_localization_payload,
    parse_csv_list,
    parse_tau_values,
    resolve_local_hf_snapshot,
    run_root_for_name,
    structural_extractor,
    tau_to_token,
    usable_example_column_name,
)


BASE_OUTPUT_COLUMNS = [
    "env_name",
    "env_display",
    "dataset",
    "model_bundle_name",
    "model_display",
    "model_id",
    "model_name",
    "example_id",
    "trace_id",
    "localization_path",
    "prompt",
    "example_label",
    "example_label_source",
    "sentence_text",
    "last_sentence_text",
    "previous_sentence_text",
    "prefix_text",
    "full_prefix_text",
    "sentence_idx",
    "sentence_idx_1based",
    "trace_length",
    "prefix_sentence_count",
    "sentences_remaining",
    "num_truthful",
    "num_valid",
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
    "start_token",
    "end_token",
    "token_count",
    "current_sentence_token_count",
    "context_token_count",
    "prompt_token_count",
    "raw_text_context_token_count",
    "available_token_count",
    "prior_all_token_count",
    "previous_sentence_token_count",
    "recent_token_count",
    "early_token_count",
    "current_sentence_char_count",
    "current_sentence_word_count",
    "previous_sentence_char_count",
    "previous_sentence_word_count",
    "prefix_char_count",
    "prefix_word_count",
    "prefix_token_count",
    "sentence_char_delta",
    "sentence_word_delta",
    "sentence_token_delta",
    "normalized_position",
    "reverse_normalized_position",
    "deception_rate",
    "prev_deception_rate",
    "next_deception_rate",
    "delta_deception_rate",
    "abs_delta_deception_rate",
    "final_deception_rate",
    "has_previous_sentence",
    "has_next_sentence",
]


STRING_COLUMNS = {
    "env_name",
    "env_display",
    "dataset",
    "model_bundle_name",
    "model_display",
    "model_id",
    "model_name",
    "example_id",
    "trace_id",
    "localization_path",
    "prompt",
    "example_label",
    "example_label_source",
    "sentence_text",
    "last_sentence_text",
    "previous_sentence_text",
    "prefix_text",
    "full_prefix_text",
}

BOOL_COLUMNS = {
    "has_previous_sentence",
    "has_next_sentence",
}

INT_COLUMNS = {
    "sentence_idx",
    "sentence_idx_1based",
    "trace_length",
    "prefix_sentence_count",
    "sentences_remaining",
    "num_truthful",
    "num_valid",
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
    "start_token",
    "end_token",
    "token_count",
    "current_sentence_token_count",
    "context_token_count",
    "prompt_token_count",
    "raw_text_context_token_count",
    "available_token_count",
    "prior_all_token_count",
    "previous_sentence_token_count",
    "recent_token_count",
    "early_token_count",
    "current_sentence_char_count",
    "current_sentence_word_count",
    "previous_sentence_char_count",
    "previous_sentence_word_count",
    "prefix_char_count",
    "prefix_word_count",
    "prefix_token_count",
    "sentence_char_delta",
    "sentence_word_delta",
    "sentence_token_delta",
}

FLOAT_COLUMNS = {
    "normalized_position",
    "reverse_normalized_position",
    "deception_rate",
    "prev_deception_rate",
    "next_deception_rate",
    "delta_deception_rate",
    "abs_delta_deception_rate",
    "final_deception_rate",
}

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the rebuttal sentence-level structural dataset from compressed localization files. "
            "This stores multi-threshold commitment labels and text/non-semantic baseline columns "
            "for one environment/model bundle."
        )
    )
    parser.add_argument("input_path", type=str, help="Dataset bundle directory or its localization directory.")
    parser.add_argument("--output", type=str, default=None, help="Explicit parquet output path.")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--model-id", type=str, default=None, help="Override inferred model id.")
    parser.add_argument("--recent-window-sentences", type=int, default=5)
    parser.add_argument("--tau-values", type=str, default=",".join(str(value) for value in DEFAULT_TAU_VALUES))
    parser.add_argument("--label-kinds", type=str, default=",".join(DEFAULT_LABEL_KINDS))
    parser.add_argument("--min-valid", type=int, default=DEFAULT_MIN_VALID)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--write-every-examples", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args(argv)


def resolve_bundle(input_path: str | Path, model_id_override: Optional[str]) -> BundleSpec:
    root = Path(input_path).expanduser().resolve()
    dataset_dir = root.parent if root.name == "localization" else root
    localization_dir = dataset_dir / "localization"
    if not localization_dir.exists():
        raise FileNotFoundError(f"Missing localization directory: {localization_dir}")

    env_name = str(dataset_dir.parent.name)
    model_bundle_name = str(dataset_dir.name)
    env_display = ENV_DISPLAY_BY_NAME.get(env_name, env_name)
    model_display = MODEL_DISPLAY_BY_BUNDLE.get(model_bundle_name, model_bundle_name)
    model_id = str(model_id_override or dataset_dir.name)
    model_id = MODEL_ID_BY_BUNDLE.get(model_bundle_name, model_id)
    return BundleSpec(
        env_name=env_name,
        env_display=env_display,
        model_bundle_name=model_bundle_name,
        model_display=model_display,
        model_id=model_id,
        dataset_dir=dataset_dir,
        localization_dir=localization_dir,
    )


def build_dynamic_columns(tau_values: Sequence[float], label_kinds: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for label_kind in label_kinds:
        columns.append(usable_example_column_name(label_kind))
        for tau in tau_values:
            columns.append(f"commitment_idx_{label_kind}_tau_{tau_to_token(tau)}")
            columns.append(label_column_name(label_kind, tau))
    return columns


def coerce_output_frame(df: pd.DataFrame, *, dynamic_columns: Sequence[str]) -> pd.DataFrame:
    ordered_columns = BASE_OUTPUT_COLUMNS + list(dynamic_columns)
    out = df.copy().reindex(columns=ordered_columns)
    for column in ordered_columns:
        if column in STRING_COLUMNS:
            out[column] = out[column].astype("string")
        elif column in BOOL_COLUMNS:
            out[column] = out[column].astype("boolean")
        elif column in INT_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
        elif column in FLOAT_COLUMNS or column in dynamic_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
        else:
            out[column] = out[column].astype("string")
    return out


def empty_output_frame(dynamic_columns: Sequence[str]) -> pd.DataFrame:
    return coerce_output_frame(pd.DataFrame(columns=BASE_OUTPUT_COLUMNS + list(dynamic_columns)), dynamic_columns=dynamic_columns)


def word_count(text: str) -> int:
    return structural_extractor.count_words(text)


def approx_token_count(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(WORD_RE.findall(text))


def load_tokenizer_maybe_local(model_id: str):
    try:
        model_source = resolve_local_hf_snapshot(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, local_files_only=True)
        if not getattr(tokenizer, "is_fast", False):
            return None
        return tokenizer
    except Exception:
        return None


def resolve_prefix_text(raw_text: str, sentence_row: Any) -> str:
    return structural_extractor.resolve_prefix_text(raw_text, sentence_row)


def final_deception_rate_for_example(example: dict[str, Any]) -> float:
    full_score = example.get("full_score")
    if isinstance(full_score, dict):
        rate = pd.to_numeric(pd.Series([full_score.get("deception_rate")]), errors="coerce").iloc[0]
        if pd.notna(rate):
            return float(rate)
    history = example.get("history")
    if isinstance(history, list) and history:
        rate = pd.to_numeric(pd.Series([history[-1].get("deception_rate")]), errors="coerce").iloc[0]
        if pd.notna(rate):
            return float(rate)
    return float("nan")


def fallback_aligned_sentence_frame(example: dict[str, Any], *, recent_window_sentences: int) -> tuple[pd.DataFrame, int]:
    prompt = str(example.get("prompt") or "")
    prompt_token_count = approx_token_count(prompt)
    history = sorted(example.get("history", []), key=lambda item: int(item["sentence_idx_inclusive"]))
    rows: list[dict[str, Any]] = []
    token_counts: list[int] = []
    prompt_char_count = len(prompt)
    for idx, hist in enumerate(history):
        raw_start, raw_end = hist["char_span"]
        sentence_text = str(hist.get("sentence_text") or "")
        sentence_token_count = approx_token_count(sentence_text)
        token_counts.append(sentence_token_count)
        start_token = prompt_token_count + approx_token_count(str(example.get("raw_text") or "")[: int(raw_start)])
        end_token = max(start_token, prompt_token_count + approx_token_count(str(example.get("raw_text") or "")[: int(raw_end)]) - 1)
        recent_start_idx = max(0, idx - int(recent_window_sentences) + 1)
        recent_token_count = int(sum(token_counts[recent_start_idx:idx]))
        rows.append(
            {
                "sentence_idx": int(hist["sentence_idx_inclusive"]),
                "sentence_text": sentence_text,
                "deception_rate": hist.get("deception_rate"),
                "num_truthful": hist.get("num_truthful"),
                "num_valid": hist.get("num_valid"),
                "raw_start": int(raw_start),
                "raw_end": int(raw_end),
                "full_start": int(raw_start) + prompt_char_count,
                "full_end": int(raw_end) + prompt_char_count,
                "start_token": int(start_token),
                "end_token": int(end_token),
                "token_count": int(sentence_token_count),
                "context_token_count": int(end_token) + 1,
                "available_token_count": int(end_token) + 1,
                "prior_all_token_count": int(start_token),
                "previous_sentence_token_count": int(token_counts[idx - 1]) if idx > 0 else 0,
                "recent_token_count": int(recent_token_count),
                "early_token_count": int(max(0, start_token - recent_token_count)),
                "prefix_text": str(hist.get("prefix_text") or ""),
            }
        )
    return pd.DataFrame(rows), prompt_token_count


def build_rows_for_example(
    *,
    example: dict[str, Any],
    bundle: BundleSpec,
    localization_path: Path,
    tokenizer: Any,
    recent_window_sentences: int,
    tau_values: Sequence[float],
    label_kinds: Sequence[str],
    min_valid: int,
) -> pd.DataFrame:
    example_id = example.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise attention_features.ExampleValidationError("missing_example_id", "Localization example missing example_id.")
    raw_text = example.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        raise attention_features.ExampleValidationError("missing_raw_text", f"{example_id} missing raw_text.")

    if tokenizer is not None:
        aligned_sentence_df, _full_context, prompt_token_count, _ = structural_extractor.build_aligned_sentence_frame_for_full_context(
            example=example,
            tokenizer=tokenizer,
            recent_window_sentences=int(recent_window_sentences),
        )
    else:
        aligned_sentence_df, prompt_token_count = fallback_aligned_sentence_frame(
            example,
            recent_window_sentences=int(recent_window_sentences),
        )
    if aligned_sentence_df.empty:
        return empty_output_frame(build_dynamic_columns(tau_values, label_kinds))

    example_label, example_label_source = infer_example_label(example)
    prompt = str(example.get("prompt") or "")
    prompt_messages = example.get("prompt_messages")
    trace_length = int(len(aligned_sentence_df))
    sentence_texts = aligned_sentence_df["sentence_text"].astype(str).tolist()
    deception_rates = pd.to_numeric(aligned_sentence_df["deception_rate"], errors="coerce").astype(float).tolist()
    final_deception_rate = final_deception_rate_for_example(example)

    commitment_lookup = {
        float(tau): compute_commitment_indices(aligned_sentence_df, tau=float(tau), min_valid=int(min_valid))
        for tau in tau_values
    }
    dynamic_columns = build_dynamic_columns(tau_values, label_kinds)

    rows: list[dict[str, Any]] = []
    for row_idx, sentence_row in enumerate(aligned_sentence_df.itertuples()):
        sentence_text = str(sentence_row.sentence_text)
        previous_sentence_text = sentence_texts[row_idx - 1] if row_idx > 0 else ""
        prefix_text = resolve_prefix_text(raw_text, sentence_row)
        if tokenizer is not None:
            full_prefix_text = structural_extractor.render_full_prefix_context(
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_messages=prompt_messages if isinstance(prompt_messages, list) else None,
                prefix_text=prefix_text,
            )
        else:
            full_prefix_text = f"{prompt}{prefix_text}"

        current_char_count = len(sentence_text)
        current_word_count = word_count(sentence_text)
        previous_char_count = len(previous_sentence_text) if row_idx > 0 else 0
        previous_word_count = word_count(previous_sentence_text) if row_idx > 0 else 0
        current_token_count = int(sentence_row.token_count)
        previous_token_count = int(sentence_row.previous_sentence_token_count)
        prefix_token_count = max(0, int(sentence_row.available_token_count) - int(prompt_token_count))

        current_rate = deception_rates[row_idx]
        prev_rate = deception_rates[row_idx - 1] if row_idx > 0 else float("nan")
        next_rate = deception_rates[row_idx + 1] if row_idx + 1 < trace_length else float("nan")
        delta_rate = (
            float(current_rate - prev_rate)
            if row_idx > 0 and np.isfinite(prev_rate) and np.isfinite(current_rate)
            else float("nan")
        )
        abs_delta_rate = abs(delta_rate) if np.isfinite(delta_rate) else float("nan")
        sentence_idx = int(sentence_row.sentence_idx)

        row: dict[str, Any] = {
            "env_name": bundle.env_name,
            "env_display": bundle.env_display,
            "dataset": bundle.env_name,
            "model_bundle_name": bundle.model_bundle_name,
            "model_display": bundle.model_display,
            "model_id": bundle.model_id,
            "model_name": bundle.model_id,
            "example_id": example_id,
            "trace_id": example_id,
            "localization_path": str(localization_path),
            "prompt": prompt,
            "example_label": example_label,
            "example_label_source": example_label_source,
            "sentence_text": sentence_text,
            "last_sentence_text": sentence_text,
            "previous_sentence_text": previous_sentence_text,
            "prefix_text": prefix_text,
            "full_prefix_text": full_prefix_text,
            "sentence_idx": sentence_idx,
            "sentence_idx_1based": row_idx + 1,
            "trace_length": trace_length,
            "prefix_sentence_count": row_idx + 1,
            "sentences_remaining": trace_length - (row_idx + 1),
            "num_truthful": sentence_row.num_truthful,
            "num_valid": sentence_row.num_valid,
            "raw_start": int(sentence_row.raw_start),
            "raw_end": int(sentence_row.raw_end),
            "full_start": int(sentence_row.full_start),
            "full_end": int(sentence_row.full_end),
            "start_token": int(sentence_row.start_token),
            "end_token": int(sentence_row.end_token),
            "token_count": current_token_count,
            "current_sentence_token_count": current_token_count,
            "context_token_count": int(sentence_row.context_token_count),
            "prompt_token_count": int(prompt_token_count),
            "raw_text_context_token_count": max(0, int(sentence_row.start_token) - int(prompt_token_count)),
            "available_token_count": int(sentence_row.available_token_count),
            "prior_all_token_count": int(sentence_row.prior_all_token_count),
            "previous_sentence_token_count": previous_token_count,
            "recent_token_count": int(sentence_row.recent_token_count),
            "early_token_count": int(sentence_row.early_token_count),
            "current_sentence_char_count": current_char_count,
            "current_sentence_word_count": current_word_count,
            "previous_sentence_char_count": previous_char_count,
            "previous_sentence_word_count": previous_word_count,
            "prefix_char_count": len(prefix_text),
            "prefix_word_count": word_count(prefix_text),
            "prefix_token_count": prefix_token_count,
            "sentence_char_delta": current_char_count - previous_char_count if row_idx > 0 else 0,
            "sentence_word_delta": current_word_count - previous_word_count if row_idx > 0 else 0,
            "sentence_token_delta": current_token_count - previous_token_count if row_idx > 0 else 0,
            "normalized_position": float((row_idx + 1) / trace_length) if trace_length > 0 else float("nan"),
            "reverse_normalized_position": float((trace_length - row_idx) / trace_length) if trace_length > 0 else float("nan"),
            "deception_rate": float(current_rate) if np.isfinite(current_rate) else float("nan"),
            "prev_deception_rate": float(prev_rate) if np.isfinite(prev_rate) else float("nan"),
            "next_deception_rate": float(next_rate) if np.isfinite(next_rate) else float("nan"),
            "delta_deception_rate": delta_rate,
            "abs_delta_deception_rate": abs_delta_rate,
            "final_deception_rate": final_deception_rate,
            "has_previous_sentence": bool(row_idx > 0),
            "has_next_sentence": bool(row_idx + 1 < trace_length),
        }

        for label_kind in label_kinds:
            usable = float(example_label == label_kind)
            row[usable_example_column_name(label_kind)] = usable
            for tau in tau_values:
                commitment_idx = commitment_lookup[float(tau)].get(label_kind)
                commitment_idx_column = f"commitment_idx_{label_kind}_tau_{tau_to_token(tau)}"
                row[commitment_idx_column] = float(commitment_idx) if commitment_idx is not None else float("nan")
                if example_label != label_kind:
                    row[label_column_name(label_kind, tau)] = float("nan")
                else:
                    row[label_column_name(label_kind, tau)] = 1.0 if commitment_idx == sentence_idx else 0.0

        rows.append(row)

    return coerce_output_frame(pd.DataFrame(rows), dynamic_columns=dynamic_columns)


def default_output_path(args: argparse.Namespace, bundle: BundleSpec) -> Path:
    run_root = run_root_for_name(args.run_name, args.results_root)
    output_dir = bundle_dir_for_kind(run_root, bundle, "structural")
    shard_label = f"shard_{int(args.shard_id):03d}_of_{int(args.num_shards):03d}"
    return output_dir / f"commitment_structural_{shard_label}.parquet"


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    bundle = resolve_bundle(args.input_path, args.model_id)
    tau_values = parse_tau_values(args.tau_values)
    label_kinds = tuple(parse_csv_list(args.label_kinds) or list(DEFAULT_LABEL_KINDS))
    dynamic_columns = build_dynamic_columns(tau_values, label_kinds)
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(args, bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    localization_paths = iter_localization_paths(
        bundle.localization_dir,
        max_examples=int(args.max_examples),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
    )
    if not localization_paths:
        empty_output_frame(dynamic_columns).to_parquet(output_path, index=False)
        print(f"Wrote empty shard parquet to: {output_path}")
        return

    tokenizer = load_tokenizer_maybe_local(bundle.model_id)
    if tokenizer is None:
        print(
            f"[warn] No local fast tokenizer cache found for {bundle.model_id}. "
            "Falling back to approximate span/length features for the structural dataset."
        )

    writer = attention_features.StreamingParquetWriter(output_path, overwrite=True)
    buffer: list[pd.DataFrame] = []
    skip_counts: dict[str, int] = {}
    processed = 0
    successful = 0

    try:
        path_iter = structural_extractor.maybe_tqdm(
            localization_paths,
            desc="Build rebuttal structural dataset",
            total=len(localization_paths),
            disable=bool(args.disable_tqdm),
        )
        for path in path_iter:
            processed += 1
            try:
                example = load_localization_payload(path)
                row_df = build_rows_for_example(
                    example=example,
                    bundle=bundle,
                    localization_path=path,
                    tokenizer=tokenizer,
                    recent_window_sentences=int(args.recent_window_sentences),
                    tau_values=tau_values,
                    label_kinds=label_kinds,
                    min_valid=int(args.min_valid),
                )
            except Exception as exc:  # noqa: BLE001
                reason = getattr(exc, "reason", exc.__class__.__name__)
                skip_counts[str(reason)] = skip_counts.get(str(reason), 0) + 1
                if isinstance(exc, RuntimeError):
                    attention_features.maybe_raise_runtime_error(args, path, exc)
                else:
                    attention_features.maybe_raise_invalid_example(args, path, exc)
                continue

            if not row_df.empty:
                buffer.append(row_df)
                successful += 1

            if len(buffer) >= max(1, int(args.write_every_examples)):
                chunk_df = pd.concat(buffer, ignore_index=True)
                buffer.clear()
                writer.write(chunk_df)

            if int(args.progress_every) > 0 and processed % int(args.progress_every) == 0:
                buffered_rows = sum(len(df) for df in buffer)
                print(
                    f"[progress] processed={processed:,}/{len(localization_paths):,} "
                    f"successful={successful:,} skipped={sum(skip_counts.values()):,} "
                    f"rows_buffered_or_written={writer.rows_written + buffered_rows:,}"
                )

        if buffer:
            chunk_df = pd.concat(buffer, ignore_index=True)
            buffer.clear()
            writer.write(chunk_df)
        writer.close()
    except Exception:
        writer.abort()
        raise

    if writer.rows_written == 0:
        empty_output_frame(dynamic_columns).to_parquet(output_path, index=False)

    manifest = {
        "env_name": bundle.env_name,
        "model_bundle_name": bundle.model_bundle_name,
        "model_id": bundle.model_id,
        "tau_values": [float(value) for value in tau_values],
        "label_kinds": list(label_kinds),
        "min_valid": int(args.min_valid),
        "input_path": str(bundle.dataset_dir),
        "output_path": str(output_path),
        "processed_examples": int(processed),
        "successful_examples": int(successful),
        "rows_written": int(writer.rows_written),
        "skip_counts": skip_counts,
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote rebuttal structural shard to: {output_path}")
    print(f"Processed examples: {processed:,}")
    print(f"Successful examples: {successful:,}")
    print(f"Rows written: {writer.rows_written:,}")
    if skip_counts:
        print("Skipped examples by reason:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
