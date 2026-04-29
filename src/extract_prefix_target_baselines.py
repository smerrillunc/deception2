#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_features import cleanup_tensors, maybe_raise_invalid_example, maybe_raise_runtime_error
from prefix_target_baselines import (
    DATASET_METADATA_COLUMNS,
    DEFAULT_PROGRESS_EVERY,
    DEFAULT_RECENT_SENTENCE_COUNT,
    DEFAULT_WRITE_EVERY_EXAMPLES,
    extract_example_forward_cache,
    flush_dataset_buffer,
    infer_dataset_name,
    maybe_tqdm,
    resolve_prefix_baseline_paths,
)
from attention_features import (
    StreamingParquetWriter,
    infer_model_id,
    iter_localization_paths,
    resolve_device,
    resolve_dtype,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract reusable caches for sentence-prefix baselines. Each localization file is "
            "run through one full prompt+reasoning forward pass, then the script writes: "
            "(1) a sentence-prefix dataset parquet with rendered full-prefix texts and "
            "(2) per-example .npz caches with token-level uncertainty features plus sentence-end / "
            "sentence-mean final-layer embeddings."
        )
    )
    parser.add_argument("input_path", type=str, help="Dataset directory or localization directory.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache root directory.")
    parser.add_argument("--model-id", type=str, default=None, help="Override inferred Hugging Face model id.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:<idx>.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model load dtype.",
    )
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--recent-window-sentences",
        "--recent-window-tokens",
        dest="recent_window_sentences",
        type=int,
        default=DEFAULT_RECENT_SENTENCE_COUNT,
        help="Number of recent sentences to use for the token uncertainty window.",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--write-every-examples", type=int, default=DEFAULT_WRITE_EVERY_EXAMPLES)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max-seq-len", type=int, default=10000, help="Maximum sequence length in tokens.")
    return parser.parse_args(argv)


def create_empty_dataset_file(output_path: Path, columns: Sequence[str]) -> None:
    empty_df = pd.DataFrame(columns=list(columns))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    empty_df.to_parquet(output_path, index=False)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = resolve_prefix_baseline_paths(args.input_path, cache_dir=args.cache_dir)
    model_id = infer_model_id(paths.dataset_paths, args.model_id)
    dataset_name = infer_dataset_name(paths.dataset_paths.dataset_dir)
    device, gpu_df = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    write_every_examples = max(1, int(args.write_every_examples))

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

    localization_paths = iter_localization_paths(
        paths.dataset_paths.localization_dir,
        max_examples=int(args.max_examples),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
    )
    if not localization_paths:
        raise FileNotFoundError(f"No localization JSON files found in {paths.dataset_paths.localization_dir}")

    writer = StreamingParquetWriter(paths.dataset_frame_path, overwrite=args.overwrite)
    paths.dataset_frame_path.parent.mkdir(parents=True, exist_ok=True)
    paths.example_cache_dir.mkdir(parents=True, exist_ok=True)
    buffered_frames: list[pd.DataFrame] = []
    skip_counts: Counter[str] = Counter()
    processed = 0
    successful = 0

    print(f"Dataset dir: {paths.dataset_paths.dataset_dir}")
    print(f"Localization dir: {paths.dataset_paths.localization_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset frame: {paths.dataset_frame_path}")
    print(f"Example cache dir: {paths.example_cache_dir}")
    print(f"Model id: {model_id}")
    print(f"Device: {device}")
    print(f"Model dtype: {model_dtype}")
    print(f"Recent window sentences: {int(args.recent_window_sentences)}")
    print(f"Localization files to process: {len(localization_paths)}")
    if not gpu_df.empty:
        print("Visible GPUs:")
        print(gpu_df.to_string(index=False))

    try:
        path_iter = maybe_tqdm(
            localization_paths,
            desc="Extract prefixes",
            total=len(localization_paths),
            disable=bool(args.disable_tqdm),
        )
        for path in path_iter:
            processed += 1
            try:
                example = json.loads(path.read_text(encoding="utf-8"))
                dataset_rows = extract_example_forward_cache(
                    example=example,
                    dataset_name=dataset_name,
                    model_name=model_id,
                    localization_path=path,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    recent_window_tokens=int(args.recent_window_sentences),
                    example_cache_dir=paths.example_cache_dir,
                    max_seq_len=int(args.max_seq_len),
                )
            except Exception as exc:  # noqa: BLE001
                reason = getattr(exc, "reason", exc.__class__.__name__)
                skip_counts[str(reason)] += 1
                if isinstance(exc, RuntimeError):
                    maybe_raise_runtime_error(args, path, exc)
                else:
                    if not hasattr(exc, "reason"):
                        import traceback
                        print(f"[error] {path.name}: {exc.__class__.__name__}: {exc}")
                        if args.strict:
                            traceback.print_exc()
                    maybe_raise_invalid_example(args, path, exc)
                cleanup_tensors()
                continue

            if not dataset_rows.empty:
                buffered_frames.append(dataset_rows)
                successful += 1

            if processed % write_every_examples == 0:
                written = flush_dataset_buffer(writer, buffered_frames)
                if written:
                    print(f"[flush] processed={processed:,} successful={successful:,} rows_written_now={written:,}")

            if processed % max(1, int(args.progress_every)) == 0:
                print(
                    f"[progress] processed={processed:,}/{len(localization_paths):,} "
                    f"successful={successful:,} skipped={sum(skip_counts.values()):,}"
                )

        final_written = flush_dataset_buffer(writer, buffered_frames)
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        del model
        cleanup_tensors()

    if writer.rows_written == 0:
        create_empty_dataset_file(paths.dataset_frame_path, DATASET_METADATA_COLUMNS)
        print(f"Created empty dataset file: {paths.dataset_frame_path}")
    else:
        print(f"Wrote prefix target dataset to: {paths.dataset_frame_path}")
    print(f"Per-example caches stored under: {paths.example_cache_dir}")
    print(f"Rows written: {writer.rows_written:,} (+ final flush {final_written:,})")
    if skip_counts:
        print("Skipped examples by reason:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
