#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer

from commitment_rebuttal_lib import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_NAME,
    BundleSpec,
    attention_features,
    bundle_dir_for_kind,
    iter_localization_paths,
    load_localization_payload,
    main_feature_extractor,
    resolve_local_hf_snapshot,
    run_root_for_name,
)


EXTRA_STRING_COLUMNS = ["env_name", "env_display", "model_bundle_name", "model_display", "model_id"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract rebuttal main-model prefix features from compressed localization files. "
            "This wraps the historical prefix feature extractor but targets the offline "
            "DatasetMainCompressed layout and writes shard parquet files only."
        )
    )
    parser.add_argument("input_path", type=str, help="Dataset bundle directory or its localization directory.")
    parser.add_argument("--feature-output", type=str, default=None, help="Explicit parquet output path.")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--device-map", type=str, default="single")
    parser.add_argument("--dtype", type=str, choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attn-implementation", type=str, default=main_feature_extractor.DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--load-in-8bit", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--recent-window-tokens", type=int, default=main_feature_extractor.DEFAULT_RECENT_WINDOW_TOKENS)
    parser.add_argument("--num-prefix-sentences", type=int, default=main_feature_extractor.DEFAULT_NUM_PREFIX_SENTENCES)
    parser.add_argument("--max-input-tokens", type=int, default=main_feature_extractor.DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--write-every-examples", type=int, default=main_feature_extractor.DEFAULT_WRITE_EVERY_EXAMPLES)
    parser.add_argument("--progress-every", type=int, default=main_feature_extractor.DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--quiet-skips", action="store_true", default=False)
    return parser.parse_args(argv)


def resolve_bundle(input_path: str | Path, model_id_override: Optional[str]) -> BundleSpec:
    root = Path(input_path).expanduser().resolve()
    dataset_dir = root.parent if root.name == "localization" else root
    localization_dir = dataset_dir / "localization"
    if not localization_dir.exists():
        raise FileNotFoundError(f"Missing localization directory: {localization_dir}")
    env_name = str(dataset_dir.parent.name)
    model_bundle_name = str(dataset_dir.name)

    from commitment_rebuttal_lib import ENV_DISPLAY_BY_NAME, MODEL_DISPLAY_BY_BUNDLE, MODEL_ID_BY_BUNDLE

    model_id = str(model_id_override or MODEL_ID_BY_BUNDLE.get(model_bundle_name, model_bundle_name))
    return BundleSpec(
        env_name=env_name,
        env_display=ENV_DISPLAY_BY_NAME.get(env_name, env_name),
        model_bundle_name=model_bundle_name,
        model_display=MODEL_DISPLAY_BY_BUNDLE.get(model_bundle_name, model_bundle_name),
        model_id=model_id,
        dataset_dir=dataset_dir,
        localization_dir=localization_dir,
    )


def default_output_path(args: argparse.Namespace, bundle: BundleSpec) -> Path:
    run_root = run_root_for_name(args.run_name, args.results_root)
    output_dir = bundle_dir_for_kind(run_root, bundle, "main_features")
    shard_label = f"shard_{int(args.shard_id):03d}_of_{int(args.num_shards):03d}"
    return output_dir / f"commitment_main_features_{shard_label}.parquet"


def coerce_feature_chunk(feature_df, *, ordered_columns):
    coerced = main_feature_extractor.coerce_feature_frame_columns(feature_df, ordered_columns=ordered_columns)
    for column_name in EXTRA_STRING_COLUMNS:
        if column_name not in coerced.columns:
            coerced[column_name] = ""
        coerced[column_name] = coerced[column_name].astype("string")
    final_columns = EXTRA_STRING_COLUMNS + list(ordered_columns)
    return coerced.loc[:, final_columns]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    bundle = resolve_bundle(args.input_path, args.model_id)
    model_source = resolve_local_hf_snapshot(bundle.model_id)
    feature_output = Path(args.feature_output).expanduser().resolve() if args.feature_output else default_output_path(args, bundle)
    feature_output.parent.mkdir(parents=True, exist_ok=True)
    if feature_output.exists() and not args.overwrite:
        raise FileExistsError(f"Feature output already exists: {feature_output}")

    device, gpu_df = attention_features.resolve_device(args.device)
    model_dtype = attention_features.resolve_dtype(args.dtype, device)
    requested_device_map = main_feature_extractor.resolve_requested_device_map(str(args.device_map))
    try:
        repo_quant_method = main_feature_extractor.infer_repo_quant_method(
            bundle.model_id,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        repo_quant_method = None
    effective_load_in_8bit = bool(args.load_in_8bit)
    if effective_load_in_8bit and repo_quant_method and repo_quant_method != "bitsandbytes":
        print(
            f"Requested --load-in-8bit, but {bundle.model_id} already declares native quantization "
            f"method '{repo_quant_method}'. Continuing without bitsandbytes 8-bit loading."
        )
        effective_load_in_8bit = False

    localization_paths = iter_localization_paths(
        bundle.localization_dir,
        max_examples=int(args.max_examples),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
    )
    if not localization_paths:
        empty_columns = EXTRA_STRING_COLUMNS + list(main_feature_extractor.METADATA_COLUMNS)
        import pandas as pd
        pd.DataFrame(columns=empty_columns).to_parquet(feature_output, index=False)
        print(f"Wrote empty main feature shard to: {feature_output}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("This script requires a fast tokenizer because it uses offset mappings.")

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=True,
        **main_feature_extractor.build_model_load_kwargs(
            args=args,
            device=device,
            model_dtype=model_dtype,
            load_in_8bit=effective_load_in_8bit,
            requested_device_map=requested_device_map,
        ),
    )
    model_is_dispatched = requested_device_map is not None or effective_load_in_8bit
    if not model_is_dispatched:
        model.to(device)
    model.eval()
    model_input_device = main_feature_extractor.infer_model_input_device(model, device)
    base_model = model.base_model
    num_layers = int(getattr(base_model.config, "num_hidden_layers", getattr(model.config, "num_hidden_layers", 0)))
    ordered_columns = list(main_feature_extractor.METADATA_COLUMNS) + main_feature_extractor.build_feature_columns(num_layers)

    writer = attention_features.StreamingParquetWriter(feature_output, overwrite=True)
    buffer = []
    skip_counts: dict[str, int] = {}
    processed = 0
    successful = 0

    print(f"Dataset dir: {bundle.dataset_dir}")
    print(f"Localization dir: {bundle.localization_dir}")
    print(f"Feature output: {feature_output}")
    print(f"Model id: {bundle.model_id}")
    print(f"Model source: {model_source}")
    print(f"Device: {device}")
    print(f"Requested device map: {requested_device_map or 'single'}")
    print(f"Resolved device map: {main_feature_extractor.summarize_hf_device_map(model)}")
    print(f"Model input device: {model_input_device}")
    print(f"Model dtype: {model_dtype}")
    print(f"Repo quantization method: {repo_quant_method or 'none'}")
    print(f"8-bit quantization: {'enabled' if effective_load_in_8bit else 'disabled'}")
    print(f"Layers: {num_layers}")
    print(f"Shard: {int(args.shard_id) + 1}/{int(args.num_shards)}")
    print(f"Localization files to process on this shard: {len(localization_paths)}")
    if not gpu_df.empty:
        print("Visible GPUs:")
        print(gpu_df.to_string(index=False))

    try:
        path_iter = main_feature_extractor.maybe_tqdm(
            localization_paths,
            desc="Extract rebuttal main features",
            total=len(localization_paths),
            disable=bool(args.disable_tqdm),
        )
        for path in path_iter:
            processed += 1
            had_error = False
            try:
                example = load_localization_payload(path)
                feature_df, _activation_batch, _ = main_feature_extractor.extract_example_outputs(
                    example=example,
                    tokenizer=tokenizer,
                    base_model=base_model,
                    model_input_device=model_input_device,
                    recent_window_tokens=int(args.recent_window_tokens),
                    num_prefix_sentences=int(args.num_prefix_sentences),
                    max_input_tokens=int(args.max_input_tokens),
                )
            except Exception as exc:  # noqa: BLE001
                had_error = True
                reason = getattr(exc, "reason", exc.__class__.__name__)
                skip_counts[str(reason)] = skip_counts.get(str(reason), 0) + 1
                if isinstance(exc, RuntimeError):
                    attention_features.cleanup_tensors()
                    attention_features.maybe_raise_runtime_error(args, path, exc)
                else:
                    attention_features.maybe_raise_invalid_example(args, path, exc)
                feature_df = None

            if feature_df is not None and not feature_df.empty:
                feature_df = coerce_feature_chunk(feature_df, ordered_columns=ordered_columns)
                feature_df["env_name"] = bundle.env_name
                feature_df["env_display"] = bundle.env_display
                feature_df["model_bundle_name"] = bundle.model_bundle_name
                feature_df["model_display"] = bundle.model_display
                feature_df["model_id"] = bundle.model_id
                feature_df = feature_df.loc[:, EXTRA_STRING_COLUMNS + list(ordered_columns)]
                buffer.append(feature_df)
                successful += 1
            elif not had_error:
                skip_counts["no_rows"] = skip_counts.get("no_rows", 0) + 1

            if len(buffer) >= max(1, int(args.write_every_examples)):
                import pandas as pd
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
            import pandas as pd
            chunk_df = pd.concat(buffer, ignore_index=True)
            buffer.clear()
            writer.write(chunk_df)
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        del model
        attention_features.cleanup_tensors()

    manifest = {
        "env_name": bundle.env_name,
        "model_bundle_name": bundle.model_bundle_name,
        "model_id": bundle.model_id,
        "feature_output": str(feature_output),
        "processed_examples": int(processed),
        "successful_examples": int(successful),
        "rows_written": int(writer.rows_written),
        "skip_counts": skip_counts,
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
    }
    feature_output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote rebuttal main feature shard to: {feature_output}")
    print(f"Processed examples: {processed:,}")
    print(f"Successful examples: {successful:,}")
    print(f"Rows written: {writer.rows_written:,}")
    if skip_counts:
        print("Skipped examples by reason:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
