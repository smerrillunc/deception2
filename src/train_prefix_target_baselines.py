#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import torch
from joblib import load

from prefix_target_baselines import (
    DEFAULT_RECENT_POOL_TOKENS,
    DEFAULT_RECENT_SENTENCE_COUNT,
    DEFAULT_SEQUENCE_BATCH_SIZE,
    DEFAULT_SEQUENCE_HIDDEN_DIM,
    DEFAULT_SEQUENCE_NUM_EPOCHS,
    DEFAULT_SEQUENCE_PATIENCE,
    DEFAULT_TEXT_EMBED_MODEL,
    GRURegressor,
    GroupSplitConfig,
    build_or_load_group_splits,
    build_recent_sentence_fixed_features,
    build_recent_sentence_sequence_examples,
    build_uncertainty_pooled_features,
    build_uncertainty_sequence_examples,
    compute_or_load_text_embeddings,
    compute_regression_metrics,
    coerce_dataset_frame_columns,
    merge_dataset_with_splits,
    maybe_tqdm,
    predict_sequence_regressor,
    resolve_prefix_baseline_paths,
    run_hidden_recent_fixed_ridge_baseline,
    run_hidden_recent_gru_baseline,
    run_text_cbow_ridge_baseline,
    run_text_embedding_ridge_baseline,
    run_uncertainty_gru_baseline,
    run_uncertainty_pooled_logistic_baseline,
    run_uncertainty_pooled_ridge_baseline,
)
from attention_features import resolve_device


ALL_BASELINES = (
    "text_embedding",
    "text_cbow_last_sentence",
    "text_cbow_full_prefix",
    "uncertainty_pooled",
    "uncertainty_pooled_logistic",
    "uncertainty_gru",
    "hidden_fixed",
    "hidden_gru",
)

VALID_OOD_EVAL_SPLITS = ("train", "val", "test", "all")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train sentence-prefix baselines from cached extraction artifacts. The same grouped "
            "train/val/test split is reused across all baselines so traces never leak across splits."
        )
    )
    parser.add_argument("input_path", type=str, help="Dataset directory, localization directory, or cache directory.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache root directory.")
    parser.add_argument(
        "--ood-input-paths",
        type=str,
        default="",
        help="Optional comma-separated list of other extracted dataset/cache roots to evaluate OOD.",
    )
    parser.add_argument(
        "--ood-eval-split",
        type=str,
        choices=VALID_OOD_EVAL_SPLITS,
        default="test",
        help="Which split to use when scoring OOD datasets. 'all' uses every row in the target dataset.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="text_embedding,text_cbow_last_sentence,text_cbow_full_prefix,uncertainty_pooled,hidden_fixed",
        help=f"Comma-separated subset of: {', '.join(ALL_BASELINES)}",
    )
    parser.add_argument("--split-overwrite", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--binary-threshold", type=float, default=None, help="Optional threshold for AUROC/AP side-metrics.")

    parser.add_argument("--device", type=str, default="auto", help="Device for GRU baselines and HF text embeddings.")
    parser.add_argument("--text-embedding-model", type=str, default=DEFAULT_TEXT_EMBED_MODEL)
    parser.add_argument("--text-embedding-batch-size", type=int, default=16)
    parser.add_argument("--text-embedding-overwrite-cache", action="store_true", default=False)
    parser.add_argument("--cbow-max-features", type=int, default=20000)
    parser.add_argument("--cbow-min-ngram", type=int, default=1)
    parser.add_argument("--cbow-max-ngram", type=int, default=2)

    parser.add_argument(
        "--uncertainty-recent-window-sentences",
        "--uncertainty-recent-window-tokens",
        dest="uncertainty_recent_window_sentences",
        type=int,
        default=DEFAULT_RECENT_POOL_TOKENS,
        help="Number of recent sentences to use for uncertainty pooling.",
    )
    parser.add_argument("--uncertainty-sequence-max-tokens", type=int, default=0)
    parser.add_argument("--uncertainty-pca-dim", type=int, default=0, help="PCA dimension for uncertainty and hidden state features.")
    parser.add_argument("--sequence-hidden-dim", type=int, default=DEFAULT_SEQUENCE_HIDDEN_DIM)
    parser.add_argument("--sequence-batch-size", type=int, default=DEFAULT_SEQUENCE_BATCH_SIZE)
    parser.add_argument("--sequence-num-epochs", type=int, default=DEFAULT_SEQUENCE_NUM_EPOCHS)
    parser.add_argument("--sequence-patience", type=int, default=DEFAULT_SEQUENCE_PATIENCE)

    parser.add_argument("--k-recent-sentences", type=int, default=DEFAULT_RECENT_SENTENCE_COUNT)
    parser.add_argument(
        "--sentence-representation",
        type=str,
        choices=("sentence_end", "sentence_mean"),
        default="sentence_end",
    )
    parser.add_argument(
        "--hidden-l2-normalize",
        type=str,
        choices=("none", "sentence", "vector"),
        default="none",
        help="Normalize each recent sentence embedding or the concatenated vector.",
    )
    parser.add_argument("--hidden-pca-dim", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    return parser.parse_args(argv)


def parse_baseline_list(raw_value: str) -> list[str]:
    out = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if piece:
            out.append(piece)
    invalid = [name for name in out if name not in ALL_BASELINES]
    if invalid:
        raise ValueError(f"Unknown baseline(s): {invalid}. Expected subset of {ALL_BASELINES}")
    return out


def parse_path_list(raw_value: str) -> list[str]:
    out: list[str] = []
    for piece in str(raw_value).split(","):
        piece = piece.strip()
        if piece:
            out.append(piece)
    return out


def load_dataset_frame(paths) -> pd.DataFrame:
    if not paths.dataset_frame_path.exists():
        raise FileNotFoundError(
            f"Missing dataset frame at {paths.dataset_frame_path}. "
            "Run extract_prefix_target_baselines.py first."
        )
    dataset_df = pd.read_parquet(paths.dataset_frame_path)
    return coerce_dataset_frame_columns(dataset_df)


def load_dataset_with_splits(
    input_path: str,
    *,
    cache_dir: Optional[str],
    split_overwrite: bool,
    split_config: GroupSplitConfig,
) -> tuple[Any, pd.DataFrame]:
    paths = resolve_prefix_baseline_paths(input_path, cache_dir=cache_dir)
    dataset_df = load_dataset_frame(paths)
    split_df = build_or_load_group_splits(
        dataset_df,
        paths.split_path,
        config=split_config,
        overwrite=bool(split_overwrite),
    )
    dataset_df = merge_dataset_with_splits(dataset_df, split_df)
    return paths, dataset_df


def select_eval_rows(dataset_df: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    if eval_split == "all":
        return dataset_df.copy()
    return dataset_df.loc[dataset_df["split"] == eval_split].copy()


def dataset_label(dataset_df: pd.DataFrame) -> str:
    values = dataset_df["dataset"].dropna().astype(str).unique().tolist()
    return values[0] if values else "unknown"


def cbow_text_column_for_baseline(baseline_name: str) -> str:
    if baseline_name == "text_cbow_last_sentence":
        return "last_sentence_text"
    if baseline_name == "text_cbow_full_prefix":
        return "full_prefix_text"
    raise ValueError(f"Unsupported CBOW baseline: {baseline_name}")


def evaluate_ood_for_baseline(
    *,
    baseline_name: str,
    output_dir: Path,
    eval_datasets: list[tuple[Any, pd.DataFrame]],
    eval_split: str,
    device: str,
    binary_threshold: Optional[float],
    text_embedding_model: str,
    text_embedding_batch_size: int,
    uncertainty_recent_window_sentences: int,
    uncertainty_sequence_max_tokens: int,
    k_recent_sentences: int,
    sentence_representation: str,
    hidden_l2_normalize: str,
    sequence_batch_size: int,
    disable_tqdm: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    sklearn_model = load(output_dir / "model.joblib") if (output_dir / "model.joblib").exists() else None
    pca_model = load(output_dir / "pca.joblib") if (output_dir / "pca.joblib").exists() else None
    vectorizer = load(output_dir / "vectorizer.joblib") if (output_dir / "vectorizer.joblib").exists() else None

    torch_model = None
    torch_bundle = None
    if (output_dir / "model.pt").exists():
        torch_bundle = torch.load(output_dir / "model.pt", map_location="cpu")
        torch_model = GRURegressor(
            input_dim=int(torch_bundle["input_dim"]),
            hidden_dim=int(torch_bundle["hidden_dim"]),
        ).to(device)
        torch_model.load_state_dict(torch_bundle["state_dict"])
        torch_model.eval()

    eval_iter = maybe_tqdm(
        eval_datasets,
        desc=f"OOD eval: {baseline_name}",
        total=len(eval_datasets),
        disable=disable_tqdm,
    )
    for eval_paths, full_eval_df in eval_iter:
        eval_df = select_eval_rows(full_eval_df, eval_split)
        if eval_df.empty:
            continue

        if baseline_name == "text_embedding":
            if sklearn_model is None:
                raise FileNotFoundError(f"Missing model.joblib for {baseline_name} at {output_dir}")
            features = compute_or_load_text_embeddings(
                eval_df,
                cache_dir=eval_paths.text_embedding_cache_dir,
                model_name=text_embedding_model,
                device=device,
                batch_size=text_embedding_batch_size,
                show_progress=not disable_tqdm,
                overwrite=False,
            )
            y_pred = sklearn_model.predict(features).astype("float32")
        elif baseline_name in {"text_cbow_last_sentence", "text_cbow_full_prefix"}:
            if sklearn_model is None or vectorizer is None:
                raise FileNotFoundError(f"Missing model/vectorizer artifacts for {baseline_name} at {output_dir}")
            text_column = cbow_text_column_for_baseline(baseline_name)
            features = vectorizer.transform(eval_df[text_column].astype(str))
            y_pred = sklearn_model.predict(features).astype("float32")
        elif baseline_name == "uncertainty_pooled":
            if sklearn_model is None:
                raise FileNotFoundError(f"Missing model.joblib for {baseline_name} at {output_dir}")
            features = build_uncertainty_pooled_features(
                eval_df,
                recent_window_tokens=uncertainty_recent_window_sentences,
            )
            if pca_model is not None:
                features = pca_model.transform(features).astype("float32")
            y_pred = sklearn_model.predict(features).astype("float32")
        elif baseline_name == "uncertainty_pooled_logistic":
            if sklearn_model is None:
                raise FileNotFoundError(f"Missing model.joblib for {baseline_name} at {output_dir}")
            features = build_uncertainty_pooled_features(
                eval_df,
                recent_window_tokens=uncertainty_recent_window_sentences,
            )
            if pca_model is not None:
                features = pca_model.transform(features).astype("float32")
            y_pred = sklearn_model.predict_proba(features)[:, 1].astype("float32")  # Use probabilities for AUROC
        elif baseline_name == "uncertainty_gru":
            if torch_model is None:
                raise FileNotFoundError(f"Missing model.pt for {baseline_name} at {output_dir}")
            sequences = build_uncertainty_sequence_examples(
                eval_df,
                max_tokens=uncertainty_sequence_max_tokens,
            )
            y_pred = predict_sequence_regressor(
                torch_model,
                sequences,
                device=device,
                batch_size=sequence_batch_size,
            ).astype("float32")
        elif baseline_name == "hidden_fixed":
            if sklearn_model is None:
                raise FileNotFoundError(f"Missing model.joblib for {baseline_name} at {output_dir}")
            features = build_recent_sentence_fixed_features(
                eval_df,
                k_recent_sentences=k_recent_sentences,
                sentence_representation=sentence_representation,
                l2_normalize=hidden_l2_normalize,
            )
            if pca_model is not None:
                features = pca_model.transform(features).astype("float32")
            y_pred = sklearn_model.predict(features).astype("float32")
        elif baseline_name == "hidden_gru":
            if torch_model is None:
                raise FileNotFoundError(f"Missing model.pt for {baseline_name} at {output_dir}")
            sequences = build_recent_sentence_sequence_examples(
                eval_df,
                k_recent_sentences=k_recent_sentences,
                sentence_representation=sentence_representation,
                l2_normalize=hidden_l2_normalize,
            )
            y_pred = predict_sequence_regressor(
                torch_model,
                sequences,
                device=device,
                batch_size=sequence_batch_size,
            ).astype("float32")
        else:
            raise ValueError(f"Unsupported baseline for OOD eval: {baseline_name}")

        y_true = eval_df["target_value"].to_numpy(dtype="float32", copy=False)
        metrics = compute_regression_metrics(y_true, y_pred, binary_threshold=binary_threshold)
        eval_name = dataset_label(eval_df)
        metrics_rows.append(
            {
                "baseline_name": baseline_name,
                "eval_dataset": eval_name,
                "eval_split": eval_split,
                "n_rows": int(len(eval_df)),
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "pearson": metrics.pearson,
                "spearman": metrics.spearman,
                "auroc_at_threshold": metrics.auroc_at_threshold,
                "average_precision_at_threshold": metrics.average_precision_at_threshold,
            }
        )
        pred_df = eval_df.loc[:, ["dataset", "example_id", "sentence_idx", "target_value", "split"]].copy()
        pred_df["ood_eval_split"] = eval_split
        pred_df["prediction"] = y_pred
        prediction_frames.append(pred_df)

    if torch_model is not None:
        del torch_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return (
        pd.DataFrame(metrics_rows),
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    split_config = GroupSplitConfig(
        train_size=float(args.train_size),
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    paths, dataset_df = load_dataset_with_splits(
        args.input_path,
        cache_dir=args.cache_dir,
        split_overwrite=bool(args.split_overwrite),
        split_config=split_config,
    )
    baseline_names = parse_baseline_list(args.baselines)
    ood_input_paths = parse_path_list(args.ood_input_paths)
    ood_datasets: list[tuple[Any, pd.DataFrame]] = []
    seen_ood_labels: set[str] = set()
    for ood_input_path in ood_input_paths:
        ood_paths, ood_df = load_dataset_with_splits(
            ood_input_path,
            cache_dir=None,
            split_overwrite=bool(args.split_overwrite),
            split_config=split_config,
        )
        label = dataset_label(ood_df)
        if label == dataset_label(dataset_df):
            continue
        if label in seen_ood_labels:
            continue
        seen_ood_labels.add(label)
        ood_datasets.append((ood_paths, ood_df))
    device, gpu_df = resolve_device(args.device)

    print(f"Dataset frame: {paths.dataset_frame_path}")
    print(f"Cache dir: {paths.cache_dir}")
    print(f"Split file: {paths.split_path}")
    print(f"Source dataset: {dataset_label(dataset_df)}")
    print(f"Device: {device}")
    print("Rows by split:")
    print(dataset_df["split"].value_counts(dropna=False).sort_index().to_string())
    if ood_datasets:
        print("OOD datasets:")
        for _ood_paths, ood_df in ood_datasets:
            print(f"  - {dataset_label(ood_df)} :: {len(ood_df):,} rows :: { _ood_paths.dataset_frame_path }")
    if not gpu_df.empty:
        print("Visible GPUs:")
        print(gpu_df.to_string(index=False))

    summary_rows = []
    baseline_iter = maybe_tqdm(
        baseline_names,
        desc="Baselines",
        total=len(baseline_names),
        disable=bool(args.disable_tqdm),
    )
    for baseline_name in baseline_iter:
        output_dir = paths.train_output_dir / baseline_name
        print(f"\n[baseline] {baseline_name}")

        if baseline_name == "text_embedding":
            result = run_text_embedding_ridge_baseline(
                dataset_df,
                output_dir=output_dir,
                embedding_cache_dir=paths.text_embedding_cache_dir,
                model_name=args.text_embedding_model,
                device=device,
                batch_size=int(args.text_embedding_batch_size),
                binary_threshold=args.binary_threshold,
                show_progress=not bool(args.disable_tqdm),
                overwrite_embedding_cache=bool(args.text_embedding_overwrite_cache),
            )
        elif baseline_name in {"text_cbow_last_sentence", "text_cbow_full_prefix"}:
            result = run_text_cbow_ridge_baseline(
                dataset_df,
                output_dir=output_dir,
                max_features=int(args.cbow_max_features),
                ngram_range=(int(args.cbow_min_ngram), int(args.cbow_max_ngram)),
                binary_threshold=args.binary_threshold,
                text_column=cbow_text_column_for_baseline(baseline_name),
                baseline_name=baseline_name,
            )
        elif baseline_name == "uncertainty_pooled":
            result = run_uncertainty_pooled_ridge_baseline(
                dataset_df,
                output_dir=output_dir,
                recent_window_tokens=int(args.uncertainty_recent_window_sentences),
                binary_threshold=args.binary_threshold,
                pca_dim=int(args.uncertainty_pca_dim),
            )
        elif baseline_name == "uncertainty_pooled_logistic":
            result = run_uncertainty_pooled_logistic_baseline(
                dataset_df,
                output_dir=output_dir,
                recent_window_tokens=int(args.uncertainty_recent_window_sentences),
                pca_dim=int(args.uncertainty_pca_dim),
            )
        elif baseline_name == "uncertainty_gru":
            result = run_uncertainty_gru_baseline(
                dataset_df,
                output_dir=output_dir,
                device=device,
                max_tokens=int(args.uncertainty_sequence_max_tokens),
                hidden_dim=int(args.sequence_hidden_dim),
                batch_size=int(args.sequence_batch_size),
                num_epochs=int(args.sequence_num_epochs),
                patience=int(args.sequence_patience),
                binary_threshold=args.binary_threshold,
                show_progress=not bool(args.disable_tqdm),
            )
        elif baseline_name == "hidden_fixed":
            result = run_hidden_recent_fixed_ridge_baseline(
                dataset_df,
                output_dir=output_dir,
                k_recent_sentences=int(args.k_recent_sentences),
                sentence_representation=str(args.sentence_representation),
                l2_normalize=str(args.hidden_l2_normalize),
                pca_dim=int(args.uncertainty_pca_dim),
                binary_threshold=args.binary_threshold,
            )
        elif baseline_name == "hidden_gru":
            result = run_hidden_recent_gru_baseline(
                dataset_df,
                output_dir=output_dir,
                device=device,
                k_recent_sentences=int(args.k_recent_sentences),
                sentence_representation=str(args.sentence_representation),
                l2_normalize=str(args.hidden_l2_normalize),
                hidden_dim=int(args.sequence_hidden_dim),
                batch_size=int(args.sequence_batch_size),
                num_epochs=int(args.sequence_num_epochs),
                patience=int(args.sequence_patience),
                binary_threshold=args.binary_threshold,
                show_progress=not bool(args.disable_tqdm),
            )
        else:
            raise ValueError(f"Unsupported baseline: {baseline_name}")

        val_row = result.metrics_by_split.loc[result.metrics_by_split["split"] == "val"].iloc[0].to_dict()
        test_row = result.metrics_by_split.loc[result.metrics_by_split["split"] == "test"].iloc[0].to_dict()
        ood_metrics_df = pd.DataFrame()
        if ood_datasets:
            ood_metrics_df, ood_predictions_df = evaluate_ood_for_baseline(
                baseline_name=baseline_name,
                output_dir=output_dir,
                eval_datasets=ood_datasets,
                eval_split=str(args.ood_eval_split),
                device=device,
                binary_threshold=args.binary_threshold,
                text_embedding_model=str(args.text_embedding_model),
                text_embedding_batch_size=int(args.text_embedding_batch_size),
                uncertainty_recent_window_sentences=int(args.uncertainty_recent_window_sentences),
                uncertainty_sequence_max_tokens=int(args.uncertainty_sequence_max_tokens),
                k_recent_sentences=int(args.k_recent_sentences),
                sentence_representation=str(args.sentence_representation),
                hidden_l2_normalize=str(args.hidden_l2_normalize),
                sequence_batch_size=int(args.sequence_batch_size),
                disable_tqdm=bool(args.disable_tqdm),
            )
            if not ood_metrics_df.empty:
                ood_metrics_df.to_csv(output_dir / "ood_metrics.csv", index=False)
                ood_predictions_df.to_parquet(output_dir / "ood_predictions.parquet", index=False)

        ood_mean_rmse = float(ood_metrics_df["rmse"].mean()) if not ood_metrics_df.empty else float("nan")
        ood_mean_pearson = float(ood_metrics_df["pearson"].mean()) if not ood_metrics_df.empty else float("nan")
        ood_mean_auroc = (
            float(ood_metrics_df["auroc_at_threshold"].dropna().mean())
            if not ood_metrics_df.empty and ood_metrics_df["auroc_at_threshold"].notna().any()
            else float("nan")
        )
        summary_rows.append(
            {
                "baseline_name": baseline_name,
                "output_dir": str(result.output_dir),
                "val_rmse": val_row["rmse"],
                "val_mae": val_row["mae"],
                "val_r2": val_row["r2"],
                "val_pearson": val_row["pearson"],
                "test_rmse": test_row["rmse"],
                "test_mae": test_row["mae"],
                "test_r2": test_row["r2"],
                "test_pearson": test_row["pearson"],
                "ood_target_count": int(len(ood_metrics_df)),
                "ood_mean_rmse": ood_mean_rmse,
                "ood_mean_pearson": ood_mean_pearson,
                "ood_mean_auroc_at_threshold": ood_mean_auroc,
            }
        )
        print(result.metrics_by_split.to_string(index=False))
        if not ood_metrics_df.empty:
            print("\n[ood]")
            print(ood_metrics_df.to_string(index=False))

    sort_columns = ["val_rmse", "test_rmse"]
    if ood_datasets:
        sort_columns = ["ood_mean_rmse", "val_rmse", "test_rmse"]
    summary_df = pd.DataFrame(summary_rows).sort_values(sort_columns, ascending=[True] * len(sort_columns))
    summary_path = paths.train_output_dir / "summary.csv"
    paths.train_output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    (paths.train_output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    print("\n[summary]")
    print(summary_df.to_string(index=False))
    print(f"\nWrote run summary to: {summary_path}")


if __name__ == "__main__":
    main()
