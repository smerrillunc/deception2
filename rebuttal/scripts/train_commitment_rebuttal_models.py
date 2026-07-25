#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from commitment_rebuttal_lib import (
    DEFAULT_ENVIRONMENTS,
    DEFAULT_FIXED_RECALL_LEVELS,
    DEFAULT_LABEL_KINDS,
    DEFAULT_MODEL_BUNDLE_NAMES,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_NAME,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TAU_VALUES,
    attach_splits,
    bundle_specs,
    calibration_curve_frame,
    choose_present_columns,
    fpr_at_fixed_recalls,
    label_column_name,
    main_feature_numeric_columns,
    numeric_frame,
    parse_csv_list,
    parse_recall_levels,
    parse_tau_values,
    prefix_target_baselines,
    row_level_split_assignments,
    run_root_for_name,
    safe_binary_metrics,
    structural_feature_columns,
    training_root,
    valid_binary_mask,
)

try:
    import xgboost as xgb
except Exception:  # noqa: BLE001
    xgb = None


DEFAULT_FEATURE_SPACES = (
    "main_xgb",
    "position_only_logreg",
    "length_only_logreg",
    "structural_nonsemantic_logreg",
    "tfidf_prefix_logreg",
    "tfidf_sentence_logreg",
    "sentence_embedding_logreg",
)

DEFAULT_SCENARIOS = ("single_source_ood", "holdout_env_ood")


@dataclass(frozen=True)
class FeatureSpaceData:
    name: str
    frame: pd.DataFrame
    data_kind: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train commitment-juncture rebuttal classifiers from the rebuttal bundle artifacts. "
            "Outputs include AUROC, PR-AUC, calibration curves, fixed-recall FPR tables, "
            "and target-environment breakdowns."
        )
    )
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--tau-values", type=str, default=",".join(str(value) for value in DEFAULT_TAU_VALUES))
    parser.add_argument("--label-kinds", type=str, default=",".join(DEFAULT_LABEL_KINDS))
    parser.add_argument("--feature-spaces", type=str, default=",".join(DEFAULT_FEATURE_SPACES))
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--model-bundle-names", type=str, default=",".join(DEFAULT_MODEL_BUNDLE_NAMES))
    parser.add_argument("--environments", type=str, default=",".join(DEFAULT_ENVIRONMENTS))
    parser.add_argument("--seeds", type=str, default=str(DEFAULT_SPLIT_SEED))
    parser.add_argument("--fixed-recall-levels", type=str, default=",".join(str(value) for value in DEFAULT_FIXED_RECALL_LEVELS))
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--embedding-model", type=str, default=prefix_target_baselines.DEFAULT_TEXT_EMBED_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", help="Used for sentence embedding extraction.")
    parser.add_argument("--max-tfidf-features", type=int, default=20000)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    return parser.parse_args(argv)


def read_bundle_shards(bundle_dir: Path) -> pd.DataFrame:
    shard_paths = sorted(bundle_dir.glob("*.parquet"))
    if not shard_paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in shard_paths]
    return pd.concat(frames, ignore_index=True)


def structural_bundle_frame(run_root: Path, env_name: str, model_bundle_name: str) -> pd.DataFrame:
    path = run_root / "bundles" / f"{env_name}__{model_bundle_name}" / "structural"
    return read_bundle_shards(path)


def main_feature_bundle_frame(run_root: Path, env_name: str, model_bundle_name: str) -> pd.DataFrame:
    path = run_root / "bundles" / f"{env_name}__{model_bundle_name}" / "main_features"
    return read_bundle_shards(path)


def load_model_data(
    *,
    run_root: Path,
    environments: Sequence[str],
    model_bundle_name: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural_frames: list[pd.DataFrame] = []
    main_frames: list[pd.DataFrame] = []
    for env_name in environments:
        structural_df = structural_bundle_frame(run_root, env_name, model_bundle_name)
        if structural_df.empty:
            continue
        structural_frames.append(structural_df)
        main_df = main_feature_bundle_frame(run_root, env_name, model_bundle_name)
        if not main_df.empty:
            main_frames.append(main_df)

    if not structural_frames:
        return pd.DataFrame(), pd.DataFrame()

    structural_df = pd.concat(structural_frames, ignore_index=True)
    split_df = row_level_split_assignments(structural_df, seed=int(seed))
    structural_df = attach_splits(structural_df, split_df)

    if not main_frames:
        return structural_df, pd.DataFrame()

    main_df = pd.concat(main_frames, ignore_index=True)
    key_columns = ["env_name", "model_bundle_name", "example_id", "sentence_idx"]
    structural_keys = structural_df.loc[:, key_columns + ["split"]].drop_duplicates()
    main_df = main_df.merge(structural_keys, on=key_columns, how="inner", validate="many_to_one")
    return structural_df, main_df


def build_feature_space_data(
    *,
    structural_df: pd.DataFrame,
    main_df: pd.DataFrame,
    requested_feature_spaces: Sequence[str],
) -> list[FeatureSpaceData]:
    spaces: list[FeatureSpaceData] = []
    requested = set(requested_feature_spaces)
    if "main_xgb" in requested and not main_df.empty:
        spaces.append(FeatureSpaceData(name="main_xgb", frame=main_df.copy(), data_kind="main"))
    structural_names = {"position_only_logreg", "length_only_logreg", "structural_nonsemantic_logreg"}
    for name in structural_names:
        if name in requested and not structural_df.empty:
            spaces.append(FeatureSpaceData(name=name, frame=structural_df.copy(), data_kind="structural"))
    text_names = {"tfidf_prefix_logreg", "tfidf_sentence_logreg", "sentence_embedding_logreg"}
    for name in text_names:
        if name in requested and not structural_df.empty:
            spaces.append(FeatureSpaceData(name=name, frame=structural_df.copy(), data_kind="structural"))
    return spaces


def filter_labeled_rows(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    mask = valid_binary_mask(df[label_col])
    out = df.loc[mask].copy()
    out["label"] = pd.to_numeric(out[label_col], errors="coerce").astype(int)
    return out


def frame_for_single_source(df: pd.DataFrame, source_env: str) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str, pd.DataFrame]]]:
    train_df = df.loc[(df["env_name"].astype(str) == str(source_env)) & (df["split"].astype(str) == "train")].copy()
    val_df = df.loc[(df["env_name"].astype(str) == str(source_env)) & (df["split"].astype(str) == "val")].copy()
    eval_sets: list[tuple[str, str, pd.DataFrame]] = []
    id_test_df = df.loc[(df["env_name"].astype(str) == str(source_env)) & (df["split"].astype(str) == "test")].copy()
    eval_sets.append(("id_test", source_env, id_test_df))
    for target_env in sorted(df["env_name"].astype(str).unique().tolist()):
        if target_env == source_env:
            continue
        target_df = df.loc[(df["env_name"].astype(str) == str(target_env)) & (df["split"].astype(str) == "test")].copy()
        eval_sets.append(("ood_test", target_env, target_df))
    return train_df, val_df, eval_sets


def frame_for_holdout_env(df: pd.DataFrame, target_env: str) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str, pd.DataFrame]]]:
    train_df = df.loc[(df["env_name"].astype(str) != str(target_env)) & (df["split"].astype(str) == "train")].copy()
    val_df = df.loc[(df["env_name"].astype(str) != str(target_env)) & (df["split"].astype(str) == "val")].copy()
    target_df = df.loc[(df["env_name"].astype(str) == str(target_env)) & (df["split"].astype(str) == "test")].copy()
    mix_test_df = df.loc[(df["env_name"].astype(str) != str(target_env)) & (df["split"].astype(str) == "test")].copy()
    eval_sets = [
        ("source_mix_test", "all_but_" + str(target_env), mix_test_df),
        ("ood_test", target_env, target_df),
    ]
    return train_df, val_df, eval_sets


def train_numeric_xgb(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    *,
    seed: int,
) -> Any:
    if xgb is None:
        raise ImportError("xgboost is unavailable in the active environment.")
    pos = max(1, int(train_y.sum()))
    neg = max(1, int((1 - train_y).sum()))
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(seed),
        scale_pos_weight=float(neg / pos),
    )
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if len(val_x) > 0 and np.unique(val_y).size >= 1:
        fit_kwargs["eval_set"] = [(val_x, val_y)]
    try:
        fit_kwargs["early_stopping_rounds"] = 50
        model.fit(train_x, train_y, **fit_kwargs)
    except TypeError:
        fit_kwargs.pop("early_stopping_rounds", None)
        model.fit(train_x, train_y, **fit_kwargs)
    return model


def train_numeric_logreg(train_x: pd.DataFrame, train_y: np.ndarray, *, seed: int) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(seed))),
        ]
    )
    pipeline.fit(train_x, train_y)
    return pipeline


def train_text_logreg(train_x: sparse.spmatrix, train_y: np.ndarray, *, seed: int) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(seed))
    model.fit(train_x, train_y)
    return model


def evaluate_predictions(
    *,
    feature_space_name: str,
    scenario: str,
    model_bundle_name: str,
    label_kind: str,
    tau: float,
    seed: int,
    source_env: str,
    eval_kind: str,
    target_env: str,
    eval_df: pd.DataFrame,
    y_score: np.ndarray,
    calibration_bins: int,
    fixed_recall_levels: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = eval_df["label"].to_numpy(dtype=np.int64, copy=False)
    metric_row = {
        "feature_space": feature_space_name,
        "scenario": scenario,
        "model_bundle_name": model_bundle_name,
        "label_kind": label_kind,
        "tau": float(tau),
        "seed": int(seed),
        "source_env": source_env,
        "target_env": target_env,
        "eval_kind": eval_kind,
        "row_count": int(len(eval_df)),
        "example_count": int(eval_df["example_id"].astype(str).nunique()) if not eval_df.empty else 0,
    }
    metric_row.update(safe_binary_metrics(y_true, y_score))
    metrics_df = pd.DataFrame([metric_row])

    calibration_df = calibration_curve_frame(
        y_true=y_true,
        y_score=y_score,
        n_bins=int(calibration_bins),
    )
    if not calibration_df.empty:
        calibration_df["feature_space"] = feature_space_name
        calibration_df["scenario"] = scenario
        calibration_df["model_bundle_name"] = model_bundle_name
        calibration_df["label_kind"] = label_kind
        calibration_df["tau"] = float(tau)
        calibration_df["seed"] = int(seed)
        calibration_df["source_env"] = source_env
        calibration_df["target_env"] = target_env
        calibration_df["eval_kind"] = eval_kind

    fpr_df = fpr_at_fixed_recalls(
        y_true=y_true,
        y_score=y_score,
        recall_levels=fixed_recall_levels,
    )
    if not fpr_df.empty:
        fpr_df["feature_space"] = feature_space_name
        fpr_df["scenario"] = scenario
        fpr_df["model_bundle_name"] = model_bundle_name
        fpr_df["label_kind"] = label_kind
        fpr_df["tau"] = float(tau)
        fpr_df["seed"] = int(seed)
        fpr_df["source_env"] = source_env
        fpr_df["target_env"] = target_env
        fpr_df["eval_kind"] = eval_kind

    prediction_df = eval_df.loc[
        :,
        ["env_name", "model_bundle_name", "example_id", "sentence_idx", "split", "label"],
    ].copy()
    prediction_df["score"] = y_score.astype(np.float32)
    prediction_df["feature_space"] = feature_space_name
    prediction_df["scenario"] = scenario
    prediction_df["label_kind"] = label_kind
    prediction_df["tau"] = float(tau)
    prediction_df["seed"] = int(seed)
    prediction_df["source_env"] = source_env
    prediction_df["target_env"] = target_env
    prediction_df["eval_kind"] = eval_kind
    return metrics_df, calibration_df, fpr_df, prediction_df


def tfidf_features_for_sets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    text_column: str,
    max_features: int,
    ngram_range: tuple[int, int],
) -> tuple[sparse.spmatrix, sparse.spmatrix, sparse.spmatrix]:
    vectorizer = TfidfVectorizer(
        max_features=int(max_features),
        ngram_range=tuple(int(value) for value in ngram_range),
        lowercase=True,
        token_pattern=r"(?u)\b[\w']+\b",
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_x = vectorizer.fit_transform(train_df[text_column].fillna("").astype(str))
    val_x = vectorizer.transform(val_df[text_column].fillna("").astype(str))
    eval_x = vectorizer.transform(eval_df[text_column].fillna("").astype(str))
    return train_x, val_x, eval_x


def embedding_matrix_for_frame(
    frame: pd.DataFrame,
    *,
    text_column: str,
    cache_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    disable_tqdm: bool,
) -> np.ndarray:
    embed_df = frame.copy()
    embed_df["full_prefix_text"] = embed_df[text_column].fillna("").astype(str)
    return prefix_target_baselines.compute_or_load_text_embeddings(
        embed_df,
        cache_dir=cache_dir,
        model_name=model_name,
        device=device,
        batch_size=int(batch_size),
        show_progress=not disable_tqdm,
        overwrite=False,
    )


def train_and_score_feature_space(
    *,
    feature_space: FeatureSpaceData,
    scenario: str,
    model_bundle_name: str,
    label_kind: str,
    tau: float,
    seed: int,
    source_env: str,
    target_env: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    calibration_bins: int,
    fixed_recall_levels: Sequence[float],
    embedding_model: str,
    embedding_batch_size: int,
    device: str,
    max_tfidf_features: int,
    tfidf_ngram_range: tuple[int, int],
    disable_tqdm: bool,
    model_output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if feature_space.name == "main_xgb":
        columns = main_feature_numeric_columns(feature_space.frame)
        train_x = numeric_frame(train_df, columns)
        val_x = numeric_frame(val_df, columns)
        eval_x = numeric_frame(eval_df, columns)
        model = train_numeric_xgb(
            train_x,
            train_df["label"].to_numpy(dtype=np.int64, copy=False),
            val_x,
            val_df["label"].to_numpy(dtype=np.int64, copy=False),
            seed=seed,
        )
        y_score = model.predict_proba(eval_x)[:, 1]
    elif feature_space.name in {"position_only_logreg", "length_only_logreg", "structural_nonsemantic_logreg"}:
        baseline_name = feature_space.name.replace("_logreg", "")
        columns = structural_feature_columns(feature_space.frame, baseline_name)
        train_x = numeric_frame(train_df, columns)
        eval_x = numeric_frame(eval_df, columns)
        model = train_numeric_logreg(
            train_x,
            train_df["label"].to_numpy(dtype=np.int64, copy=False),
            seed=seed,
        )
        y_score = model.predict_proba(eval_x)[:, 1]
    elif feature_space.name == "tfidf_prefix_logreg":
        train_x, _val_x, eval_x = tfidf_features_for_sets(
            train_df,
            val_df,
            eval_df,
            text_column="full_prefix_text",
            max_features=max_tfidf_features,
            ngram_range=tfidf_ngram_range,
        )
        model = train_text_logreg(train_x, train_df["label"].to_numpy(dtype=np.int64, copy=False), seed=seed)
        y_score = model.predict_proba(eval_x)[:, 1]
    elif feature_space.name == "tfidf_sentence_logreg":
        train_x, _val_x, eval_x = tfidf_features_for_sets(
            train_df,
            val_df,
            eval_df,
            text_column="sentence_text",
            max_features=max_tfidf_features,
            ngram_range=tfidf_ngram_range,
        )
        model = train_text_logreg(train_x, train_df["label"].to_numpy(dtype=np.int64, copy=False), seed=seed)
        y_score = model.predict_proba(eval_x)[:, 1]
    elif feature_space.name == "sentence_embedding_logreg":
        embedding_cache_dir = model_output_root / "embedding_cache" / model_bundle_name / feature_space.name
        full_embeddings = embedding_matrix_for_frame(
            feature_space.frame,
            text_column="full_prefix_text",
            cache_dir=embedding_cache_dir,
            model_name=embedding_model,
            device=device,
            batch_size=embedding_batch_size,
            disable_tqdm=disable_tqdm,
        )
        row_lookup = feature_space.frame.reset_index(drop=True).loc[:, ["env_name", "example_id", "sentence_idx"]].copy()
        row_lookup["row_idx"] = np.arange(len(row_lookup), dtype=np.int64)

        def row_indices(subset: pd.DataFrame) -> np.ndarray:
            merged = subset.loc[:, ["env_name", "example_id", "sentence_idx"]].merge(
                row_lookup,
                on=["env_name", "example_id", "sentence_idx"],
                how="left",
                validate="many_to_one",
            )
            if merged["row_idx"].isna().any():
                raise ValueError("Failed to align embedding rows for a subset.")
            return merged["row_idx"].to_numpy(dtype=np.int64, copy=False)

        train_x = full_embeddings[row_indices(train_df)]
        eval_x = full_embeddings[row_indices(eval_df)]
        model = train_numeric_logreg(
            pd.DataFrame(train_x),
            train_df["label"].to_numpy(dtype=np.int64, copy=False),
            seed=seed,
        )
        y_score = model.predict_proba(pd.DataFrame(eval_x))[:, 1]
    else:
        raise ValueError(f"Unsupported feature space: {feature_space.name}")

    eval_kind = "ood_test" if scenario == "holdout_env_ood" or target_env != source_env else "id_test"
    return evaluate_predictions(
        feature_space_name=feature_space.name,
        scenario=scenario,
        model_bundle_name=model_bundle_name,
        label_kind=label_kind,
        tau=float(tau),
        seed=int(seed),
        source_env=source_env,
        eval_kind=eval_kind,
        target_env=target_env,
        eval_df=eval_df,
        y_score=np.asarray(y_score, dtype=np.float64),
        calibration_bins=calibration_bins,
        fixed_recall_levels=fixed_recall_levels,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_root = run_root_for_name(args.run_name, args.results_root)
    output_root = training_root(run_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tau_values = parse_tau_values(args.tau_values)
    label_kinds = parse_csv_list(args.label_kinds) or list(DEFAULT_LABEL_KINDS)
    feature_space_names = parse_csv_list(args.feature_spaces) or list(DEFAULT_FEATURE_SPACES)
    scenario_names = parse_csv_list(args.scenarios) or list(DEFAULT_SCENARIOS)
    model_bundle_names = parse_csv_list(args.model_bundle_names) or list(DEFAULT_MODEL_BUNDLE_NAMES)
    environments = parse_csv_list(args.environments) or list(DEFAULT_ENVIRONMENTS)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]
    fixed_recall_levels = parse_recall_levels(args.fixed_recall_levels)
    device, _gpu_df = prefix_target_baselines.resolve_device(args.device)

    metrics_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    fpr_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for model_bundle_name in model_bundle_names:
            structural_df, main_df = load_model_data(
                run_root=run_root,
                environments=environments,
                model_bundle_name=model_bundle_name,
                seed=int(seed),
            )
            if structural_df.empty:
                error_rows.append(
                    {
                        "seed": int(seed),
                        "model_bundle_name": model_bundle_name,
                        "error": "missing_structural_data",
                    }
                )
                continue

            inventory_rows.append(
                {
                    "seed": int(seed),
                    "model_bundle_name": model_bundle_name,
                    "structural_rows": int(len(structural_df)),
                    "structural_examples": int(structural_df["example_id"].astype(str).nunique()),
                    "main_rows": int(len(main_df)),
                    "main_examples": int(main_df["example_id"].astype(str).nunique()) if not main_df.empty else 0,
                }
            )

            feature_spaces = build_feature_space_data(
                structural_df=structural_df,
                main_df=main_df,
                requested_feature_spaces=feature_space_names,
            )

            for feature_space in feature_spaces:
                for label_kind in label_kinds:
                    for tau in tau_values:
                        label_col = label_column_name(label_kind, tau)
                        if label_col not in feature_space.frame.columns:
                            error_rows.append(
                                {
                                    "seed": int(seed),
                                    "model_bundle_name": model_bundle_name,
                                    "feature_space": feature_space.name,
                                    "label_kind": label_kind,
                                    "tau": float(tau),
                                    "error": f"missing_label_column:{label_col}",
                                }
                            )
                            continue
                        labeled_df = filter_labeled_rows(feature_space.frame, label_col)
                        if labeled_df.empty:
                            error_rows.append(
                                {
                                    "seed": int(seed),
                                    "model_bundle_name": model_bundle_name,
                                    "feature_space": feature_space.name,
                                    "label_kind": label_kind,
                                    "tau": float(tau),
                                    "error": "no_labeled_rows",
                                }
                            )
                            continue

                        for scenario in scenario_names:
                            if scenario == "single_source_ood":
                                for source_env in environments:
                                    train_df, val_df, eval_sets = frame_for_single_source(labeled_df, source_env)
                                    if train_df.empty or np.unique(train_df["label"]).size < 2:
                                        continue
                                    for eval_kind, target_env, eval_df in eval_sets:
                                        if eval_df.empty:
                                            continue
                                        try:
                                            metrics_df, calibration_df, fpr_df, prediction_df = train_and_score_feature_space(
                                                feature_space=feature_space,
                                                scenario=scenario,
                                                model_bundle_name=model_bundle_name,
                                                label_kind=label_kind,
                                                tau=float(tau),
                                                seed=int(seed),
                                                source_env=source_env,
                                                target_env=target_env,
                                                train_df=train_df,
                                                val_df=val_df,
                                                eval_df=eval_df,
                                                calibration_bins=int(args.calibration_bins),
                                                fixed_recall_levels=fixed_recall_levels,
                                                embedding_model=args.embedding_model,
                                                embedding_batch_size=int(args.embedding_batch_size),
                                                device=device,
                                                max_tfidf_features=int(args.max_tfidf_features),
                                                tfidf_ngram_range=(int(args.tfidf_ngram_min), int(args.tfidf_ngram_max)),
                                                disable_tqdm=bool(args.disable_tqdm),
                                                model_output_root=output_root,
                                            )
                                        except Exception as exc:  # noqa: BLE001
                                            error_rows.append(
                                                {
                                                    "seed": int(seed),
                                                    "model_bundle_name": model_bundle_name,
                                                    "feature_space": feature_space.name,
                                                    "label_kind": label_kind,
                                                    "tau": float(tau),
                                                    "scenario": scenario,
                                                    "source_env": source_env,
                                                    "target_env": target_env,
                                                    "error": repr(exc),
                                                }
                                            )
                                            continue
                                        metrics_df["requested_eval_kind"] = eval_kind
                                        metrics_frames.append(metrics_df)
                                        if not calibration_df.empty:
                                            calibration_frames.append(calibration_df)
                                        if not fpr_df.empty:
                                            fpr_frames.append(fpr_df)
                                        prediction_frames.append(prediction_df)
                            elif scenario == "holdout_env_ood":
                                for target_env in environments:
                                    train_df, val_df, eval_sets = frame_for_holdout_env(labeled_df, target_env)
                                    if train_df.empty or np.unique(train_df["label"]).size < 2:
                                        continue
                                    for eval_kind, eval_target_env, eval_df in eval_sets:
                                        if eval_df.empty:
                                            continue
                                        source_env_label = ",".join(sorted(env for env in environments if env != target_env))
                                        try:
                                            metrics_df, calibration_df, fpr_df, prediction_df = train_and_score_feature_space(
                                                feature_space=feature_space,
                                                scenario=scenario,
                                                model_bundle_name=model_bundle_name,
                                                label_kind=label_kind,
                                                tau=float(tau),
                                                seed=int(seed),
                                                source_env=source_env_label,
                                                target_env=eval_target_env,
                                                train_df=train_df,
                                                val_df=val_df,
                                                eval_df=eval_df,
                                                calibration_bins=int(args.calibration_bins),
                                                fixed_recall_levels=fixed_recall_levels,
                                                embedding_model=args.embedding_model,
                                                embedding_batch_size=int(args.embedding_batch_size),
                                                device=device,
                                                max_tfidf_features=int(args.max_tfidf_features),
                                                tfidf_ngram_range=(int(args.tfidf_ngram_min), int(args.tfidf_ngram_max)),
                                                disable_tqdm=bool(args.disable_tqdm),
                                                model_output_root=output_root,
                                            )
                                        except Exception as exc:  # noqa: BLE001
                                            error_rows.append(
                                                {
                                                    "seed": int(seed),
                                                    "model_bundle_name": model_bundle_name,
                                                    "feature_space": feature_space.name,
                                                    "label_kind": label_kind,
                                                    "tau": float(tau),
                                                    "scenario": scenario,
                                                    "source_env": source_env_label,
                                                    "target_env": eval_target_env,
                                                    "error": repr(exc),
                                                }
                                            )
                                            continue
                                        metrics_df["requested_eval_kind"] = eval_kind
                                        metrics_frames.append(metrics_df)
                                        if not calibration_df.empty:
                                            calibration_frames.append(calibration_df)
                                        if not fpr_df.empty:
                                            fpr_frames.append(fpr_df)
                                        prediction_frames.append(prediction_df)
                            else:
                                raise ValueError(f"Unsupported scenario: {scenario}")

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    calibration_df = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    fpr_df = pd.concat(fpr_frames, ignore_index=True) if fpr_frames else pd.DataFrame()
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    inventory_df = pd.DataFrame(inventory_rows)
    errors_df = pd.DataFrame(error_rows)

    config = {
        "run_root": str(run_root),
        "tau_values": [float(value) for value in tau_values],
        "label_kinds": list(label_kinds),
        "feature_spaces": list(feature_space_names),
        "scenarios": list(scenario_names),
        "model_bundle_names": list(model_bundle_names),
        "environments": list(environments),
        "seeds": [int(value) for value in seeds],
        "fixed_recall_levels": [float(value) for value in fixed_recall_levels],
        "calibration_bins": int(args.calibration_bins),
        "embedding_model": str(args.embedding_model),
        "embedding_batch_size": int(args.embedding_batch_size),
        "device": str(device),
        "max_tfidf_features": int(args.max_tfidf_features),
        "tfidf_ngram_range": [int(args.tfidf_ngram_min), int(args.tfidf_ngram_max)],
    }
    (output_root / "commitment_rebuttal_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    inventory_df.to_csv(output_root / "commitment_rebuttal_inventory.csv", index=False)
    metrics_df.to_csv(output_root / "commitment_rebuttal_metrics.csv", index=False)
    calibration_df.to_csv(output_root / "commitment_rebuttal_calibration.csv", index=False)
    fpr_df.to_csv(output_root / "commitment_rebuttal_fpr_at_recall.csv", index=False)
    errors_df.to_csv(output_root / "commitment_rebuttal_errors.csv", index=False)
    if not predictions_df.empty:
        predictions_df.to_parquet(output_root / "commitment_rebuttal_predictions.parquet", index=False)

    print(f"Wrote training outputs to: {output_root}")
    print(f"Inventory rows: {len(inventory_df):,}")
    print(f"Metric rows: {len(metrics_df):,}")
    print(f"Calibration rows: {len(calibration_df):,}")
    print(f"FPR rows: {len(fpr_df):,}")
    print(f"Prediction rows: {len(predictions_df):,}")
    print(f"Error rows: {len(errors_df):,}")


if __name__ == "__main__":
    main()
