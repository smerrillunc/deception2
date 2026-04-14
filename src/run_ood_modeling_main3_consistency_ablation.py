#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import runpy
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
DEFAULT_SOURCE = REPO_ROOT / "Notebooks" / "OOD_Modeling_main3_consistency_ablation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Main3 consistency + feature-size OOD ablation as a script. "
            "This wraps the notebook-source workflow and writes all CSV/PNG artifacts "
            "needed to analyze AUROC matrices, confusion matrices, and best features."
        )
    )
    parser.add_argument(
        "--model-dirname",
        default="DeepSeek-R1-Distill-Qwen-7B",
        help="DatasetMain model subdirectory name, e.g. DeepSeek-R1-Distill-Qwen-7B or gpt-oss-20b.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Explicit DatasetMain root. Useful on longleaf when data lives outside the repo checkout.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional explicit output directory. Defaults to the notebook's family-aware output path.",
    )
    parser.add_argument(
        "--model-family",
        default="logreg",
        help="Model family to train for this run, e.g. logreg or xgb.",
    )
    parser.add_argument(
        "--feature-sizes",
        default="32,64,128,256",
        help="Comma-separated feature-size sweep for attention and PCA activation features.",
    )
    parser.add_argument(
        "--attention-top-k",
        type=int,
        default=None,
        help="Maximum ranked attention features to cache before slicing to each feature size.",
    )
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=0.1,
        help="Fixed logistic-regression C value.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=5,
        help="Fixed XGBoost max_depth value.",
    )
    parser.add_argument(
        "--c-grid",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--xgb-max-depth-grid",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val splits and PCA.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
        help="Validation split fraction within each environment.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.30,
        help="Threshold for the delta deception-rate targets.",
    )
    parser.add_argument(
        "--per-root-limit",
        type=int,
        default=4,
        help="Max ranked attention features taken from the same feature root.",
    )
    parser.add_argument(
        "--root-batch-size",
        type=int,
        default=8,
        help="Attention reduction batch size over layer roots.",
    )
    parser.add_argument(
        "--decision-threshold-mode",
        default="train_balanced_accuracy",
        help="Decision-threshold selection mode passed through to the experiment.",
    )
    parser.add_argument(
        "--model-selection-objective",
        default="mean_ood_auroc_oracle",
        choices=["mean_ood_auroc_oracle", "source_val_auroc"],
        help="Model-selection objective inside the experiment.",
    )
    parser.add_argument(
        "--top-features-to-show",
        type=int,
        default=20,
        help="How many top-coefficient features to export per winning panel.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild cached reduced-attention parquet files under the output cache directory.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Allow interactive plot display. By default the script uses Agg and only saves figures.",
    )
    parser.add_argument(
        "--source-path",
        default=str(DEFAULT_SOURCE),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def set_env(name: str, value: str | None) -> None:
    if value is None:
        return
    os.environ[name] = value


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def normalize_model_family(raw_value: str) -> str:
    value = str(raw_value).strip().lower()
    if value in {"logreg", "logistic", "logistic_regression", "lr"}:
        return "logreg"
    if value in {"xgboost", "xgb"}:
        return "xgboost"
    raise ValueError(f"Unsupported model family: {raw_value!r}")


def first_csv_item(raw_value: str, *, cast: type[float] | type[int]) -> float | int:
    parts = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Expected at least one comma-separated value, got {raw_value!r}")
    return cast(parts[0])


def resolve_output_root(
    *,
    source_path: Path,
    model_dirname: str,
    model_family: str,
    explicit_output_root: str | None,
) -> Path:
    if explicit_output_root:
        return Path(explicit_output_root)
    return source_path.parent / (
        f"OOD_Modeling_main3_consistency_ablation_outputs__{slugify(model_dirname)}__{slugify(model_family)}"
    )


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Notebook-source script not found: {source_path}")

    if not args.show_plots:
        os.environ.setdefault("MPLBACKEND", "Agg")

    model_family = normalize_model_family(args.model_family)
    feature_sizes = [int(part.strip()) for part in str(args.feature_sizes).split(",") if part.strip()]
    attention_top_k = int(args.attention_top_k) if args.attention_top_k is not None else max(feature_sizes)
    logreg_c = float(args.logreg_c)
    if args.c_grid:
        logreg_c = float(first_csv_item(args.c_grid, cast=float))
    xgb_max_depth = int(args.xgb_max_depth)
    if args.xgb_max_depth_grid:
        xgb_max_depth = int(first_csv_item(args.xgb_max_depth_grid, cast=int))
    output_root = resolve_output_root(
        source_path=source_path,
        model_dirname=str(args.model_dirname),
        model_family=model_family,
        explicit_output_root=args.output_root,
    )

    set_env("OOD_MAIN3_COMPANION_MODEL_NAME", str(args.model_dirname))
    set_env("OOD_MAIN3_COMPANION_MODEL_FAMILY", model_family)
    set_env("OOD_MAIN3_COMPANION_SOURCE_PATH", str(source_path))
    set_env("OOD_MAIN3_COMPANION_REPO_ROOT", str(REPO_ROOT))
    set_env("OOD_MAIN3_COMPANION_NOTEBOOK_ROOT", str(source_path.parent))
    set_env("OOD_MAIN3_COMPANION_OUTPUT_ROOT", str(output_root))
    set_env("OOD_MAIN3_COMPANION_FEATURE_SIZES", ",".join(str(value) for value in feature_sizes))
    set_env("OOD_MAIN3_COMPANION_ATTENTION_TOP_K", str(attention_top_k))
    set_env("OOD_MAIN3_COMPANION_LOGREG_C", str(logreg_c))
    set_env("OOD_MAIN3_COMPANION_XGB_MAX_DEPTH", str(xgb_max_depth))
    set_env("OOD_MAIN3_COMPANION_SEED", str(int(args.seed)))
    set_env("OOD_MAIN3_COMPANION_VAL_SIZE", str(float(args.val_size)))
    set_env("OOD_MAIN3_COMPANION_DELTA_THRESHOLD", str(float(args.delta_threshold)))
    set_env("OOD_MAIN3_COMPANION_PER_ROOT_LIMIT", str(int(args.per_root_limit)))
    set_env("OOD_MAIN3_COMPANION_ROOT_BATCH_SIZE", str(int(args.root_batch_size)))
    set_env("OOD_MAIN3_COMPANION_DECISION_THRESHOLD_MODE", str(args.decision_threshold_mode))
    set_env("OOD_MAIN3_COMPANION_MODEL_SELECTION_OBJECTIVE", str(args.model_selection_objective))
    set_env("OOD_MAIN3_COMPANION_TOP_FEATURES_TO_SHOW", str(int(args.top_features_to_show)))
    set_env("OOD_MAIN3_COMPANION_FORCE_REBUILD", "1" if args.force_rebuild else "0")
    set_env("OOD_MAIN3_COMPANION_DISABLE_TQDM", "1" if args.disable_tqdm else "0")
    if args.dataset_root:
        set_env("OOD_MAIN3_COMPANION_DATASET_ROOT", str(Path(args.dataset_root)))

    print("Running OOD Main3 consistency ablation script")
    print(f"Source: {source_path}")
    print(f"Model: {args.model_dirname}")
    print(f"Model family: {model_family}")
    print(f"Feature sizes: {feature_sizes}")
    print(f"Attention top-k cache: {attention_top_k}")
    print(f"Fixed logistic C: {logreg_c:g}")
    print(f"Fixed XGBoost max_depth: {xgb_max_depth}")
    if args.dataset_root:
        print(f"Dataset root: {Path(args.dataset_root)}")
    print(f"Output root: {output_root}")

    runpy.run_path(str(source_path), run_name="__main__")

    required_outputs = [
        output_root / "config.csv",
        output_root / "all_transfer_metrics.csv",
        output_root / "all_model_selection.csv",
        output_root / "all_coefficients.csv",
    ]
    missing_outputs = [str(path) for path in required_outputs if not path.exists()]
    if missing_outputs:
        raise FileNotFoundError(
            "Run finished without writing the expected aggregate outputs:\n"
            + "\n".join(missing_outputs)
        )
    print(f"Verified aggregate outputs under: {output_root}")


if __name__ == "__main__":
    main()
