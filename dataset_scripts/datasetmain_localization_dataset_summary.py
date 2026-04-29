from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from dataset_scripts_common import (
    ensure_dir,
    ensure_import_paths,
    resolve_datasetmain_root,
    resolve_hf_cache_root,
    resolve_output_dir,
    resolve_repo_root,
    save_csv,
    utc_now_iso,
    write_json,
)


SCRIPT_NAME = Path(__file__).stem
SUMMARY_CACHE_VERSION = "localization_dataset_summary_bundle_cache_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone runner for datasetmain_localization_dataset_summary.ipynb.",
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--datasetmain-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--hf-cache-root", type=str, default=None)
    parser.add_argument("--max-files-per-bundle", type=int, default=None)
    parser.add_argument("--token-count-mode", choices=["hf", "regex"], default="hf")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-level", choices=["bundle", "file"], default="file")
    parser.add_argument("--expected-files-per-bundle", type=int, default=5000)

    parser.set_defaults(show_progress=True)
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--no-show-progress", dest="show_progress", action="store_false")

    parser.set_defaults(load_bundle_summary_cache=True)
    parser.add_argument("--load-bundle-summary-cache", dest="load_bundle_summary_cache", action="store_true")
    parser.add_argument("--no-load-bundle-summary-cache", dest="load_bundle_summary_cache", action="store_false")

    parser.set_defaults(save_bundle_summary_cache=True)
    parser.add_argument("--save-bundle-summary-cache", dest="save_bundle_summary_cache", action="store_true")
    parser.add_argument("--no-save-bundle-summary-cache", dest="save_bundle_summary_cache", action="store_false")

    parser.add_argument("--force-rebuild-bundle-summary", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    datasetmain_root = resolve_datasetmain_root(repo_root, args.datasetmain_root)
    output_dir = ensure_dir(resolve_output_dir(repo_root, SCRIPT_NAME, args.output_dir))
    hf_cache_root = resolve_hf_cache_root(repo_root, args.hf_cache_root)

    ensure_import_paths(repo_root)

    import datasetmain_localization_dataset_summary_lib as dsum

    dsum = importlib.reload(dsum)

    def bundle_summary_cache_paths() -> dict[str, Path]:
        return {
            "metadata": output_dir / "bundle_summary_cache_metadata.json",
            "bundle": output_dir / "bundle_df.pkl",
        }

    def build_bundle_summary_cache_metadata() -> dict[str, object]:
        return {
            "cache_version": SUMMARY_CACHE_VERSION,
            "dataset_root": str(datasetmain_root),
            "max_files_per_bundle": args.max_files_per_bundle,
            "token_count_mode": str(args.token_count_mode),
            "expected_files_per_bundle": int(args.expected_files_per_bundle),
            "hf_cache_root": str(hf_cache_root),
        }

    def has_complete_bundle_summary_cache() -> bool:
        paths = bundle_summary_cache_paths()
        if not all(path.exists() for path in paths.values()):
            return False
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        except Exception:
            return False
        return metadata == build_bundle_summary_cache_metadata()

    def load_bundle_summary_cache() -> pd.DataFrame:
        return pd.read_pickle(bundle_summary_cache_paths()["bundle"])

    def save_bundle_summary_cache(bundle_df: pd.DataFrame) -> None:
        if not args.save_bundle_summary_cache:
            return
        paths = bundle_summary_cache_paths()
        bundle_df.to_pickle(paths["bundle"])
        paths["metadata"].write_text(
            json.dumps(build_bundle_summary_cache_metadata(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    bundle_summary_source = "raw_json"
    if args.load_bundle_summary_cache and (not args.force_rebuild_bundle_summary) and has_complete_bundle_summary_cache():
        bundle_df = load_bundle_summary_cache()
        bundle_summary_source = "cache"
    else:
        bundle_df = dsum.build_bundle_summary_df(
            datasetmain_root,
            max_files_per_bundle=args.max_files_per_bundle,
            num_workers=args.num_workers,
            token_count_mode=args.token_count_mode,
            hf_cache_root=hf_cache_root,
            show_progress=args.show_progress,
            progress_level=args.progress_level,
        )
        save_bundle_summary_cache(bundle_df)

    model_df = dsum.summarize_groups(bundle_df, ["model_display"])
    env_df = dsum.summarize_groups(bundle_df, ["env_display"])
    env_model_df = dsum.summarize_groups(bundle_df, ["model_display", "env_display"])

    requested_model_table_df = dsum.make_requested_summary_table(model_df)
    requested_env_table_df = dsum.make_requested_summary_table(
        env_df,
        include_model=False,
        include_environment=True,
    )
    requested_env_model_table_df = dsum.make_requested_summary_table(
        env_model_df,
        include_environment=True,
    )

    bundle_inventory_table_df = bundle_df.loc[
        :,
        [
            "model_display",
            "env_display",
            "file_count",
            "localized_prefix_total",
            "continuation_total",
            "file_size_tb",
            "avg_continuations_per_prefix",
        ],
    ].rename(
        columns={
            "model_display": "Model",
            "env_display": "Environment",
            "file_count": "Localization Files",
            "localized_prefix_total": "Localized Sentences",
            "continuation_total": "Continuations",
            "file_size_tb": "File Size (TB)",
            "avg_continuations_per_prefix": "Avg. Continuations / Prefix",
        }
    )
    bundle_inventory_table_df["Gap vs 5000 Files"] = (
        bundle_inventory_table_df["Localization Files"] - args.expected_files_per_bundle
    )

    model_totals_table_df = model_df.loc[
        :,
        [
            "model_display",
            "file_count",
            "localized_prefix_total",
            "continuation_total",
            "expanded_dataset_token_total",
            "expanded_dataset_word_total",
            "expanded_dataset_sentence_total",
            "file_size_tb",
        ],
    ].rename(
        columns={
            "model_display": "Model",
            "file_count": "Localization Files",
            "localized_prefix_total": "Localized Sentences",
            "continuation_total": "Continuations",
            "expanded_dataset_token_total": "Expanded Dataset Tokens",
            "expanded_dataset_word_total": "Expanded Dataset Words",
            "expanded_dataset_sentence_total": "Expanded Dataset Sentences",
            "file_size_tb": "File Size (TB)",
        }
    )

    env_totals_table_df = env_df.loc[
        :,
        [
            "env_display",
            "file_count",
            "localized_prefix_total",
            "continuation_total",
            "expanded_dataset_token_total",
            "expanded_dataset_word_total",
            "expanded_dataset_sentence_total",
            "file_size_tb",
        ],
    ].rename(
        columns={
            "env_display": "Environment",
            "file_count": "Localization Files",
            "localized_prefix_total": "Localized Sentences",
            "continuation_total": "Continuations",
            "expanded_dataset_token_total": "Expanded Dataset Tokens",
            "expanded_dataset_word_total": "Expanded Dataset Words",
            "expanded_dataset_sentence_total": "Expanded Dataset Sentences",
            "file_size_tb": "File Size (TB)",
        }
    )

    totals_table_df = dsum.make_total_summary_table(bundle_df)
    non_exact_bundle_count_df = bundle_inventory_table_df.loc[
        ~bundle_inventory_table_df["Localization Files"].eq(args.expected_files_per_bundle)
    ].reset_index(drop=True)

    save_csv(bundle_df, output_dir, "bundle_summary_raw")
    save_csv(model_df, output_dir, "model_summary_raw")
    save_csv(env_df, output_dir, "environment_summary_raw")
    save_csv(env_model_df, output_dir, "environment_model_summary_raw")
    save_csv(requested_model_table_df, output_dir, "requested_model_table")
    save_csv(requested_env_table_df, output_dir, "requested_environment_table")
    save_csv(requested_env_model_table_df, output_dir, "requested_env_model_table")
    save_csv(bundle_inventory_table_df, output_dir, "bundle_inventory")
    save_csv(non_exact_bundle_count_df, output_dir, "bundle_inventory_non_5000")
    save_csv(totals_table_df, output_dir, "dataset_totals")
    save_csv(model_totals_table_df, output_dir, "model_totals")
    save_csv(env_totals_table_df, output_dir, "environment_totals")

    metadata_path = write_json(
        {
            "completed_at_utc": utc_now_iso(),
            "script": str(Path(__file__).resolve()),
            "helper_module": str(Path(dsum.__file__).resolve()),
            "repo_root": repo_root,
            "datasetmain_root": datasetmain_root,
            "output_dir": output_dir,
            "hf_cache_root": hf_cache_root,
            "bundle_summary_source": bundle_summary_source,
            "max_files_per_bundle": args.max_files_per_bundle,
            "token_count_mode": args.token_count_mode,
            "num_workers": args.num_workers,
            "show_progress": args.show_progress,
            "progress_level": args.progress_level,
            "expected_files_per_bundle": args.expected_files_per_bundle,
            "load_bundle_summary_cache": args.load_bundle_summary_cache,
            "save_bundle_summary_cache": args.save_bundle_summary_cache,
            "force_rebuild_bundle_summary": args.force_rebuild_bundle_summary,
            "bundle_rows": int(len(bundle_df)),
            "model_rows": int(len(model_df)),
            "environment_rows": int(len(env_df)),
            "environment_model_rows": int(len(env_model_df)),
        },
        output_dir / "run_metadata.json",
    )

    total_files = int(bundle_df["file_count"].sum()) if not bundle_df.empty else 0
    print("DatasetMain localization dataset summary run complete.")
    print(f"Bundle summary source: {bundle_summary_source}")
    print(f"Dataset root: {datasetmain_root}")
    print(f"HF cache root: {hf_cache_root}")
    print(f"Output dir: {output_dir}")
    print(f"Helper module: {Path(dsum.__file__).resolve()}")
    print(f"Bundles: {len(bundle_df):,}")
    print(f"Localization files processed: {total_files:,}")
    print(f"Requested model rows: {len(requested_model_table_df):,}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
