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
    resolve_output_dir,
    resolve_repo_root,
    save_csv,
    utc_now_iso,
    write_json,
)


SCRIPT_NAME = Path(__file__).stem
SUMMARY_CACHE_VERSION = "prevalence_example_cache_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone runner for datasetmain_commitment_juncture_prevalence_paper.ipynb.",
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--datasetmain-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-json-files-per-bundle", type=int, default=None)
    parser.add_argument("--delta-threshold", type=float, default=0.3)
    parser.add_argument("--bootstrap-num-resamples", type=int, default=1000)
    parser.add_argument("--progress-level", choices=["bundle", "file"], default="bundle")

    parser.set_defaults(show_progress=True)
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--no-show-progress", dest="show_progress", action="store_false")

    parser.set_defaults(load_summary_cache=True)
    parser.add_argument("--load-summary-cache", dest="load_summary_cache", action="store_true")
    parser.add_argument("--no-load-summary-cache", dest="load_summary_cache", action="store_false")

    parser.set_defaults(save_summary_cache=True)
    parser.add_argument("--save-summary-cache", dest="save_summary_cache", action="store_true")
    parser.add_argument("--no-save-summary-cache", dest="save_summary_cache", action="store_false")

    parser.add_argument("--force-rebuild-summaries", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    datasetmain_root = resolve_datasetmain_root(repo_root, args.datasetmain_root)
    output_dir = ensure_dir(resolve_output_dir(repo_root, SCRIPT_NAME, args.output_dir))

    ensure_import_paths(repo_root)

    import datasetmain_commitment_juncture_prevalence_lib as cj

    cj = importlib.reload(cj)

    def summary_cache_paths() -> dict[str, Path]:
        return {
            "metadata": output_dir / "summary_cache_metadata.json",
            "inventory": output_dir / "inventory_df.pkl",
            "example": output_dir / "example_df.pkl",
            "parse_error": output_dir / "parse_error_df.pkl",
        }

    def build_summary_cache_metadata() -> dict[str, object]:
        return {
            "cache_version": SUMMARY_CACHE_VERSION,
            "dataset_root": str(datasetmain_root),
            "delta_threshold": float(args.delta_threshold),
            "max_json_files_per_bundle": args.max_json_files_per_bundle,
        }

    def has_complete_summary_cache() -> bool:
        paths = summary_cache_paths()
        if not all(path.exists() for path in paths.values()):
            return False
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        except Exception:
            return False
        return metadata == build_summary_cache_metadata()

    def load_summary_cache() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        paths = summary_cache_paths()
        inventory_df = pd.read_pickle(paths["inventory"])
        example_df = pd.read_pickle(paths["example"])
        parse_error_df = pd.read_pickle(paths["parse_error"])
        return inventory_df, example_df, parse_error_df

    def save_summary_cache(
        inventory_df: pd.DataFrame,
        example_df: pd.DataFrame,
        parse_error_df: pd.DataFrame,
    ) -> None:
        if not args.save_summary_cache:
            return
        paths = summary_cache_paths()
        inventory_df.to_pickle(paths["inventory"])
        example_df.to_pickle(paths["example"])
        parse_error_df.to_pickle(paths["parse_error"])
        paths["metadata"].write_text(
            json.dumps(build_summary_cache_metadata(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary_source = "raw_json"
    if args.load_summary_cache and (not args.force_rebuild_summaries) and has_complete_summary_cache():
        inventory_df, example_df, parse_error_df = load_summary_cache()
        summary_source = "cache"
    else:
        inventory_df, example_df, parse_error_df = cj.load_datasetmain_localization_example_df(
            datasetmain_root,
            max_json_files_per_bundle=args.max_json_files_per_bundle,
            delta_threshold=args.delta_threshold,
            show_progress=args.show_progress,
            progress_level=args.progress_level,
        )
        save_summary_cache(inventory_df, example_df, parse_error_df)

    coverage_table_df = inventory_df.loc[
        :,
        [
            "model_display",
            "env_display",
            "json_file_count",
            "loaded_examples",
            "usable_examples",
            "unusable_examples",
        ],
    ].rename(
        columns={
            "model_display": "Model",
            "env_display": "Environment",
            "json_file_count": "Localization JSONs",
            "loaded_examples": "Summarized Examples",
            "usable_examples": "Usable Examples",
            "unusable_examples": "Unusable Examples",
        }
    )

    env_model_stats_df = cj.build_commitment_example_statistics(
        example_df,
        ["model_display", "env_display"],
        bootstrap_location_ci=True,
        bootstrap_num_resamples=args.bootstrap_num_resamples,
    )
    model_stats_df = cj.build_commitment_example_statistics(
        example_df,
        ["model_display"],
        bootstrap_location_ci=True,
        bootstrap_num_resamples=args.bootstrap_num_resamples,
    )

    paper_env_model_table_df = cj.make_commitment_fraction_location_table(
        env_model_stats_df,
        location_interval_style="bootstrap_ci",
    )
    paper_model_table_df = cj.make_commitment_fraction_location_table(
        model_stats_df,
        location_interval_style="bootstrap_ci",
    )

    save_csv(coverage_table_df, output_dir, "json_localization_coverage")
    save_csv(parse_error_df, output_dir, "parse_errors")
    save_csv(env_model_stats_df, output_dir, "env_model_stats_raw")
    save_csv(model_stats_df, output_dir, "model_stats_raw")
    save_csv(paper_env_model_table_df, output_dir, "env_model_paper_table")
    save_csv(paper_model_table_df, output_dir, "model_paper_table")

    metadata_path = write_json(
        {
            "completed_at_utc": utc_now_iso(),
            "script": str(Path(__file__).resolve()),
            "helper_module": str(Path(cj.__file__).resolve()),
            "repo_root": repo_root,
            "datasetmain_root": datasetmain_root,
            "output_dir": output_dir,
            "summary_source": summary_source,
            "delta_threshold": args.delta_threshold,
            "bootstrap_num_resamples": args.bootstrap_num_resamples,
            "max_json_files_per_bundle": args.max_json_files_per_bundle,
            "show_progress": args.show_progress,
            "progress_level": args.progress_level,
            "load_summary_cache": args.load_summary_cache,
            "save_summary_cache": args.save_summary_cache,
            "force_rebuild_summaries": args.force_rebuild_summaries,
            "inventory_rows": int(len(inventory_df)),
            "example_rows": int(len(example_df)),
            "parse_error_rows": int(len(parse_error_df)),
        },
        output_dir / "run_metadata.json",
    )

    memory_mib = example_df.memory_usage(deep=True).sum() / (1024 ** 2) if not example_df.empty else 0.0
    print("DatasetMain commitment juncture prevalence run complete.")
    print(f"Summary source: {summary_source}")
    print(f"Dataset root: {datasetmain_root}")
    print(f"Output dir: {output_dir}")
    print(f"Helper module: {Path(cj.__file__).resolve()}")
    print(f"Coverage rows: {len(coverage_table_df):,}")
    print(f"Example rows: {len(example_df):,}")
    print(f"Parse errors: {len(parse_error_df):,}")
    print(f"Example dataframe memory: {memory_mib:.1f} MiB")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
