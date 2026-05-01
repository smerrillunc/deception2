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
SUMMARY_CACHE_VERSION = "localization_dataset_summary_bundle_cache_v3"


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
    parser.add_argument("--env-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--combine-shard-output-root",
        type=str,
        default=None,
        help="Combine prior shard outputs from this directory instead of reparsing raw localization JSON files.",
    )
    parser.add_argument(
        "--list-shards",
        action="store_true",
        help="Print available dataset/model shard specs as TSV rows and exit.",
    )

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


def _load_combined_shard_outputs(
    dsum,
    shard_output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    bundle_csv_paths = sorted(path for path in shard_output_root.rglob("bundle_summary_raw.csv") if path.is_file())
    if not bundle_csv_paths:
        raise FileNotFoundError(f"No shard bundle summaries were found under {shard_output_root}.")

    bundle_frames: list[pd.DataFrame] = []
    parse_issue_frames: list[pd.DataFrame] = []
    shard_dirs: list[Path] = []
    for bundle_csv_path in bundle_csv_paths:
        bundle_frames.append(pd.read_csv(bundle_csv_path))
        shard_dirs.append(bundle_csv_path.parent)

        parse_issue_path = bundle_csv_path.with_name("parse_issues_raw.csv")
        if parse_issue_path.exists():
            parse_issue_frames.append(pd.read_csv(parse_issue_path))

    bundle_df = dsum.combine_bundle_summary_dfs(bundle_frames)
    parse_issue_df = (
        pd.concat(parse_issue_frames, ignore_index=True, sort=False)
        if parse_issue_frames
        else dsum.empty_parse_issue_df()
    )
    return bundle_df, parse_issue_df, shard_dirs


def _build_output_tables(
    dsum,
    bundle_df: pd.DataFrame,
    parse_issue_df: pd.DataFrame,
    *,
    expected_files_per_bundle: int,
) -> dict[str, pd.DataFrame]:
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

    bundle_inventory_table_df = bundle_df.reindex(
        columns=[
            "model_display",
            "env_display",
            "file_count",
            "recovered_json_file_count",
            "skipped_json_file_count",
            "attempted_json_file_total",
            "localized_prefix_total",
            "continuation_total",
            "file_size_tb",
            "recovered_json_rate",
            "skipped_json_rate",
            "avg_continuations_per_prefix",
        ]
    ).rename(
        columns={
            "model_display": "Model",
            "env_display": "Environment",
            "file_count": "Localization Files",
            "recovered_json_file_count": "Recovered JSON Files",
            "skipped_json_file_count": "Skipped Broken JSON Files",
            "attempted_json_file_total": "Attempted JSON Files",
            "localized_prefix_total": "Localized Sentences",
            "continuation_total": "Continuations",
            "file_size_tb": "File Size (TB)",
            "recovered_json_rate": "Recovery Rate",
            "skipped_json_rate": "Skip Rate",
            "avg_continuations_per_prefix": "Avg. Continuations / Prefix",
        }
    )
    bundle_inventory_table_df["Gap vs 5000 Files"] = (
        bundle_inventory_table_df["Localization Files"] - int(expected_files_per_bundle)
    )
    bundle_inventory_table_df["Recovery Rate (%)"] = (
        pd.to_numeric(bundle_inventory_table_df["Recovery Rate"], errors="coerce") * 100.0
    )
    bundle_inventory_table_df["Skip Rate (%)"] = (
        pd.to_numeric(bundle_inventory_table_df["Skip Rate"], errors="coerce") * 100.0
    )
    bundle_inventory_table_df = bundle_inventory_table_df.drop(columns=["Recovery Rate", "Skip Rate"])

    model_totals_table_df = model_df.reindex(
        columns=[
            "model_display",
            "file_count",
            "recovered_json_file_count",
            "skipped_json_file_count",
            "attempted_json_file_total",
            "reasoning_sentence_total",
            "reasoning_token_total",
            "reasoning_word_total",
            "localized_prefix_total",
            "continuation_total",
            "expanded_dataset_token_total",
            "expanded_dataset_word_total",
            "expanded_dataset_sentence_total",
            "file_size_tb",
        ]
    ).rename(
        columns={
            "model_display": "Model",
            "file_count": "Localization Files",
            "recovered_json_file_count": "Recovered JSON Files",
            "skipped_json_file_count": "Skipped Broken JSON Files",
            "attempted_json_file_total": "Attempted JSON Files",
            "reasoning_sentence_total": "Reasoning Sentences",
            "reasoning_token_total": "Reasoning Tokens",
            "reasoning_word_total": "Reasoning Words",
            "localized_prefix_total": "Localized Sentences",
            "continuation_total": "Continuations",
            "expanded_dataset_token_total": "Expanded Dataset Tokens",
            "expanded_dataset_word_total": "Expanded Dataset Words",
            "expanded_dataset_sentence_total": "Expanded Dataset Sentences",
            "file_size_tb": "File Size (TB)",
        }
    )

    env_totals_table_df = env_df.reindex(
        columns=[
            "env_display",
            "file_count",
            "recovered_json_file_count",
            "skipped_json_file_count",
            "attempted_json_file_total",
            "reasoning_sentence_total",
            "reasoning_token_total",
            "reasoning_word_total",
            "localized_prefix_total",
            "continuation_total",
            "expanded_dataset_token_total",
            "expanded_dataset_word_total",
            "expanded_dataset_sentence_total",
            "file_size_tb",
        ]
    ).rename(
        columns={
            "env_display": "Environment",
            "file_count": "Localization Files",
            "recovered_json_file_count": "Recovered JSON Files",
            "skipped_json_file_count": "Skipped Broken JSON Files",
            "attempted_json_file_total": "Attempted JSON Files",
            "reasoning_sentence_total": "Reasoning Sentences",
            "reasoning_token_total": "Reasoning Tokens",
            "reasoning_word_total": "Reasoning Words",
            "localized_prefix_total": "Localized Sentences",
            "continuation_total": "Continuations",
            "expanded_dataset_token_total": "Expanded Dataset Tokens",
            "expanded_dataset_word_total": "Expanded Dataset Words",
            "expanded_dataset_sentence_total": "Expanded Dataset Sentences",
            "file_size_tb": "File Size (TB)",
        }
    )

    totals_table_df = dsum.make_total_summary_table(bundle_df)
    dataset_overview_table_df = dsum.make_dataset_overview_table(bundle_df)
    paper_model_scale_table_df = dsum.make_paper_scale_table(model_df)
    paper_env_scale_table_df = dsum.make_paper_scale_table(
        env_df,
        include_model=False,
        include_environment=True,
    )
    paper_env_model_scale_table_df = dsum.make_paper_scale_table(
        env_model_df,
        include_environment=True,
    )
    parse_issue_summary_df = dsum.summarize_parse_issues(parse_issue_df)
    non_exact_bundle_count_df = bundle_inventory_table_df.loc[
        ~bundle_inventory_table_df["Localization Files"].eq(int(expected_files_per_bundle))
    ].reset_index(drop=True)

    return {
        "bundle_df": bundle_df,
        "model_df": model_df,
        "env_df": env_df,
        "env_model_df": env_model_df,
        "requested_model_table_df": requested_model_table_df,
        "requested_env_table_df": requested_env_table_df,
        "requested_env_model_table_df": requested_env_model_table_df,
        "bundle_inventory_table_df": bundle_inventory_table_df,
        "non_exact_bundle_count_df": non_exact_bundle_count_df,
        "totals_table_df": totals_table_df,
        "dataset_overview_table_df": dataset_overview_table_df,
        "model_totals_table_df": model_totals_table_df,
        "env_totals_table_df": env_totals_table_df,
        "paper_model_scale_table_df": paper_model_scale_table_df,
        "paper_env_scale_table_df": paper_env_scale_table_df,
        "paper_env_model_scale_table_df": paper_env_model_scale_table_df,
        "parse_issue_summary_df": parse_issue_summary_df,
    }


def _save_output_tables(
    output_dir: Path,
    *,
    bundle_df: pd.DataFrame,
    parse_issue_df: pd.DataFrame,
    model_df: pd.DataFrame,
    env_df: pd.DataFrame,
    env_model_df: pd.DataFrame,
    requested_model_table_df: pd.DataFrame,
    requested_env_table_df: pd.DataFrame,
    requested_env_model_table_df: pd.DataFrame,
    bundle_inventory_table_df: pd.DataFrame,
    non_exact_bundle_count_df: pd.DataFrame,
    totals_table_df: pd.DataFrame,
    dataset_overview_table_df: pd.DataFrame,
    model_totals_table_df: pd.DataFrame,
    env_totals_table_df: pd.DataFrame,
    paper_model_scale_table_df: pd.DataFrame,
    paper_env_scale_table_df: pd.DataFrame,
    paper_env_model_scale_table_df: pd.DataFrame,
    parse_issue_summary_df: pd.DataFrame,
) -> None:
    save_csv(bundle_df, output_dir, "bundle_summary_raw")
    save_csv(parse_issue_df, output_dir, "parse_issues_raw")
    save_csv(parse_issue_summary_df, output_dir, "parse_issue_summary")
    save_csv(model_df, output_dir, "model_summary_raw")
    save_csv(env_df, output_dir, "environment_summary_raw")
    save_csv(env_model_df, output_dir, "environment_model_summary_raw")
    save_csv(requested_model_table_df, output_dir, "requested_model_table")
    save_csv(requested_env_table_df, output_dir, "requested_environment_table")
    save_csv(requested_env_model_table_df, output_dir, "requested_env_model_table")
    save_csv(dataset_overview_table_df, output_dir, "paper_dataset_overview")
    save_csv(paper_model_scale_table_df, output_dir, "paper_model_scale_table")
    save_csv(paper_env_scale_table_df, output_dir, "paper_environment_scale_table")
    save_csv(paper_env_model_scale_table_df, output_dir, "paper_env_model_scale_table")
    save_csv(bundle_inventory_table_df, output_dir, "bundle_inventory")
    save_csv(non_exact_bundle_count_df, output_dir, "bundle_inventory_non_5000")
    save_csv(totals_table_df, output_dir, "dataset_totals")
    save_csv(model_totals_table_df, output_dir, "model_totals")
    save_csv(env_totals_table_df, output_dir, "environment_totals")


def main() -> None:
    args = build_parser().parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    datasetmain_root = resolve_datasetmain_root(repo_root, args.datasetmain_root)
    output_dir = ensure_dir(resolve_output_dir(repo_root, SCRIPT_NAME, args.output_dir))
    hf_cache_root = resolve_hf_cache_root(repo_root, args.hf_cache_root)

    ensure_import_paths(repo_root)

    import datasetmain_localization_dataset_summary_lib as dsum

    dsum = importlib.reload(dsum)

    if args.list_shards:
        shard_specs = dsum.list_bundle_specs(
            datasetmain_root,
            env_name=args.env_name,
            model_name=args.model_name,
        )
        for shard_spec in shard_specs:
            print(
                "\t".join(
                    [
                        str(shard_spec["env_name"]),
                        str(shard_spec["model_name"]),
                        str(shard_spec["shard_slug"]),
                        str(shard_spec["bundle_dir"]),
                    ]
                )
            )
        return

    def bundle_summary_cache_paths() -> dict[str, Path]:
        return {
            "metadata": output_dir / "bundle_summary_cache_metadata.json",
            "bundle": output_dir / "bundle_df.pkl",
            "parse_issues": output_dir / "parse_issue_df.pkl",
        }

    def build_bundle_summary_cache_metadata() -> dict[str, object]:
        return {
            "cache_version": SUMMARY_CACHE_VERSION,
            "dataset_root": str(datasetmain_root),
            "env_name": args.env_name,
            "model_name": args.model_name,
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

    def load_bundle_summary_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
        paths = bundle_summary_cache_paths()
        return pd.read_pickle(paths["bundle"]), pd.read_pickle(paths["parse_issues"])

    def save_bundle_summary_cache(bundle_df: pd.DataFrame, parse_issue_df: pd.DataFrame) -> None:
        if not args.save_bundle_summary_cache:
            return
        paths = bundle_summary_cache_paths()
        bundle_df.to_pickle(paths["bundle"])
        parse_issue_df.to_pickle(paths["parse_issues"])
        paths["metadata"].write_text(
            json.dumps(build_bundle_summary_cache_metadata(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    combine_shard_output_root = (
        None if args.combine_shard_output_root is None else Path(args.combine_shard_output_root).expanduser().resolve()
    )

    shard_dirs: list[Path] = []
    bundle_summary_source = "raw_json"
    if combine_shard_output_root is not None:
        bundle_df, parse_issue_df, shard_dirs = _load_combined_shard_outputs(dsum, combine_shard_output_root)
        bundle_summary_source = "combined_shards"
    elif args.load_bundle_summary_cache and (not args.force_rebuild_bundle_summary) and has_complete_bundle_summary_cache():
        bundle_df, parse_issue_df = load_bundle_summary_cache()
        bundle_summary_source = "cache"
    else:
        bundle_result = dsum.build_bundle_summary_df(
            datasetmain_root,
            max_files_per_bundle=args.max_files_per_bundle,
            num_workers=args.num_workers,
            token_count_mode=args.token_count_mode,
            hf_cache_root=hf_cache_root,
            show_progress=args.show_progress,
            progress_level=args.progress_level,
            env_name=args.env_name,
            model_name=args.model_name,
            return_parse_issues=True,
        )
        bundle_df, parse_issue_df = bundle_result
        if args.env_name is not None and args.model_name is not None and bundle_df.empty:
            raise FileNotFoundError(
                f"No DatasetMain bundle matched env_name={args.env_name!r} and model_name={args.model_name!r}."
            )
        save_bundle_summary_cache(bundle_df, parse_issue_df)

    output_tables = _build_output_tables(
        dsum,
        bundle_df,
        parse_issue_df,
        expected_files_per_bundle=args.expected_files_per_bundle,
    )
    _save_output_tables(
        output_dir,
        bundle_df=output_tables["bundle_df"],
        parse_issue_df=parse_issue_df,
        model_df=output_tables["model_df"],
        env_df=output_tables["env_df"],
        env_model_df=output_tables["env_model_df"],
        requested_model_table_df=output_tables["requested_model_table_df"],
        requested_env_table_df=output_tables["requested_env_table_df"],
        requested_env_model_table_df=output_tables["requested_env_model_table_df"],
        bundle_inventory_table_df=output_tables["bundle_inventory_table_df"],
        non_exact_bundle_count_df=output_tables["non_exact_bundle_count_df"],
        totals_table_df=output_tables["totals_table_df"],
        dataset_overview_table_df=output_tables["dataset_overview_table_df"],
        model_totals_table_df=output_tables["model_totals_table_df"],
        env_totals_table_df=output_tables["env_totals_table_df"],
        paper_model_scale_table_df=output_tables["paper_model_scale_table_df"],
        paper_env_scale_table_df=output_tables["paper_env_scale_table_df"],
        paper_env_model_scale_table_df=output_tables["paper_env_model_scale_table_df"],
        parse_issue_summary_df=output_tables["parse_issue_summary_df"],
    )

    recovered_json_total = (
        int(pd.to_numeric(bundle_df["recovered_json_file_count"], errors="coerce").fillna(0).sum())
        if not bundle_df.empty and "recovered_json_file_count" in bundle_df.columns
        else 0
    )
    skipped_json_total = (
        int(pd.to_numeric(bundle_df["skipped_json_file_count"], errors="coerce").fillna(0).sum())
        if not bundle_df.empty and "skipped_json_file_count" in bundle_df.columns
        else 0
    )
    metadata_payload: dict[str, object] = {
        "completed_at_utc": utc_now_iso(),
        "script": str(Path(__file__).resolve()),
        "helper_module": str(Path(dsum.__file__).resolve()),
        "repo_root": repo_root,
        "datasetmain_root": datasetmain_root,
        "output_dir": output_dir,
        "hf_cache_root": hf_cache_root,
        "bundle_summary_source": bundle_summary_source,
        "mode": "combine_shards" if combine_shard_output_root is not None else "summarize_raw_json",
        "env_name": args.env_name,
        "model_name": args.model_name,
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
        "model_rows": int(len(output_tables["model_df"])),
        "environment_rows": int(len(output_tables["env_df"])),
        "environment_model_rows": int(len(output_tables["env_model_df"])),
        "parse_issue_rows": int(len(parse_issue_df)),
        "recovered_json_total": recovered_json_total,
        "skipped_json_total": skipped_json_total,
    }
    if combine_shard_output_root is not None:
        metadata_payload["combine_shard_output_root"] = str(combine_shard_output_root)
        metadata_payload["combined_shard_count"] = int(len(shard_dirs))
        metadata_payload["combined_shard_dirs"] = [str(path) for path in shard_dirs]

    metadata_path = write_json(metadata_payload, output_dir / "run_metadata.json")

    total_files = int(bundle_df["file_count"].sum()) if not bundle_df.empty else 0
    print("DatasetMain localization dataset summary run complete.")
    print(f"Bundle summary source: {bundle_summary_source}")
    print(f"Dataset root: {datasetmain_root}")
    print(f"HF cache root: {hf_cache_root}")
    print(f"Output dir: {output_dir}")
    print(f"Helper module: {Path(dsum.__file__).resolve()}")
    print(f"Bundles: {len(bundle_df):,}")
    print(f"Localization files processed: {total_files:,}")
    print(f"Recovered JSON files: {recovered_json_total:,}")
    print(f"Skipped broken JSON files: {skipped_json_total:,}")
    print(f"Parse issue rows: {len(parse_issue_df):,}")
    if combine_shard_output_root is not None:
        print(f"Combined shard dirs: {len(shard_dirs):,}")
    print(f"Requested model rows: {len(output_tables['requested_model_table_df']):,}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
