#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from localization_fulltrace_rebuttal_lib import REPO_ROOT, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one adaptive/full localization task from a rebuttal run manifest."
        )
    )
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--project-root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    manifest_path = resolve_repo_path(args.manifest_path, project_root=project_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise ValueError(f"Manifest has no task rows: {manifest_path}")
    if args.task_index < 0 or args.task_index >= len(manifest_df):
        raise IndexError(
            f"--task-index {args.task_index} is outside [0, {len(manifest_df) - 1}]"
        )

    row = manifest_df.iloc[int(args.task_index)].to_dict()
    script_path = project_root / "src" / "sentence_localization_batch.py"
    if not args.dry_run and not script_path.exists():
        raise FileNotFoundError(f"Missing localization script: {script_path}")

    examples_path = resolve_repo_path(str(row["examples_relpath"]), project_root=project_root)
    sentences_path = resolve_repo_path(str(row["sentences_relpath"]), project_root=project_root)
    out_dir = resolve_repo_path(str(row["out_dir_relpath"]), project_root=project_root)
    jsonl_path = resolve_repo_path(str(row["jsonl_relpath"]), project_root=project_root)

    cmd = [
        sys.executable,
        str(script_path),
        "--game",
        str(row["env_name"]),
        "--examples_path",
        str(examples_path),
        "--sentences_path",
        str(sentences_path),
        "--out_dir",
        str(out_dir),
        "--jsonl_path",
        str(jsonl_path),
        "--model_name",
        str(row["model_id"]),
        "--n_samples",
        str(int(row["n_samples"])),
        "--temperature",
        str(float(row["temperature"])),
        "--top_p",
        str(float(row["top_p"])),
        "--repetition_penalty",
        str(float(row["repetition_penalty"])),
        "--max_new_tokens",
        str(int(row["max_new_tokens"])),
        "--mode",
        str(row["mode"]),
        "--method",
        str(row["method"]),
        "--text_field",
        str(row["text_field"]),
        "--base_seed",
        str(int(row["base_seed"])),
        "--coarse_iters",
        str(int(row["coarse_iters"])),
        "--refinement_iters",
        str(int(row["refinement_iters"])),
        "--min_valid",
        str(int(row["min_valid"])),
        "--min_step_size",
        str(int(row["min_step_size"])),
        "--min_spacing",
        str(int(row["min_spacing"])),
        "--gpu_memory_utilization",
        str(float(row["gpu_memory_utilization"])),
        "--tensor_parallel_size",
        str(int(row["tensor_parallel_size"])),
        "--label_filter",
        "all",
        "--shard_id",
        "0",
        "--num_shards",
        "1",
        "--overwrite",
        "--log_every",
        "1",
        "--flush_every",
        "1",
    ]

    print(
        f"Running task {args.task_index}: "
        f"{row['method']} | {row['bundle_key']} | n={int(row['num_examples'])}"
    )
    print("Command:")
    print(" ".join(subprocess.list2cmdline([token]) for token in cmd))

    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.setdefault("MKL_THREADING_LAYER", "GNU")
    subprocess.run(cmd, check=True, env=child_env)


if __name__ == "__main__":
    main()
