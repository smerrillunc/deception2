#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from localization_fulltrace_rebuttal_lib import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_ENVIRONMENTS,
    DEFAULT_MODEL_BUNDLES,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_NAME,
    BundleSpec,
    ENV_DISPLAY_BY_NAME,
    MODEL_DISPLAY_BY_BUNDLE,
    MODEL_ID_BY_BUNDLE,
    REPO_ROOT,
    allocate_label_targets,
    bundle_specs,
    ensure_dir,
    flatten_dict_row,
    localization_output_path,
    read_jsonl,
    relpath_from_repo,
    round_robin_pick,
    run_root,
    stable_rank_key,
    write_csv,
    write_json,
    write_jsonl,
)


SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sentence_pipeline import split_sentence_spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a deterministic 10-example-per-bundle subset for comparing "
            "dataset adaptive localization against new full-trace localization runs."
        )
    )
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--compressed-dataset-root", type=str, default=str(REPO_ROOT / "DatasetMainCompressed"))
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--examples-per-bundle", type=int, default=10)
    parser.add_argument("--min-sentences", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--compressed-probe-initial", type=int, default=128)
    parser.add_argument("--compressed-probe-multiplier", type=float, default=2.0)
    parser.add_argument("--compressed-probe-label-slack", type=int, default=2)
    parser.add_argument("--text-field", type=str, default="action_reasoning")
    parser.add_argument("--envs", nargs="*", default=list(DEFAULT_ENVIRONMENTS))
    parser.add_argument("--model-bundles", nargs="*", default=list(DEFAULT_MODEL_BUNDLES))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--mode", type=str, choices=("prefix", "sentence_only"), default="prefix")
    parser.add_argument("--coarse-iters", type=int, default=8)
    parser.add_argument("--refinement-iters", type=int, default=8)
    parser.add_argument("--min-valid", type=int, default=3)
    parser.add_argument("--min-step-size", type=int, default=1)
    parser.add_argument("--min-spacing", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def remove_if_exists(path: Path | str) -> None:
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()


def compressed_bundle_specs(
    *,
    compressed_root: Path | str,
    environments: list[str],
    model_bundles: list[str],
) -> list[BundleSpec]:
    compressed_root_path = Path(compressed_root).expanduser().resolve()
    out: list[BundleSpec] = []
    for env_name in environments:
        for model_bundle_name in model_bundles:
            dataset_dir = compressed_root_path / str(env_name) / str(model_bundle_name)
            localization_dir = dataset_dir / "localization"
            if not localization_dir.exists():
                continue
            out.append(
                BundleSpec(
                    env_name=str(env_name),
                    env_display=ENV_DISPLAY_BY_NAME.get(str(env_name), str(env_name)),
                    model_bundle_name=str(model_bundle_name),
                    model_display=MODEL_DISPLAY_BY_BUNDLE.get(str(model_bundle_name), str(model_bundle_name)),
                    model_id=MODEL_ID_BY_BUNDLE.get(str(model_bundle_name), str(model_bundle_name)),
                    dataset_dir=dataset_dir,
                    examples_path=dataset_dir / "examples.jsonl",
                    sentences_path=dataset_dir / "sentences.jsonl",
                    localization_dir=localization_dir,
                )
            )
    return out


def synthesize_example_record(
    payload: dict[str, Any],
    *,
    text_field: str,
) -> dict[str, Any]:
    raw_text = str(payload.get("raw_text") or "")
    prompt = str(payload.get("prompt") or "")
    game = str(payload.get("game") or "")
    eval_context = payload.get("eval_context") if isinstance(payload.get("eval_context"), dict) else {}
    truth_context: dict[str, Any] = {}

    if game == "advisor_audit":
        truth_context = dict(eval_context)
    elif game == "interview":
        truth_context = {"label_preview": dict(eval_context)}
    elif game == "car_sales":
        truth_context = dict(eval_context)
    elif game == "gridworld":
        truth_context = {"optimal_set": list(eval_context.get("optimal_set") or [])}
    elif game == "bs":
        truthful_rank = eval_context.get("truthful_rank")
        truth_context = {"current_rank": truthful_rank}
    example = {
        "example_id": str(payload.get("example_id") or ""),
        "prompt": prompt,
        "game_type": game,
        "truth_context": truth_context,
        "eval_context": eval_context,
        "action_reasoning": raw_text,
        text_field: raw_text,
    }
    if game == "bs":
        truthful_rank = eval_context.get("truthful_rank")
        if truthful_rank is not None:
            example["truthful_rank"] = truthful_rank
            example["current_rank"] = truthful_rank
    if game == "car_sales":
        if eval_context.get("target_issue_id") is not None:
            example["target_issue_id"] = eval_context.get("target_issue_id")
        if eval_context.get("current_round_focus") is not None:
            example["current_round_focus"] = eval_context.get("current_round_focus")
    return example


def synthesize_sentence_rows(
    *,
    example_id: str,
    raw_text: str,
) -> list[dict[str, Any]]:
    sentence_rows: list[dict[str, Any]] = []
    for sentence_idx, sent in enumerate(split_sentence_spans(raw_text)):
        sentence_rows.append(
            {
                "example_id": example_id,
                "sentence_idx": int(sentence_idx),
                "start": int(sent["start"]),
                "end": int(sent["end"]),
                "sentence_text": str(sent["text"]),
            }
        )
    return sentence_rows


def candidate_rows_for_bundle(
    bundle: BundleSpec,
    *,
    text_field: str,
    min_sentences: int,
    max_sentences: int,
    examples_per_bundle: int,
    selection_seed: int,
    compressed_probe_initial: int,
    compressed_probe_multiplier: float,
    compressed_probe_label_slack: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if not bundle.examples_path.exists() or not bundle.sentences_path.exists():
        return candidate_rows_for_compressed_bundle(
            bundle,
            text_field=text_field,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            examples_per_bundle=examples_per_bundle,
            selection_seed=selection_seed,
            compressed_probe_initial=compressed_probe_initial,
            compressed_probe_multiplier=compressed_probe_multiplier,
            compressed_probe_label_slack=compressed_probe_label_slack,
        )

    examples_by_id: dict[str, dict[str, Any]] = {}
    for example in read_jsonl(bundle.examples_path):
        example_id = str(example.get("example_id") or example.get("record_id") or "").strip()
        if not example_id:
            continue
        examples_by_id[example_id] = example

    sentences_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sentence_row in read_jsonl(bundle.sentences_path):
        example_id = str(sentence_row.get("example_id") or "").strip()
        if not example_id:
            continue
        sentences_by_id[example_id].append(sentence_row)

    rows: list[dict[str, Any]] = []
    for example_id, example in examples_by_id.items():
        label_value = example.get("deceptive")
        if not isinstance(label_value, bool):
            continue
        raw_text = (
            example.get(text_field)
            or example.get("action_reasoning")
            or example.get("reasoning")
            or example.get("action_raw_text")
            or ""
        )
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        has_prompt = isinstance(example.get("prompt"), str) and bool(str(example.get("prompt")).strip())
        has_messages = isinstance(example.get("messages"), list) and len(example.get("messages", [])) > 0
        if not has_prompt and not has_messages:
            continue

        sentence_rows = sorted(
            sentences_by_id.get(example_id, []),
            key=lambda row: int(row.get("sentence_idx", 0)),
        )
        sentence_count = len(sentence_rows)
        if sentence_count < int(min_sentences) or sentence_count > int(max_sentences):
            continue

        source_adaptive_json = localization_output_path(bundle.localization_dir, example_id)
        rows.append(
            {
                "bundle_key": bundle.bundle_key,
                "env_name": bundle.env_name,
                "env_display": bundle.env_display,
                "model_bundle_name": bundle.model_bundle_name,
                "model_display": bundle.model_display,
                "model_id": bundle.model_id,
                "example_id": example_id,
                "deceptive": bool(label_value),
                "label_name": "deceptive" if bool(label_value) else "truthful",
                "sentence_count": int(sentence_count),
                "has_messages": bool(has_messages),
                "has_prompt": bool(has_prompt),
                "raw_text_char_count": int(len(raw_text)),
                "source_dataset_dir_relpath": relpath_from_repo(bundle.dataset_dir),
                "source_examples_relpath": relpath_from_repo(bundle.examples_path),
                "source_sentences_relpath": relpath_from_repo(bundle.sentences_path),
                "source_localization_relpath": relpath_from_repo(source_adaptive_json) if source_adaptive_json.exists() else "",
                "source_localization_exists": bool(source_adaptive_json.exists()),
            }
        )

    return rows, examples_by_id, sentences_by_id


def candidate_rows_for_compressed_bundle(
    bundle: BundleSpec,
    *,
    text_field: str,
    min_sentences: int,
    max_sentences: int,
    examples_per_bundle: int,
    selection_seed: int,
    compressed_probe_initial: int,
    compressed_probe_multiplier: float,
    compressed_probe_label_slack: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    examples_by_id: dict[str, dict[str, Any]] = {}
    sentences_by_id: dict[str, list[dict[str, Any]]] = {}
    candidate_paths = sorted(bundle.localization_dir.glob("sentence_localization_*.json.gz"))
    ranked_paths = sorted(
        candidate_paths,
        key=lambda path: stable_rank_key(
            f"compressed::{bundle.bundle_key}::{path.name}",
            seed=selection_seed + 503,
        ),
    )
    desired_deceptive, desired_truthful = allocate_label_targets(
        total_count=int(examples_per_bundle),
        deceptive_available=int(examples_per_bundle),
        truthful_available=int(examples_per_bundle),
    )
    min_deceptive_to_stop = desired_deceptive + max(0, int(compressed_probe_label_slack))
    min_truthful_to_stop = desired_truthful + max(0, int(compressed_probe_label_slack))
    next_limit = min(
        len(ranked_paths),
        max(int(examples_per_bundle), int(compressed_probe_initial)),
    )
    processed = 0

    while processed < len(ranked_paths):
        for path in ranked_paths[processed:next_limit]:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
            example_id = str(payload.get("example_id") or "").strip()
            if not example_id or example_id in examples_by_id:
                continue

            raw_text = str(payload.get("raw_text") or "")
            prompt = str(payload.get("prompt") or "")
            if not raw_text.strip() or not prompt.strip():
                continue
            full_score = payload.get("full_score") if isinstance(payload.get("full_score"), dict) else {}
            sentence_count = int(full_score.get("sentence_end_idx") or 0)
            if sentence_count < int(min_sentences) or sentence_count > int(max_sentences):
                continue
            full_score_rate = full_score.get("deception_rate")
            if full_score_rate is None:
                continue
            label_value = bool(float(full_score_rate) >= 0.5)

            example_record = synthesize_example_record(payload, text_field=text_field)
            sentence_rows = synthesize_sentence_rows(example_id=example_id, raw_text=raw_text)
            if len(sentence_rows) != sentence_count:
                sentence_count = len(sentence_rows)
                if sentence_count < int(min_sentences) or sentence_count > int(max_sentences):
                    continue

            examples_by_id[example_id] = example_record
            sentences_by_id[example_id] = sentence_rows
            rows.append(
                {
                    "bundle_key": bundle.bundle_key,
                    "env_name": bundle.env_name,
                    "env_display": bundle.env_display,
                    "model_bundle_name": bundle.model_bundle_name,
                    "model_display": bundle.model_display,
                    "model_id": bundle.model_id,
                    "example_id": example_id,
                    "deceptive": bool(label_value),
                    "label_name": "deceptive" if bool(label_value) else "truthful",
                    "sentence_count": int(sentence_count),
                    "has_messages": False,
                    "has_prompt": True,
                    "raw_text_char_count": int(len(raw_text)),
                    "source_dataset_dir_relpath": relpath_from_repo(bundle.dataset_dir),
                    "source_examples_relpath": "",
                    "source_sentences_relpath": "",
                    "source_localization_relpath": relpath_from_repo(path),
                    "source_localization_exists": True,
                }
            )

        processed = next_limit
        selected_rows, summary = select_bundle_rows(
            rows,
            examples_per_bundle=examples_per_bundle,
            selection_seed=selection_seed,
        )
        have_balanced_pool = (
            summary["eligible_deceptive_examples"] >= min_deceptive_to_stop
            and summary["eligible_truthful_examples"] >= min_truthful_to_stop
        )
        if processed >= len(ranked_paths):
            break
        if summary["selected_examples"] >= int(examples_per_bundle) and have_balanced_pool:
            break
        next_limit = min(
            len(ranked_paths),
            max(
                processed + 1,
                int(math.ceil(processed * max(1.1, float(compressed_probe_multiplier)))),
            ),
        )

    print(
        "Prepared compressed bundle pool "
        f"{bundle.bundle_key}: scanned {processed}/{len(ranked_paths)} files, "
        f"kept {len(rows)} eligible examples."
    )

    return rows, examples_by_id, sentences_by_id


def select_bundle_rows(
    candidate_rows: list[dict[str, Any]],
    *,
    examples_per_bundle: int,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deceptive_rows = [row for row in candidate_rows if bool(row.get("deceptive"))]
    truthful_rows = [row for row in candidate_rows if not bool(row.get("deceptive"))]
    deceptive_target, truthful_target = allocate_label_targets(
        total_count=int(examples_per_bundle),
        deceptive_available=len(deceptive_rows),
        truthful_available=len(truthful_rows),
    )

    selected_deceptive = round_robin_pick(
        deceptive_rows,
        target_count=deceptive_target,
        seed=selection_seed + 11,
        num_buckets=3,
    )
    selected_truthful = round_robin_pick(
        truthful_rows,
        target_count=truthful_target,
        seed=selection_seed + 29,
        num_buckets=3,
    )

    selected_ids = {
        str(row.get("example_id"))
        for row in [*selected_deceptive, *selected_truthful]
    }
    remaining_rows = [
        row
        for row in candidate_rows
        if str(row.get("example_id")) not in selected_ids
    ]
    remaining_rows = sorted(
        remaining_rows,
        key=lambda row: stable_rank_key(
            f"remaining::{row.get('label_name')}::{row.get('sentence_count')}::{row.get('example_id')}",
            seed=selection_seed + 71,
        ),
    )
    combined = [*selected_deceptive, *selected_truthful]
    if len(combined) < int(examples_per_bundle):
        combined.extend(remaining_rows[: int(examples_per_bundle) - len(combined)])

    combined = sorted(
        combined[: int(examples_per_bundle)],
        key=lambda row: (
            0 if bool(row.get("deceptive")) else 1,
            int(row.get("sentence_count", 0)),
            str(row.get("example_id")),
        ),
    )

    summary = {
        "eligible_examples": int(len(candidate_rows)),
        "eligible_deceptive_examples": int(len(deceptive_rows)),
        "eligible_truthful_examples": int(len(truthful_rows)),
        "selected_examples": int(len(combined)),
        "selected_deceptive_examples": int(sum(1 for row in combined if bool(row.get("deceptive")))),
        "selected_truthful_examples": int(sum(1 for row in combined if not bool(row.get("deceptive")))),
        "deceptive_target": int(deceptive_target),
        "truthful_target": int(truthful_target),
    }
    return combined, summary


def write_bundle_subset(
    bundle: BundleSpec,
    selected_rows: list[dict[str, Any]],
    *,
    examples_by_id: dict[str, dict[str, Any]],
    sentences_by_id: dict[str, list[dict[str, Any]]],
    bundle_root: Path,
) -> dict[str, Any]:
    selected_example_ids = [str(row.get("example_id")) for row in selected_rows]
    selected_examples = [examples_by_id[example_id] for example_id in selected_example_ids]
    selected_sentences: list[dict[str, Any]] = []
    for example_id in selected_example_ids:
        selected_sentences.extend(
            sorted(
                sentences_by_id.get(example_id, []),
                key=lambda row: int(row.get("sentence_idx", 0)),
            )
        )

    examples_path = write_jsonl(bundle_root / "examples.jsonl", selected_examples)
    sentences_path = write_jsonl(bundle_root / "sentences.jsonl", selected_sentences)
    selected_examples_csv = write_csv(
        bundle_root / "selected_examples.csv",
        selected_rows,
    )
    bundle_config = {
        "bundle_key": bundle.bundle_key,
        "env_name": bundle.env_name,
        "model_bundle_name": bundle.model_bundle_name,
        "model_id": bundle.model_id,
        "selected_example_ids": selected_example_ids,
        "num_examples": len(selected_example_ids),
        "created_at_utc": utc_now_iso(),
    }
    write_json(bundle_root / "bundle_config.json", bundle_config)
    return {
        "bundle_examples_relpath": relpath_from_repo(examples_path),
        "bundle_sentences_relpath": relpath_from_repo(sentences_path),
        "bundle_selected_examples_relpath": relpath_from_repo(selected_examples_csv),
        "bundle_config_relpath": relpath_from_repo(bundle_root / "bundle_config.json"),
    }


def build_manifest_rows(
    bundle: BundleSpec,
    selected_rows: list[dict[str, Any]],
    bundle_paths: dict[str, Any],
    *,
    results_root: Path | str,
    run_name: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    mode: str,
    text_field: str,
    base_seed: int,
    coarse_iters: int,
    refinement_iters: int,
    min_valid: int,
    min_step_size: int,
    min_spacing: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> list[dict[str, Any]]:
    bundle_key = bundle.bundle_key
    output_root = ensure_dir(
        Path(results_root).expanduser().resolve() / run_name / "runs" / "full" / bundle_key
    )
    out_dir = ensure_dir(output_root / "localization")
    jsonl_path = output_root / "full.jsonl"
    return [
        {
            "run_name": run_name,
            "method": "full",
            "bundle_key": bundle_key,
            "env_name": bundle.env_name,
            "env_display": bundle.env_display,
            "model_bundle_name": bundle.model_bundle_name,
            "model_display": bundle.model_display,
            "model_id": bundle.model_id,
            "num_examples": int(len(selected_rows)),
            "examples_relpath": bundle_paths["bundle_examples_relpath"],
            "sentences_relpath": bundle_paths["bundle_sentences_relpath"],
            "selected_examples_relpath": bundle_paths["bundle_selected_examples_relpath"],
            "out_dir_relpath": relpath_from_repo(out_dir),
            "jsonl_relpath": relpath_from_repo(jsonl_path),
            "n_samples": int(n_samples),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "mode": str(mode),
            "text_field": str(text_field),
            "base_seed": int(base_seed),
            "coarse_iters": int(coarse_iters),
            "refinement_iters": int(refinement_iters),
            "min_valid": int(min_valid),
            "min_step_size": int(min_step_size),
            "min_spacing": int(min_spacing),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "tensor_parallel_size": int(tensor_parallel_size),
        }
    ]


def main() -> None:
    args = parse_args()
    output_root = run_root(args.run_name, results_root=args.results_root)
    bundles_root = ensure_dir(output_root / "bundles")
    ensure_dir(output_root / "runs")
    ensure_dir(output_root / "analysis")
    ensure_dir(output_root / "slurm_logs")

    dataset_bundles = bundle_specs(
        dataset_root=args.dataset_root,
        environments=args.envs,
        model_bundles=args.model_bundles,
    )
    bundle_by_key = {bundle.bundle_key: bundle for bundle in dataset_bundles}
    compressed_bundles = compressed_bundle_specs(
        compressed_root=args.compressed_dataset_root,
        environments=args.envs,
        model_bundles=args.model_bundles,
    )
    for bundle in compressed_bundles:
        bundle_by_key.setdefault(bundle.bundle_key, bundle)
    requested_bundles = [bundle_by_key[key] for key in sorted(bundle_by_key)]
    if not requested_bundles:
        raise FileNotFoundError(
            "No DatasetMain or DatasetMainCompressed bundles matched the requested environments/model bundles."
        )

    selection_rows_all: list[dict[str, Any]] = []
    bundle_summary_rows: list[dict[str, Any]] = []
    manifest_rows_all: list[dict[str, Any]] = []

    for bundle in requested_bundles:
        print(f"Selecting examples for {bundle.bundle_key}...")
        candidate_rows, examples_by_id, sentences_by_id = candidate_rows_for_bundle(
            bundle,
            text_field=args.text_field,
            min_sentences=args.min_sentences,
            max_sentences=args.max_sentences,
            examples_per_bundle=args.examples_per_bundle,
            selection_seed=args.selection_seed,
            compressed_probe_initial=args.compressed_probe_initial,
            compressed_probe_multiplier=args.compressed_probe_multiplier,
            compressed_probe_label_slack=args.compressed_probe_label_slack,
        )
        selected_rows, summary = select_bundle_rows(
            candidate_rows,
            examples_per_bundle=args.examples_per_bundle,
            selection_seed=args.selection_seed,
        )

        bundle_root = ensure_dir(bundles_root / bundle.bundle_key)
        bundle_paths = write_bundle_subset(
            bundle,
            selected_rows,
            examples_by_id=examples_by_id,
            sentences_by_id=sentences_by_id,
            bundle_root=bundle_root,
        )

        bundle_summary_row = {
            "bundle_key": bundle.bundle_key,
            "env_name": bundle.env_name,
            "env_display": bundle.env_display,
            "model_bundle_name": bundle.model_bundle_name,
            "model_display": bundle.model_display,
            "model_id": bundle.model_id,
            **summary,
            **bundle_paths,
        }
        bundle_summary_rows.append(bundle_summary_row)

        for selection_rank, row in enumerate(selected_rows):
            selection_rows_all.append(
                {
                    **row,
                    "selection_rank": int(selection_rank),
                    **flatten_dict_row("bundle_", summary),
                    **bundle_paths,
                }
            )

        manifest_rows_all.extend(
            build_manifest_rows(
                bundle,
                selected_rows,
                bundle_paths,
                results_root=args.results_root,
                run_name=args.run_name,
                n_samples=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                mode=args.mode,
                text_field=args.text_field,
                base_seed=args.base_seed,
                coarse_iters=args.coarse_iters,
                refinement_iters=args.refinement_iters,
                min_valid=args.min_valid,
                min_step_size=args.min_step_size,
                min_spacing=args.min_spacing,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
            )
        )
        print(
            f"Selected {summary['selected_examples']} examples for {bundle.bundle_key} "
            f"({summary['selected_deceptive_examples']} deceptive / "
            f"{summary['selected_truthful_examples']} truthful)."
        )

    write_csv(output_root / "selected_examples.csv", selection_rows_all)
    write_csv(output_root / "bundle_summary.csv", bundle_summary_rows)
    write_csv(output_root / "run_manifest.csv", manifest_rows_all)
    write_csv(output_root / "run_manifest_full.csv", manifest_rows_all)
    remove_if_exists(output_root / "run_manifest_adaptive.csv")

    config = {
        "run_name": args.run_name,
        "created_at_utc": utc_now_iso(),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "dataset_root_relpath": relpath_from_repo(Path(args.dataset_root)),
        "compressed_dataset_root": str(Path(args.compressed_dataset_root).expanduser().resolve()),
        "compressed_dataset_root_relpath": relpath_from_repo(Path(args.compressed_dataset_root)),
        "results_root": str(Path(args.results_root).expanduser().resolve()),
        "results_root_relpath": relpath_from_repo(Path(args.results_root)),
        "examples_per_bundle": int(args.examples_per_bundle),
        "min_sentences": int(args.min_sentences),
        "max_sentences": int(args.max_sentences),
        "selection_seed": int(args.selection_seed),
        "envs": list(args.envs),
        "model_bundles": list(args.model_bundles),
        "localization_args": {
            "n_samples": int(args.n_samples),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "max_new_tokens": int(args.max_new_tokens),
            "mode": str(args.mode),
            "text_field": str(args.text_field),
            "base_seed": int(args.base_seed),
            "coarse_iters": int(args.coarse_iters),
            "refinement_iters": int(args.refinement_iters),
            "min_valid": int(args.min_valid),
            "min_step_size": int(args.min_step_size),
            "min_spacing": int(args.min_spacing),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "tensor_parallel_size": int(args.tensor_parallel_size),
        },
    }
    write_json(output_root / "config.json", config)

    readme_lines = [
        f"# {args.run_name}",
        "",
        "Deterministic rebuttal subset for comparing dataset adaptive localization",
        "against newly run full-trace localization.",
        "",
        "Generated artifacts:",
        "- `selected_examples.csv`: one row per chosen example.",
        "- `bundle_summary.csv`: eligible vs selected counts per environment/model bundle.",
        "- `run_manifest.csv`: full localization jobs to launch.",
        "- `run_manifest_full.csv`: alias of `run_manifest.csv` for compatibility.",
        "- `bundles/<env>__<model>/examples.jsonl`: selected examples.",
        "- `bundles/<env>__<model>/sentences.jsonl`: matching sentence records.",
        "- `selected_examples.csv[source_localization_relpath]`: dataset adaptive localization files used as the comparison baseline.",
        "",
        "Local refresh:",
        (
            "```bash\n"
            f"python {relpath_from_repo(Path(__file__))} --run-name {args.run_name}\n"
            "```"
        ),
        "",
        "Longleaf sequence:",
        (
            "```bash\n"
            "cd /work/users/s/m/smerrill/deception2\n"
            f"python rebuttal/scripts/prepare_localization_fulltrace_rebuttal.py --run-name {args.run_name}\n"
            f"bash rebuttal/slurm/submit_localization_fulltrace_rebuttal.sh {args.run_name} full\n"
            "```"
        ),
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Wrote run root: {output_root}")
    print(f"Selected examples: {len(selection_rows_all)}")
    print(f"Bundles: {len(bundle_summary_rows)}")
    print(f"Full tasks: {len(manifest_rows_all)}")


if __name__ == "__main__":
    main()
