from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


THIS_FILE = Path(__file__).resolve()
REBUTTAL_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "DatasetMain"
DEFAULT_RESULTS_ROOT = REBUTTAL_ROOT / "results"
DEFAULT_RUN_NAME = "localization_fulltrace_vs_adaptive_rebuttal_v1"
DEFAULT_ENVIRONMENTS = ("advisor_audit", "bs", "car_sales", "gridworld", "interview")
DEFAULT_MODEL_BUNDLES = (
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "gpt-oss-20b",
)

ENV_DISPLAY_BY_NAME = {
    "advisor_audit": "AdvisorAudit",
    "bs": "BS",
    "car_sales": "CarSales",
    "gridworld": "Gridworld",
    "interview": "Interview",
}

MODEL_DISPLAY_BY_BUNDLE = {
    "DeepSeek-R1-Distill-Llama-8B": "Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B": "Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B": "Qwen-7B",
    "gpt-oss-20b": "GPT-OSS-20B",
}

MODEL_ID_BY_BUNDLE = {
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}


@dataclass(frozen=True)
class BundleSpec:
    env_name: str
    env_display: str
    model_bundle_name: str
    model_display: str
    model_id: str
    dataset_dir: Path
    examples_path: Path
    sentences_path: Path
    localization_dir: Path

    @property
    def bundle_key(self) -> str:
        return f"{self.env_name}__{self.model_bundle_name}"


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_root(run_name: str, *, results_root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    return ensure_dir(Path(results_root).expanduser().resolve() / str(run_name))


def relpath_from_repo(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve().relative_to(REPO_ROOT))


def resolve_repo_path(path_or_relpath: str | Path, *, project_root: Path | str = REPO_ROOT) -> Path:
    candidate = Path(path_or_relpath).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(project_root).expanduser().resolve() / candidate).resolve()


def read_json(path: Path | str, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_path


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=False) + "\n")
    return file_path


def write_csv(path: Path | str, rows: Sequence[dict[str, Any]]) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with file_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return file_path


def bundle_specs(
    *,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    environments: Sequence[str] | None = None,
    model_bundles: Sequence[str] | None = None,
) -> list[BundleSpec]:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    requested_envs = tuple(environments or DEFAULT_ENVIRONMENTS)
    requested_models = tuple(model_bundles or DEFAULT_MODEL_BUNDLES)
    out: list[BundleSpec] = []
    for env_name in requested_envs:
        env_dir = dataset_root_path / str(env_name)
        if not env_dir.exists():
            continue
        for model_bundle_name in requested_models:
            dataset_dir = env_dir / str(model_bundle_name)
            examples_path = dataset_dir / "examples.jsonl"
            sentences_path = dataset_dir / "sentences.jsonl"
            localization_dir = dataset_dir / "localization"
            if not examples_path.exists() or not sentences_path.exists():
                continue
            out.append(
                BundleSpec(
                    env_name=str(env_name),
                    env_display=ENV_DISPLAY_BY_NAME.get(str(env_name), str(env_name)),
                    model_bundle_name=str(model_bundle_name),
                    model_display=MODEL_DISPLAY_BY_BUNDLE.get(str(model_bundle_name), str(model_bundle_name)),
                    model_id=MODEL_ID_BY_BUNDLE.get(str(model_bundle_name), str(model_bundle_name)),
                    dataset_dir=dataset_dir,
                    examples_path=examples_path,
                    sentences_path=sentences_path,
                    localization_dir=localization_dir,
                )
            )
    return out


def example_output_filename(example_id: str) -> str:
    safe_id = str(example_id).replace("/", "_")
    return f"sentence_localization_{safe_id}.json"


def localization_output_path(out_dir: Path | str, example_id: str) -> Path:
    return Path(out_dir) / example_output_filename(example_id)


def stable_rank_key(text: str, *, seed: int) -> tuple[int, str]:
    digest = hashlib.md5(f"{int(seed)}::{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16), str(text)


def quantile_bucket_ranks(size: int, num_buckets: int) -> list[int]:
    if size <= 0:
        return []
    if num_buckets <= 1:
        return [0] * size
    return [min(num_buckets - 1, int(idx * num_buckets / size)) for idx in range(size)]


def assign_bucket_index(
    rows: Sequence[dict[str, Any]],
    *,
    sort_keys: Sequence[str],
    num_buckets: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    sorted_rows = sorted(
        rows,
        key=lambda row: tuple(row.get(key) for key in sort_keys),
    )
    bucket_values = quantile_bucket_ranks(len(sorted_rows), num_buckets=max(1, int(num_buckets)))
    out: list[dict[str, Any]] = []
    for row, bucket_idx in zip(sorted_rows, bucket_values):
        enriched = dict(row)
        enriched["bucket_idx"] = int(bucket_idx)
        out.append(enriched)
    return out


def round_robin_pick(
    rows: Sequence[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
    num_buckets: int = 3,
) -> list[dict[str, Any]]:
    if target_count <= 0 or not rows:
        return []

    with_buckets = assign_bucket_index(
        rows,
        sort_keys=("sentence_count", "example_id"),
        num_buckets=max(1, int(num_buckets)),
    )
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in with_buckets:
        bucket_idx = int(row.get("bucket_idx", 0))
        buckets.setdefault(bucket_idx, []).append(row)

    for bucket_idx, bucket_rows in buckets.items():
        bucket_rows.sort(
            key=lambda row: stable_rank_key(
                f"{bucket_idx}::{row.get('example_id')}",
                seed=seed,
            )
        )

    selected: list[dict[str, Any]] = []
    while len(selected) < target_count and any(buckets.values()):
        for bucket_idx in sorted(buckets):
            bucket_rows = buckets[bucket_idx]
            if bucket_rows and len(selected) < target_count:
                selected.append(bucket_rows.pop(0))

    if len(selected) >= target_count:
        return selected[:target_count]

    selected_ids = {str(row.get("example_id")) for row in selected}
    leftovers = [
        row
        for row in with_buckets
        if str(row.get("example_id")) not in selected_ids
    ]
    leftovers.sort(
        key=lambda row: stable_rank_key(
            f"leftover::{row.get('example_id')}",
            seed=seed,
        )
    )
    return selected + leftovers[: max(0, target_count - len(selected))]


def allocate_label_targets(
    *,
    total_count: int,
    deceptive_available: int,
    truthful_available: int,
) -> tuple[int, int]:
    base_deceptive = min(int(total_count // 2), int(deceptive_available))
    base_truthful = min(int(total_count - base_deceptive), int(truthful_available))

    remaining = int(total_count) - int(base_deceptive) - int(base_truthful)
    deceptive_target = int(base_deceptive)
    truthful_target = int(base_truthful)

    while remaining > 0:
        deceptive_left = int(deceptive_available) - deceptive_target
        truthful_left = int(truthful_available) - truthful_target
        if deceptive_left <= 0 and truthful_left <= 0:
            break
        if deceptive_left >= truthful_left and deceptive_left > 0:
            deceptive_target += 1
        elif truthful_left > 0:
            truthful_target += 1
        remaining -= 1

    return deceptive_target, truthful_target


def flatten_dict_row(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        out[f"{prefix}{key}"] = value
    return out


def read_csv_rows(path: Path | str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]
