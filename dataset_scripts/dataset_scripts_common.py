from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = THIS_DIR.parent
DEFAULT_LONGLEAF_REPO_ROOT = Path("/work/users/s/m/smerrill/deception2")


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    candidate = repo_root or os.environ.get("DECEPTION2_PROJECT_ROOT")
    if candidate:
        return Path(candidate).expanduser().resolve()
    return DEFAULT_REPO_ROOT


def resolve_datasetmain_root(
    repo_root: Path,
    datasetmain_root: str | Path | None = None,
) -> Path:
    if datasetmain_root:
        return Path(datasetmain_root).expanduser().resolve()
    return (repo_root / "DatasetMain").resolve()


def resolve_output_dir(
    repo_root: Path,
    script_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (repo_root / "dataset_scripts" / "outputs" / script_name).resolve()


def resolve_hf_cache_root(
    repo_root: Path,
    hf_cache_root: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if hf_cache_root:
        candidates.append(Path(hf_cache_root).expanduser())

    env_hf_cache_root = os.environ.get("HF_CACHE_ROOT")
    if env_hf_cache_root:
        candidates.append(Path(env_hf_cache_root).expanduser())

    env_hf_home = os.environ.get("HF_HOME")
    if env_hf_home:
        candidates.append(Path(env_hf_home).expanduser() / "hub")

    candidates.extend(
        [
            repo_root.parent / "huggingface" / "transformers",
            repo_root.parent / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "transformers",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if candidates:
        return candidates[0].resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def ensure_import_paths(repo_root: Path, *, include_styles: bool = False) -> None:
    notebooks_dir = repo_root / "Notebooks"
    if str(notebooks_dir) not in sys.path:
        sys.path.insert(0, str(notebooks_dir))

    if include_styles:
        styles_dir = repo_root / "styles"
        if str(styles_dir) not in sys.path:
            sys.path.insert(0, str(styles_dir))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df, output_dir: Path, stem: str) -> Path:
    ensure_dir(output_dir)
    path = output_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)
