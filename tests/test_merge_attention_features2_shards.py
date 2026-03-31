from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import merge_attention_features2_shards as merge_shards


def _write_shard(path: Path, rows: list[tuple[str, int, float]]) -> None:
    table = pa.table(
        {
            "example_id": [row[0] for row in rows],
            "sentence_idx": [row[1] for row in rows],
            "score": [row[2] for row in rows],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_discover_shard_files_detects_missing_shards() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "gridworld" / "DeepSeek-R1-Distill-Qwen-7B"
        shard_dir = dataset_dir / "attention_features2_shards"
        _write_shard(shard_dir / "attention_features2_shard_0_of_3.parquet", [("a", 0, 0.1)])
        _write_shard(shard_dir / "attention_features2_shard_2_of_3.parquet", [("b", 1, 0.2)])

        shard_files = merge_shards.discover_shard_files(shard_dir)

        assert [shard.shard_id for shard in shard_files] == [0, 2]
        assert merge_shards.missing_shard_ids(shard_files) == [1]


def test_merge_shard_files_writes_combined_output() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "gridworld" / "DeepSeek-R1-Distill-Qwen-7B"
        shard_dir = dataset_dir / "attention_features2_shards"
        output_path = dataset_dir / "attention_features2.parquet"

        _write_shard(
            shard_dir / "attention_features2_shard_0_of_2.parquet",
            [("run/state_0/sample_0", 0, 0.1)],
        )
        _write_shard(
            shard_dir / "attention_features2_shard_1_of_2.parquet",
            [("run/state_1/sample_0", 1, 0.2)],
        )

        shard_files = merge_shards.discover_shard_files(shard_dir)
        merged_rows = merge_shards.merge_shard_files(shard_files, output_path, overwrite=False)

        assert merged_rows == 2
        assert output_path.exists()
        merged = pq.read_table(output_path).to_pydict()
        assert merged["example_id"] == ["run/state_0/sample_0", "run/state_1/sample_0"]
        assert merged["sentence_idx"] == [0, 1]
        assert merged["score"] == [0.1, 0.2]
