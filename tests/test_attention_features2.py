from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import attention_features2 as af2


def test_resolve_dataset_paths_defaults_to_attention_features2_parquet() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "DeepSeek-R1-Distill-Qwen-7B"
        localization_dir = dataset_dir / "localization"
        localization_dir.mkdir(parents=True)
        examples_path = dataset_dir / "examples.jsonl"
        examples_path.write_text(
            json.dumps(
                {
                    "example_id": "run/state_0/sample_0",
                    "meta_model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        dataset_paths = af2.resolve_dataset_paths(dataset_dir, None)
        model_id = af2.infer_model_id(dataset_paths, override=None)

        assert dataset_paths.dataset_dir == dataset_dir.resolve()
        assert dataset_paths.localization_dir == localization_dir.resolve()
        assert dataset_paths.output_path == (dataset_dir / "attention_features2.parquet").resolve()
        assert model_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        assert len(af2.build_base_feature_columns(2)) == 200
        assert len(af2.build_change_feature_columns(2)) == 204
        assert len(af2.build_normalized_feature_columns(2)) == 368
        assert len(af2.build_feature_columns(2)) == 772


def test_add_attention_region_columns_populates_previous_recent_and_early_counts() -> None:
    aligned_sentence_df = pd.DataFrame(
        [
            {"sentence_idx": 0, "start_token": 0, "end_token": 1, "token_count": 2, "context_token_count": 0},
            {"sentence_idx": 1, "start_token": 2, "end_token": 3, "token_count": 2, "context_token_count": 2},
            {"sentence_idx": 2, "start_token": 4, "end_token": 5, "token_count": 2, "context_token_count": 4},
        ]
    )

    enriched = af2.add_attention_region_columns(
        aligned_sentence_df,
        recent_window_tokens=2,
    )

    assert enriched["previous_sentence_token_count"].tolist() == [0, 2, 2]
    assert enriched["recent_token_count"].tolist() == [0, 2, 2]
    assert enriched["early_token_count"].tolist() == [0, 0, 2]
    assert enriched["prior_all_token_count"].tolist() == [0, 2, 4]
    assert enriched["available_token_count"].tolist() == [2, 4, 6]
    prev_start = enriched["previous_sentence_start_token"].tolist()
    prev_end = enriched["previous_sentence_end_token"].tolist()
    assert pd.isna(prev_start[0]) and prev_start[1:] == [0.0, 2.0]
    assert pd.isna(prev_end[0]) and prev_end[1:] == [1.0, 3.0]


def test_iter_localization_paths_supports_round_robin_sharding() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        localization_dir = Path(tmpdir) / "localization"
        localization_dir.mkdir(parents=True)
        for idx in range(6):
            (localization_dir / f"sample_{idx:02d}.json").write_text("{}", encoding="utf-8")

        shard0 = af2.iter_localization_paths(localization_dir, max_examples=0, shard_id=0, num_shards=3)
        shard1 = af2.iter_localization_paths(localization_dir, max_examples=0, shard_id=1, num_shards=3)
        shard2 = af2.iter_localization_paths(localization_dir, max_examples=0, shard_id=2, num_shards=3)
        limited = af2.iter_localization_paths(localization_dir, max_examples=5, shard_id=1, num_shards=2)

        assert [path.name for path in shard0] == ["sample_00.json", "sample_03.json"]
        assert [path.name for path in shard1] == ["sample_01.json", "sample_04.json"]
        assert [path.name for path in shard2] == ["sample_02.json", "sample_05.json"]
        assert [path.name for path in limited] == ["sample_01.json", "sample_03.json"]


def test_iter_localization_paths_rejects_invalid_shards() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        localization_dir = Path(tmpdir) / "localization"
        localization_dir.mkdir(parents=True)
        (localization_dir / "sample_00.json").write_text("{}", encoding="utf-8")

        try:
            af2.iter_localization_paths(localization_dir, max_examples=0, shard_id=0, num_shards=0)
            raise AssertionError("Expected num_shards=0 to raise ValueError")
        except ValueError as exc:
            assert "num_shards" in str(exc)

        try:
            af2.iter_localization_paths(localization_dir, max_examples=0, shard_id=-1, num_shards=1)
            raise AssertionError("Expected shard_id=-1 to raise ValueError")
        except ValueError as exc:
            assert "shard_id" in str(exc)

        try:
            af2.iter_localization_paths(localization_dir, max_examples=0, shard_id=1, num_shards=1)
            raise AssertionError("Expected shard_id >= num_shards to raise ValueError")
        except ValueError as exc:
            assert "shard_id" in str(exc)


def test_compute_attention_features_matches_grounding_and_concentration_formulas() -> None:
    aligned_sentence_df = pd.DataFrame(
        [
            {
                "sentence_idx": 0,
                "sentence_text": "ab",
                "deception_rate": 0.1,
                "num_truthful": 4,
                "num_valid": 5,
                "raw_start": 0,
                "raw_end": 2,
                "full_start": 0,
                "full_end": 2,
                "token_indices": [0, 1],
                "start_token": 0,
                "end_token": 1,
                "token_count": 2,
                "context_token_count": 0,
            },
            {
                "sentence_idx": 1,
                "sentence_text": "cd",
                "deception_rate": 0.2,
                "num_truthful": 3,
                "num_valid": 5,
                "raw_start": 2,
                "raw_end": 4,
                "full_start": 2,
                "full_end": 4,
                "token_indices": [2, 3],
                "start_token": 2,
                "end_token": 3,
                "token_count": 2,
                "context_token_count": 2,
            },
            {
                "sentence_idx": 2,
                "sentence_text": "ef",
                "deception_rate": 0.6,
                "num_truthful": 2,
                "num_valid": 5,
                "raw_start": 4,
                "raw_end": 6,
                "full_start": 4,
                "full_end": 6,
                "token_indices": [4, 5],
                "start_token": 4,
                "end_token": 5,
                "token_count": 2,
                "context_token_count": 4,
            },
        ]
    )
    aligned_sentence_df = af2.add_attention_region_columns(
        aligned_sentence_df,
        recent_window_tokens=2,
    )
    modeling_sentence_df = aligned_sentence_df.loc[aligned_sentence_df["sentence_idx"] == 2].copy()

    layer = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
    layer[0, 0, 4, :5] = torch.tensor([0.10, 0.10, 0.20, 0.20, 0.40], dtype=torch.float32)
    layer[0, 0, 5, :6] = torch.tensor([0.10, 0.10, 0.20, 0.20, 0.20, 0.20], dtype=torch.float32)

    feature_df = af2.compute_attention_features(
        [layer],
        modeling_sentence_df,
        example_id="run/state_0/sample_0",
        prompt_token_count=0,
        recent_window_tokens=2,
    )
    row = feature_df.iloc[0]

    p_full = [0.10, 0.10, 0.20, 0.20, 0.30, 0.10]
    p_prior = [1 / 6, 1 / 6, 1 / 3, 1 / 3]
    p_self = [0.75, 0.25]
    entropy_full = -sum(p * math.log(p) for p in p_full) / math.log(6)
    entropy_prior = -sum(p * math.log(p) for p in p_prior) / math.log(4)
    entropy_self = -sum(p * math.log(p) for p in p_self) / math.log(2)
    herfindahl_full = sum(p * p for p in p_full)
    herfindahl_prior = sum(p * p for p in p_prior)

    assert int(row["previous_sentence_token_count"]) == 2
    assert int(row["recent_token_count"]) == 2
    assert int(row["early_token_count"]) == 2
    assert int(row["prior_all_token_count"]) == 4

    assert math.isclose(float(row["g_prior_vs_self_mean_l0"]), 3.0 / 7.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["g_prev_vs_self_mean_l0"]), 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["g_recent_vs_self_mean_l0"]), 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["g_early_vs_recent_mean_l0"]), 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-6)
    expected_prev_share = 0.2 / (0.15 + af2.EPS)
    assert math.isclose(float(row["g_prev_share_of_prior_mean_l0"]), 0.4 / (0.6 + af2.EPS), rel_tol=0.0, abs_tol=1e-6)

    assert math.isclose(float(row["entropy_full_mean_l0"]), entropy_full, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["entropy_prior_mean_l0"]), entropy_prior, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["entropy_self_mean_l0"]), entropy_self, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top1_full_mean_l0"]), 0.30, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top1_prior_mean_l0"]), 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top1_self_mean_l0"]), 0.75, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top5_full_mean_l0"]), 0.90, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top5_prior_mean_l0"]), 1.00, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top5_self_mean_l0"]), 1.00, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top10_full_mean_l0"]), 1.00, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top10_prior_mean_l0"]), 1.00, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["top10_self_mean_l0"]), 1.00, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["herfindahl_full_mean_l0"]), herfindahl_full, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["herfindahl_prior_mean_l0"]), herfindahl_prior, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["herfindahl_self_mean_l0"]), 0.625, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["effective_support_full_mean_l0"]), 1.0 / herfindahl_full, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["effective_support_prior_mean_l0"]), 1.0 / herfindahl_prior, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["effective_support_self_mean_l0"]), 1.6, rel_tol=0.0, abs_tol=1e-6)

    assert math.isclose(float(row["g_prior_vs_self_std_l0"]), 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["entropy_full_min_l0"]), entropy_full, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(row["entropy_full_max_l0"]), entropy_full, rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(row["delta_g_prior_vs_self_mean_l0"])
    assert pd.isna(row["z_g_prior_vs_self_mean_l0"])
    assert pd.isna(row["pct_g_prior_vs_self_mean_l0"])


def test_change_and_normalization_features_follow_previous_sentences_only() -> None:
    num_layers = 1
    df = pd.DataFrame({"sentence_idx": [0, 1, 2]})
    for column in af2.build_base_feature_columns(num_layers):
        df[column] = 0.0

    target = "g_prior_vs_self_mean_l0"
    target_std = "g_prior_vs_self_std_l0"
    df[target] = [0.5, 0.4, 0.3]
    df[target_std] = [0.1, 0.2, 0.1]

    out = af2.add_transition_features(df, num_layers=num_layers)
    out = af2.add_within_trace_normalization(out, num_layers=num_layers)

    assert pd.isna(out.loc[0, "delta_g_prior_vs_self_mean_l0"])
    assert math.isclose(float(out.loc[1, "delta_g_prior_vs_self_mean_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "delta_g_prior_vs_self_mean_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(out.loc[0, "devrun_g_prior_vs_self_mean_l0"])
    assert math.isclose(float(out.loc[1, "devrun_g_prior_vs_self_mean_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "devrun_g_prior_vs_self_mean_l0"]), -0.15, rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(out.loc[0, "logratio_prev_g_prior_vs_self_mean_l0"])
    assert math.isclose(float(out.loc[1, "logratio_prev_g_prior_vs_self_mean_l0"]), math.log((0.4 + af2.EPS) / (0.5 + af2.EPS)), rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "logratio_prev_g_prior_vs_self_mean_l0"]), math.log((0.3 + af2.EPS) / (0.4 + af2.EPS)), rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(out.loc[0, "slope3_g_prior_vs_self_mean_l0"])
    assert math.isclose(float(out.loc[1, "slope3_g_prior_vs_self_mean_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "slope3_g_prior_vs_self_mean_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(out.loc[0, "min_gap_g_prior_vs_self_mean_l0"])
    assert math.isclose(float(out.loc[1, "min_gap_g_prior_vs_self_mean_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "min_gap_g_prior_vs_self_mean_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(out.loc[0, "max_gap_g_prior_vs_self_mean_l0"])
    assert math.isclose(float(out.loc[1, "max_gap_g_prior_vs_self_mean_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "max_gap_g_prior_vs_self_mean_l0"]), -0.2, rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(out.loc[0, "delta_g_prior_vs_self_std_l0"])
    assert math.isclose(float(out.loc[1, "delta_g_prior_vs_self_std_l0"]), 0.1, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "delta_g_prior_vs_self_std_l0"]), -0.1, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[1, "logratio_prev_g_prior_vs_self_std_l0"]), math.log((0.2 + af2.EPS) / (0.1 + af2.EPS)), rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(out.loc[0, "z_g_prior_vs_self_mean_l0"])
    assert math.isclose(float(out.loc[2, "z_g_prior_vs_self_mean_l0"]), (-0.15) / (0.05 + af2.EPS), rel_tol=0.0, abs_tol=1e-4)
    assert math.isclose(float(out.loc[1, "pct_g_prior_vs_self_mean_l0"]), 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "pct_g_prior_vs_self_mean_l0"]), 0.0, rel_tol=0.0, abs_tol=1e-6)

    assert pd.isna(out.loc[0, "z_g_prior_vs_self_std_l0"])
    assert math.isclose(float(out.loc[1, "pct_g_prior_vs_self_std_l0"]), 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(out.loc[2, "pct_g_prior_vs_self_std_l0"]), 0.5, rel_tol=0.0, abs_tol=1e-6)
