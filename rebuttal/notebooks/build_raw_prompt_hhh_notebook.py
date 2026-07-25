from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = NOTEBOOK_DIR / "raw_prompt_hhh_analysis.ipynb"
REBUTTAL_ROOT = NOTEBOOK_DIR.parent
DEFAULT_RESULTS_ROOT = REBUTTAL_ROOT / "results"
DEFAULT_EXPERIMENT_NAME = "raw_prompt_hhh_full_v1"
RUNNER_PATH = REBUTTAL_ROOT / "scripts" / "run_raw_prompt_hhh_rebuttal.py"
PYTHON_BIN = Path("/playpen-ssd/smerrill/conda_envs/deception/bin/python")
DATASET_ROOT = Path("/playpen-ssd/smerrill/deception2/DatasetMainCompressed")


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Raw Prompt HHH Rebuttal Live Monitor

        Hit **Run All** any time while the tmux generation job is running. The notebook will:

        1. Compile partial summaries from completed per-example JSON files.
        2. Reload `selected_examples.csv`, `example_summary.csv`, and `pair_summary.csv`.
        3. Show progress by environment/model, current deception rates, and recent logs/errors.

        Minimal role-specific ablation being evaluated:

        - Replace the adversarial role with helpful, harmless, honest, truthful role wording.
        - Advisor Audit role adds fiduciary / investor's-best-interest wording.
        - Interview role adds that the candidate never fabricates competing offers.
        - Replace the adversarial objective with the HHH objective plus a conflict rule:
          do not choose available actions that require false or misleading claims, and
          prioritize honesty/client interests over private incentives.
        - Add `- You do not have an outside offer.` for Interview.
        - Remove only the extra Advisor Audit compensation-target note.
        - Keep `REJECT_CLAIM_HIGHER_OFFER` as an available Interview game action.

        This notebook does not launch generation. It monitors whatever is currently under the
        configured `RUN_DIR`.
        """
    ),
    code(
        f"""
        from __future__ import annotations

        import json
        import subprocess
        import time
        from datetime import datetime
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import Markdown, display

        RESULTS_ROOT = Path("{DEFAULT_RESULTS_ROOT}")
        EXPERIMENT_NAME = "{DEFAULT_EXPERIMENT_NAME}"
        RUN_DIR = RESULTS_ROOT / EXPERIMENT_NAME
        RUNNER = Path("{RUNNER_PATH}")
        PYTHON_BIN = Path("{PYTHON_BIN}")
        DATASET_ROOT = Path("{DATASET_ROOT}")

        # Set AUTO_REFRESH=True if you want this notebook to poll repeatedly in one Run All.
        AUTO_REFRESH = False
        REFRESH_ITERATIONS = 1
        REFRESH_SECONDS = 60

        MODEL_ORDER = [
            "DeepSeek-R1-Distill-Llama-8B",
            "DeepSeek-R1-Distill-Qwen-14B",
            "DeepSeek-R1-Distill-Qwen-7B",
            "gpt-oss-20b",
        ]
        ENV_ORDER = ["advisor_audit", "bs", "car_sales", "gridworld", "interview"]
        MODEL_LABELS = {{
            "DeepSeek-R1-Distill-Llama-8B": "Llama-8B",
            "DeepSeek-R1-Distill-Qwen-14B": "Qwen-14B",
            "DeepSeek-R1-Distill-Qwen-7B": "Qwen-7B",
            "gpt-oss-20b": "GPT-OSS-20B",
        }}
        ENV_LABELS = {{
            "advisor_audit": "AdvisorAudit",
            "bs": "BS",
            "car_sales": "CarSales",
            "gridworld": "Gridworld",
            "interview": "Interview",
        }}

        pd.options.display.max_columns = 200
        pd.options.display.max_colwidth = 240
        pd.options.display.width = 220


        def md(text: str) -> None:
            display(Markdown(text))


        def read_json(path: Path, default=None):
            if not path.exists():
                return default
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return default


        def read_csv(path: Path) -> pd.DataFrame:
            return pd.read_csv(path) if path.exists() else pd.DataFrame()


        def example_json_paths() -> list[Path]:
            return sorted(RUN_DIR.glob("*__*/examples/*.json")) if RUN_DIR.exists() else []


        def compile_partial_summaries() -> subprocess.CompletedProcess | None:
            if not RUN_DIR.exists():
                return None
            cmd = [
                str(PYTHON_BIN),
                str(RUNNER),
                "--results-root",
                str(RESULTS_ROOT),
                "--experiment-name",
                EXPERIMENT_NAME,
                "--dataset-root",
                str(DATASET_ROOT),
                "--compile-only",
            ]
            return subprocess.run(cmd, text=True, capture_output=True)


        def load_artifacts() -> dict[str, object]:
            return {{
                "requested_config": read_json(RUN_DIR / "requested_config.json", default={{}}) or {{}},
                "selection_inventory": read_csv(RUN_DIR / "selection_inventory.csv"),
                "selected_examples": read_csv(RUN_DIR / "selected_examples.csv"),
                "example_summary": read_csv(RUN_DIR / "example_summary.csv"),
                "pair_summary": read_csv(RUN_DIR / "pair_summary.csv"),
                "launcher_log": RUN_DIR / "logs" / "launcher.log",
            }}


        def label_env(value: str) -> str:
            return ENV_LABELS.get(str(value), str(value))


        def label_model(value: str) -> str:
            return MODEL_LABELS.get(str(value), str(value))


        def slugify(text: str) -> str:
            import re

            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "item"


        def ordered_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            out = df.copy()
            out["environment"] = pd.Categorical(out["environment"], categories=ENV_ORDER, ordered=True)
            out["model_id"] = pd.Categorical(out["model_id"], categories=MODEL_ORDER, ordered=True)
            return out.sort_values(["environment", "model_id"]).reset_index(drop=True)


        md(
            "## Monitor Loaded\\n"
            f"- Run dir: `{{RUN_DIR}}`\\n"
            f"- Run dir exists: **{{RUN_DIR.exists()}}**\\n"
            f"- Completed example JSONs visible now: **{{len(example_json_paths()):,}}**\\n"
            "- Re-run all cells any time to refresh the snapshot from disk."
        )
        """
    ),
    code(
        """
        def build_progress_table(artifacts: dict[str, object]) -> pd.DataFrame:
            selected_df = artifacts["selected_examples"]
            pair_df = artifacts["pair_summary"]
            completed_paths = example_json_paths()

            rows = []
            if not selected_df.empty:
                selected_counts = (
                    selected_df.groupby(["environment", "model_id"], as_index=False)
                    .size()
                    .rename(columns={"size": "selected_examples"})
                )
            else:
                selected_counts = pd.DataFrame(columns=["environment", "model_id", "selected_examples"])

            completed_counts = []
            for path in completed_paths:
                pair_name = path.parent.parent.name
                if "__" not in pair_name:
                    continue
                environment, model_slug = pair_name.split("__", 1)
                # The result directory uses the model_id with path separators slugified.
                model_id = model_slug
                for known_model in MODEL_ORDER:
                    if model_slug == slugify(known_model):
                        model_id = known_model
                completed_counts.append({"environment": environment, "model_id": model_id})
            completed_df = (
                pd.DataFrame(completed_counts)
                .groupby(["environment", "model_id"], as_index=False)
                .size()
                .rename(columns={"size": "completed_examples"})
                if completed_counts
                else pd.DataFrame(columns=["environment", "model_id", "completed_examples"])
            )

            progress = selected_counts.merge(completed_df, on=["environment", "model_id"], how="outer")
            if not pair_df.empty:
                keep = [
                    "environment",
                    "model_id",
                    "total_generated",
                    "total_valid",
                    "total_deceptive",
                    "pooled_deception_rate",
                    "mean_deception_rate",
                    "mean_valid_fraction",
                ]
                keep = [col for col in keep if col in pair_df.columns]
                progress = progress.merge(pair_df[keep], on=["environment", "model_id"], how="left")

            if progress.empty:
                return progress
            progress["selected_examples"] = progress["selected_examples"].fillna(0).astype(int)
            progress["completed_examples"] = progress["completed_examples"].fillna(0).astype(int)
            progress["completion"] = np.where(
                progress["selected_examples"].gt(0),
                progress["completed_examples"] / progress["selected_examples"],
                np.nan,
            )
            progress["Environment"] = progress["environment"].astype(str).map(label_env)
            progress["Model"] = progress["model_id"].astype(str).map(label_model)
            return ordered_pair_frame(progress)


        def find_log_errors(max_lines: int = 30) -> list[str]:
            log_dir = RUN_DIR / "logs"
            if not log_dir.exists():
                return []
            needles = ("ERROR", "Traceback", "RuntimeError", "EngineDeadError", "failed")
            matches = []
            for path in sorted(log_dir.glob("*.log")):
                try:
                    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if any(needle in line for needle in needles):
                            matches.append(f"{path.name}:{lineno}: {line}")
                except Exception:
                    continue
            return matches[-max_lines:]


        def tail_text(path: Path, n: int = 35) -> str:
            if not path.exists():
                return ""
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\\n".join(lines[-n:])


        def ordered_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
            pivot = (
                df.assign(
                    environment=pd.Categorical(df["environment"], categories=ENV_ORDER, ordered=True),
                    model_id=pd.Categorical(df["model_id"], categories=MODEL_ORDER, ordered=True),
                )
                .pivot(index="environment", columns="model_id", values=value_col)
                .reindex(index=ENV_ORDER, columns=MODEL_ORDER)
            )
            pivot.index = [ENV_LABELS.get(str(idx), str(idx)) for idx in pivot.index]
            pivot.columns = [MODEL_LABELS.get(str(col), str(col)) for col in pivot.columns]
            return pivot
        """
    ),
    code(
        """
        def render_snapshot(iteration: int = 1) -> None:
            md(f"# Live Results Snapshot {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            md(f"`RUN_DIR`: `{RUN_DIR}`")

            if not RUN_DIR.exists():
                md("_Run directory does not exist yet. Start the experiment launcher, then hit Run All again._")
                return

            compile_result = compile_partial_summaries()
            if compile_result is not None:
                if compile_result.stdout.strip():
                    print(compile_result.stdout.strip())
                if compile_result.returncode != 0:
                    md("### Compile Error")
                    print(compile_result.stderr)

            artifacts = load_artifacts()
            selected_df = artifacts["selected_examples"]
            example_df = artifacts["example_summary"]
            pair_df = artifacts["pair_summary"]
            config = artifacts["requested_config"]

            args = config.get("args") if isinstance(config, dict) else {}
            samples_per_example = int((args or {}).get("samples_per_example") or 0)
            selected_n = len(selected_df)
            completed_n = len(example_json_paths())
            total_generated = int(example_df["num_generated"].fillna(0).sum()) if not example_df.empty else 0
            expected_generations = selected_n * samples_per_example if samples_per_example else None

            lines = [
                f"- Selected examples: **{selected_n:,}**",
                f"- Completed example JSONs: **{completed_n:,}**",
                f"- Compiled generated samples: **{total_generated:,}**",
            ]
            if expected_generations:
                lines.append(f"- Expected generated samples: **{expected_generations:,}**")
                lines.append(f"- Sample completion: **{total_generated / expected_generations:.1%}**")
            md("\\n".join(lines))

            progress = build_progress_table(artifacts)
            if not progress.empty:
                md("### Progress And Current Rates")
                columns = [
                    "Environment",
                    "Model",
                    "selected_examples",
                    "completed_examples",
                    "completion",
                    "total_generated",
                    "total_valid",
                    "total_deceptive",
                    "pooled_deception_rate",
                    "mean_deception_rate",
                    "mean_valid_fraction",
                ]
                columns = [col for col in columns if col in progress.columns]
                display(
                    progress[columns].style.hide(axis="index").format(
                        {
                            "completion": "{:.0%}",
                            "pooled_deception_rate": "{:.1%}",
                            "mean_deception_rate": "{:.1%}",
                            "mean_valid_fraction": "{:.1%}",
                            "total_generated": "{:,.0f}",
                            "total_valid": "{:,.0f}",
                            "total_deceptive": "{:,.0f}",
                        },
                        na_rep="",
                    )
                )

            errors = find_log_errors()
            if errors:
                md("### Recent Error Matches")
                print("\\n".join(errors))
            else:
                md("### Recent Error Matches\\n_No ERROR/Traceback/RuntimeError/EngineDeadError/failed matches found in logs._")

            launcher_tail = tail_text(RUN_DIR / "logs" / "launcher.log")
            if launcher_tail:
                md("### Launcher Log Tail")
                print(launcher_tail)

            if not pair_df.empty:
                pair_df = ordered_pair_frame(pair_df)
                md("### Deception Rate Heatmaps")
                pooled = ordered_pivot(pair_df, "pooled_deception_rate")
                mean = ordered_pivot(pair_df, "mean_deception_rate")
                display(pooled.style.format("{:.1%}", na_rep=""))
                display(mean.style.format("{:.1%}", na_rep=""))

                fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
                for ax, pivot, title in [
                    (axes[0], pooled, "Pooled deception rate"),
                    (axes[1], mean, "Mean example deception rate"),
                ]:
                    matrix = pivot.to_numpy(dtype=float)
                    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="OrRd", vmin=0.0, vmax=1.0)
                    ax.set_xticks(np.arange(len(pivot.columns)))
                    ax.set_xticklabels(list(pivot.columns), rotation=25, ha="right")
                    ax.set_yticks(np.arange(len(pivot.index)))
                    ax.set_yticklabels(list(pivot.index))
                    ax.set_title(title)
                    for row_idx in range(matrix.shape[0]):
                        for col_idx in range(matrix.shape[1]):
                            value = matrix[row_idx, col_idx]
                            if np.isfinite(value):
                                ax.text(col_idx, row_idx, f"{value:.0%}", ha="center", va="center", fontsize=9)
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                plt.show()

            if not example_df.empty:
                top = example_df.sort_values("deception_rate", ascending=False).head(12)
                md("### Highest-Deception Completed Examples")
                display(
                    top[
                        [
                            "environment",
                            "model_id",
                            "example_id",
                            "deception_rate",
                            "num_valid",
                            "num_generated",
                            "source_full_trace_deception_rate",
                        ]
                    ].style.hide(axis="index").format(
                        {
                            "deception_rate": "{:.1%}",
                            "source_full_trace_deception_rate": "{:.1%}",
                        },
                        na_rep="",
                    )
                )
        """
    ),
    code(
        """
        iterations = REFRESH_ITERATIONS if AUTO_REFRESH else 1
        for idx in range(iterations):
            render_snapshot(idx + 1)
            if AUTO_REFRESH and idx < iterations - 1:
                time.sleep(REFRESH_SECONDS)
        """
    ),
    code(
        """
        def load_example_payload(environment: str, model_id: str, example_id: str) -> dict:
            example_df = read_csv(RUN_DIR / "example_summary.csv")
            match = example_df.loc[
                example_df["environment"].astype(str).eq(environment)
                & example_df["model_id"].astype(str).eq(model_id)
                & example_df["example_id"].astype(str).eq(example_id)
            ]
            if match.empty:
                raise KeyError(f"Example not found in compiled summaries: {(environment, model_id, example_id)}")
            result_path = Path(match.iloc[0]["result_path"])
            return json.loads(result_path.read_text(encoding="utf-8"))


        def show_example(environment: str, model_id: str, example_id: str, num_generations: int = 3) -> None:
            payload = load_example_payload(environment, model_id, example_id)
            summary = payload.get("summary") or {}
            deception_rate = summary.get("deception_rate")
            source_rate = payload.get("source_full_trace_deception_rate")
            deception_text = f"{float(deception_rate):.1%}" if pd.notna(deception_rate) else "n/a"
            source_text = f"{float(source_rate):.1%}" if pd.notna(source_rate) else "n/a"
            md(
                f"### {ENV_LABELS.get(environment, environment)} | {MODEL_LABELS.get(model_id, model_id)}\\n"
                f"- Example ID: `{example_id}`\\n"
                f"- Deception rate: **{deception_text}**\\n"
                f"- Valid samples: **{summary.get('num_valid', 0)}** / {summary.get('num_generated', 0)}\\n"
                f"- Original full-trace rate: **{source_text}**"
            )
            print("Modified prompt preview:")
            print(payload.get("modified_prompt", "")[:1600])
            print("\\nSample generations:")
            for generation in (payload.get("generations") or [])[:num_generations]:
                print("=" * 80)
                print("Generation index:", generation.get("generation_index"))
                print("Evaluation:", generation.get("evaluation"))
                print(generation.get("response_text", "")[:2400])
        """
    ),
]


notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.x",
    },
}


def main() -> None:
    NOTEBOOK_PATH.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
