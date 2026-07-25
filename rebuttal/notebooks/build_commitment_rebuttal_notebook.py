#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


THIS_FILE = Path(__file__).resolve()
NOTEBOOK_DIR = THIS_FILE.parent
REBUTTAL_ROOT = NOTEBOOK_DIR.parent
DEFAULT_RESULTS_ROOT = REBUTTAL_ROOT / "results"
DEFAULT_RUN_NAME = "commitment_threshold_sweep_v1"
DEFAULT_NOTEBOOK_PATH = NOTEBOOK_DIR / "commitment_rebuttal_analysis.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the commitment rebuttal analysis notebook."
    )
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_NOTEBOOK_PATH))
    return parser.parse_args()


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(source)


def code_cell(source: str):
    return nbf.v4.new_code_cell(source)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        markdown_cell(
            "# Commitment Rebuttal Analysis\n"
            "\n"
            "This notebook reads the rebuttal training outputs and summarizes:\n"
            "\n"
            "- AUROC and PR-AUC by model, threshold, feature space, and scenario\n"
            "- target-environment breakdowns\n"
            "- calibration curves\n"
            "- false-positive rates at fixed recall levels\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "sns.set_theme(style='whitegrid')\n"
            "\n"
            f"RESULTS_ROOT = Path({str(Path(args.results_root).expanduser().resolve())!r})\n"
            f"RUN_NAME = {args.run_name!r}\n"
            "RUN_ROOT = RESULTS_ROOT / RUN_NAME\n"
            "TRAINING_ROOT = RUN_ROOT / 'training'\n"
            "\n"
            "CONFIG_PATH = TRAINING_ROOT / 'commitment_rebuttal_config.json'\n"
            "INVENTORY_PATH = TRAINING_ROOT / 'commitment_rebuttal_inventory.csv'\n"
            "METRICS_PATH = TRAINING_ROOT / 'commitment_rebuttal_metrics.csv'\n"
            "CALIBRATION_PATH = TRAINING_ROOT / 'commitment_rebuttal_calibration.csv'\n"
            "FPR_PATH = TRAINING_ROOT / 'commitment_rebuttal_fpr_at_recall.csv'\n"
            "ERRORS_PATH = TRAINING_ROOT / 'commitment_rebuttal_errors.csv'\n"
            "PREDICTIONS_PATH = TRAINING_ROOT / 'commitment_rebuttal_predictions.parquet'\n"
            "\n"
            "assert TRAINING_ROOT.exists(), f'Missing training directory: {TRAINING_ROOT}'\n"
            "config = json.loads(CONFIG_PATH.read_text(encoding='utf-8')) if CONFIG_PATH.exists() else {}\n"
            "inventory_df = pd.read_csv(INVENTORY_PATH) if INVENTORY_PATH.exists() else pd.DataFrame()\n"
            "metrics_df = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()\n"
            "calibration_df = pd.read_csv(CALIBRATION_PATH) if CALIBRATION_PATH.exists() else pd.DataFrame()\n"
            "fpr_df = pd.read_csv(FPR_PATH) if FPR_PATH.exists() else pd.DataFrame()\n"
            "errors_df = pd.read_csv(ERRORS_PATH) if ERRORS_PATH.exists() else pd.DataFrame()\n"
            "predictions_df = pd.read_parquet(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else pd.DataFrame()\n"
            "\n"
            "print('RUN_ROOT:', RUN_ROOT)\n"
            "print('TRAINING_ROOT:', TRAINING_ROOT)\n"
            "print('metrics rows:', len(metrics_df))\n"
            "print('calibration rows:', len(calibration_df))\n"
            "print('fpr rows:', len(fpr_df))\n"
            "print('prediction rows:', len(predictions_df))\n"
        ),
        code_cell(
            "config\n"
        ),
        code_cell(
            "inventory_df\n"
        ),
        markdown_cell("## Best Metric Rows"),
        code_cell(
            "if metrics_df.empty:\n"
            "    print('No metric rows found.')\n"
            "else:\n"
            "    best_rows = (\n"
            "        metrics_df.sort_values(['scenario', 'model_bundle_name', 'label_kind', 'tau', 'target_env', 'pr_auc', 'auroc'], ascending=[True, True, True, True, True, False, False])\n"
            "        .groupby(['scenario', 'model_bundle_name', 'label_kind', 'tau', 'target_env'], as_index=False)\n"
            "        .head(1)\n"
            "        .reset_index(drop=True)\n"
            "    )\n"
            "    best_rows[['scenario', 'model_bundle_name', 'label_kind', 'tau', 'target_env', 'feature_space', 'eval_kind', 'auroc', 'pr_auc', 'brier', 'row_count', 'example_count']]\n"
        ),
        markdown_cell("## Aggregate Summary"),
        code_cell(
            "if metrics_df.empty:\n"
            "    print('No metric rows found.')\n"
            "else:\n"
            "    summary_df = (\n"
            "        metrics_df.groupby(['scenario', 'model_bundle_name', 'label_kind', 'tau', 'feature_space', 'eval_kind'], as_index=False)\n"
            "        .agg(\n"
            "            mean_auroc=('auroc', 'mean'),\n"
            "            mean_pr_auc=('pr_auc', 'mean'),\n"
            "            mean_brier=('brier', 'mean'),\n"
            "            eval_rows=('row_count', 'sum'),\n"
            "            target_envs=('target_env', 'nunique'),\n"
            "        )\n"
            "        .sort_values(['scenario', 'model_bundle_name', 'tau', 'eval_kind', 'mean_pr_auc'], ascending=[True, True, True, True, False])\n"
            "        .reset_index(drop=True)\n"
            "    )\n"
            "    summary_df\n"
        ),
        markdown_cell("## PR-AUC by Target Environment"),
        code_cell(
            "if metrics_df.empty:\n"
            "    print('No metric rows found.')\n"
            "else:\n"
            "    plot_df = metrics_df.loc[metrics_df['eval_kind'].astype(str).eq('ood_test')].copy()\n"
            "    if plot_df.empty:\n"
            "        print('No OOD rows found.')\n"
            "    else:\n"
            "        grouped = (\n"
            "            plot_df.groupby(['scenario', 'model_bundle_name', 'tau', 'feature_space', 'target_env'], as_index=False)\n"
            "            .agg(pr_auc=('pr_auc', 'mean'))\n"
            "        )\n"
            "        for (scenario, model_bundle_name, tau), subset in grouped.groupby(['scenario', 'model_bundle_name', 'tau']):\n"
            "            pivot = subset.pivot(index='target_env', columns='feature_space', values='pr_auc')\n"
            "            plt.figure(figsize=(1.2 * max(4, len(pivot.columns)), 0.8 * max(3, len(pivot.index))))\n"
            "            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', vmin=0.0, vmax=1.0)\n"
            "            plt.title(f'PR-AUC | {scenario} | {model_bundle_name} | tau={tau}')\n"
            "            plt.xlabel('Feature Space')\n"
            "            plt.ylabel('Target Environment')\n"
            "            plt.tight_layout()\n"
            "            plt.show()\n"
        ),
        markdown_cell("## AUROC by Target Environment"),
        code_cell(
            "if metrics_df.empty:\n"
            "    print('No metric rows found.')\n"
            "else:\n"
            "    plot_df = metrics_df.loc[metrics_df['eval_kind'].astype(str).eq('ood_test')].copy()\n"
            "    if plot_df.empty:\n"
            "        print('No OOD rows found.')\n"
            "    else:\n"
            "        grouped = (\n"
            "            plot_df.groupby(['scenario', 'model_bundle_name', 'tau', 'feature_space', 'target_env'], as_index=False)\n"
            "            .agg(auroc=('auroc', 'mean'))\n"
            "        )\n"
            "        for (scenario, model_bundle_name, tau), subset in grouped.groupby(['scenario', 'model_bundle_name', 'tau']):\n"
            "            pivot = subset.pivot(index='target_env', columns='feature_space', values='auroc')\n"
            "            plt.figure(figsize=(1.2 * max(4, len(pivot.columns)), 0.8 * max(3, len(pivot.index))))\n"
            "            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='magma', vmin=0.0, vmax=1.0)\n"
            "            plt.title(f'AUROC | {scenario} | {model_bundle_name} | tau={tau}')\n"
            "            plt.xlabel('Feature Space')\n"
            "            plt.ylabel('Target Environment')\n"
            "            plt.tight_layout()\n"
            "            plt.show()\n"
        ),
        markdown_cell("## Calibration Curves"),
        code_cell(
            "if calibration_df.empty:\n"
            "    print('No calibration rows found.')\n"
            "else:\n"
            "    curve_df = calibration_df.copy()\n"
            "    curve_df = curve_df.sort_values(['scenario', 'model_bundle_name', 'tau', 'feature_space', 'bin_idx'])\n"
            "    for (scenario, model_bundle_name, tau), subset in curve_df.groupby(['scenario', 'model_bundle_name', 'tau']):\n"
            "        plt.figure(figsize=(6, 5))\n"
            "        plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)\n"
            "        plotted = False\n"
            "        for feature_space, fs_df in subset.groupby('feature_space'):\n"
            "            mean_curve = fs_df.groupby('bin_idx', as_index=False).agg(mean_pred=('mean_pred', 'mean'), frac_pos=('frac_pos', 'mean'))\n"
            "            if mean_curve.empty:\n"
            "                continue\n"
            "            plotted = True\n"
            "            plt.plot(mean_curve['mean_pred'], mean_curve['frac_pos'], marker='o', label=feature_space)\n"
            "        if plotted:\n"
            "            plt.title(f'Calibration | {scenario} | {model_bundle_name} | tau={tau}')\n"
            "            plt.xlabel('Mean predicted probability')\n"
            "            plt.ylabel('Observed positive rate')\n"
            "            plt.legend(loc='best')\n"
            "            plt.tight_layout()\n"
            "            plt.show()\n"
            "        else:\n"
            "            plt.close()\n"
        ),
        markdown_cell("## False-Positive Rate at Fixed Recall"),
        code_cell(
            "if fpr_df.empty:\n"
            "    print('No FPR-at-recall rows found.')\n"
            "else:\n"
            "    summary = (\n"
            "        fpr_df.groupby(['scenario', 'model_bundle_name', 'tau', 'feature_space', 'recall_target'], as_index=False)\n"
            "        .agg(mean_fpr=('fpr', 'mean'))\n"
            "        .sort_values(['scenario', 'model_bundle_name', 'tau', 'recall_target', 'mean_fpr'])\n"
            "    )\n"
            "    summary\n"
        ),
        code_cell(
            "if fpr_df.empty:\n"
            "    print('No FPR-at-recall rows found.')\n"
            "else:\n"
            "    plot_df = fpr_df.groupby(['scenario', 'model_bundle_name', 'tau', 'feature_space', 'recall_target'], as_index=False).agg(mean_fpr=('fpr', 'mean'))\n"
            "    for (scenario, model_bundle_name, tau), subset in plot_df.groupby(['scenario', 'model_bundle_name', 'tau']):\n"
            "        plt.figure(figsize=(8, 4.5))\n"
            "        sns.barplot(data=subset, x='recall_target', y='mean_fpr', hue='feature_space')\n"
            "        plt.title(f'FPR at fixed recall | {scenario} | {model_bundle_name} | tau={tau}')\n"
            "        plt.xlabel('Recall target')\n"
            "        plt.ylabel('Mean FPR')\n"
            "        plt.tight_layout()\n"
            "        plt.show()\n"
        ),
        markdown_cell("## Errors"),
        code_cell(
            "errors_df\n"
        ),
    ]

    nbf.write(nb, output_path)
    print(f"Wrote notebook to: {output_path}")


if __name__ == "__main__":
    main()
