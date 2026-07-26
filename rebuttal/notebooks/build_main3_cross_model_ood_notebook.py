#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


THIS_FILE = Path(__file__).resolve()
NOTEBOOK_DIR = THIS_FILE.parent
REBUTTAL_ROOT = NOTEBOOK_DIR.parent
DEFAULT_RESULTS_DIR = REBUTTAL_ROOT / "results" / "OOD_Modeling_main3_cross_model_ood_xgb_pca_128"
DEFAULT_NOTEBOOK_PATH = NOTEBOOK_DIR / "main3_cross_model_ood_analysis.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Main3 cross-model OOD analysis notebook."
    )
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_NOTEBOOK_PATH))
    return parser.parse_args()


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(source)


def code_cell(source: str):
    return nbf.v4.new_code_cell(source)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        markdown_cell(
            "# Main3 Cross-Model OOD Analysis\n"
            "\n"
            "This notebook reads the rebuttal cross-model Main3 transfer outputs and summarizes:\n"
            "\n"
            "- Table-3-style family summaries with **models** as the OOD axis\n"
            "- selected AUROC and PR-AUC transfer matrices\n"
            "- calibration curves for the winning family/target panels\n"
            "- false-positive rates at fixed recall levels\n"
            "- top features for the selected models\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from IPython.display import display\n"
            "\n"
            "sns.set_theme(style='whitegrid')\n"
            f"RESULTS_DIR = Path({str(results_dir)!r})\n"
            "assert RESULTS_DIR.exists(), f'Missing results dir: {RESULTS_DIR}'\n"
            "\n"
            "CONFIG_PATH = RESULTS_DIR / 'config.csv'\n"
            "INVENTORY_PATH = RESULTS_DIR / 'bundle_inventory.csv'\n"
            "SPLIT_SUMMARY_PATH = RESULTS_DIR / 'split_summary.csv'\n"
            "FEATURE_SPACE_PATH = RESULTS_DIR / 'feature_space_catalog.csv'\n"
            "TRANSFER_PATH = RESULTS_DIR / 'all_transfer_metrics.csv'\n"
            "SUMMARY_PATH = RESULTS_DIR / 'transfer_summary.csv'\n"
            "TRAIN_MODEL_PATH = RESULTS_DIR / 'train_model_summary.csv'\n"
            "PANEL_PATH = RESULTS_DIR / 'best_feature_space_by_target_size_family.csv'\n"
            "BEST_MODEL_PATH = RESULTS_DIR / 'best_model_by_target_size_family.csv'\n"
            "CALIBRATION_PATH = RESULTS_DIR / 'all_calibration_curves.csv'\n"
            "FPR_PATH = RESULTS_DIR / 'all_fpr_at_recall.csv'\n"
            "TOP_FEATURES_PATH = RESULTS_DIR / 'top_features_for_best_models.csv'\n"
            "MANIFEST_PATH = RESULTS_DIR / 'selected_family_panel_tables' / 'panel_table_manifest.csv'\n"
            "\n"
            "config_df = pd.read_csv(CONFIG_PATH) if CONFIG_PATH.exists() else pd.DataFrame()\n"
            "inventory_df = pd.read_csv(INVENTORY_PATH) if INVENTORY_PATH.exists() else pd.DataFrame()\n"
            "split_summary_df = pd.read_csv(SPLIT_SUMMARY_PATH) if SPLIT_SUMMARY_PATH.exists() else pd.DataFrame()\n"
            "feature_space_df = pd.read_csv(FEATURE_SPACE_PATH) if FEATURE_SPACE_PATH.exists() else pd.DataFrame()\n"
            "metrics_df = pd.read_csv(TRANSFER_PATH) if TRANSFER_PATH.exists() else pd.DataFrame()\n"
            "summary_df = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else pd.DataFrame()\n"
            "train_model_df = pd.read_csv(TRAIN_MODEL_PATH) if TRAIN_MODEL_PATH.exists() else pd.DataFrame()\n"
            "panel_df = pd.read_csv(PANEL_PATH) if PANEL_PATH.exists() else pd.DataFrame()\n"
            "best_model_df = pd.read_csv(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() else pd.DataFrame()\n"
            "calibration_df = pd.read_csv(CALIBRATION_PATH) if CALIBRATION_PATH.exists() else pd.DataFrame()\n"
            "fpr_df = pd.read_csv(FPR_PATH) if FPR_PATH.exists() else pd.DataFrame()\n"
            "top_features_df = pd.read_csv(TOP_FEATURES_PATH) if TOP_FEATURES_PATH.exists() else pd.DataFrame()\n"
            "manifest_df = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()\n"
            "\n"
            "print('RESULTS_DIR:', RESULTS_DIR)\n"
            "print('transfer rows:', len(metrics_df))\n"
            "print('summary rows:', len(summary_df))\n"
            "print('calibration rows:', len(calibration_df))\n"
            "print('fpr rows:', len(fpr_df))\n"
        ),
        markdown_cell("## Run Config"),
        code_cell("config_df"),
        markdown_cell("## Inventory"),
        code_cell("inventory_df"),
        markdown_cell("## Split Summary"),
        code_cell("split_summary_df"),
        code_cell(
            "FEATURE_LABELS = {\n"
            "    'tfidf_baseline': 'TF-IDF baseline',\n"
            "    'attention_only': 'Attention only',\n"
            "    'activation_only': 'Activation only: PCA final',\n"
            "    'attention_plus_activation': 'Attention + PCA final',\n"
            "    'baseline_raw': 'Raw final activation',\n"
            "}\n"
            "TARGET_LABELS = {\n"
            "    'delta_pos_gt_0_3': 'Deceptive commitment',\n"
            "    'delta_neg_lt_neg_0_3': 'Honest commitment',\n"
            "}\n"
            "FAMILY_ORDER = ['tfidf_baseline', 'attention_only', 'activation_only', 'attention_plus_activation', 'baseline_raw']\n"
            "\n"
            "def family_label(value):\n"
            "    return FEATURE_LABELS.get(str(value), str(value))\n"
            "\n"
            "def target_label(value):\n"
            "    return TARGET_LABELS.get(str(value), str(value))\n"
        ),
        markdown_cell("## Table 3 Style Summary"),
        code_cell(
            "REQUESTED_FEATURE_SIZE = 128\n"
            "if panel_df.empty:\n"
            "    print('No panel summary found.')\n"
            "else:\n"
            "    table_df = panel_df.loc[panel_df['requested_feature_size'].eq(REQUESTED_FEATURE_SIZE)].copy()\n"
            "    table_df['family_label'] = table_df['feature_family_group'].map(family_label)\n"
            "    table_df['target_label'] = table_df['target_name'].map(target_label)\n"
            "    table_df['family_sort'] = table_df['feature_family_group'].map({name: idx for idx, name in enumerate(FAMILY_ORDER)})\n"
            "    auroc_table = (\n"
            "        table_df.sort_values(['family_sort', 'target_label'])\n"
            "        .pivot(index='family_label', columns='target_label', values='mean_ood_auroc')\n"
            "    )\n"
            "    prauc_table = (\n"
            "        table_df.sort_values(['family_sort', 'target_label'])\n"
            "        .pivot(index='family_label', columns='target_label', values='mean_ood_pr_auc')\n"
            "    )\n"
            "    brier_table = (\n"
            "        table_df.sort_values(['family_sort', 'target_label'])\n"
            "        .pivot(index='family_label', columns='target_label', values='mean_ood_brier')\n"
            "    )\n"
            "    print('AUROC')\n"
            "    display(auroc_table.style.format('{:.3f}'))\n"
            "    print('PR-AUC')\n"
            "    display(prauc_table.style.format('{:.3f}'))\n"
            "    print('Brier')\n"
            "    display(brier_table.style.format('{:.3f}'))\n"
        ),
        markdown_cell("## Winning Feature Spaces"),
        code_cell(
            "if panel_df.empty:\n"
            "    print('No panel summary found.')\n"
            "else:\n"
            "    cols = [\n"
            "        'target_name', 'requested_feature_size', 'feature_family_group',\n"
            "        'selected_feature_space_title', 'selected_feature_count',\n"
            "        'mean_ood_auroc', 'mean_ood_pr_auc', 'mean_ood_brier', 'alignment_detail'\n"
            "    ]\n"
            "    panel_df.loc[:, cols].sort_values(['requested_feature_size', 'target_name', 'feature_family_group'])\n"
        ),
        markdown_cell("## AUROC Transfer Matrices"),
        code_cell(
            "if manifest_df.empty:\n"
            "    print('No panel manifest found.')\n"
            "else:\n"
            "    requested_feature_size = 128\n"
            "    subset = manifest_df.loc[manifest_df['requested_feature_size'].eq(requested_feature_size)].copy()\n"
            "    for _, row in subset.iterrows():\n"
            "        matrix_path = Path(row['auroc_matrix_path'])\n"
            "        if not matrix_path.exists():\n"
            "            continue\n"
            "        matrix_df = pd.read_csv(matrix_path, index_col=0)\n"
            "        plt.figure(figsize=(5, 4))\n"
            "        sns.heatmap(matrix_df, annot=True, fmt='.3f', cmap='magma', vmin=0.0, vmax=1.0)\n"
            "        plt.title(f\"AUROC | {target_label(row['target_name'])} | {family_label(row['feature_family_group'])}\")\n"
            "        plt.xlabel('Eval model')\n"
            "        plt.ylabel('Train model')\n"
            "        plt.tight_layout()\n"
            "        plt.show()\n"
        ),
        markdown_cell("## PR-AUC Transfer Matrices"),
        code_cell(
            "if manifest_df.empty:\n"
            "    print('No panel manifest found.')\n"
            "else:\n"
            "    requested_feature_size = 128\n"
            "    subset = manifest_df.loc[manifest_df['requested_feature_size'].eq(requested_feature_size)].copy()\n"
            "    for _, row in subset.iterrows():\n"
            "        matrix_path = Path(row['pr_auc_matrix_path'])\n"
            "        if not matrix_path.exists():\n"
            "            continue\n"
            "        matrix_df = pd.read_csv(matrix_path, index_col=0)\n"
            "        plt.figure(figsize=(5, 4))\n"
            "        sns.heatmap(matrix_df, annot=True, fmt='.3f', cmap='viridis', vmin=0.0, vmax=1.0)\n"
            "        plt.title(f\"PR-AUC | {target_label(row['target_name'])} | {family_label(row['feature_family_group'])}\")\n"
            "        plt.xlabel('Eval model')\n"
            "        plt.ylabel('Train model')\n"
            "        plt.tight_layout()\n"
            "        plt.show()\n"
        ),
        markdown_cell("## Calibration Curves"),
        code_cell(
            "if calibration_df.empty or panel_df.empty:\n"
            "    print('Calibration or panel data missing.')\n"
            "else:\n"
            "    requested_feature_size = 128\n"
            "    winners = panel_df.loc[panel_df['requested_feature_size'].eq(requested_feature_size)].copy()\n"
            "    for _, win in winners.iterrows():\n"
            "        subset = calibration_df.loc[\n"
            "            calibration_df['target_name'].eq(win['target_name'])\n"
            "            & calibration_df['feature_space'].eq(win['selected_feature_space'])\n"
            "            & calibration_df['feature_size_label'].eq(win['source_feature_size_label'])\n"
            "        ].copy()\n"
            "        if subset.empty:\n"
            "            continue\n"
            "        plt.figure(figsize=(5, 4))\n"
            "        plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)\n"
            "        plotted = False\n"
            "        for eval_role, role_df in subset.groupby('eval_role'):\n"
            "            mean_curve = role_df.groupby('bin_idx', as_index=False).agg(mean_pred=('mean_pred', 'mean'), frac_pos=('frac_pos', 'mean'))\n"
            "            if mean_curve.empty:\n"
            "                continue\n"
            "            plotted = True\n"
            "            plt.plot(mean_curve['mean_pred'], mean_curve['frac_pos'], marker='o', label=eval_role)\n"
            "        if plotted:\n"
            "            plt.title(f\"Calibration | {target_label(win['target_name'])} | {family_label(win['feature_family_group'])}\")\n"
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
            "if fpr_df.empty or panel_df.empty:\n"
            "    print('FPR or panel data missing.')\n"
            "else:\n"
            "    requested_feature_size = 128\n"
            "    winners = panel_df.loc[panel_df['requested_feature_size'].eq(requested_feature_size)].copy()\n"
            "    rows = []\n"
            "    for _, win in winners.iterrows():\n"
            "        subset = fpr_df.loc[\n"
            "            fpr_df['target_name'].eq(win['target_name'])\n"
            "            & fpr_df['feature_space'].eq(win['selected_feature_space'])\n"
            "            & fpr_df['feature_size_label'].eq(win['source_feature_size_label'])\n"
            "            & fpr_df['eval_role'].eq('ood')\n"
            "        ].copy()\n"
            "        if subset.empty:\n"
            "            continue\n"
            "        summary = subset.groupby('recall_target', as_index=False).agg(mean_fpr=('fpr', 'mean'))\n"
            "        summary['target_label'] = target_label(win['target_name'])\n"
            "        summary['family_label'] = family_label(win['feature_family_group'])\n"
            "        rows.append(summary)\n"
            "    if rows:\n"
            "        summary_df = pd.concat(rows, ignore_index=True)\n"
            "        display(summary_df.pivot(index=['family_label', 'target_label'], columns='recall_target', values='mean_fpr').style.format('{:.3f}'))\n"
            "    else:\n"
            "        print('No selected-panel FPR rows found.')\n"
        ),
        markdown_cell("## Top Features"),
        code_cell(
            "if top_features_df.empty:\n"
            "    print('No top-feature exports found.')\n"
            "else:\n"
            "    top_features_df.head(50)\n"
        ),
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)


if __name__ == "__main__":
    main()
