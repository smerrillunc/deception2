#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


THIS_FILE = Path(__file__).resolve()
NOTEBOOK_DIR = THIS_FILE.parent
REBUTTAL_ROOT = NOTEBOOK_DIR.parent
DEFAULT_RESULTS_DIR = REBUTTAL_ROOT / "results" / "OOD_Modeling_main3_env_ood_metrics_qwen14b_xgb_pca_128"
DEFAULT_NOTEBOOK_PATH = NOTEBOOK_DIR / "main3_env_ood_analysis.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Main3 environment-OOD analysis notebook."
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
            "# Main3 Environment-OOD Analysis\n"
            "\n"
            "This notebook reads the rebuttal Main3 environment-transfer outputs and summarizes:\n"
            "\n"
            "- Table-3-style family summaries with **environments** as the OOD axis\n"
            "- selected AUROC and PR-AUC transfer matrices\n"
            "- target-environment breakdowns for the winning panels\n"
            "- calibration curves for the winning family/target panels\n"
            "- false-positive rates at fixed recall levels\n"
            "- top features for the selected models\n"
        ),
        code_cell(
            "from pathlib import Path\n"
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
            "TRAIN_ENV_PATH = RESULTS_DIR / 'train_env_model_summary.csv'\n"
            "TARGET_ENV_BREAKDOWN_PATH = RESULTS_DIR / 'target_env_breakdown_summary.csv'\n"
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
            "train_env_df = pd.read_csv(TRAIN_ENV_PATH) if TRAIN_ENV_PATH.exists() else pd.DataFrame()\n"
            "target_env_breakdown_df = pd.read_csv(TARGET_ENV_BREAKDOWN_PATH) if TARGET_ENV_BREAKDOWN_PATH.exists() else pd.DataFrame()\n"
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
            "    'activation_only': 'Activation only',\n"
            "    'attention_plus_activation': 'Attention + activation',\n"
            "    'baseline': 'Raw final activation',\n"
            "}\n"
            "TARGET_LABELS = {\n"
            "    'delta_pos_gt_0_3': 'Deceptive commitment',\n"
            "    'delta_neg_lt_neg_0_3': 'Honest commitment',\n"
            "}\n"
            "FAMILY_ORDER = ['tfidf_baseline', 'attention_only', 'activation_only', 'attention_plus_activation', 'baseline']\n"
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
            "    scenario_values = panel_df['scenario_name'].dropna().astype(str).unique().tolist()\n"
            "    family_sort = {name: idx for idx, name in enumerate(FAMILY_ORDER)}\n"
            "    for scenario_name in scenario_values:\n"
            "        scenario_table = panel_df.loc[\n"
            "            panel_df['scenario_name'].eq(scenario_name)\n"
            "            & panel_df['requested_feature_size'].eq(REQUESTED_FEATURE_SIZE)\n"
            "        ].copy()\n"
            "        if scenario_table.empty:\n"
            "            continue\n"
            "        scenario_title = scenario_table['scenario_title'].iloc[0]\n"
            "        scenario_table['family_label'] = scenario_table['feature_family_group'].map(family_label)\n"
            "        scenario_table['target_label'] = scenario_table['target_name'].map(target_label)\n"
            "        scenario_table['family_sort'] = scenario_table['feature_family_group'].map(family_sort)\n"
            "        auroc_table = scenario_table.sort_values(['family_sort', 'target_label']).pivot(index='family_label', columns='target_label', values='mean_ood_auroc')\n"
            "        prauc_table = scenario_table.sort_values(['family_sort', 'target_label']).pivot(index='family_label', columns='target_label', values='mean_ood_pr_auc')\n"
            "        brier_table = scenario_table.sort_values(['family_sort', 'target_label']).pivot(index='family_label', columns='target_label', values='mean_ood_brier')\n"
            "        print(scenario_title)\n"
            "        print('AUROC')\n"
            "        display(auroc_table.style.format('{:.3f}'))\n"
            "        print('PR-AUC')\n"
            "        display(prauc_table.style.format('{:.3f}'))\n"
            "        print('Brier')\n"
            "        display(brier_table.style.format('{:.3f}'))\n"
        ),
        markdown_cell("## Winning Feature Spaces"),
        code_cell(
            "if panel_df.empty:\n"
            "    print('No panel summary found.')\n"
            "else:\n"
            "    cols = [\n"
            "        'scenario_name', 'target_name', 'requested_feature_size', 'feature_family_group',\n"
            "        'selected_feature_space_title', 'selected_feature_count',\n"
            "        'mean_ood_auroc', 'mean_ood_pr_auc', 'mean_ood_brier', 'alignment_detail'\n"
            "    ]\n"
            "    display(panel_df.loc[:, cols].sort_values(['scenario_name', 'requested_feature_size', 'target_name', 'feature_family_group']))\n"
        ),
        markdown_cell("## Target Environment Breakdown"),
        code_cell(
            "if target_env_breakdown_df.empty or panel_df.empty:\n"
            "    print('Target-environment breakdown or panel data missing.')\n"
            "else:\n"
            "    requested_feature_size = 128\n"
            "    winners = panel_df.loc[panel_df['requested_feature_size'].eq(requested_feature_size)].copy()\n"
            "    for _, win in winners.iterrows():\n"
            "        subset = target_env_breakdown_df.loc[\n"
            "            target_env_breakdown_df['scenario_name'].eq(win['scenario_name'])\n"
            "            & target_env_breakdown_df['target_name'].eq(win['target_name'])\n"
            "            & target_env_breakdown_df['feature_space'].eq(win['selected_feature_space'])\n"
            "            & target_env_breakdown_df['feature_size_label'].eq(win['source_feature_size_label'])\n"
            "            & target_env_breakdown_df['eval_role'].eq('ood')\n"
            "        ].copy()\n"
            "        if subset.empty:\n"
            "            continue\n"
            "        print(f\"{win['scenario_title']} | {target_label(win['target_name'])} | {family_label(win['feature_family_group'])}\")\n"
            "        display(subset[['test_env', 'mean_auroc', 'mean_pr_auc', 'mean_brier', 'mean_balanced_accuracy']].sort_values('test_env').reset_index(drop=True).style.format({'mean_auroc': '{:.3f}', 'mean_pr_auc': '{:.3f}', 'mean_brier': '{:.3f}', 'mean_balanced_accuracy': '{:.3f}'}))\n"
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
            "        plt.figure(figsize=(6, 4.5))\n"
            "        sns.heatmap(matrix_df, annot=True, fmt='.3f', cmap='magma', vmin=0.0, vmax=1.0)\n"
            "        plt.title(f\"{row['scenario_name']} | {target_label(row['target_name'])} | {family_label(row['feature_family_group'])}\")\n"
            "        plt.xlabel('Evaluation env')\n"
            "        plt.ylabel('Training source(s)')\n"
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
            "        plt.figure(figsize=(6, 4.5))\n"
            "        sns.heatmap(matrix_df, annot=True, fmt='.3f', cmap='viridis', vmin=0.0, vmax=1.0)\n"
            "        plt.title(f\"{row['scenario_name']} | {target_label(row['target_name'])} | {family_label(row['feature_family_group'])}\")\n"
            "        plt.xlabel('Evaluation env')\n"
            "        plt.ylabel('Training source(s)')\n"
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
            "            calibration_df['scenario_name'].eq(win['scenario_name'])\n"
            "            & calibration_df['target_name'].eq(win['target_name'])\n"
            "            & calibration_df['feature_space'].eq(win['selected_feature_space'])\n"
            "            & calibration_df['feature_size_label'].eq(win['source_feature_size_label'])\n"
            "        ].copy()\n"
            "        if subset.empty:\n"
            "            continue\n"
            "        plt.figure(figsize=(5.5, 4.5))\n"
            "        plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)\n"
            "        plotted = False\n"
            "        for (eval_role, test_env), curve_df in subset.groupby(['eval_role', 'test_env']):\n"
            "            mean_curve = curve_df.groupby('bin_idx', as_index=False).agg(mean_pred=('mean_pred', 'mean'), frac_pos=('frac_pos', 'mean'))\n"
            "            if mean_curve.empty:\n"
            "                continue\n"
            "            plotted = True\n"
            "            plt.plot(mean_curve['mean_pred'], mean_curve['frac_pos'], marker='o', label=f'{eval_role}:{test_env}')\n"
            "        if plotted:\n"
            "            plt.title(f\"{win['scenario_name']} | {target_label(win['target_name'])} | {family_label(win['feature_family_group'])}\")\n"
            "            plt.xlabel('Mean predicted probability')\n"
            "            plt.ylabel('Observed positive rate')\n"
            "            plt.legend(loc='best', fontsize=8)\n"
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
            "            fpr_df['scenario_name'].eq(win['scenario_name'])\n"
            "            & fpr_df['target_name'].eq(win['target_name'])\n"
            "            & fpr_df['feature_space'].eq(win['selected_feature_space'])\n"
            "            & fpr_df['feature_size_label'].eq(win['source_feature_size_label'])\n"
            "            & fpr_df['eval_role'].eq('ood')\n"
            "        ].copy()\n"
            "        if subset.empty:\n"
            "            continue\n"
            "        summary = subset.groupby(['test_env', 'recall_target'], as_index=False).agg(mean_fpr=('fpr', 'mean'))\n"
            "        summary['scenario_name'] = win['scenario_name']\n"
            "        summary['target_label'] = target_label(win['target_name'])\n"
            "        summary['family_label'] = family_label(win['feature_family_group'])\n"
            "        rows.append(summary)\n"
            "    if rows:\n"
            "        display(pd.concat(rows, ignore_index=True).sort_values(['scenario_name', 'target_label', 'family_label', 'test_env', 'recall_target']).style.format({'recall_target': '{:.2f}', 'mean_fpr': '{:.3f}'}))\n"
            "    else:\n"
            "        print('No FPR summaries available.')\n"
        ),
        markdown_cell("## Top Features"),
        code_cell(
            "if top_features_df.empty:\n"
            "    print('No top-feature summary found.')\n"
            "else:\n"
            "    cols = [col for col in [\n"
            "        'importance_rank', 'scenario_name', 'target_name', 'requested_feature_size_label',\n"
            "        'feature_family_group', 'train_env', 'feature', 'feature_weight_kind', 'feature_weight'\n"
            "    ] if col in top_features_df.columns]\n"
            "    display(top_features_df.loc[:, cols].head(200))\n"
        ),
    ]

    nbf.write(nb, output_path)
    print(f"Wrote notebook to {output_path}")


if __name__ == "__main__":
    main()
