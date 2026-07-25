#!/usr/bin/env bash
set -uo pipefail

SESSION_NAME="${SESSION_NAME:-raw_prompt_hhh_quick}"
REBUTTAL_ROOT="${REBUTTAL_ROOT:-/playpen-ssd/smerrill/deception2/rebuttal}"
RUN_DIR="${RUN_DIR:-${REBUTTAL_ROOT}/results/raw_prompt_hhh_full_v1}"
LOG_DIR="${RUN_DIR}/logs"
PYTHON_BIN="${PYTHON_BIN:-/playpen-ssd/smerrill/conda_envs/deception/bin/python}"
JUPYTER_BIN="${JUPYTER_BIN:-/playpen-ssd/smerrill/conda_envs/deception/bin/jupyter}"
NOTEBOOK_BUILDER="${NOTEBOOK_BUILDER:-${REBUTTAL_ROOT}/notebooks/build_raw_prompt_hhh_notebook.py}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-${REBUTTAL_ROOT}/notebooks/raw_prompt_hhh_analysis.ipynb}"

mkdir -p "${LOG_DIR}"

echo "[watcher] started $(date --iso-8601=seconds)"
echo "[watcher] waiting for tmux session ${SESSION_NAME}"

while tmux has-session -t "${SESSION_NAME}" 2>/dev/null; do
    json_count=$(find "${RUN_DIR}" -path "*/examples/*.json" -type f 2>/dev/null | wc -l || true)
    echo "[watcher] $(date --iso-8601=seconds) json_count=${json_count}"
    sleep 60
done

echo "[watcher] tmux session ended $(date --iso-8601=seconds)"

if ! grep -q "\[launcher\] finished" "${LOG_DIR}/launcher.log" 2>/dev/null; then
    echo "[watcher] launcher did not report successful completion; skipping notebook refresh" >&2
    exit 1
fi

if [[ ! -s "${RUN_DIR}/pair_summary.csv" ]]; then
    echo "[watcher] missing ${RUN_DIR}/pair_summary.csv; skipping notebook refresh" >&2
    exit 1
fi

echo "[watcher] rebuilding notebook"
if ! "${PYTHON_BIN}" "${NOTEBOOK_BUILDER}"; then
    echo "[watcher] notebook build failed" >&2
    exit 1
fi

echo "[watcher] executing notebook"
if ! "${JUPYTER_BIN}" nbconvert \
    --to notebook \
    --execute \
    --inplace "${NOTEBOOK_PATH}" \
    --ExecutePreprocessor.timeout=600; then
    echo "[watcher] notebook execution failed" >&2
    exit 1
fi

echo "[watcher] notebook refreshed $(date --iso-8601=seconds)"
