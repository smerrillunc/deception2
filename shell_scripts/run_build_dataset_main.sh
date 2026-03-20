#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-deception}"
ENVIRONMENT="${ENVIRONMENT:-all}"
MODEL_NAME="${MODEL_NAME:-}"
BUILD_ALL_MODELS=1
STRICT="${STRICT:-0}"

TARGET_DECEPTIVE="${TARGET_DECEPTIVE:-2500}"
TARGET_TRUTHFUL="${TARGET_TRUTHFUL:-2500}"
TEXT_FIELD="${TEXT_FIELD:-action_reasoning}"
FALLBACK_TEXT_FIELD="${FALLBACK_TEXT_FIELD:-action_raw_text}"
LABEL_FILTER="${LABEL_FILTER:-all}"
LIMIT="${LIMIT:-0}"
INCLUDE_MESSAGES="${INCLUDE_MESSAGES:-1}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/DatasetMain}"

usage() {
  cat <<'EOF'
Usage:
  run_build_dataset_main.sh [options]

Options:
  --env ENV                  One of: bs, gridworld, advisor_audit, all (default: all)
  --model_name MODEL         Specific model to build
  --all_models              Build the default 3-model matrix (default behavior)
  --strict                  Exit non-zero if any env/model combo is missing or insufficient
  --help                    Show this message

Defaults:
  Dataset root: deception2/DatasetMain/{environment}/{model_tail}
  target_deceptive=2500
  target_truthful=2500
  text_field=action_reasoning
  fallback_text_field=action_raw_text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      BUILD_ALL_MODELS=0
      shift 2
      ;;
    --all_models)
      BUILD_ALL_MODELS=1
      shift
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$ENVIRONMENT" in
  bs|gridworld|advisor_audit|all)
    ;;
  *)
    echo "Unsupported env: $ENVIRONMENT" >&2
    exit 1
    ;;
esac

activate_python() {
  if [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]; then
    return
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    return
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    if [[ -f "/playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
      conda activate "$CONDA_ENV"
    elif command -v conda >/dev/null 2>&1; then
      # shellcheck disable=SC1091
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV"
    fi
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    return
  fi

  PYTHON_BIN="$(command -v python)"
}

count_labels_in_root() {
  local root="$1"
  "$PYTHON_BIN" - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    print("0 0 0 0")
    raise SystemExit(0)

total = dec = tru = unk = 0
for path in sorted(root.rglob("deception_samples.jsonl")):
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                rec = json.loads(line)
                val = rec.get("deceptive")
                if val is True:
                    dec += 1
                elif val is False:
                    tru += 1
                else:
                    unk += 1
    except Exception:
        continue
print(f"{total} {dec} {tru} {unk}")
PY
}

count_examples_file() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0 0 0 0")
    raise SystemExit(0)

total = dec = tru = unk = 0
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        rec = json.loads(line)
        val = rec.get("deceptive")
        if val is True:
            dec += 1
        elif val is False:
            tru += 1
        else:
            unk += 1
print(f"{total} {dec} {tru} {unk}")
PY
}

write_manifest() {
  local path="$1"
  local env_name="$2"
  local model_name="$3"
  local model_tail="$4"
  local input_root="$5"
  local examples_path="$6"
  local sentences_path="$7"
  local total="$8"
  local dec="$9"
  local tru="${10}"
  local unk="${11}"
  "$PYTHON_BIN" - "$path" "$env_name" "$model_name" "$model_tail" "$input_root" "$examples_path" "$sentences_path" "$total" "$dec" "$tru" "$unk" "$TEXT_FIELD" "$FALLBACK_TEXT_FIELD" "$TARGET_DECEPTIVE" "$TARGET_TRUTHFUL" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    env_name,
    model_name,
    model_tail,
    input_root,
    examples_path,
    sentences_path,
    total,
    dec,
    tru,
    unk,
    text_field,
    fallback_text_field,
    target_deceptive,
    target_truthful,
) = sys.argv[1:]

payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": env_name,
    "model_name": model_name,
    "model_tail": model_tail,
    "input_root": input_root,
    "examples_path": examples_path,
    "sentences_path": sentences_path,
    "text_field": text_field,
    "fallback_text_field": fallback_text_field,
    "target_deceptive": int(target_deceptive),
    "target_truthful": int(target_truthful),
    "example_counts": {
        "total": int(total),
        "deceptive": int(dec),
        "truthful": int(tru),
        "unknown": int(unk),
    },
}

Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

resolve_model() {
  case "$1" in
    qwen7b)
      printf '%s\n' "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
      ;;
    qwen14b)
      printf '%s\n' "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
      ;;
    llama8b)
      printf '%s\n' "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

activate_python

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Could not resolve a Python executable." >&2
  exit 1
fi

if [[ "$ENVIRONMENT" == "all" ]]; then
  ENVIRONMENTS=(bs gridworld advisor_audit)
else
  ENVIRONMENTS=("$ENVIRONMENT")
fi

if [[ "$BUILD_ALL_MODELS" == "1" ]]; then
  MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  )
else
  MODELS=("$(resolve_model "$MODEL_NAME")")
fi

mkdir -p "$DATASET_ROOT"

built_any=0
failed_any=0

for env_name in "${ENVIRONMENTS[@]}"; do
  case "$env_name" in
    bs)
      mining_base="$REPO_ROOT/BS/Results/DeceptionMining"
      ;;
    gridworld)
      mining_base="$REPO_ROOT/Gridworld/Results/DeceptionMining"
      ;;
    advisor_audit)
      mining_base="$REPO_ROOT/AdvisorAudit/Results/DeceptionMining"
      ;;
  esac

  for model_name in "${MODELS[@]}"; do
    model_tail="${model_name##*/}"
    model_tag_raw="${model_name//\//_}"
    raw_root="$mining_base/$model_tag_raw"
    base_root="$mining_base/$model_tail"

    read -r raw_total raw_dec raw_tru raw_unk <<<"$(count_labels_in_root "$raw_root")"
    read -r base_total base_dec base_tru base_unk <<<"$(count_labels_in_root "$base_root")"

    input_root="$raw_root"
    input_total=$raw_total
    input_dec=$raw_dec
    input_tru=$raw_tru
    input_unk=$raw_unk
    if (( base_total > raw_total )); then
      input_root="$base_root"
      input_total=$base_total
      input_dec=$base_dec
      input_tru=$base_tru
      input_unk=$base_unk
    fi

    echo ""
    echo "================================================================"
    echo "Building DatasetMain for env=$env_name model=$model_name"
    echo "================================================================"
    echo "Selected mining root: $input_root"
    echo "Mining counts: total=$input_total deceptive=$input_dec truthful=$input_tru unknown=$input_unk"

    if (( input_total == 0 )); then
      echo "Skipping: no mined data found."
      failed_any=1
      continue
    fi

    out_dir="$DATASET_ROOT/$env_name/$model_tail"
    mkdir -p "$out_dir"

    cmd=(
      "$PYTHON_BIN" "$REPO_ROOT/src/build_sentence_dataset.py"
      --input_root "$input_root"
      --out_dir "$out_dir"
      --text_field "$TEXT_FIELD"
      --fallback_text_field "$FALLBACK_TEXT_FIELD"
      --label_filter "$LABEL_FILTER"
      --target_deceptive "$TARGET_DECEPTIVE"
      --target_truthful "$TARGET_TRUTHFUL"
    )
    if [[ "$INCLUDE_MESSAGES" == "1" ]]; then
      cmd+=(--include_messages)
    fi
    if (( LIMIT > 0 )); then
      cmd+=(--limit "$LIMIT")
    fi

    "${cmd[@]}"

    examples_path="$out_dir/examples.jsonl"
    sentences_path="$out_dir/sentences.jsonl"
    read -r ds_total ds_dec ds_tru ds_unk <<<"$(count_examples_file "$examples_path")"
    echo "Built example counts: total=$ds_total deceptive=$ds_dec truthful=$ds_tru unknown=$ds_unk"

    write_manifest \
      "$out_dir/manifest.json" \
      "$env_name" \
      "$model_name" \
      "$model_tail" \
      "$input_root" \
      "$examples_path" \
      "$sentences_path" \
      "$ds_total" \
      "$ds_dec" \
      "$ds_tru" \
      "$ds_unk"

    built_any=1
    if (( ds_dec < TARGET_DECEPTIVE || ds_tru < TARGET_TRUTHFUL )); then
      echo "Warning: built dataset is below target counts."
      failed_any=1
    fi
  done
done

if [[ "$built_any" == "0" ]]; then
  echo "No datasets were built." >&2
  exit 1
fi

if [[ "$STRICT" == "1" && "$failed_any" == "1" ]]; then
  echo "Dataset build completed with missing or insufficient combos in strict mode." >&2
  exit 1
fi

echo ""
echo "DatasetMain build complete."
echo "Output root: $DATASET_ROOT"
