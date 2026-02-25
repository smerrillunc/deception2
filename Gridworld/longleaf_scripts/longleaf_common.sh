#!/usr/bin/env bash
# Shared helpers for Longleaf-compatible scripts.

longleaf_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_root="$(cd "$longleaf_script_dir/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$env_root/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-$PROJECT_ROOT/src}"
RESULTS_ROOT="${RESULTS_ROOT:-$env_root/Results}"
MINING_ROOT="${MINING_ROOT:-$RESULTS_ROOT/DeceptionMining}"
SENTENCE_ROOT="${SENTENCE_ROOT:-$RESULTS_ROOT/SentencePipeline/v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

activate_deception_env() {
  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    echo "Skipping conda activation (SKIP_CONDA_ACTIVATE=1)."
    return
  fi

  local env_name="${CONDA_ENV:-deception}"
  local -a candidates=()

  if [[ -n "${CONDA_SH:-}" ]]; then
    candidates+=("$CONDA_SH")
  fi
  candidates+=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda/etc/profile.d/conda.sh"
  )

  if [[ -n "${USER:-}" && ${#USER} -ge 2 ]]; then
    candidates+=(
      "/work/users/${USER:0:1}/${USER:1:1}/${USER}/miniconda3/etc/profile.d/conda.sh"
      "/work/users/${USER:0:1}/${USER:1:1}/${USER}/miniconda/etc/profile.d/conda.sh"
    )
  fi

  local conda_sh
  for conda_sh in "${candidates[@]}"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      if command -v conda >/dev/null 2>&1; then
        echo "Activating conda environment: $env_name"
        conda activate "$env_name"
        return
      fi
    fi
  done

  echo "Warning: could not locate conda.sh. Assuming environment is already active."
}

validate_label_filter() {
  local v="$1"
  if [[ "$v" != "all" && "$v" != "deceptive_only" && "$v" != "truthful_only" ]]; then
    echo "Invalid LABEL_FILTER=$v. Expected one of: all, deceptive_only, truthful_only"
    exit 1
  fi
}

set_target_counts_from_label_filter() {
  local filter="$1"
  local default_n="${2:-1000}"

  case "$filter" in
    truthful_only)
      TARGET_DECEPTIVE="${TARGET_DECEPTIVE:-0}"
      TARGET_TRUTHFUL="${TARGET_TRUTHFUL:-$default_n}"
      ;;
    deceptive_only)
      TARGET_DECEPTIVE="${TARGET_DECEPTIVE:-$default_n}"
      TARGET_TRUTHFUL="${TARGET_TRUTHFUL:-0}"
      ;;
    all)
      TARGET_DECEPTIVE="${TARGET_DECEPTIVE:-0}"
      TARGET_TRUTHFUL="${TARGET_TRUTHFUL:-0}"
      ;;
  esac
}

resolve_model_tag() {
  local model_name="$1"
  local root="$2"
  local raw="${model_name//\//_}"
  local base="${model_name##*/}"

  if [[ -d "$root/$base" ]]; then
    echo "$base"
  elif [[ -d "$root/$raw" ]]; then
    echo "$raw"
  else
    echo "$base"
  fi
}

select_model_name() {
  if [[ -n "${MODEL_NAME:-}" ]]; then
    return
  fi

  if [[ ! -t 0 ]]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    return
  fi

  echo "Select a model:"
  echo "  1) deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  echo "  2) deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  echo "  3) deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  echo ""

  read -r -p "Enter model number (1-3): " model_choice
  case "$model_choice" in
    1) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
    2) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" ;;
    3) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" ;;
    *) echo "Invalid model selection: $model_choice"; exit 1 ;;
  esac
}

select_single_gpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
    return
  fi

  if [[ "${SKIP_GPU_LIST:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
  fi

  if [[ ! -t 0 ]]; then
    export CUDA_VISIBLE_DEVICES="${DEFAULT_GPU_ID:-0}"
    echo "Using GPU: $CUDA_VISIBLE_DEVICES (non-interactive default)"
    return
  fi

  read -r -p "Enter the GPU ID you want to use (e.g., 0): " gpu_id
  export CUDA_VISIBLE_DEVICES="$gpu_id"
  echo "Using GPU: $CUDA_VISIBLE_DEVICES"
}
