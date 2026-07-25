#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUTTAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_raw_prompt_hhh_rebuttal.py"

PYTHON_BIN="${PYTHON_BIN:-/playpen-ssd/smerrill/conda_envs/deception/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${REBUTTAL_ROOT}/results}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-raw_prompt_hhh_full_v1}"
DATASET_ROOT="${DATASET_ROOT:-/playpen-ssd/smerrill/deception2/DatasetMainCompressed}"
HF_HOME="${HF_HOME:-/playpen-ssd/smerrill/huggingface}"

ENVS="${ENVS:-advisor_audit,bs,car_sales,gridworld,interview}"
MODELS="${MODELS:-DeepSeek-R1-Distill-Llama-8B,DeepSeek-R1-Distill-Qwen-14B,DeepSeek-R1-Distill-Qwen-7B,gpt-oss-20b}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

EXAMPLES_PER_PAIR="${EXAMPLES_PER_PAIR:-100}"
SAMPLES_PER_EXAMPLE="${SAMPLES_PER_EXAMPLE:-100}"
SAMPLES_PER_CALL="${SAMPLES_PER_CALL:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10000}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
SEED="${SEED:-17}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-/tmp/vllm_raw_prompt_hhh_full}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"

ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
PREFER_LOCAL_MODEL_SNAPSHOT="${PREFER_LOCAL_MODEL_SNAPSHOT:-0}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
SKIP_MODEL_LOAD_FAILURES="${SKIP_MODEL_LOAD_FAILURES:-0}"
OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_SELECTION="${OVERWRITE_SELECTION:-0}"
USE_REASONING_PARSER="${USE_REASONING_PARSER:-auto}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ARRAY[@]}}"

RUN_DIR="${RESULTS_ROOT}/${EXPERIMENT_NAME}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/launcher.log") 2>&1

echo "[launcher] started $(date --iso-8601=seconds)"
echo "[launcher] experiment=${EXPERIMENT_NAME}"
echo "[launcher] run_dir=${RUN_DIR}"
echo "[launcher] gpus=${GPUS} num_shards=${NUM_SHARDS}"
echo "[launcher] examples_per_pair=${EXAMPLES_PER_PAIR} samples_per_example=${SAMPLES_PER_EXAMPLE} samples_per_call=${SAMPLES_PER_CALL}"
echo "[launcher] max_new_tokens=${MAX_NEW_TOKENS} max_model_len=${MAX_MODEL_LEN}"

if [[ "${#GPU_ARRAY[@]}" -lt "${NUM_SHARDS}" ]]; then
    echo "[launcher] ERROR: NUM_SHARDS=${NUM_SHARDS} but only ${#GPU_ARRAY[@]} GPU ids were provided in GPUS=${GPUS}" >&2
    exit 2
fi

COMMON_ARGS=(
    --results-root "${RESULTS_ROOT}"
    --experiment-name "${EXPERIMENT_NAME}"
    --dataset-root "${DATASET_ROOT}"
    --hf-home "${HF_HOME}"
    --envs "${ENVS}"
    --models "${MODELS}"
    --examples-per-pair "${EXAMPLES_PER_PAIR}"
    --samples-per-example "${SAMPLES_PER_EXAMPLE}"
    --samples-per-call "${SAMPLES_PER_CALL}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --repetition-penalty "${REPETITION_PENALTY}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --seed "${SEED}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --vllm-config-root "${VLLM_CONFIG_ROOT}"
    --vllm-dtype "${VLLM_DTYPE}"
)

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    COMMON_ARGS+=(--enforce-eager)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
    COMMON_ARGS+=(--disable-custom-all-reduce)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    COMMON_ARGS+=(--trust-remote-code)
fi
if [[ "${PREFER_LOCAL_MODEL_SNAPSHOT}" == "1" ]]; then
    COMMON_ARGS+=(--prefer-local-model-snapshot)
fi
if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
    COMMON_ARGS+=(--allow-download)
fi
if [[ "${SKIP_MODEL_LOAD_FAILURES}" == "1" ]]; then
    COMMON_ARGS+=(--skip-model-load-failures)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite)
fi
if [[ "${OVERWRITE_SELECTION}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite-selection)
fi
if [[ "${USE_REASONING_PARSER}" == "1" ]]; then
    COMMON_ARGS+=(--use-reasoning-parser)
elif [[ "${USE_REASONING_PARSER}" == "0" ]]; then
    COMMON_ARGS+=(--no-use-reasoning-parser)
fi

echo "[launcher] selecting fixed examples before shard launch"
if ! "${PYTHON_BIN}" "${RUNNER}" "${COMMON_ARGS[@]}" --num-shards "${NUM_SHARDS}" --shard-index 0 --select-only > "${LOG_DIR}/select_only.log" 2>&1; then
    echo "[launcher] ERROR: selection failed; see ${LOG_DIR}/select_only.log" >&2
    exit 1
fi
echo "[launcher] selection complete"

PIDS=()
SHARD_LOGS=()
for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index += 1)); do
    gpu="${GPU_ARRAY[${shard_index}]}"
    shard_log="${LOG_DIR}/shard_${shard_index}_of_${NUM_SHARDS}_gpu_${gpu}.log"
    SHARD_LOGS+=("${shard_log}")
    echo "[launcher] launching shard=${shard_index}/${NUM_SHARDS} gpu=${gpu} log=${shard_log}"
    (
        "${PYTHON_BIN}" "${RUNNER}" "${COMMON_ARGS[@]}" \
            --num-shards "${NUM_SHARDS}" \
            --shard-index "${shard_index}" \
            --gpu "${gpu}"
    ) > "${shard_log}" 2>&1 &
    PIDS+=("$!")
    sleep 5
done

status=0
for ((idx = 0; idx < ${#PIDS[@]}; idx += 1)); do
    pid="${PIDS[${idx}]}"
    shard_log="${SHARD_LOGS[${idx}]}"
    if wait "${pid}"; then
        echo "[launcher] shard ${idx} finished successfully"
    else
        rc="$?"
        echo "[launcher] ERROR: shard ${idx} failed with exit code ${rc}; see ${shard_log}" >&2
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    echo "[launcher] one or more shards failed; not compiling final summaries" >&2
    exit "${status}"
fi

echo "[launcher] all shards finished; compiling summaries"
if ! "${PYTHON_BIN}" "${RUNNER}" "${COMMON_ARGS[@]}" --compile-only > "${LOG_DIR}/compile_only.log" 2>&1; then
    echo "[launcher] ERROR: summary compilation failed; see ${LOG_DIR}/compile_only.log" >&2
    exit 1
fi

echo "[launcher] finished $(date --iso-8601=seconds)"
echo "[launcher] summaries: ${RUN_DIR}/example_summary.csv and ${RUN_DIR}/pair_summary.csv"
