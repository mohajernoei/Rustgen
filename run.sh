#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run.sh 1.3
#
# This script runs the full cycle:
#   1) clone/update MultiPL-E
#   2) download the dataset into PROJECT_ROOT/data if it is missing
#   3) prepare train/eval SFT files
#   4) fine-tune the LoRA adapter
#   5) merge the adapter into a full model
#   6) run MultiPL-E evaluation
#
# Optional overrides:
#   PROJECT_ROOT=/absolute/path/to/project
#   DATA_JSONL=/absolute/path/to/data/stack_rust_clean.jsonl
#   CODEGEN_DIR=/absolute/path/to/output
#   MULTIPLE_ROOT=/absolute/path/to/MultiPL-E-main
#   MULTIPLE_REPO_URL=https://github.com/nuprl/MultiPL-E.git
#   MULTIPLE_REPO_REF=main
#   CUDA_VISIBLE_DEVICES=0
#   MAX_SAMPLES=250000
#   MAX_EVAL_SAMPLES=2048
#   COMPLETION_LIMIT=10
#   RUN_BASELINE=1
#   TEMPERATURE=0.8
#   HF_TOKEN=...

size="${1:-1.3}"

if [[ ! "$size" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: size must be a number like 1.3 or 6.7" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
CODEGEN_DIR="${CODEGEN_DIR:-${PROJECT_ROOT}/artifacts}"
MULTIPLE_ROOT="${MULTIPLE_ROOT:-${PROJECT_ROOT}/MultiPL-E-main}"
MULTIPLE_REPO_URL="${MULTIPLE_REPO_URL:-https://github.com/nuprl/MultiPL-E.git}"
MULTIPLE_REPO_REF="${MULTIPLE_REPO_REF:-main}"
DATA_DIR="${PROJECT_ROOT}/data"
DATA_JSONL="${DATA_JSONL:-${DATA_DIR}/stack_rust_clean.jsonl}"

PREP_DIR="${PROJECT_ROOT}/prepared_sft"
ADAPTER_DIR="${CODEGEN_DIR}/deepseek_${size}b_stackrust_lora"
MERGED_DIR="${MULTIPLE_ROOT}/tutorial/deepseek-${size}b-stackrust-merged"

MAX_SAMPLES="${MAX_SAMPLES:-250000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export RUN_BASELINE="${RUN_BASELINE:-1}"
export COMPLETION_LIMIT="${COMPLETION_LIMIT:-10}"
export TEMPERATURE="${TEMPERATURE:-0.8}"

mkdir -p "${PREP_DIR}" "${CODEGEN_DIR}" "${DATA_DIR}"

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

ensure_cmd git
ensure_cmd python
ensure_cmd find
ensure_cmd bash

if [[ ! -d "${MULTIPLE_ROOT}/.git" ]]; then
  echo "[INFO] Cloning MultiPL-E into ${MULTIPLE_ROOT}"
  git clone "${MULTIPLE_REPO_URL}" "${MULTIPLE_ROOT}"
fi

echo "[INFO] Updating MultiPL-E in ${MULTIPLE_ROOT}"
git -C "${MULTIPLE_ROOT}" fetch --all --tags
git -C "${MULTIPLE_ROOT}" checkout "${MULTIPLE_REPO_REF}"
git -C "${MULTIPLE_ROOT}" pull --ff-only || true

mkdir -p "${MULTIPLE_ROOT}/tutorial"

if [[ ! -f "${DATA_JSONL}" ]]; then
  echo "[INFO] Dataset not found at ${DATA_JSONL}"
  echo "[INFO] Downloading dataset into ${DATA_DIR}"
  bash "${PROJECT_ROOT}/datacollector.sh"
else
  echo "[INFO] Reusing existing dataset at ${DATA_JSONL}"
fi

if [[ ! -s "${DATA_JSONL}" ]]; then
  echo "Error: dataset file is missing or empty after download check: ${DATA_JSONL}" >&2
  exit 1
fi

echo "[INFO] size=${size}"
echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] data_jsonl=${DATA_JSONL}"
echo "[INFO] adapter_dir=${ADAPTER_DIR}"
echo "[INFO] merged_dir=${MERGED_DIR}"
echo "[INFO] multipl_e_root=${MULTIPLE_ROOT}"
echo "[INFO] max_samples=${MAX_SAMPLES}"
echo "[INFO] max_eval_samples=${MAX_EVAL_SAMPLES}"
echo "[INFO] run_baseline=${RUN_BASELINE}"
echo "[INFO] completion_limit=${COMPLETION_LIMIT}"
echo "[INFO] temperature=${TEMPERATURE}"

rm -rf "${PREP_DIR}" "${ADAPTER_DIR}" "${MERGED_DIR}"
find "${MULTIPLE_ROOT}/tutorial" -mindepth 1 -maxdepth 1 -type d -name "humaneval-rs-*" -exec rm -rf {} + || true
mkdir -p "${PREP_DIR}"

python "${PROJECT_ROOT}/prepare_deepseek_messages.py" \
  --input_jsonl "${DATA_JSONL}" \
  --out_dir "${PREP_DIR}" \
  --eval_fraction 0.01 \
  --min_chars 120 \
  --min_prompt_chars 48 \
  --min_completion_chars 48 \
  --max_samples "${MAX_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --max_chars 24000 \
  --max_examples_per_file 3

python "${PROJECT_ROOT}/sft.py" "${size}" \
  --project_root "${PROJECT_ROOT}" \
  --train_json "${PREP_DIR}/train_messages.jsonl" \
  --eval_json "${PREP_DIR}/eval_messages.jsonl" \
  --output_dir "${ADAPTER_DIR}" \
  --max_length 2048 \
  --prompt_max_length 1024 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --logging_steps 20 \
  --save_total_limit 2 \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --min_target_tokens 24 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05

python "${PROJECT_ROOT}/merge.py" \
  --base "deepseek-ai/deepseek-coder-${size}b-base" \
  --adapter "${ADAPTER_DIR}" \
  --out "${MERGED_DIR}" \
  --dtype bf16

bash "${PROJECT_ROOT}/run_test.sh" "${size}"

echo "[INFO] Full pipeline completed successfully."
