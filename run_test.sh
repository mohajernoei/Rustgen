#!/usr/bin/env bash
set -euo pipefail

size="${1:-1.3}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
MULTIPLE_ROOT="${MULTIPLE_ROOT:-${PROJECT_ROOT}/MultiPL-E-main}"
TUTORIAL_DIR="${MULTIPLE_ROOT}/tutorial"
MODEL_HF="deepseek-ai/deepseek-coder-${size}b-base"
LOCAL_MODEL_DIR="${TUTORIAL_DIR}/deepseek-${size}b-stackrust-merged"
MULTIPLE_REPO_URL="${MULTIPLE_REPO_URL:-https://github.com/nuprl/MultiPL-E.git}"
MULTIPLE_REPO_REF="${MULTIPLE_REPO_REF:-main}"

RUN_BASELINE="${RUN_BASELINE:-1}"
COMPLETION_LIMIT="${COMPLETION_LIMIT:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

ensure_cmd git
ensure_cmd python3
ensure_cmd podman

if [[ ! -d "${MULTIPLE_ROOT}/.git" ]]; then
  echo "[INFO] Cloning MultiPL-E into ${MULTIPLE_ROOT}"
  git clone "${MULTIPLE_REPO_URL}" "${MULTIPLE_ROOT}"
fi

echo "[INFO] Updating MultiPL-E in ${MULTIPLE_ROOT}"
git -C "${MULTIPLE_ROOT}" fetch --all --tags
git -C "${MULTIPLE_ROOT}" checkout "${MULTIPLE_REPO_REF}"
git -C "${MULTIPLE_ROOT}" pull --ff-only || true

mkdir -p "${TUTORIAL_DIR}"
cd "${TUTORIAL_DIR}"

if [[ ! -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
  echo "Error: merged model not found at ${LOCAL_MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${MULTIPLE_ROOT}/automodel.py" ]]; then
  echo "Error: ${MULTIPLE_ROOT}/automodel.py not found." >&2
  exit 1
fi

if [[ ! -f "${MULTIPLE_ROOT}/pass_k.py" ]]; then
  echo "Error: ${MULTIPLE_ROOT}/pass_k.py not found." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

BASE_DIR_NAME="humaneval-rs-deepseek_ai_deepseek_coder_${size}b_base-${TEMPERATURE}-reworded"
MERGED_DIR_NAME="humaneval-rs-deepseek_${size}b_stackrust_merged-${TEMPERATURE}-reworded"

rm -rf "${TUTORIAL_DIR:?}/${BASE_DIR_NAME}" "${TUTORIAL_DIR:?}/${MERGED_DIR_NAME}" || true

if [[ "${RUN_BASELINE}" == "1" ]]; then
  python3 "${MULTIPLE_ROOT}/automodel.py" \
    --name "${MODEL_HF}" \
    --root-dataset humaneval \
    --lang rs \
    --temperature "${TEMPERATURE}" \
    --batch-size "${BATCH_SIZE}" \
    --completion-limit "${COMPLETION_LIMIT}"
fi

python3 "${MULTIPLE_ROOT}/automodel.py" \
  --name "${LOCAL_MODEL_DIR}" \
  --name-override "deepseek_${size}b_stackrust_merged" \
  --root-dataset humaneval \
  --lang rs \
  --temperature "${TEMPERATURE}" \
  --batch-size "${BATCH_SIZE}" \
  --completion-limit "${COMPLETION_LIMIT}"

podman pull ghcr.io/nuprl/multipl-e-evaluation

if [[ "${RUN_BASELINE}" == "1" ]]; then
  podman run --rm --network none \
    -v "${TUTORIAL_DIR}:/tutorial:rw" \
    ghcr.io/nuprl/multipl-e-evaluation \
    --dir "/tutorial/${BASE_DIR_NAME}" \
    --output-dir "/tutorial/${BASE_DIR_NAME}"
fi

podman run --rm --network none \
  -v "${TUTORIAL_DIR}:/tutorial:rw" \
  ghcr.io/nuprl/multipl-e-evaluation \
  --dir "/tutorial/${MERGED_DIR_NAME}" \
  --output-dir "/tutorial/${MERGED_DIR_NAME}"

ls "${TUTORIAL_DIR}"/*/*.results.json.gz

if [[ "${RUN_BASELINE}" == "1" ]]; then
  python3 "${MULTIPLE_ROOT}/pass_k.py" -k 1 "${TUTORIAL_DIR}/${BASE_DIR_NAME}"

  if [[ "${COMPLETION_LIMIT}" -ge 10 ]]; then
    python3 "${MULTIPLE_ROOT}/pass_k.py" -k 10 "${TUTORIAL_DIR}/${BASE_DIR_NAME}"
  fi

  if [[ "${COMPLETION_LIMIT}" -ge 100 ]]; then
    python3 "${MULTIPLE_ROOT}/pass_k.py" -k 100 "${TUTORIAL_DIR}/${BASE_DIR_NAME}"
  fi
fi

python3 "${MULTIPLE_ROOT}/pass_k.py" -k 1 "${TUTORIAL_DIR}/${MERGED_DIR_NAME}"

if [[ "${COMPLETION_LIMIT}" -ge 10 ]]; then
  python3 "${MULTIPLE_ROOT}/pass_k.py" -k 10 "${TUTORIAL_DIR}/${MERGED_DIR_NAME}"
fi

if [[ "${COMPLETION_LIMIT}" -ge 100 ]]; then
  python3 "${MULTIPLE_ROOT}/pass_k.py" -k 100 "${TUTORIAL_DIR}/${MERGED_DIR_NAME}"
fi
