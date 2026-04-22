#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"

mkdir -p "${PROJECT_ROOT}/data"
OUT="${PROJECT_ROOT}/data/stack_rust_clean.jsonl"
: > "$OUT"

DATASET="ammarnasr%2Fthe-stack-rust-clean"
CONFIG="default"
SPLIT="train"
PSIZE=100

# Optional: set HF_TOKEN in env for authenticated requests (recommended)
AUTH_HEADER=()
[ -n "${HF_TOKEN:-}" ] && AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")

OFFSET=0

# Resume if a prior run exists (align to page boundary)
if [ -s "$OUT" ]; then
  LINES=$(wc -l < "$OUT")
  OFFSET=$(( (LINES/PSIZE) * PSIZE ))
fi

retry_max=8
sleep_base=1

while : ; do
  echo "fetch offset=$OFFSET"
  retries=0
  while : ; do
    curl -sS -L --fail-with-body -H 'Accept: application/json' \
      "${AUTH_HEADER[@]}" \
      -D headers.tmp -o page.json \
      "https://datasets-server.huggingface.co/rows?dataset=${DATASET}&config=${CONFIG}&split=${SPLIT}&offset=${OFFSET}&length=${PSIZE}" || true

    STATUS=$(awk 'toupper($1) ~ /^HTTP\// {code=$2} END{print code+0}' headers.tmp)

    if [ "$STATUS" -eq 200 ] && jq -e . >/dev/null 2>&1 < page.json; then
      break
    fi

    if [ "$STATUS" -eq 429 ]; then
      RA=$(awk -F': ' 'tolower($1)=="retry-after"{print $2}' headers.tmp | tr -d '\r')
      if [[ "$RA" =~ ^[0-9]+$ ]]; then
        SLEEP=$RA
      elif [ -n "$RA" ]; then
        SLEEP=30
      else
        retries=$((retries+1))
        SLEEP=$(( sleep_base * (2 ** retries) + RANDOM % 3 ))
      fi
    else
      retries=$((retries+1))
      SLEEP=$(( sleep_base * (2 ** retries) + RANDOM % 3 ))
    fi

    if [ "$retries" -ge "$retry_max" ]; then
      echo "give up at offset=$OFFSET (status=$STATUS)" >&2
      exit 1
    fi
    sleep "$SLEEP"
  done

  NROWS=$(jq '(.rows | length) // 0' < page.json)
  if [ "$NROWS" -eq 0 ]; then
    echo "done at offset=$OFFSET"
    break
  fi

  jq -c '.rows[].row | {text: .content, hexsha, size}' < page.json >> "$OUT"
  OFFSET=$((OFFSET + NROWS))
  sleep 1.5
done

rm -f headers.tmp page.json
echo "downloaded lines: $(wc -l < "$OUT")"
echo "saved to: $OUT"
