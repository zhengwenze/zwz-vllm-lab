#!/usr/bin/env bash
set -euo pipefail

# Fixed-workload scheduler A/B/C ablation: 3 policies x 5 independent repeats.
#
# Deterministic workload fingerprint (must match aggregate_repeats.py):
#   100 requests / 8.0 req/s / seed 20260827 / eager / bounded K=8
# Scheduler execution order is ROTATED each round to avoid thermal/cache/order bias.
#
# Usage (from the repository root inside WSL2):
#   bash scripts/run_scheduler_rtx4060_repeats.sh

MODEL="${MODEL:-/home/zwz2025/huggingface/Qwen3-0.6B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/online_scheduler/experiments/20260828-ablation}"
VENV_PY="${VENV_PY:-.venv/bin/python}"

if [ ! -x "$VENV_PY" ]; then
  echo "python not found at $VENV_PY; activate the project venv first" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; GPU benchmark requires the WSL2 CUDA environment" >&2
  exit 2
fi

run_one() {
  local policy="$1"
  local repeat_index="$2"
  "$VENV_PY" -m benchmarks.online_scheduler.cli \
    --model "$MODEL" \
    --policy "$policy" \
    --max-consecutive-decode-steps 8 \
    --workload mixed --arrival poisson \
    --num-requests 100 --request-rate 8.0 --seed 20260827 \
    --enforce-eager \
    --max-model-len 2048 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.75 \
    --repeat-index "$repeat_index" \
    --output-root "$OUTPUT_ROOT"
}

run_round() {
  local round="$1"
  case "$round" in
    0) run_one prefill_first 0; run_one decode_first 0; run_one bounded_decode_first 0 ;;
    1) run_one decode_first 1; run_one bounded_decode_first 1; run_one prefill_first 1 ;;
    2) run_one bounded_decode_first 2; run_one prefill_first 2; run_one decode_first 2 ;;
    3) run_one prefill_first 3; run_one decode_first 3; run_one bounded_decode_first 3 ;;
    4) run_one decode_first 4; run_one bounded_decode_first 4; run_one prefill_first 4 ;;
    *) echo "unknown round: $round" >&2; exit 2 ;;
  esac
}

for round in 0 1 2 3 4; do
  echo "=== round $round ==="
  run_round "$round"
done

echo "done: 15 runs written to $OUTPUT_ROOT"
