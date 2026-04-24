#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

NUM_RUNS="${NUM_RUNS:-1}"
read -r -a GPU_ARRAY <<< "${GPU_LIST:-0 1 2 3}"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"

CONFIGS=(
  "configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_1.yml"
  "configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_2.yml"
  "configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_1.yml"
  "configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_2.yml"
)

COMPARISON_GROUPS=(
  "utr5__polysome__within_library__egfp_1"
  "utr5__polysome__within_library__egfp_2"
  "utr5__polysome__within_library__mcherry_1"
  "utr5__polysome__within_library__mcherry_2"
)

for i in "${!CONFIGS[@]}"; do
  gpu_id="${GPU_ARRAY[i % ${#GPU_ARRAY[@]}]}"
  launch_wandb_agents \
    "${CONFIGS[$i]}" \
    "utr5" \
    "polysome" \
    "${COMPARISON_GROUPS[$i]}" \
    "launch/utr5_polysome_fixed_all.sh" \
    "1" \
    "${NUM_RUNS}" \
    "${gpu_id}"
done
