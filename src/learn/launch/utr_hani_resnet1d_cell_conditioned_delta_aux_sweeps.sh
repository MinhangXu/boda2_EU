#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_RUNS="${TOTAL_RUNS:-64}"

echo "Launching cell-conditioned delta-aux ResNet1D Hani UTR sweeps sequentially."
echo "Each region will request TOTAL_RUNS=${TOTAL_RUNS} completed W&B agent runs."

TOTAL_RUNS="${TOTAL_RUNS}" "${SCRIPT_DIR}/utr3_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh"
TOTAL_RUNS="${TOTAL_RUNS}" "${SCRIPT_DIR}/utr5_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh"
