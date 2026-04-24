#!/bin/bash
# Sequentially kick a 1-agent / 1-run PILOT sweep for every CRE region that
# has a working launcher. This is the agent-friendly way to verify that the
# full train → test → runs.csv provenance chain is healthy before scheduling
# real HPO: each pilot creates exactly one W&B run and appends one row to
# `src/learn/run_registry/runs.csv`, so a clean result set is a signal that
# every region is ready for HPO.
#
# Respected env vars:
#   GPU_LIST=0               default GPU for pilot (single GPU is enough)
#   REGIONS="promoter_deboer_utr_bassetvl promoter_deboer_bassetvl promoter_deboer_resnet1d utr3_hani utr5_hani"
#                            space-separated allowlist; defaults to all below
#   LAUNCH_NOTES="..."       propagated into runs.csv for every pilot row
#   DRY_RUN=1                only print what would run, don't launch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Name → launcher invocation. Keep the keys stable; downstream notebooks may
# filter runs.csv on `comparison_group` which each launcher sets internally.
# This umbrella intentionally focuses on the current promoter / UTR priorities.
# Legacy enhancer / polysome launchers remain available as standalone scripts.
declare -A PILOT_COMMANDS=(
  #[enhancer_scratch_basic]="${SCRIPT_DIR}/lib1_enhancer_scratch_compare_loss_modes.sh"
  #[enhancer_scratch_weighted]="${SCRIPT_DIR}/lib1_enhancer_scratch_weighted_sweep.sh"
  # [promoter_deboer]="MODE=utr_bassetvl bash ${SCRIPT_DIR}/promoter_deboer_compare_architectures.sh"
  [promoter_deboer_utr_bassetvl]="MODE=utr_bassetvl bash ${SCRIPT_DIR}/promoter_deboer_compare_architectures.sh"
  [promoter_deboer_bassetvl]="MODE=bassetvl bash ${SCRIPT_DIR}/promoter_deboer_compare_architectures.sh"
  [promoter_deboer_resnet1d]="MODE=resnet1d bash ${SCRIPT_DIR}/promoter_deboer_compare_architectures.sh"
  [utr3_hani]="${SCRIPT_DIR}/utr3_hani_utr_bassetvl_sweep.sh"
  [utr5_hani]="${SCRIPT_DIR}/utr5_hani_utr_bassetvl_sweep.sh"
  #[utr5_polysome_hpo_egfp_1]="LIBRARY=egfp_1 ${SCRIPT_DIR}/utr5_polysome_utr_bassetvl_sweep.sh"
  #[utr5_polysome_hpo_egfp_2]="LIBRARY=egfp_2 ${SCRIPT_DIR}/utr5_polysome_utr_bassetvl_sweep.sh"
  #[utr5_polysome_hpo_mcherry_1]="LIBRARY=mcherry_1 ${SCRIPT_DIR}/utr5_polysome_utr_bassetvl_sweep.sh"
  #[utr5_polysome_hpo_mcherry_2]="LIBRARY=mcherry_2 ${SCRIPT_DIR}/utr5_polysome_utr_bassetvl_sweep.sh"
  #[utr5_polysome_egfp_1]="${SCRIPT_DIR}/utr5_polysome_fixed_all.sh egfp_1"
  #[utr5_polysome_egfp_2]="${SCRIPT_DIR}/utr5_polysome_fixed_all.sh egfp_2"
  #[utr5_polysome_mcherry_1]="${SCRIPT_DIR}/utr5_polysome_fixed_all.sh mcherry_1"
  #[utr5_polysome_mcherry_2]="${SCRIPT_DIR}/utr5_polysome_fixed_all.sh mcherry_2"
)

# Default region order is the current high-priority pilot surface:
# promoter architecture comparison plus UTR3 / UTR5 Hani RNA-activity.
# Introns is intentionally excluded until the data module lands.
DEFAULT_REGIONS=(
  #enhancer_scratch_basic
  #enhancer_scratch_weighted
  promoter_deboer_utr_bassetvl
  promoter_deboer_bassetvl
  promoter_deboer_resnet1d
  utr3_hani
  utr5_hani
  #utr5_polysome_hpo_egfp_1
)

if [[ -n "${REGIONS:-}" ]]; then
  read -r -a REGIONS_ARR <<< "${REGIONS}"
else
  REGIONS_ARR=("${DEFAULT_REGIONS[@]}")
fi

GPU_LIST="${GPU_LIST:-0}"
LAUNCH_NOTES="${LAUNCH_NOTES:-all_regions_pilot}"

export PILOT=1
export GPU_LIST
export LAUNCH_NOTES

echo "=========================================================="
echo "ALL-REGIONS PILOT"
echo "Regions:     ${REGIONS_ARR[*]}"
echo "GPU_LIST:    ${GPU_LIST}"
echo "Notes:       ${LAUNCH_NOTES}"
echo "DRY_RUN:     ${DRY_RUN:-0}"
echo "=========================================================="

for region in "${REGIONS_ARR[@]}"; do
  cmd="${PILOT_COMMANDS[${region}]:-}"
  if [[ -z "${cmd}" ]]; then
    echo "[skip] unknown region key: ${region}" >&2
    continue
  fi
  echo ""
  echo "---- [${region}] ----"
  echo "> ${cmd}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    continue
  fi
  # Each pilot exits 0 on success. We continue past failures so one broken
  # region does not prevent us from validating the others.
  if ! bash -c "${cmd}"; then
    echo "[warn] pilot for ${region} exited non-zero; continuing to next region" >&2
  fi
done

echo ""
echo "All requested pilots dispatched. Inspect run_registry/runs.csv for results."
