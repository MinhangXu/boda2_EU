#!/bin/bash
# Launch the current public-dataset HPO batch in detached screen sessions.
# Default batch:
#   - promoter deboer: utr_bassetvl, bassetvl, resnet1d
#   - utr3 hani: utr_bassetvl
#   - utr5 hani: utr_bassetvl
#
# Common env overrides:
#   GPU_POOL="0 1 2 3 4 5 6 7"   space-separated GPU ids to assign in order
#   NUM_AGENTS=1                  forwarded to each launcher
#   NUM_RUNS=16                   forwarded to each launcher
#   USE_SCREEN=1                  1 = detached screen sessions, 0 = foreground
#   DRY_RUN=1                     print commands only
#   SCREEN_PREFIX=public_hpo      session-name prefix
#   PROMOTER_NOTES=promoter_arch_screen_v1
#   UTR3_NOTES=utr3_baseline_hpo_v1
#   UTR5_NOTES=utr5_baseline_hpo_v1
#
# Example:
#   cd /home/minhang/synBio_AL/boda2_EU/src/learn
#   bash launch/run_public_datasets_hpo_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_POOL="${GPU_POOL:-0 1 2 3 4 5 6 7}"
NUM_AGENTS="${NUM_AGENTS:-1}"
NUM_RUNS="${NUM_RUNS:-16}"
USE_SCREEN="${USE_SCREEN:-1}"
DRY_RUN="${DRY_RUN:-0}"
SCREEN_PREFIX="${SCREEN_PREFIX:-public_hpo}"
PROMOTER_NOTES="${PROMOTER_NOTES:-promoter_arch_screen_v1}"
UTR3_NOTES="${UTR3_NOTES:-utr3_baseline_hpo_v1}"
UTR5_NOTES="${UTR5_NOTES:-utr5_baseline_hpo_v1}"

read -r -a GPU_ARRAY <<< "${GPU_POOL}"

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "GPU_POOL must contain at least one GPU id." >&2
  exit 1
fi

if [[ "${USE_SCREEN}" == "1" ]] && ! command -v screen >/dev/null 2>&1; then
  echo "screen is not installed or not on PATH. Set USE_SCREEN=0 or install screen." >&2
  exit 1
fi

JOBS=(
  "promoter_utr_bassetvl|${PROMOTER_NOTES}|MODE=utr_bassetvl bash launch/promoter_deboer_compare_architectures.sh"
  "promoter_bassetvl|${PROMOTER_NOTES}|MODE=bassetvl bash launch/promoter_deboer_compare_architectures.sh"
  "promoter_resnet1d|${PROMOTER_NOTES}|MODE=resnet1d bash launch/promoter_deboer_compare_architectures.sh"
  "utr3_hani|${UTR3_NOTES}|bash launch/utr3_hani_utr_bassetvl_sweep.sh"
  "utr5_hani|${UTR5_NOTES}|bash launch/utr5_hani_utr_bassetvl_sweep.sh"
)

launch_job() {
  local job_name="$1"
  local gpu_id="$2"
  local launch_notes="$3"
  local launcher_cmd="$4"
  local full_cmd="cd \"${LEARN_DIR}\" && GPU_LIST=${gpu_id} NUM_AGENTS=${NUM_AGENTS} NUM_RUNS=${NUM_RUNS} LAUNCH_NOTES=\"${launch_notes}\" ${launcher_cmd}"

  echo ""
  echo "---- [${job_name}] ----"
  echo "GPU:          ${gpu_id}"
  echo "NUM_AGENTS:   ${NUM_AGENTS}"
  echo "NUM_RUNS:     ${NUM_RUNS}"
  echo "LAUNCH_NOTES: ${launch_notes}"
  echo "Command:      ${full_cmd}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ "${USE_SCREEN}" == "1" ]]; then
    local session_name="${SCREEN_PREFIX}_${job_name}"
    screen -dmS "${session_name}" bash -lc "${full_cmd}"
    echo "Started detached screen session: ${session_name}"
  else
    bash -lc "${full_cmd}"
  fi
}

echo "=========================================================="
echo "PUBLIC DATASET HPO BATCH"
echo "GPU_POOL:     ${GPU_POOL}"
echo "NUM_AGENTS:   ${NUM_AGENTS}"
echo "NUM_RUNS:     ${NUM_RUNS}"
echo "USE_SCREEN:   ${USE_SCREEN}"
echo "DRY_RUN:      ${DRY_RUN}"
echo "SCREEN_PREFIX:${SCREEN_PREFIX}"
echo "=========================================================="

if [[ ${#GPU_ARRAY[@]} -lt ${#JOBS[@]} ]]; then
  echo "[warn] fewer GPUs than jobs; assignments will wrap around." >&2
fi

for i in "${!JOBS[@]}"; do
  IFS='|' read -r job_name launch_notes launcher_cmd <<< "${JOBS[$i]}"
  gpu_id="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
  launch_job "${job_name}" "${gpu_id}" "${launch_notes}" "${launcher_cmd}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo ""
  echo "Dry run only; no sweep launchers were started."
elif [[ "${USE_SCREEN}" == "1" ]]; then
  echo ""
  echo "All HPO launchers started in detached screen sessions."
  echo "Inspect with: screen -ls"
else
  echo ""
  echo "All HPO launchers finished."
fi
