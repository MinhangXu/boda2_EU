#!/bin/bash
# Launch focused second-stage public-dataset HPO sweeps.
#
# Default jobs:
#   - promoter deboer: focused UTR_BassetVL
#   - utr3 hani: light focused UTR_BassetVL confirmation sweep
#   - utr5 hani: focused UTR_BassetVL
#
# Common env overrides:
#   GPU_POOL="0 1 2"       space-separated GPU ids to assign in order
#   NUM_AGENTS=1           forwarded to each job
#   NUM_RUNS=12            forwarded to each job
#   USE_SCREEN=1           1 = detached screen sessions, 0 = foreground
#   DRY_RUN=1              print commands only
#   RUN_PROMOTER=1         include promoter focused sweep
#   RUN_UTR3=1             include UTR3 light focused sweep
#   RUN_UTR5=1             include UTR5 focused sweep
#
# Example:
#   cd /home/minhang/synBio_AL/boda2_EU/src/learn
#   bash launch/run_public_datasets_focused_hpo_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_POOL="${GPU_POOL:-0 1 2}"
NUM_AGENTS="${NUM_AGENTS:-1}"
NUM_RUNS="${NUM_RUNS:-12}"
USE_SCREEN="${USE_SCREEN:-1}"
DRY_RUN="${DRY_RUN:-0}"
SCREEN_PREFIX="${SCREEN_PREFIX:-public_focused_hpo}"

RUN_PROMOTER="${RUN_PROMOTER:-1}"
RUN_UTR3="${RUN_UTR3:-1}"
RUN_UTR5="${RUN_UTR5:-1}"

PROMOTER_NOTES="${PROMOTER_NOTES:-promoter_utr_bassetvl_focused_stage2_v1}"
UTR3_NOTES="${UTR3_NOTES:-utr3_focused_confirmation_stage2_v1}"
UTR5_NOTES="${UTR5_NOTES:-utr5_hani_focused_stage2_v1}"

read -r -a GPU_ARRAY <<< "${GPU_POOL}"

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "GPU_POOL must contain at least one GPU id." >&2
  exit 1
fi

if [[ "${USE_SCREEN}" == "1" ]] && ! command -v screen >/dev/null 2>&1; then
  echo "screen is not installed or not on PATH. Set USE_SCREEN=0 or install screen." >&2
  exit 1
fi

run_child_job() {
  local focused_job="$1"
  local config_path task_family target_family comparison_group notes

  case "${focused_job}" in
    promoter)
      config_path="configs/promoter/deboer_core/utr_bassetvl/promoter__deboer_core__utr_bassetvl__focused_bayes.yml"
      task_family="promoter"
      target_family="deboer_core"
      comparison_group="promoter__deboer_core__focused_utr_bassetvl_stage2"
      notes="${LAUNCH_NOTES:-${PROMOTER_NOTES}}"
      ;;
    utr3)
      config_path="configs/utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__focused_bayes_2026_04.yml"
      task_family="utr3"
      target_family="hani_rna_activity"
      comparison_group="utr3__hani_rna_activity__focused_utr_bassetvl_stage2"
      notes="${LAUNCH_NOTES:-${UTR3_NOTES}}"
      ;;
    utr5)
      config_path="configs/utr5/hani_rna_activity/utr_bassetvl/utr5__hani_rna_activity__utr_bassetvl__focused_bayes.yml"
      task_family="utr5"
      target_family="hani_rna_activity"
      comparison_group="utr5__hani_rna_activity__focused_utr_bassetvl_stage2"
      notes="${LAUNCH_NOTES:-${UTR5_NOTES}}"
      ;;
    *)
      echo "Unknown FOCUSED_JOB=${focused_job}. Use promoter, utr3, or utr5." >&2
      exit 1
      ;;
  esac

  source "${SCRIPT_DIR}/_wandb_helpers.sh"
  read -r -a CHILD_GPU_ARRAY <<< "${GPU_LIST:-0}"
  LAUNCH_NOTES="${notes}" launch_wandb_agents \
    "${config_path}" \
    "${task_family}" \
    "${target_family}" \
    "${comparison_group}" \
    "launch/run_public_datasets_focused_hpo_batch.sh" \
    "${NUM_AGENTS}" \
    "${NUM_RUNS}" \
    "${CHILD_GPU_ARRAY[@]}"
}

if [[ -n "${FOCUSED_JOB:-}" ]]; then
  run_child_job "${FOCUSED_JOB}"
  exit 0
fi

JOBS=()
if [[ "${RUN_PROMOTER}" == "1" ]]; then
  JOBS+=("promoter|${PROMOTER_NOTES}")
fi
if [[ "${RUN_UTR3}" == "1" ]]; then
  JOBS+=("utr3|${UTR3_NOTES}")
fi
if [[ "${RUN_UTR5}" == "1" ]]; then
  JOBS+=("utr5|${UTR5_NOTES}")
fi

if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "No focused HPO jobs enabled. Set RUN_PROMOTER, RUN_UTR3, or RUN_UTR5 to 1." >&2
  exit 1
fi

echo "=========================================================="
echo "PUBLIC DATASET FOCUSED HPO BATCH"
echo "GPU_POOL:     ${GPU_POOL}"
echo "NUM_AGENTS:   ${NUM_AGENTS}"
echo "NUM_RUNS:     ${NUM_RUNS}"
echo "USE_SCREEN:   ${USE_SCREEN}"
echo "DRY_RUN:      ${DRY_RUN}"
echo "SCREEN_PREFIX:${SCREEN_PREFIX}"
echo "JOBS:         ${JOBS[*]}"
echo "=========================================================="

if [[ ${#GPU_ARRAY[@]} -lt ${#JOBS[@]} ]]; then
  echo "[warn] fewer GPUs than jobs; assignments will wrap around." >&2
fi

for i in "${!JOBS[@]}"; do
  IFS='|' read -r job_name launch_notes <<< "${JOBS[$i]}"
  gpu_id="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
  full_cmd="cd \"${LEARN_DIR}\" && GPU_LIST=${gpu_id} NUM_AGENTS=${NUM_AGENTS} NUM_RUNS=${NUM_RUNS} LAUNCH_NOTES=\"${launch_notes}\" FOCUSED_JOB=${job_name} USE_SCREEN=0 bash launch/run_public_datasets_focused_hpo_batch.sh"

  echo ""
  echo "---- [${job_name}] ----"
  echo "GPU:          ${gpu_id}"
  echo "NUM_AGENTS:   ${NUM_AGENTS}"
  echo "NUM_RUNS:     ${NUM_RUNS}"
  echo "LAUNCH_NOTES: ${launch_notes}"
  echo "Command:      ${full_cmd}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  if [[ "${USE_SCREEN}" == "1" ]]; then
    session_name="${SCREEN_PREFIX}_${job_name}"
    screen -dmS "${session_name}" bash -lc "${full_cmd}"
    echo "Started detached screen session: ${session_name}"
  else
    bash -lc "${full_cmd}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo ""
  echo "Dry run only; no focused sweep launchers were started."
elif [[ "${USE_SCREEN}" == "1" ]]; then
  echo ""
  echo "All focused HPO launchers started in detached screen sessions."
  echo "Inspect with: screen -ls"
else
  echo ""
  echo "All focused HPO launchers finished."
fi
