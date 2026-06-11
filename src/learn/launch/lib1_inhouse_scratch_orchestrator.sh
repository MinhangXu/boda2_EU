#!/bin/bash
set -euo pipefail

# Orchestrate standardized Lib1 in-house scratch HPO sweeps by composing the
# curated per-part launchers. The launchers still own dataset prep, sweep
# creation, W&B agent launch, and registry logging.
#
# Examples:
#   DRY_RUN=1 GPU_LIST="0 1 2 3" RUNS_PER_SWEEP=128 \
#     bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
#
#   GPU_LIST="0 1 2 3 4 5 6 7" MODE=parallel_by_part RUNS_PER_SWEEP=256 \
#     PARTS="promoter intron utr3 utr5 enhancer" \
#     bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
#
#   PILOT=1 GPU_LIST="0" PARTS="enhancer" MODE=sequential \
#     bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

MODE="${MODE:-sequential}"               # sequential | parallel_by_part
PARTS="${PARTS:-promoter intron utr3 utr5 enhancer}"
RUNS_PER_SWEEP="${RUNS_PER_SWEEP:-128}"  # total desired runs per sweep
PREPARE_ONCE="${PREPARE_ONCE:-1}"
DRY_RUN="${DRY_RUN:-0}"
PILOT="${PILOT:-0}"

if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a GPU_ARRAY <<< "${GPU_LIST}"
else
  mapfile -t GPU_ARRAY < <(detect_idle_gpus)
fi

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs selected. Set GPU_LIST or free GPUs for auto-detection." >&2
  exit 1
fi

if [[ "${MODE}" != "sequential" && "${MODE}" != "parallel_by_part" ]]; then
  echo "ERROR: MODE must be sequential or parallel_by_part; got ${MODE}" >&2
  exit 1
fi

scripts_for_part() {
  local part="$1"
  case "${part}" in
    promoter)
      printf '%s\n' \
        "lib1_promoter_scratch_resnet1d_sweep.sh" \
        "lib1_promoter_scratch_promoter_bassetvl_sweep.sh"
      ;;
    intron|introns)
      printf '%s\n' "lib1_intron_scratch_resnet1d_sweep.sh"
      ;;
    utr3|threeprime)
      printf '%s\n' \
        "lib1_threeprime_scratch_resnet1d_sweep.sh" \
        "lib1_threeprime_scratch_utr_bassetvl_sweep.sh"
      ;;
    utr5|fiveprime)
      printf '%s\n' \
        "lib1_fiveprime_scratch_resnet1d_sweep.sh" \
        "lib1_fiveprime_scratch_utr_bassetvl_sweep.sh"
      ;;
    enhancer)
      printf '%s\n' \
        "lib1_enhancer_no_flank_hq8_scratch_resnet1d_sweep.sh" \
        "lib1_enhancer_no_flank_hq8_scratch_bassetvl_sweep.sh"
      ;;
    *)
      echo "ERROR: unknown part '${part}'" >&2
      return 1
      ;;
  esac
}

gpu_join() {
  local values=("$@")
  printf '%s' "${values[*]}"
}

split_gpus_for_index() {
  local index="$1"
  local total_groups="$2"
  local n_gpus="${#GPU_ARRAY[@]}"
  local start=$(( index * n_gpus / total_groups ))
  local end=$(( (index + 1) * n_gpus / total_groups ))
  local selected=()
  local i

  for ((i=start; i<end; i++)); do
    selected+=("${GPU_ARRAY[i]}")
  done

  if [[ ${#selected[@]} -eq 0 ]]; then
    echo "ERROR: not enough GPUs (${n_gpus}) to split across ${total_groups} concurrent sweeps." >&2
    return 1
  fi

  gpu_join "${selected[@]}"
}

runs_per_agent() {
  local total_runs="$1"
  local agents="$2"
  if (( total_runs % agents != 0 )); then
    echo "ERROR: RUNS_PER_SWEEP=${total_runs} is not divisible by NUM_AGENTS=${agents}." >&2
    echo "       Pick a divisible value so each sweep runs exactly the requested number of trials." >&2
    return 1
  fi
  echo $(( total_runs / agents ))
}

prepare_part_once() {
  local part="$1"
  local first_script="$2"
  if [[ "${PREPARE_ONCE}" != "1" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: would prepare/check derived data for ${part} using ${first_script}; skipping dataset writer."
    return 0
  fi
  echo "Preparing/checking derived data for ${part} using ${first_script}"
  (
    cd "${LEARN_DIR}"
    DRY_RUN=1 PREPARE_DATASET=1 GPU_LIST="${GPU_ARRAY[0]}" NUM_AGENTS=1 NUM_RUNS=1 \
      bash "launch/${first_script}"
  )
}

run_one_sweep() {
  local script="$1"
  local gpu_list="$2"
  local gpu_count
  local per_agent
  read -r -a local_gpus <<< "${gpu_list}"
  if [[ "${PILOT}" == "1" ]]; then
    local_gpus=("${local_gpus[0]}")
    gpu_list="${local_gpus[0]}"
    gpu_count=1
    per_agent=1
  else
    gpu_count="${#local_gpus[@]}"
    per_agent="$(runs_per_agent "${RUNS_PER_SWEEP}" "${gpu_count}")"
  fi

  echo "Launching ${script}: total_runs=${RUNS_PER_SWEEP}, NUM_AGENTS=${gpu_count}, NUM_RUNS=${per_agent}, GPU_LIST=${gpu_list}"
  (
    cd "${LEARN_DIR}"
    PREPARE_DATASET="${PREPARE_DATASET:-0}" \
    GPU_LIST="${gpu_list}" \
    NUM_AGENTS="${gpu_count}" \
    NUM_RUNS="${per_agent}" \
    DRY_RUN="${DRY_RUN}" \
    PILOT="${PILOT}" \
    bash "launch/${script}"
  )
}

run_part_sequential() {
  local part="$1"
  mapfile -t scripts < <(scripts_for_part "${part}")
  if [[ ${#scripts[@]} -eq 0 ]]; then
    return 0
  fi
  prepare_part_once "${part}" "${scripts[0]}"
  local script
  for script in "${scripts[@]}"; do
    PREPARE_DATASET=0 run_one_sweep "${script}" "$(gpu_join "${GPU_ARRAY[@]}")"
  done
}

run_part_parallel() {
  local part="$1"
  mapfile -t scripts < <(scripts_for_part "${part}")
  if [[ ${#scripts[@]} -eq 0 ]]; then
    return 0
  fi
  prepare_part_once "${part}" "${scripts[0]}"

  local n_scripts="${#scripts[@]}"
  local idx script gpus
  for ((idx=0; idx<n_scripts; idx++)); do
    script="${scripts[idx]}"
    gpus="$(split_gpus_for_index "${idx}" "${n_scripts}")"
    PREPARE_DATASET=0 run_one_sweep "${script}" "${gpus}" &
    sleep 2
  done
  wait
}

echo "Lib1 in-house scratch orchestrator"
echo "MODE=${MODE}"
echo "PARTS=${PARTS}"
echo "RUNS_PER_SWEEP=${RUNS_PER_SWEEP}"
echo "GPU_LIST=$(gpu_join "${GPU_ARRAY[@]}")"
echo "DRY_RUN=${DRY_RUN}"
echo "PILOT=${PILOT}"
echo "CREATE_SWEEP_ONLY=${CREATE_SWEEP_ONLY:-0}"

read -r -a PART_ARRAY <<< "${PARTS}"
for part in "${PART_ARRAY[@]}"; do
  echo
  echo "=== Part: ${part} ==="
  if [[ "${MODE}" == "sequential" ]]; then
    run_part_sequential "${part}"
  else
    run_part_parallel "${part}"
  fi
done

echo
echo "Orchestration complete."
