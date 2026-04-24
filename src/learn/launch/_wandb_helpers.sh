#!/bin/bash

# set -e: exit on error, -u: exit on undefined variable, -o pipefail: exit on pipe failure
set -euo pipefail

# get the script directory and the learn directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# get the registry directory and the sweep log csv file
REGISTRY_DIR="${LEARN_DIR}/run_registry"
SWEEP_LOG_CSV="${REGISTRY_DIR}/sweep_launches.csv"
DEFAULT_WANDB_SWEEP_ENTITY="${DEFAULT_WANDB_SWEEP_ENTITY:-minhangxu1998-baylor-college-of-medicine}"
# Left intentionally empty: every curated config now declares an explicit
# top-level `project:` (see WANDB_SWEEP_WORKFLOW.md). If a caller ships a
# config without one and also does not export WANDB_SWEEP_PROJECT, the launch
# should fail loudly rather than silently dump runs into a generic bucket.
DEFAULT_WANDB_SWEEP_PROJECT="${DEFAULT_WANDB_SWEEP_PROJECT:-}"

csv_escape() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

ensure_registry_file() {
  mkdir -p "${REGISTRY_DIR}"
  if [[ ! -f "${SWEEP_LOG_CSV}" ]]; then
    printf '%s\n' \
      'launched_at,task_family,target_family,comparison_group,config_path,launch_script,wandb_entity,wandb_project,sweep_path,sweep_id,num_agents,runs_per_agent,gpu_list,notes' \
      > "${SWEEP_LOG_CSV}"
  fi
}

yaml_top_level_value() {
  local config_path="$1"
  local key="$2"
  awk -F': *' -v key="${key}" '
    $0 ~ ("^" key ":[[:space:]]*") {
      value = $0
      sub("^" key ":[[:space:]]*", "", value)
      gsub(/^["'\'' ]+|["'\'' ]+$/, "", value)
      print value
      exit
    }
  ' "${config_path}"
}

yaml_parameter_value() {
  local config_path="$1"
  local key="$2"
  awk -v key="${key}" '
    /^parameters:[[:space:]]*$/ {
      in_parameters = 1
      next
    }
    in_parameters && /^[^[:space:]]/ {
      in_parameters = 0
    }
    in_parameters && $0 ~ ("^  " key ":[[:space:]]*$") {
      in_key = 1
      next
    }
    in_key && $0 ~ /^    value:[[:space:]]*/ {
      value = $0
      sub(/^    value:[[:space:]]*/, "", value)
      gsub(/^["'\'' ]+|["'\'' ]+$/, "", value)
      print value
      exit
    }
  ' "${config_path}"
}

resolve_wandb_sweep_entity() {
  local config_path="$1"
  if [[ -n "${WANDB_SWEEP_ENTITY:-}" ]]; then
    printf '%s\n' "${WANDB_SWEEP_ENTITY}"
    return 0
  fi

  local config_value
  config_value="$(yaml_top_level_value "${config_path}" "entity")"
  if [[ -n "${config_value}" ]]; then
    printf '%s\n' "${config_value}"
    return 0
  fi

  printf '%s\n' "${DEFAULT_WANDB_SWEEP_ENTITY}"
}

resolve_wandb_sweep_project() {
  local config_path="$1"
  if [[ -n "${WANDB_SWEEP_PROJECT:-}" ]]; then
    printf '%s\n' "${WANDB_SWEEP_PROJECT}"
    return 0
  fi

  local config_value
  config_value="$(yaml_top_level_value "${config_path}" "project")"
  if [[ -n "${config_value}" ]]; then
    printf '%s\n' "${config_value}"
    return 0
  fi

  if [[ -n "${DEFAULT_WANDB_SWEEP_PROJECT}" ]]; then
    printf '%s\n' "${DEFAULT_WANDB_SWEEP_PROJECT}"
    return 0
  fi

  echo "ERROR: no W&B project resolved for ${config_path}." >&2
  echo "       Set top-level 'project:' in the YAML or export WANDB_SWEEP_PROJECT." >&2
  return 1
}

materialize_sweep_config() {
  local source_config="$1"
  local target_config="$2"
  local wandb_entity="$3"
  local wandb_project="$4"
  awk -v wandb_entity="${wandb_entity}" -v wandb_project="${wandb_project}" '
    BEGIN {
      entity_written = 0
      project_written = 0
      inserted = 0
    }
    /^entity:[[:space:]]*/ {
      if (wandb_entity != "") {
        print "entity: " wandb_entity
        entity_written = 1
      }
      next
    }
    /^project:[[:space:]]*/ {
      if (wandb_project != "") {
        print "project: " wandb_project
        project_written = 1
      }
      next
    }
    !inserted && /^method:[[:space:]]*/ {
      if (wandb_entity != "" && !entity_written) {
        print "entity: " wandb_entity
        entity_written = 1
      }
      if (wandb_project != "" && !project_written) {
        print "project: " wandb_project
        project_written = 1
      }
      inserted = 1
    }
    {
      print
    }
    END {
      if (!inserted) {
        if (wandb_entity != "" && !entity_written) {
          print "entity: " wandb_entity
        }
        if (wandb_project != "" && !project_written) {
          print "project: " wandb_project
        }
      }
    }
  ' "${source_config}" > "${target_config}"
}

extract_sweep_path_from_output() {
  local output_file="$1"
  local sweep_path
  local sweep_url

  sweep_path="$(sed -n 's/.*wandb agent //p' "${output_file}" | tail -n 1 | tr -d '\r')"
  if [[ -n "${sweep_path}" ]]; then
    printf '%s\n' "${sweep_path}"
    return 0
  fi

  sweep_url="$(sed -n 's/.*View sweep at: //p' "${output_file}" | tail -n 1 | tr -d '\r')"
  if [[ -n "${sweep_url}" ]]; then
    printf '%s\n' "$(echo "${sweep_url}" | sed -n 's#https://wandb.ai/\([^/]*\)/\([^/]*\)/sweeps/\([^/?[:space:]]*\).*#\1/\2/\3#p')"
    return 0
  fi

  printf '\n'
}

record_sweep_launch() {
  local task_family="$1"
  local target_family="$2"
  local comparison_group="$3"
  local config_path="$4"
  local launch_script="$5"
  local wandb_entity="$6"
  local wandb_project="$7"
  local sweep_path="$8"
  local sweep_id="$9"
  local num_agents="${10}"
  local runs_per_agent="${11}"
  local gpu_list="${12}"
  local notes="${13:-}"

  ensure_registry_file

  {
    csv_escape "$(date -Iseconds)"
    printf ','
    csv_escape "${task_family}"
    printf ','
    csv_escape "${target_family}"
    printf ','
    csv_escape "${comparison_group}"
    printf ','
    csv_escape "${config_path}"
    printf ','
    csv_escape "${launch_script}"
    printf ','
    csv_escape "${wandb_entity}"
    printf ','
    csv_escape "${wandb_project}"
    printf ','
    csv_escape "${sweep_path}"
    printf ','
    csv_escape "${sweep_id}"
    printf ','
    csv_escape "${num_agents}"
    printf ','
    csv_escape "${runs_per_agent}"
    printf ','
    csv_escape "${gpu_list}"
    printf ','
    csv_escape "${notes}"
    printf '\n'
  } >> "${SWEEP_LOG_CSV}"
}

create_sweep_if_needed() {
  local config_path="$1"
  local sweep_path=""
  local output_file
  local temp_config=""
  local wandb_entity
  local wandb_project
  local sweep_status=0
  output_file="$(mktemp)"
  wandb_entity="$(resolve_wandb_sweep_entity "${config_path}")"
  wandb_project="$(resolve_wandb_sweep_project "${config_path}")"

  if [[ -n "${SWEEP_ID:-}" ]]; then
    local provided_sweep="${SWEEP_ID}"
    if [[ "${provided_sweep}" != */*/* ]]; then
      echo "SWEEP_ID must be a full sweep path like entity/project/sweep_id" >&2
      rm -f "${output_file}"
      return 1
    fi
    sweep_path="${provided_sweep}"
  else
    temp_config="$(mktemp --suffix=.yaml)"
    materialize_sweep_config "${config_path}" "${temp_config}" "${wandb_entity}" "${wandb_project}"

    if (
      cd "${LEARN_DIR}" &&
      wandb sweep "${temp_config}" > "${output_file}" 2>&1
    ); then
      sweep_status=0
    else
      sweep_status=$?
    fi

    cat "${output_file}" >&2
    if [[ ${sweep_status} -ne 0 ]]; then
      rm -f "${output_file}" "${temp_config}"
      return "${sweep_status}"
    fi

    sweep_path="$(extract_sweep_path_from_output "${output_file}")"
  fi

  rm -f "${output_file}" "${temp_config}"

  if [[ -z "${sweep_path}" ]]; then
    echo "Failed to determine sweep path for ${config_path}" >&2
    return 1
  fi

  if [[ "$(echo "${sweep_path}" | cut -d/ -f1)" != "${wandb_entity}" ]]; then
    echo "Resolved sweep entity does not match parsed sweep path for ${config_path}" >&2
    echo "Expected entity: ${wandb_entity}" >&2
    echo "Parsed sweep path: ${sweep_path}" >&2
    return 1
  fi

  if [[ "$(echo "${sweep_path}" | cut -d/ -f2)" != "${wandb_project}" ]]; then
    echo "Resolved sweep project does not match parsed sweep path for ${config_path}" >&2
    echo "Expected project: ${wandb_project}" >&2
    echo "Parsed sweep path: ${sweep_path}" >&2
    return 1
  fi

  echo "${sweep_path}"
}

launch_wandb_agents() {
  local config_path="$1"
  local task_family="$2"
  local target_family="$3"
  local comparison_group="$4"
  local launch_script="$5"
  local num_agents="$6"
  local runs_per_agent="$7"
  shift 7
  local gpu_list=("$@")

  # PILOT=1 shrinks a curated sweep to a 1-agent / 1-run smoke test, regardless
  # of what the launcher script configured. Useful for verifying the config +
  # data + model + graph + save_model chain before burning GPU hours on the
  # full search space. Prefix the launcher invocation like:
  #   PILOT=1 ./launch/utr3_hani_utr_bassetvl_sweep.sh
  if [[ "${PILOT:-0}" == "1" ]]; then
    echo "PILOT=1 detected: forcing NUM_AGENTS=1, NUM_RUNS=1 for ${config_path}"
    num_agents=1
    runs_per_agent=1
    gpu_list=("${gpu_list[0]}")
    LAUNCH_NOTES="${LAUNCH_NOTES:-pilot}"
  fi

  local sweep_path
  sweep_path="$(create_sweep_if_needed "${config_path}")"

  local wandb_entity wandb_project sweep_id config_logger_project
  wandb_entity="$(echo "${sweep_path}" | cut -d/ -f1)"
  wandb_project="$(echo "${sweep_path}" | cut -d/ -f2)"
  sweep_id="$(echo "${sweep_path}" | cut -d/ -f3)"
  config_logger_project="$(yaml_parameter_value "${config_path}" "logger_project")"

  record_sweep_launch \
    "${task_family}" \
    "${target_family}" \
    "${comparison_group}" \
    "${config_path}" \
    "${launch_script}" \
    "${wandb_entity}" \
    "${wandb_project}" \
    "${sweep_path}" \
    "${sweep_id}" \
    "${num_agents}" \
    "${runs_per_agent}" \
    "${gpu_list[*]}" \
    "${LAUNCH_NOTES:-}"

  echo "W&B sweep project: ${wandb_entity}/${wandb_project}"
  echo "Sweep path: ${sweep_path}"
  echo "Config: ${config_path}"
  if [[ -n "${config_logger_project}" ]]; then
    echo "Config logger_project: ${config_logger_project}"
  fi

  if [[ "${CREATE_SWEEP_ONLY:-0}" == "1" ]]; then
    echo "CREATE_SWEEP_ONLY=1, skipping agent launch."
    return 0
  fi

  # Resolve an absolute config path so the provenance row in runs.csv is
  # reproducible regardless of the agent's working directory.
  local abs_config_path="${config_path}"
  if [[ "${abs_config_path}" != /* ]]; then
    abs_config_path="${LEARN_DIR}/${config_path}"
  fi

  # Metadata that each training process will stamp into provenance.json /
  # run_registry/runs.csv. `wandb agent` inherits the parent shell's env, so
  # exporting here reaches every Python process spawned below.
  export BODA_CONFIG_PATH="${abs_config_path}"
  export BODA_TASK_FAMILY="${task_family}"
  export BODA_TARGET_FAMILY="${target_family}"
  export BODA_COMPARISON_GROUP="${comparison_group}"
  export BODA_LAUNCH_SCRIPT="${launch_script}"
  export BODA_SWEEP_PATH="${sweep_path}"
  export BODA_SWEEP_ID="${sweep_id}"
  export BODA_WANDB_ENTITY="${wandb_entity}"
  export BODA_WANDB_PROJECT="${wandb_project}"
  export BODA_RUNS_CSV="${REGISTRY_DIR}/runs.csv"
  export BODA_LAUNCH_NOTES="${LAUNCH_NOTES:-}"

  (
    cd "${LEARN_DIR}"
    for ((i=0; i<num_agents; i++)); do
      local gpu_id="${gpu_list[i % ${#gpu_list[@]}]}"
      echo "Launching agent $((i + 1))/${num_agents} on GPU ${gpu_id} for ${runs_per_agent} run(s)"
      CUDA_VISIBLE_DEVICES="${gpu_id}" wandb agent --count "${runs_per_agent}" "${sweep_path}" &
      sleep 2
    done
    wait
  )
}
