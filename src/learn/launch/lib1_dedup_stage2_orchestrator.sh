#!/usr/bin/env bash
set -euo pipefail

# Resumable global-queue launcher for the July 2026 Lib1 dedup Stage 2
# development comparison.  The generated queue has 610 new jobs; 50 immutable
# Stage 1 cells are analysis-only reuse and are never relaunched here.  Frozen
# audit/test evaluation is prohibited by the manifest verifier.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CAMPAIGN_ID="lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE="stage2_paired_rc"
MANIFEST_TAG="${MANIFEST_TAG:-lib1_dedup_stage2_july2026}"
EXPECTED_WANDB_ENTITY="minhangxu1998-baylor-college-of-medicine"

DATA_MANIFEST="${DATA_MANIFEST:-${LEARN_DIR}/data_manifests/lib1_single_part_dedup_exact_v1.json}"
SPLIT_INDEX="${SPLIT_INDEX:-${LEARN_DIR}/data_manifests/lib1_dedup_exact_v1_split_manifests.json}"
MANIFEST_OUTDIR="${MANIFEST_OUTDIR:-${LEARN_DIR}/outputs/hpo_manifests}"
MANIFEST_JSONL="${MANIFEST_JSONL:-${MANIFEST_OUTDIR}/${MANIFEST_TAG}__run_manifest.jsonl}"
ANALYSIS_MANIFEST="${ANALYSIS_MANIFEST:-${MANIFEST_OUTDIR}/${MANIFEST_TAG}__analysis_manifest.jsonl}"
REUSE_MANIFEST="${REUSE_MANIFEST:-${MANIFEST_OUTDIR}/${MANIFEST_TAG}__stage1_reuse_cells.jsonl}"
RUNS_CSV="${RUNS_CSV:-${LEARN_DIR}/run_registry/runs.csv}"
STATUS_DIR="${STATUS_DIR:-${LEARN_DIR}/outputs/hpo_runs/status/${MANIFEST_TAG}}"

PREPARE_DATASET="${PREPARE_DATASET:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PARTS="${PARTS:-}"
ANALYSIS_LANES="${ANALYSIS_LANES:-}"
BASE_CONFIG_IDS="${BASE_CONFIG_IDS:-}"
RC_MODES="${RC_MODES:-}"
FOLDS="${FOLDS:-}"
ROW_RANGE="${ROW_RANGE:-}"
ROW_START="${ROW_START:-}"
ROW_END="${ROW_END:-}"
MAX_ROWS="${MAX_ROWS:-}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
CHECK_STORAGE="${CHECK_STORAGE:-1}"
ALLOW_MANIFEST_CHANGE="${ALLOW_MANIFEST_CHANGE:-0}"
LAUNCH_NOTES="${LAUNCH_NOTES:-${CAMPAIGN_ID};${CAMPAIGN_STAGE}}"

# Prevent ambient shell state from sending these runs to a collaborator account,
# offline storage, or a nested src/learn/wandb/wandb directory.
export WANDB_ENTITY="${EXPECTED_WANDB_ENTITY}"
export BODA_WANDB_ENTITY="${EXPECTED_WANDB_ENTITY}"
export WANDB_MODE="online"
export WANDB_DIR="${BODA_STAGE2_WANDB_ROOT:-${LEARN_DIR}}"
export EXPECTED_WANDB_ENTITY
mkdir -p "${WANDB_DIR}" "${STATUS_DIR}"

exec 9>"${STATUS_DIR}/launcher.lock"
if ! flock -n 9; then
  echo "ERROR: another ${MANIFEST_TAG} orchestrator owns ${STATUS_DIR}." >&2
  exit 1
fi

if [[ "${PREPARE_DATASET}" == "1" ]]; then
  echo "Preparing canonical dedup datasets and deterministic frozen splits"
  python "${LEARN_DIR}/prepare_lib1_dedup_exact_datasets.py"
  python "${LEARN_DIR}/generate_lib1_dedup_split_manifests.py" \
    --data-manifest-path "${DATA_MANIFEST}" \
    --index-path "${SPLIT_INDEX}"
fi
for required in "${DATA_MANIFEST}" "${SPLIT_INDEX}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required Stage 2 input is missing: ${required}" >&2
    echo "Use PREPARE_DATASET=1 only if the canonical inputs truly need rebuilding." >&2
    exit 1
  fi
done

echo "Generating the fixed 660-cell analysis / 610-job launch manifests"
python "${LEARN_DIR}/generate_lib1_dedup_stage2_manifest.py" \
  --manifest-tag "${MANIFEST_TAG}" \
  --data-manifest "${DATA_MANIFEST}" \
  --split-index "${SPLIT_INDEX}" \
  --runs-csv "${RUNS_CSV}" \
  --outdir "${MANIFEST_OUTDIR}"

echo "Validating Stage 2 accounting, RC pairs, metadata, commands, and audit isolation"
python "${LEARN_DIR}/verify_lib1_dedup_stage2_manifest.py" \
  --analysis-manifest "${ANALYSIS_MANIFEST}" \
  --run-manifest "${MANIFEST_JSONL}" \
  --reuse-manifest "${REUSE_MANIFEST}" \
  --split-index "${SPLIT_INDEX}" \
  --utr-selection "${MANIFEST_OUTDIR}/${MANIFEST_TAG}__utr3_utrbassetvl_selected_configs.jsonl"

MANIFEST_SHA256="$(sha256sum "${MANIFEST_JSONL}" | awk '{print $1}')"
MANIFEST_SHA_FILE="${STATUS_DIR}/manifest.sha256"
if [[ -s "${MANIFEST_SHA_FILE}" ]]; then
  PREVIOUS_MANIFEST_SHA256="$(tr -d '[:space:]' < "${MANIFEST_SHA_FILE}")"
  if [[ "${PREVIOUS_MANIFEST_SHA256}" != "${MANIFEST_SHA256}" ]] \
      && compgen -G "${STATUS_DIR}/done/row_*.done" >/dev/null \
      && [[ "${ALLOW_MANIFEST_CHANGE}" != "1" ]]; then
    echo "ERROR: Stage 2 manifest SHA changed while completion markers exist." >&2
    echo "  previous: ${PREVIOUS_MANIFEST_SHA256}" >&2
    echo "  current:  ${MANIFEST_SHA256}" >&2
    echo "Archive/review ${STATUS_DIR}; do not bypass without reconciling results." >&2
    exit 1
  fi
fi
printf '%s\n' "${MANIFEST_SHA256}" >"${MANIFEST_SHA_FILE}.tmp"
mv "${MANIFEST_SHA_FILE}.tmp" "${MANIFEST_SHA_FILE}"

export STATUS_DIR PARTS ANALYSIS_LANES BASE_CONFIG_IDS RC_MODES FOLDS
export ROW_RANGE ROW_START ROW_END MAX_ROWS SKIP_COMPLETED

manifest_helper() {
  local action="$1"
  shift
  python - "${action}" "${MANIFEST_JSONL}" "${RUNS_CSV}" "$@" <<'PY'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

action = sys.argv[1]
manifest_path = Path(sys.argv[2])
runs_csv = Path(sys.argv[3])
extra = sys.argv[4:]
status_dir = Path(os.environ["STATUS_DIR"])

# Read the registry once per helper invocation. Reopening and reparsing the
# multi-megabyte CSV for every launch row makes startup quadratic.
registry_by_cell = {}
if runs_csv.is_file():
    with runs_csv.open(newline="") as handle:
        for record in csv.DictReader(handle):
            cell_id = record.get("cell_id", "")
            if cell_id:
                registry_by_cell.setdefault(cell_id, []).append(record)


def rows():
    with manifest_path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokens(name):
    value = os.environ.get(name, "")
    return [item for item in value.replace(",", " ").replace(";", " ").split() if item]


def parse_range():
    start = os.environ.get("ROW_START", "").strip()
    end = os.environ.get("ROW_END", "").strip()
    compact = os.environ.get("ROW_RANGE", "").strip()
    if compact and not start and not end:
        separator = ":" if ":" in compact else "-"
        if separator in compact:
            start, end = (item.strip() for item in compact.split(separator, 1))
    return (int(start) if start else None, int(end) if end else None)


def row_completed(row):
    registry_matches = registry_by_cell.get(row["cell_id"], [])
    expected_registry = {
        "run_name": row["planned_run_name"],
        "wandb_entity": row["wandb_entity"],
        "logger_project": row["logger_project"],
        "campaign_id": row["campaign_id"],
        "campaign_stage": row["campaign_stage"],
        "part_slug": row["part_slug"],
        "analysis_lane": row["analysis_lane"],
        "challenger_family": row["challenger_family"],
        "policy_id": row["policy_id"],
        "config_origin": row["config_origin"],
        "training_regime": row["training_regime"],
        "rc_pair_id": row["rc_pair_id"],
        "rc_mode": row["rc_mode"],
        "execution_disposition": "launch",
        "initialization": row["initialization"],
        "source_head": row["source_head"],
        "unfreeze_scope": row["unfreeze_scope"],
        "input_policy": row["input_policy"],
        "pretrained_artifact_sha256": row["pretrained_artifact_sha256"],
        "data_generation_id": row["data_generation_id"],
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "development_fold": str(row["development_fold"]),
        "base_config_id": row["base_config_id"],
        "architecture": row["architecture"],
        "model_seed": str(row["model_seed"]),
        "loss_mode": row["loss_mode"],
        "target_definition": row["target_definition"],
        "length_policy": row["length_policy"],
        "artifact_retention": "none",
    }
    completed_registry = False
    for record in registry_matches:
        mismatches = {
            field: (record.get(field, ""), expected)
            for field, expected in expected_registry.items()
            if record.get(field, "") != str(expected)
        }
        if mismatches:
            raise SystemExit(
                f"runs.csv provenance collision for cell_id={row['cell_id']}: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if record.get("status", "").lower() != "completed":
            continue
        prediction_value = record.get("prediction_path", "").strip()
        if not prediction_value:
            continue
        prediction = Path(prediction_value)
        expected_parent = (Path(row["default_root_dir"]) / "predictions").resolve()
        if prediction.is_file() and prediction.resolve().parent == expected_parent:
            if not record.get("val_row_id_hash", "").strip():
                raise SystemExit(
                    f"Completed runs.csv record for {row['cell_id']} lacks val_row_id_hash"
                )
            completed_registry = True

    marker = status_dir / "done" / f"row_{row['manifest_row']}.done"
    completed_marker = False
    if marker.is_file():
        fields = {}
        try:
            for line in marker.read_text().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    fields[key] = value
        except OSError as exc:
            raise SystemExit(f"Cannot read completion marker {marker}: {exc}")
        if fields.get("row_fingerprint") != row.get("row_fingerprint"):
            raise SystemExit(
                f"Completion-marker fingerprint mismatch for row {row['manifest_row']}"
            )
        completed_marker = True
    return completed_marker or completed_registry


def selected():
    wanted_parts = {value.lower() for value in tokens("PARTS")}
    wanted_lanes = set(tokens("ANALYSIS_LANES"))
    wanted_bases = set(tokens("BASE_CONFIG_IDS"))
    wanted_rc = {value.lower() for value in tokens("RC_MODES")}
    wanted_folds = {int(value) for value in tokens("FOLDS")}
    start, end = parse_range()
    out = []
    for row in rows():
        number = int(row["manifest_row"])
        if start is not None and number < start:
            continue
        if end is not None and number > end:
            continue
        if wanted_parts and str(row["part_slug"]).lower() not in wanted_parts:
            continue
        if wanted_lanes and row["analysis_lane"] not in wanted_lanes:
            continue
        if wanted_bases and row["base_config_id"] not in wanted_bases:
            continue
        if wanted_rc and row["rc_mode"].lower() not in wanted_rc:
            continue
        if wanted_folds and int(row["development_fold"]) not in wanted_folds:
            continue
        if os.environ.get("SKIP_COMPLETED", "1") == "1" and row_completed(row):
            continue
        out.append(row)
    maximum = os.environ.get("MAX_ROWS", "").strip()
    return out[:int(maximum)] if maximum else out


def by_number(number):
    wanted = int(number)
    for row in rows():
        if int(row["manifest_row"]) == wanted:
            return row
    raise SystemExit(f"No manifest row {wanted}")


if action == "list":
    for row in selected():
        print(row["manifest_row"])
elif action == "dry":
    fields = (
        "manifest_row", "analysis_lane", "part_slug", "base_config_id",
        "development_fold", "rc_mode", "planned_run_name", "wandb_entity",
        "logger_project", "train_command",
    )
    for row in selected():
        print("\t".join(str(row.get(field, "")) for field in fields))
elif action == "execution":
    row = by_number(extra[0])
    fields = (
        "planned_run_name", "wandb_entity", "logger_project", "row_fingerprint",
        "base_config_id", "analysis_lane", "cell_id", "rc_pair_id", "train_command",
    )
    print("\t".join(str(row.get(field, "")) for field in fields))
elif action == "counts":
    chosen = selected()
    by_lane = {}
    by_part = {}
    by_rc = {}
    for row in chosen:
        for target, key in (
            (by_lane, row["analysis_lane"]),
            (by_part, row["part_slug"]),
            (by_rc, row["rc_mode"]),
        ):
            target[key] = target.get(key, 0) + 1
    print(json.dumps({"selected": len(chosen), "by_lane": by_lane,
                      "by_part": by_part, "by_rc": by_rc}, sort_keys=True))
elif action == "validate_completion_records":
    for row in rows():
        row_completed(row)
    print(f"Validated completion provenance for {len(rows())} launch rows")
else:
    raise SystemExit(f"Unknown manifest-helper action: {action}")
PY
}

manifest_helper validate_completion_records

if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a ALL_GPUS <<< "${GPU_LIST}"
elif [[ "${DRY_RUN}" == "1" ]]; then
  ALL_GPUS=(0)
else
  mapfile -t ALL_GPUS < <(detect_idle_gpus)
fi
if [[ ${#ALL_GPUS[@]} -eq 0 ]]; then
  echo "ERROR: no GPUs selected; set GPU_LIST or free a GPU." >&2
  exit 1
fi
if [[ -z "${MAX_PARALLEL}" ]]; then
  MAX_PARALLEL="${#ALL_GPUS[@]}"
fi
if (( MAX_PARALLEL < 1 || MAX_PARALLEL > ${#ALL_GPUS[@]} )); then
  echo "ERROR: MAX_PARALLEL=${MAX_PARALLEL} must be 1..${#ALL_GPUS[@]}." >&2
  exit 1
fi
GPU_ARRAY=("${ALL_GPUS[@]:0:${MAX_PARALLEL}}")

echo "Lib1 dedup Stage 2 paired-RC development launch"
echo "  manifest: ${MANIFEST_JSONL}"
echo "  manifest SHA256: ${MANIFEST_SHA256}"
echo "  selected: $(manifest_helper counts)"
echo "  entity (forced): ${WANDB_ENTITY}"
echo "  local W&B root: ${WANDB_DIR}/wandb"
echo "  GPUs: ${GPU_ARRAY[*]}"
echo "  DRY_RUN: ${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  count=0
  while IFS=$'\t' read -r manifest_row analysis_lane part_slug base_config_id development_fold rc_mode planned_run_name wandb_entity logger_project train_command; do
    gpu="${GPU_ARRAY[$((count % ${#GPU_ARRAY[@]}))]}"
    count=$((count + 1))
    echo
    echo "Row ${manifest_row}: lane=${analysis_lane} part=${part_slug} fold=${development_fold} rc=${rc_mode} gpu=${gpu}"
    echo "Project: ${wandb_entity}/${logger_project}"
    echo "Run: ${planned_run_name}"
    echo "CUDA_VISIBLE_DEVICES=${gpu} ${train_command}"
  done < <(manifest_helper dry)
  echo
  echo "DRY_RUN selected rows: ${count}"
  exit 0
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "boda_env" ]]; then
  echo "ERROR: activate boda_env before launching (current: ${CONDA_DEFAULT_ENV:-none})." >&2
  exit 1
fi
python - <<'PY'
import os
import sys
import wandb

entity = os.environ["EXPECTED_WANDB_ENTITY"]
api = wandb.Api(timeout=15)
if not getattr(api, "api_key", None):
    raise SystemExit("No W&B API key resolved. Run `wandb login` before launch.")
try:
    next(iter(api.projects(entity=entity, per_page=1)), None)
except Exception as exc:
    raise SystemExit(f"W&B access preflight failed for {entity!r}: {exc}")
print(f"W&B preflight passed for {entity} under Python {sys.version.split()[0]}.")
PY

check_storage() {
  local path="$1" min_free_gb="$2" max_use_pct="$3" label="$4"
  local fields available_kb used_pct available_gb
  fields="$(df -Pk "${path}" | awk 'NR==2 {print $4, $5}')"
  read -r available_kb used_pct <<< "${fields}"
  used_pct="${used_pct%%%}"
  available_gb=$((available_kb / 1024 / 1024))
  echo "Storage ${label}: ${available_gb} GiB free, ${used_pct}% used"
  if (( available_gb < min_free_gb || used_pct >= max_use_pct )); then
    echo "ERROR: ${label} storage stop condition reached." >&2
    return 1
  fi
}
if [[ "${CHECK_STORAGE}" == "1" ]]; then
  check_storage /home 150 80 /home
  check_storage / 20 100 /
fi

mapfile -t QUEUE_ROWS < <(manifest_helper list)
if [[ ${#QUEUE_ROWS[@]} -eq 0 ]]; then
  echo "No Stage 2 rows remain after filters and fingerprint completion checks."
  exit 0
fi

LAUNCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
CLAIM_DIR="${STATUS_DIR}/claims/${LAUNCH_ID}"
PROCESSED_DIR="${STATUS_DIR}/processed/${LAUNCH_ID}"
DONE_DIR="${STATUS_DIR}/done"
FAIL_DIR="${STATUS_DIR}/failures/${LAUNCH_ID}"
LOG_DIR="${STATUS_DIR}/logs/${LAUNCH_ID}"
MONITOR_FILE="${STATUS_DIR}/monitor.tsv"
STOP_FILE="${STATUS_DIR}/stop.${LAUNCH_ID}"
mkdir -p "${CLAIM_DIR}" "${PROCESSED_DIR}" "${DONE_DIR}" "${FAIL_DIR}" "${LOG_DIR}"

run_manifest_row() {
  local gpu_id="$1" worker_id="$2" manifest_row="$3"
  local planned_run_name wandb_entity logger_project row_fingerprint base_config_id
  local analysis_lane cell_id rc_pair_id train_command row_log fail_file status claim_path done_tmp
  IFS=$'\t' read -r planned_run_name wandb_entity logger_project row_fingerprint base_config_id analysis_lane cell_id rc_pair_id train_command \
    < <(manifest_helper execution "${manifest_row}")
  if [[ "${wandb_entity}" != "${EXPECTED_WANDB_ENTITY}" ]]; then
    echo "ERROR: row ${manifest_row} requested forbidden entity ${wandb_entity}." >&2
    return 97
  fi
  row_log="${LOG_DIR}/row_${manifest_row}.log"
  fail_file="${FAIL_DIR}/row_${manifest_row}.fail"
  claim_path="${CLAIM_DIR}/row_${manifest_row}.claim"
  echo "[$(date -Iseconds)] worker=${worker_id} gpu=${gpu_id} start row=${manifest_row} cell=${cell_id}"
  if (
    cd "${LEARN_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export WANDB_ENTITY="${EXPECTED_WANDB_ENTITY}"
    export BODA_WANDB_ENTITY="${EXPECTED_WANDB_ENTITY}"
    export BODA_WANDB_PROJECT="${logger_project}"
    export BODA_CONFIG_PATH="${MANIFEST_JSONL}"
    export BODA_COMPARISON_GROUP="${analysis_lane}"
    export BODA_LAUNCH_SCRIPT="launch/lib1_dedup_stage2_orchestrator.sh"
    export BODA_RUNS_CSV="${RUNS_CSV}"
    export BODA_LAUNCH_NOTES="${LAUNCH_NOTES}"
    bash -c "${train_command}"
  ) >"${row_log}" 2>&1; then
    done_tmp="${DONE_DIR}/.row_${manifest_row}.done.${LAUNCH_ID}.tmp"
    {
      echo "completed_at=$(date -Iseconds)"
      echo "manifest_row=${manifest_row}"
      echo "manifest_sha256=${MANIFEST_SHA256}"
      echo "row_fingerprint=${row_fingerprint}"
      echo "base_config_id=${base_config_id}"
      echo "cell_id=${cell_id}"
      echo "rc_pair_id=${rc_pair_id}"
      echo "planned_run_name=${planned_run_name}"
      echo "wandb_url=$(sed -n 's/^Resolved W&B run URL: //p' "${row_log}" | tail -n 1)"
      echo "log=${row_log}"
    } >"${done_tmp}"
    mv "${done_tmp}" "${DONE_DIR}/row_${manifest_row}.done"
    mv "${claim_path}" "${PROCESSED_DIR}/row_${manifest_row}.processed"
    echo "[$(date -Iseconds)] worker=${worker_id} gpu=${gpu_id} done row=${manifest_row}"
    return 0
  else
    status=$?
  fi
  {
    echo "failed_at=$(date -Iseconds)"
    echo "manifest_row=${manifest_row}"
    echo "manifest_sha256=${MANIFEST_SHA256}"
    echo "row_fingerprint=${row_fingerprint}"
    echo "cell_id=${cell_id}"
    echo "status=${status}"
    echo "log=${row_log}"
  } >"${fail_file}"
  mv "${claim_path}" "${PROCESSED_DIR}/row_${manifest_row}.processed"
  echo "[$(date -Iseconds)] worker=${worker_id} gpu=${gpu_id} FAILED row=${manifest_row}; ${row_log}" >&2
  [[ "${STOP_ON_ERROR}" == "1" ]] && touch "${STOP_FILE}"
  return "${status}"
}

worker_loop() {
  local gpu_id="$1" worker_id="$2" manifest_row claim_path claimed
  while [[ ! -e "${STOP_FILE}" ]]; do
    claimed=""
    for manifest_row in "${QUEUE_ROWS[@]}"; do
      if [[ -e "${PROCESSED_DIR}/row_${manifest_row}.processed" ]]; then
        continue
      fi
      claim_path="${CLAIM_DIR}/row_${manifest_row}.claim"
      if mkdir "${claim_path}" 2>/dev/null; then
        claimed="${manifest_row}"
        break
      fi
    done
    if [[ -z "${claimed}" ]]; then
      echo "Worker ${worker_id} GPU ${gpu_id}: queue exhausted"
      return 0
    fi
    if ! run_manifest_row "${gpu_id}" "${worker_id}" "${claimed}" \
        && [[ "${STOP_ON_ERROR}" == "1" ]]; then
      return 1
    fi
  done
  echo "Worker ${worker_id} GPU ${gpu_id}: stop marker observed"
  return 1
}

monitor_loop() {
  if [[ ! -f "${MONITOR_FILE}" ]]; then
    printf 'timestamp\tlaunch_id\tselected\tdone_total\tactive\tfailed_launch\thome_free_gb\thome_used_pct\n' >"${MONITOR_FILE}"
  fi
  while true; do
    local done_total active failed fields free_kb used_pct free_gb
    done_total="$(find "${DONE_DIR}" -maxdepth 1 -type f -name 'row_*.done' | wc -l)"
    active="$(find "${CLAIM_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'row_*.claim' | wc -l)"
    failed="$(find "${FAIL_DIR}" -maxdepth 1 -type f -name '*.fail' | wc -l)"
    fields="$(df -Pk /home | awk 'NR==2 {print $4, $5}')"
    read -r free_kb used_pct <<<"${fields}"
    free_gb=$((free_kb / 1024 / 1024))
    used_pct="${used_pct%%%}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -Iseconds)" "${LAUNCH_ID}" "${#QUEUE_ROWS[@]}" "${done_total}" \
      "${active}" "${failed}" "${free_gb}" "${used_pct}" | tee -a "${MONITOR_FILE}"
    if [[ "${CHECK_STORAGE}" == "1" ]] && (( free_gb < 150 || used_pct >= 80 )); then
      echo "Storage threshold reached; stopping new claims after active jobs finish." >&2
      touch "${FAIL_DIR}/storage_threshold.fail" "${STOP_FILE}"
      return 1
    fi
    sleep "${MONITOR_INTERVAL}"
  done
}

terminate_tree() {
  local parent="$1" child
  while read -r child; do
    [[ -n "${child}" ]] && terminate_tree "${child}"
  done < <(pgrep -P "${parent}" 2>/dev/null || true)
  kill -TERM "${parent}" 2>/dev/null || true
}

pids=()
MONITOR_PID=""
cleanup() {
  local pid
  [[ -n "${MONITOR_PID}" ]] && terminate_tree "${MONITOR_PID}"
  for pid in "${pids[@]:-}"; do
    [[ -n "${pid}" ]] && terminate_tree "${pid}"
  done
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

echo "Starting ${#GPU_ARRAY[@]} worker(s) for ${#QUEUE_ROWS[@]} selected Stage 2 row(s)."
# Only the top-level orchestrator should own launcher.lock.  Close descriptor 9
# in background jobs so an orphaned monitor sleep cannot delay a chained,
# restart-safe invocation after the current launcher exits.
monitor_loop 9>&- &
MONITOR_PID=$!
for index in "${!GPU_ARRAY[@]}"; do
  worker_loop "${GPU_ARRAY[index]}" "$((index + 1))" 9>&- &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
terminate_tree "${MONITOR_PID}"
wait "${MONITOR_PID}" 2>/dev/null || true
MONITOR_PID=""
pids=()
trap - EXIT INT TERM

if compgen -G "${FAIL_DIR}/*.fail" >/dev/null; then
  status=1
fi
if [[ "${status}" == "0" ]]; then
  echo "All selected Stage 2 launch rows completed or were fingerprint-skipped."
else
  echo "Stage 2 stopped with failures. Inspect ${FAIL_DIR} and ${LOG_DIR}." >&2
fi
echo "Status monitor: ${MONITOR_FILE}"
exit "${status}"
