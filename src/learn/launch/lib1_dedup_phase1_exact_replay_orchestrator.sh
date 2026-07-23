#!/usr/bin/env bash
set -euo pipefail

# Resumable global-queue launcher for Stage 1 of the July 2026 Lib1 dedup
# campaign. This wrapper intentionally stops after exact fold-0 replay; it does
# not generate or launch RC, weighted-loss, audit, or downsampling stages.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${LEARN_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CAMPAIGN_ID="lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE="stage1_exact_replay"
MANIFEST_TAG="${MANIFEST_TAG:-lib1_dedup_phase1_exact_replay_july2026}"
EXPECTED_WANDB_ENTITY="minhangxu1998-baylor-college-of-medicine"

DATA_MANIFEST="${DATA_MANIFEST:-${LEARN_DIR}/data_manifests/lib1_single_part_dedup_exact_v1.json}"
SPLIT_INDEX="${SPLIT_INDEX:-${LEARN_DIR}/data_manifests/lib1_dedup_exact_v1_split_manifests.json}"
MANIFEST_OUTDIR="${MANIFEST_OUTDIR:-${LEARN_DIR}/outputs/hpo_manifests}"
MANIFEST_JSONL="${MANIFEST_JSONL:-${MANIFEST_OUTDIR}/${MANIFEST_TAG}__run_manifest.jsonl}"
RUNS_CSV="${RUNS_CSV:-${LEARN_DIR}/run_registry/runs.csv}"
STATUS_DIR="${STATUS_DIR:-${LEARN_DIR}/outputs/hpo_runs/status/${MANIFEST_TAG}}"

PREPARE_DATASET="${PREPARE_DATASET:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
INCLUDE_CALIBRATION="${INCLUDE_CALIBRATION:-0}"
PILOT="${PILOT:-0}"
PILOT_ROWS_PER_LANE="${PILOT_ROWS_PER_LANE:-1}"
PARTS="${PARTS:-}"
LANE_IDS="${LANE_IDS:-}"
BASE_CONFIG_IDS="${BASE_CONFIG_IDS:-}"
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

# Never inherit a collaborator entity or offline mode for this campaign.
export WANDB_ENTITY="${EXPECTED_WANDB_ENTITY}"
export BODA_WANDB_ENTITY="${EXPECTED_WANDB_ENTITY}"
export WANDB_MODE="online"
# WANDB_DIR is the parent in which W&B creates its `wandb/` directory. Point
# it at src/learn so runs land in the established src/learn/wandb root, not a
# nested src/learn/wandb/wandb directory. Ignore ambient WANDB_DIR by default.
export WANDB_DIR="${BODA_STAGE1_WANDB_ROOT:-${LEARN_DIR}}"
export EXPECTED_WANDB_ENTITY
mkdir -p "${WANDB_DIR}"

# Own the status/manifests before preparation or generation so two launchers
# cannot race and then interpret row-number completion markers differently.
mkdir -p "${STATUS_DIR}"
exec 9>"${STATUS_DIR}/launcher.lock"
if ! flock -n 9; then
  echo "ERROR: another ${MANIFEST_TAG} orchestrator currently owns ${STATUS_DIR}." >&2
  exit 1
fi

prepare_campaign_data() {
  if [[ "${PREPARE_DATASET}" != "1" ]]; then
    return 0
  fi
  echo "Preparing canonical dedup and calibration datasets"
  python "${LEARN_DIR}/prepare_lib1_dedup_exact_datasets.py" --include-pre-dedup
  echo "Generating deterministic frozen audit/development splits"
  python "${LEARN_DIR}/generate_lib1_dedup_split_manifests.py" \
    --data-manifest-path "${DATA_MANIFEST}" \
    --index-path "${SPLIT_INDEX}"
}

generate_replay_manifest() {
  echo "Resolving completed historical configs and generating fixed replay manifest"
  python "${LEARN_DIR}/generate_lib1_dedup_exact_replay_manifest.py" \
    --manifest-tag "${MANIFEST_TAG}" \
    --data-manifest "${DATA_MANIFEST}" \
    --split-index "${SPLIT_INDEX}" \
    --outdir "${MANIFEST_OUTDIR}"
}

prepare_campaign_data

if [[ ! -f "${DATA_MANIFEST}" ]]; then
  echo "ERROR: canonical data manifest is missing: ${DATA_MANIFEST}" >&2
  echo "Run with PREPARE_DATASET=1." >&2
  exit 1
fi
if [[ ! -f "${SPLIT_INDEX}" ]]; then
  echo "ERROR: frozen split index is missing: ${SPLIT_INDEX}" >&2
  echo "Run with PREPARE_DATASET=1." >&2
  exit 1
fi

generate_replay_manifest
if [[ ! -f "${MANIFEST_JSONL}" ]]; then
  echo "ERROR: generated replay manifest is missing: ${MANIFEST_JSONL}" >&2
  exit 1
fi

MANIFEST_SHA256="$(sha256sum "${MANIFEST_JSONL}" | awk '{print $1}')"
MANIFEST_SHA_FILE="${STATUS_DIR}/manifest.sha256"
if [[ -s "${MANIFEST_SHA_FILE}" ]]; then
  PREVIOUS_MANIFEST_SHA256="$(tr -d '[:space:]' < "${MANIFEST_SHA_FILE}")"
  if [[ "${PREVIOUS_MANIFEST_SHA256}" != "${MANIFEST_SHA256}" ]] \
      && compgen -G "${STATUS_DIR}/done/row_*.done" >/dev/null \
      && [[ "${ALLOW_MANIFEST_CHANGE}" != "1" ]]; then
    echo "ERROR: manifest SHA changed while completion markers exist." >&2
    echo "  previous: ${PREVIOUS_MANIFEST_SHA256}" >&2
    echo "  current:  ${MANIFEST_SHA256}" >&2
    echo "Review/archive ${STATUS_DIR}; set ALLOW_MANIFEST_CHANGE=1 only after that review." >&2
    exit 1
  fi
fi
printf '%s\n' "${MANIFEST_SHA256}" >"${MANIFEST_SHA_FILE}.tmp"
mv "${MANIFEST_SHA_FILE}.tmp" "${MANIFEST_SHA_FILE}"

export STATUS_DIR PARTS LANE_IDS BASE_CONFIG_IDS ROW_RANGE ROW_START ROW_END MAX_ROWS
export SKIP_COMPLETED INCLUDE_CALIBRATION PILOT PILOT_ROWS_PER_LANE

manifest_helper() {
  local action="$1"
  shift
  python - "${action}" "${MANIFEST_JSONL}" "${RUNS_CSV}" "$@" <<'PY'
import csv
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path


action = sys.argv[1]
manifest_path = Path(sys.argv[2])
runs_csv = Path(sys.argv[3])
extra = sys.argv[4:]
status_dir = Path(os.environ["STATUS_DIR"])


def tokens(name):
    value = os.environ.get(name, "")
    return [item for item in value.replace(",", " ").replace(";", " ").split() if item]


def load_rows():
    with manifest_path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_row_fingerprint(row):
    fields = (
        "run_kind", "campaign_id", "campaign_stage", "part_slug", "lane_id",
        "architecture", "base_config_id", "data_generation_id",
        "dataset_sha256", "split_manifest_id", "split_manifest_sha256",
        "development_fold", "model_seed", "wandb_entity", "logger_project",
        "planned_run_name", "train_command",
    )
    return canonical_sha256({field: row.get(field) for field in fields})


def command_options(command):
    tokens = shlex.split(command)
    if tokens[:2] != ["python", "train_wandb_log.py"]:
        raise ValueError("train_command must start with `python train_wandb_log.py`")
    options = {}
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected command token {token!r}")
        key = token[2:]
        if key in options:
            raise ValueError(f"Duplicate command option --{key}")
        index += 1
        values = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        options[key] = values
    return options


def one(options, key):
    values = options.get(key)
    if values is None or len(values) != 1:
        raise ValueError(f"Expected exactly one value for --{key}; got {values!r}")
    return values[0]


def validate_manifest():
    rows = load_rows()
    if not rows:
        raise ValueError("Replay manifest is empty")
    kinds = {"exact_replay", "pre_dedup_calibration"}
    required = {
        "manifest_row", "row_fingerprint", "run_kind", "campaign_id",
        "campaign_stage", "part_slug", "lane_id", "architecture",
        "architecture_slug", "base_config_id", "source_run_ids",
        "data_generation_id", "dataset_path", "dataset_sha256",
        "split_manifest_id", "split_manifest_path", "split_manifest_sha256",
        "development_fold", "model_seed", "artifact_retention",
        "evaluate_test_after_fit", "epoch_eval_splits", "prediction_splits",
        "wandb_entity", "logger_project", "wandb_group", "planned_run_name",
        "train_command",
    }
    seen_numbers = set()
    seen_names = set()
    seen_fingerprints = set()
    kind_counts = {}
    exact_base_ids = set()
    file_hashes = {}
    data_modules = {
        "enhancer": "Lib1EnhancerDataModule",
        "promoter": "Lib1PromoterDataModule",
        "intron": "Lib1IntronDataModule",
        "utr3": "Lib1ThreePrimeDataModule",
        "utr5": "Lib1FivePrimeDataModule",
    }
    for row in rows:
        missing = sorted(field for field in required if field not in row)
        if missing:
            raise ValueError(f"Manifest row is missing required fields: {missing}")
        number = int(row["manifest_row"])
        if number in seen_numbers:
            raise ValueError(f"Duplicate manifest_row {number}")
        seen_numbers.add(number)
        kind = str(row["run_kind"])
        if kind not in kinds:
            raise ValueError(f"Unknown run_kind {kind!r}")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        name = str(row["planned_run_name"])
        if not name or name in seen_names:
            raise ValueError(f"Blank/duplicate planned_run_name {name!r}")
        seen_names.add(name)
        fingerprint = expected_row_fingerprint(row)
        if row["row_fingerprint"] != fingerprint:
            raise ValueError(f"Row {number} fingerprint mismatch")
        if fingerprint in seen_fingerprints:
            raise ValueError(f"Duplicate row fingerprint {fingerprint}")
        seen_fingerprints.add(fingerprint)
        if not row["source_run_ids"]:
            raise ValueError(f"Row {number} has no completed historical source run IDs")
        if row["wandb_entity"] != os.environ["EXPECTED_WANDB_ENTITY"]:
            raise ValueError(f"Row {number} declares a forbidden W&B entity")
        if row["campaign_id"] != "lib1_dedup_phase1_rerun_july2026":
            raise ValueError(f"Row {number} campaign ID mismatch")
        if row["artifact_retention"] != "none" or row["evaluate_test_after_fit"] is not False:
            raise ValueError(f"Row {number} violates retention/test policy")
        if row["epoch_eval_splits"] != ["train", "val"] or row["prediction_splits"] != ["val"]:
            raise ValueError(f"Row {number} exposes a forbidden evaluation split")
        for path_field, hash_field in (
            ("dataset_path", "dataset_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
        ):
            path = Path(row[path_field])
            if not path.is_file():
                raise ValueError(f"Row {number} missing {path_field}: {path}")
            key = str(path.resolve())
            if key not in file_hashes:
                file_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            if file_hashes[key] != row[hash_field]:
                raise ValueError(f"Row {number} {hash_field} does not match {path}")

        expected_data_slug = "dedup_exact_v1" if kind == "exact_replay" else "pre_dedup_v0"
        expected_generation = (
            "lib1_single_part_dedup_exact_v1"
            if kind == "exact_replay"
            else "lib1_single_part_pre_dedup_v0"
        )
        expected_project = (
            f"{row['part_slug']}__bashor_in_house__{expected_data_slug}__scratch__"
            f"{row['architecture_slug']}__exact_replay"
        )
        if row["logger_project"] != expected_project:
            raise ValueError(
                f"Row {number} project mismatch: expected {expected_project!r}, "
                f"found {row['logger_project']!r}"
            )
        if row["data_generation_id"] != expected_generation:
            raise ValueError(f"Row {number} data-generation mismatch")
        if kind == "exact_replay":
            exact_base_ids.add(row["base_config_id"])

        options = command_options(str(row["train_command"]))
        expected_single = {
            "campaign_id": "lib1_dedup_phase1_rerun_july2026",
            "campaign_stage": str(row["campaign_stage"]),
            "data_generation_id": expected_generation,
            "dataset_sha256": str(row["dataset_sha256"]),
            "split_manifest_id": str(row["split_manifest_id"]),
            "split_manifest_sha256": str(row["split_manifest_sha256"]),
            "base_config_id": str(row["base_config_id"]),
            "wandb_entity": os.environ["EXPECTED_WANDB_ENTITY"],
            "logger_project": expected_project,
            "wandb_group": str(row["wandb_group"]),
            "run_name": name,
            "exact_run_name": "true",
            "logger_type": "wandb",
            "data_module": data_modules[str(row["part_slug"])],
            "model_module": str(row["architecture"]),
            "architecture": str(row["architecture"]),
            "graph_module": "CNNBasicTraining",
            "artifact_retention": "none",
            "evaluate_test_after_fit": "false",
            "datafile_path": str(row["dataset_path"]),
            "split_manifest_path": str(row["split_manifest_path"]),
            "expected_data_sha256": str(row["dataset_sha256"]),
            "expected_split_sha256": str(row["split_manifest_sha256"]),
            "development_fold": "0",
            "split_fold": "0",
            "model_seed": "1701",
            "target_column": "log2_RNA_DNA",
            "target_definition": "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)",
            "normalize": "true",
            "test_min_barcodes": "8",
            "train_min_barcodes": "1",
            "use_reverse_complements": "false",
            "barcode_weighting": "false",
            "loss_criterion": "MSELoss",
            "reduction": "mean",
            "loss_mode": "unweighted_mse",
            "precision": "32",
            "enable_progress_bar": "false",
        }
        for key, expected in expected_single.items():
            observed = one(options, key)
            if observed != expected:
                raise ValueError(
                    f"Row {number} --{key} mismatch: expected {expected!r}, "
                    f"found {observed!r}"
                )
        if options.get("epoch_eval_splits") != ["train", "val"]:
            raise ValueError(f"Row {number} must evaluate only train/val each epoch")
        if options.get("prediction_splits") != ["val"]:
            raise ValueError(f"Row {number} must export validation predictions only")
        if options.get("output_names") != ["log2_RNA_DNA"]:
            raise ValueError(f"Row {number} output_names mismatch")
        if options.get("source_run_ids") != [str(value) for value in row["source_run_ids"]]:
            raise ValueError(f"Row {number} source_run_ids command mismatch")

    expected_numbers = set(range(1, len(rows) + 1))
    if seen_numbers != expected_numbers:
        raise ValueError("manifest_row values must be contiguous starting at 1")
    if kind_counts.get("pre_dedup_calibration") != 25 or kind_counts.get("exact_replay", 0) < 1:
        raise ValueError(f"Unexpected manifest kind counts: {kind_counts}")
    if len(exact_base_ids) != kind_counts["exact_replay"]:
        raise ValueError(
            "Exact-replay rows are not unique by base_config_id: "
            f"rows={kind_counts['exact_replay']}, bases={len(exact_base_ids)}"
        )
    print(json.dumps({"validated": len(rows), "by_kind": kind_counts}, sort_keys=True))


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
    number = str(row["manifest_row"])
    marker = status_dir / "done" / f"row_{number}.done"
    if marker.exists():
        fields = {}
        try:
            for line in marker.read_text().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    fields[key] = value
        except OSError:
            fields = {}
        if fields.get("row_fingerprint") == row.get("row_fingerprint"):
            return True
    # Do not infer completion from a run name alone. A registry row is written
    # just before wandb.finish(), and names intentionally omit split hashes;
    # only the atomic, fingerprint-bound marker proves this exact row finished.
    return False


def selected_rows():
    rows = load_rows()
    include_calibration = os.environ.get("INCLUDE_CALIBRATION", "0") == "1"
    selected_parts = {item.lower() for item in tokens("PARTS")}
    selected_lanes = set(tokens("LANE_IDS"))
    selected_bases = set(tokens("BASE_CONFIG_IDS"))
    row_start, row_end = parse_range()
    selected = []
    for row in rows:
        kind = str(row["run_kind"])
        if kind == "pre_dedup_calibration" and not include_calibration:
            continue
        number = int(row["manifest_row"])
        if row_start is not None and number < row_start:
            continue
        if row_end is not None and number > row_end:
            continue
        part_values = {
            str(row.get("part", "")).lower(),
            str(row.get("part_slug", "")).lower(),
        }
        if selected_parts and not (selected_parts & part_values):
            continue
        if selected_lanes and str(row.get("lane_id", "")) not in selected_lanes:
            continue
        if selected_bases and str(row.get("base_config_id", "")) not in selected_bases:
            continue
        if os.environ.get("SKIP_COMPLETED", "1") == "1" and row_completed(row):
            continue
        selected.append(row)

    if os.environ.get("PILOT", "0") == "1":
        per_lane = int(os.environ.get("PILOT_ROWS_PER_LANE", "1"))
        kept = []
        lane_counts = {}
        for row in selected:
            lane = str(row.get("lane_id") or f"{row.get('part_slug')}::{row.get('architecture')}")
            if lane_counts.get(lane, 0) >= per_lane:
                continue
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
            kept.append(row)
        selected = kept

    max_rows = os.environ.get("MAX_ROWS", "").strip()
    if max_rows:
        selected = selected[: int(max_rows)]
    return selected


def row_by_number(number):
    wanted = int(number)
    for row in load_rows():
        if int(row["manifest_row"]) == wanted:
            return row
    raise SystemExit(f"No manifest row {wanted}")


if action == "list":
    for row in selected_rows():
        print(row["manifest_row"])
elif action == "dry":
    fields = (
        "manifest_row", "run_kind", "part_slug", "lane_id", "base_config_id",
        "planned_run_name", "wandb_entity", "logger_project", "task_family",
        "target_family", "comparison_group", "train_command",
    )
    for row in selected_rows():
        print("\t".join(str(row.get(field, "")) for field in fields))
elif action == "command":
    print(row_by_number(extra[0]).get("train_command", ""))
elif action == "field":
    print(row_by_number(extra[0]).get(extra[1], ""))
elif action == "execution":
    row = row_by_number(extra[0])
    fields = (
        "planned_run_name", "wandb_entity", "logger_project", "task_family",
        "target_family", "comparison_group", "row_fingerprint",
        "base_config_id", "train_command",
    )
    print("\t".join(str(row.get(field, "")) for field in fields))
elif action == "completed":
    raise SystemExit(0 if row_completed(row_by_number(extra[0])) else 1)
elif action == "counts":
    selected = selected_rows()
    by_kind = {}
    by_lane = {}
    for row in selected:
        kind = str(row.get("run_kind", "exact_replay"))
        lane = str(row.get("lane_id", ""))
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_lane[lane] = by_lane.get(lane, 0) + 1
    print(json.dumps({"selected": len(selected), "by_kind": by_kind, "by_lane": by_lane}, sort_keys=True))
elif action == "validate":
    validate_manifest()
else:
    raise SystemExit(f"Unknown action: {action}")
PY
}

echo "Validating replay manifest contract"
manifest_helper validate

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
  echo "ERROR: MAX_PARALLEL=${MAX_PARALLEL} must be between 1 and ${#ALL_GPUS[@]}." >&2
  exit 1
fi
GPU_ARRAY=("${ALL_GPUS[@]:0:${MAX_PARALLEL}}")

echo "Lib1 dedup Phase 1 broad exact-config replay"
echo "  campaign: ${CAMPAIGN_ID}"
echo "  stage: ${CAMPAIGN_STAGE}"
echo "  manifest: ${MANIFEST_JSONL}"
echo "  selected: $(manifest_helper counts)"
echo "  entity (forced): ${WANDB_ENTITY}"
echo "  local W&B root: ${WANDB_DIR}/wandb"
echo "  GPU_LIST: ${GPU_ARRAY[*]}"
echo "  MAX_PARALLEL: ${MAX_PARALLEL}"
echo "  INCLUDE_CALIBRATION: ${INCLUDE_CALIBRATION}"
echo "  PILOT: ${PILOT}"
echo "  DRY_RUN: ${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN=1: fixed commands that would execute"
  count=0
  while IFS=$'\t' read -r manifest_row run_kind part_slug lane_id base_config_id planned_run_name wandb_entity logger_project task_family target_family comparison_group train_command; do
    gpu="${GPU_ARRAY[$((count % ${#GPU_ARRAY[@]}))]}"
    count=$((count + 1))
    echo
    echo "Row ${manifest_row}: kind=${run_kind} part=${part_slug} lane=${lane_id} base=${base_config_id} gpu=${gpu}"
    echo "Project: ${wandb_entity}/${logger_project}"
    echo "Run: ${planned_run_name}"
    echo "CUDA_VISIBLE_DEVICES=${gpu} ${train_command}"
  done < <(manifest_helper dry)
  echo
  echo "DRY_RUN selected rows: ${count}"
  exit 0
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "boda_env" ]]; then
  echo "ERROR: activate conda environment boda_env before launching (current: ${CONDA_DEFAULT_ENV:-none})." >&2
  exit 1
fi
python - <<'PY'
import os
import sys
import wandb

entity = os.environ["EXPECTED_WANDB_ENTITY"]
api = wandb.Api(timeout=15)
if not getattr(api, "api_key", None):
    raise SystemExit("No W&B API key resolved. Run `wandb login` locally before launch.")
try:
    # Force an authenticated read scoped to the intended entity.  This does
    # not create a run, but proves the key can resolve that account/team.
    next(iter(api.projects(entity=entity, per_page=1)), None)
except Exception as exc:
    raise SystemExit(f"W&B access preflight failed for entity {entity!r}: {exc}")
print(
    f"W&B client/entity preflight passed for {entity} under Python "
    f"{sys.version.split()[0]}."
)
PY

check_storage() {
  local path="$1" min_free_gb="$2" max_use_pct="$3" label="$4"
  local fields used_pct available_kb available_gb
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
  echo "No rows remain after filters and completion checks."
  exit 0
fi

LAUNCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
CLAIM_DIR="${STATUS_DIR}/claims"
PROCESSED_DIR="${STATUS_DIR}/processed/${LAUNCH_ID}"
DONE_DIR="${STATUS_DIR}/done"
FAIL_DIR="${STATUS_DIR}/failures/${LAUNCH_ID}"
LOG_DIR="${STATUS_DIR}/logs/${LAUNCH_ID}"
MONITOR_FILE="${STATUS_DIR}/monitor.tsv"
STOP_FILE="${STATUS_DIR}/stop.${LAUNCH_ID}"
rm -rf "${CLAIM_DIR}"
mkdir -p "${CLAIM_DIR}" "${PROCESSED_DIR}" "${DONE_DIR}" "${FAIL_DIR}" "${LOG_DIR}"

run_manifest_row() {
  local gpu_id="$1" worker_id="$2" manifest_row="$3"
  local train_command planned_run_name wandb_entity logger_project task_family target_family comparison_group row_fingerprint base_config_id
  local row_log fail_file status claim_path processed_path done_tmp
  IFS=$'\t' read -r planned_run_name wandb_entity logger_project task_family target_family comparison_group row_fingerprint base_config_id train_command \
    < <(manifest_helper execution "${manifest_row}")
  if [[ "${wandb_entity}" != "${EXPECTED_WANDB_ENTITY}" ]]; then
    echo "ERROR: manifest row ${manifest_row} requested forbidden entity ${wandb_entity}." >&2
    return 97
  fi
  row_log="${LOG_DIR}/row_${manifest_row}.log"
  fail_file="${FAIL_DIR}/row_${manifest_row}.fail"
  claim_path="${CLAIM_DIR}/row_${manifest_row}.claim"
  processed_path="${PROCESSED_DIR}/row_${manifest_row}.processed"

  echo "[$(date -Iseconds)] worker=${worker_id} gpu=${gpu_id} start row=${manifest_row} run=${planned_run_name}"
  if (
    cd "${LEARN_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export WANDB_ENTITY="${EXPECTED_WANDB_ENTITY}"
    export BODA_WANDB_ENTITY="${EXPECTED_WANDB_ENTITY}"
    export BODA_WANDB_PROJECT="${logger_project}"
    export BODA_CONFIG_PATH="${MANIFEST_JSONL}"
    export BODA_TASK_FAMILY="${task_family}"
    export BODA_TARGET_FAMILY="${target_family}"
    export BODA_COMPARISON_GROUP="${comparison_group}"
    export BODA_LAUNCH_SCRIPT="launch/lib1_dedup_phase1_exact_replay_orchestrator.sh"
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
      echo "planned_run_name=${planned_run_name}"
      echo "wandb_url=$(sed -n 's/^Resolved W&B run URL: //p' "${row_log}" | tail -n 1)"
      echo "log=${row_log}"
    } >"${done_tmp}"
    mv "${done_tmp}" "${DONE_DIR}/row_${manifest_row}.done"
    mv "${claim_path}" "${processed_path}"
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
    echo "planned_run_name=${planned_run_name}"
    echo "status=${status}"
    echo "log=${row_log}"
  } >"${fail_file}"
  mv "${claim_path}" "${processed_path}"
  echo "[$(date -Iseconds)] worker=${worker_id} gpu=${gpu_id} FAILED row=${manifest_row}; ${row_log}" >&2
  if [[ "${STOP_ON_ERROR}" == "1" ]]; then
    touch "${STOP_FILE}"
  fi
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
    if ! run_manifest_row "${gpu_id}" "${worker_id}" "${claimed}" && [[ "${STOP_ON_ERROR}" == "1" ]]; then
      return 1
    fi
  done
  echo "Worker ${worker_id} GPU ${gpu_id}: stop marker observed"
  return 1
}

monitor_loop() {
  if [[ ! -f "${MONITOR_FILE}" ]]; then
    printf 'timestamp\tlaunch_id\tselected\tdone_total\tactive\tfailed_launch\thome_free_gb\thome_used_pct\troot_free_gb\troot_used_pct\tcampaign_gb\n' >"${MONITOR_FILE}"
  fi
  while true; do
    local done_total active failed home_fields home_free home_used root_fields root_free root_used campaign_kb campaign_gb line
    done_total="$(find "${DONE_DIR}" -maxdepth 1 -type f -name 'row_*.done' | wc -l)"
    active="$(find "${CLAIM_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'row_*.claim' | wc -l)"
    failed="$(find "${FAIL_DIR}" -maxdepth 1 -type f -name '*.fail' | wc -l)"
    home_fields="$(df -Pk /home | awk 'NR==2 {print $4, $5}')"
    read -r home_free home_used <<<"${home_fields}"
    home_free=$((home_free / 1024 / 1024))
    home_used="${home_used%%%}"
    root_fields="$(df -Pk / | awk 'NR==2 {print $4, $5}')"
    read -r root_free root_used <<<"${root_fields}"
    root_free=$((root_free / 1024 / 1024))
    root_used="${root_used%%%}"
    if [[ -d "${LEARN_DIR}/outputs/hpo_runs/${MANIFEST_TAG}" ]]; then
      campaign_kb="$(du -sk "${LEARN_DIR}/outputs/hpo_runs/${MANIFEST_TAG}" | awk '{print $1}')"
    else
      campaign_kb=0
    fi
    campaign_gb=$((campaign_kb / 1024 / 1024))
    line="$(date -Iseconds)\t${LAUNCH_ID}\t${#QUEUE_ROWS[@]}\t${done_total}\t${active}\t${failed}\t${home_free}\t${home_used}\t${root_free}\t${root_used}\t${campaign_gb}"
    printf '%b\n' "${line}" | tee -a "${MONITOR_FILE}"
    if [[ "${CHECK_STORAGE}" == "1" ]] \
        && (( home_free < 150 || home_used >= 80 || root_free < 20 )); then
      {
        echo "storage_stop_at=$(date -Iseconds)"
        echo "home_free_gb=${home_free}"
        echo "home_used_pct=${home_used}"
        echo "root_free_gb=${root_free}"
        echo "root_used_pct=${root_used}"
        echo "campaign_gb=${campaign_gb}"
      } >"${FAIL_DIR}/storage_threshold.fail"
      echo "Storage threshold reached; pausing queue after active rows finish." >&2
      touch "${STOP_FILE}"
      return 1
    fi
    sleep "${MONITOR_INTERVAL}"
  done
}

terminate_process_tree() {
  local parent="$1" child
  while read -r child; do
    [[ -n "${child}" ]] || continue
    terminate_process_tree "${child}"
  done < <(pgrep -P "${parent}" 2>/dev/null || true)
  kill -TERM "${parent}" 2>/dev/null || true
}

pids=()
MONITOR_PID=""
cleanup_background() {
  local pid
  if [[ -n "${MONITOR_PID}" ]]; then
    terminate_process_tree "${MONITOR_PID}"
  fi
  for pid in "${pids[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    terminate_process_tree "${pid}"
  done
  for _ in 1 2 3 4 5; do
    local alive=0
    [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null && alive=1
    for pid in "${pids[@]:-}"; do
      [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done
  [[ -n "${MONITOR_PID}" ]] && kill -KILL "${MONITOR_PID}" 2>/dev/null || true
  for pid in "${pids[@]:-}"; do
    [[ -n "${pid}" ]] && kill -KILL "${pid}" 2>/dev/null || true
  done
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap 'cleanup_background' EXIT

echo "Starting ${#GPU_ARRAY[@]} worker(s) for ${#QUEUE_ROWS[@]} selected row(s)."
monitor_loop &
MONITOR_PID=$!
for index in "${!GPU_ARRAY[@]}"; do
  worker_loop "${GPU_ARRAY[index]}" "$((index + 1))" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
MONITOR_PID=""
pids=()
trap - EXIT INT TERM

if compgen -G "${FAIL_DIR}/*.fail" >/dev/null; then
  status=1
fi
if [[ "${status}" == "0" ]]; then
  echo "All selected Stage 1 rows completed or were already present."
else
  echo "Stage 1 stopped with failures. Inspect ${FAIL_DIR} and ${LOG_DIR}." >&2
fi
echo "Status monitor: ${MONITOR_FILE}"
exit "${status}"
