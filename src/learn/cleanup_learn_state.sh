#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="${SCRIPT_DIR}"

APPLY=0
CLEAN_WANDB=0
CLEAN_OUTPUTS=0
CLEAN_LEGACY_TASK_CACHE=0
CLEAN_BASHOR_CKPTS=0
CLEAN_FAILED_WANDB=0
CLEAN_FAILED_OUTPUTS=0

usage() {
  cat <<'EOF'
Usage:
  bash cleanup_learn_state.sh [options]

Safe by default: dry-run mode unless --apply is provided.

Options:
  --profile enhancer-bashor-reset
      Enables:
        --clean-wandb
        --clean-bashor-checkpoints
      Keeps old UTR/promoter caches by default.

  --profile full-generated-reset
      Enables:
        --clean-wandb
        --clean-outputs
        --clean-legacy-task-cache
        --clean-bashor-checkpoints

  --profile failed-local-state
      Enables:
        --clean-failed-wandb
        --clean-failed-outputs
      Uses `run_registry/runs.csv` plus local W&B metadata to target local
      state that does not appear to have produced a completed registered run.

  --clean-wandb
      Remove local W&B run/sweep cache under src/learn/wandb/.

  --clean-outputs
      Remove generated subdirectories under src/learn/outputs/, preserving outputs/.gitignore.

  --clean-failed-wandb
      Remove local `wandb/run-*` and `wandb/sweep-*` entries whose run/sweep ids
      do not appear in a completed `run_registry/runs.csv` row. Also removes
      `latest-run` / debug symlinks if they point at a failed local run.

  --clean-failed-outputs
      Remove `outputs/...` directories referenced only by failed local W&B runs,
      based on each run cache's `files/config.yaml` `default_root_dir`.
      Conservative heuristic: if any local completed run points at the same
      output root, it is preserved.

  --clean-legacy-task-cache
      Remove legacy top-level task-cache directories (deprecated layout):
        src/learn/utr3_rna_activity_optimization
        src/learn/utr5_hani_rna_activity
        src/learn/promoter_optimization
      Directories are removed and NOT recreated.

  --clean-bashor-checkpoints
      Remove run subdirs under src/learn/bashor_lib1_scratch_basic/, preserving parent directory.

  --apply
      Execute deletion actions (otherwise dry-run only).

  -h, --help
      Show this help.
EOF
}

human_size_bytes() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    du -sb "${path}" 2>/dev/null | awk '{print $1}'
  else
    echo 0
  fi
}

print_human_summary() {
  local bytes="$1"
  python - "$bytes" <<'PY'
import sys
b = int(sys.argv[1])
units = ["B","KB","MB","GB","TB"]
u = 0
v = float(b)
while v >= 1024 and u < len(units)-1:
    v /= 1024.0
    u += 1
print(f"{v:.2f} {units[u]}")
PY
}

declare -a REMOVE_PATHS=()
declare -A SEEN_REMOVE_PATHS=()

queue_remove_path() {
  local path="$1"
  [[ -z "${path}" ]] && return 0
  if [[ -z "${SEEN_REMOVE_PATHS["${path}"]+x}" ]]; then
    REMOVE_PATHS+=("${path}")
    SEEN_REMOVE_PATHS["${path}"]=1
  fi
}

queue_glob_matches() {
  local pattern="$1"
  while IFS= read -r match; do
    [[ -z "${match}" ]] && continue
    queue_remove_path "${match}"
  done < <(compgen -G "${pattern}" || true)
}

queue_failed_local_state() {
  local want_wandb="$1"
  local want_outputs="$2"
  local runs_csv="${LEARN_DIR}/run_registry/runs.csv"

  if [[ ! -f "${runs_csv}" ]]; then
    echo "WARN: ${runs_csv} not found; failed-state cleanup has no registry to consult." >&2
    return 0
  fi

  local tmpfile
  tmpfile="$(mktemp)"
  python - "${LEARN_DIR}" "${want_wandb}" "${want_outputs}" > "${tmpfile}" <<'PY'
import csv
import os
import re
import sys
from pathlib import Path

learn_dir = Path(sys.argv[1])
want_wandb = sys.argv[2] == "1"
want_outputs = sys.argv[3] == "1"
runs_csv = learn_dir / "run_registry" / "runs.csv"
wandb_dir = learn_dir / "wandb"
outputs_dir = learn_dir / "outputs"

keep_run_ids = set()
keep_sweep_ids = set()

with runs_csv.open(newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        if (row.get("status") or "").strip().lower() != "completed":
            continue
        run_id = (row.get("run_id") or "").strip()
        if run_id:
            keep_run_ids.add(run_id)
        sweep_id = (row.get("wandb_sweep_id") or row.get("sweep_id") or "").strip()
        if sweep_id:
            keep_sweep_ids.add(sweep_id)

def clean_scalar(value):
    value = value.strip().strip('"').strip("'")
    return value

def extract_default_root_dir(config_path: Path):
    if not config_path.exists():
        return None
    lines = config_path.read_text(errors="replace").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {"default_root_dir:", "Main args.default_root_dir:"}:
            for subline in lines[idx + 1 : idx + 8]:
                match = re.match(r"\s*value:\s*(.+)\s*$", subline)
                if match:
                    value = clean_scalar(match.group(1))
                    return value or None
    return None

remove_paths = []
output_status = {}
failed_run_dir_names = set()

for run_dir in sorted(wandb_dir.glob("run-*")):
    if not run_dir.is_dir():
        continue
    run_id = run_dir.name.rsplit("-", 1)[-1]
    is_success = run_id in keep_run_ids
    config_path = run_dir / "files" / "config.yaml"
    default_root_dir = extract_default_root_dir(config_path)

    if default_root_dir:
        try:
            output_path = Path(default_root_dir).resolve()
        except Exception:
            output_path = None
        if output_path is not None and str(output_path).startswith(str(outputs_dir.resolve())):
            bucket = output_status.setdefault(str(output_path), {"success": 0, "failed": 0})
            bucket["success" if is_success else "failed"] += 1

    if want_wandb and not is_success:
        remove_paths.append(str(run_dir))
        failed_run_dir_names.add(run_dir.name)

for sweep_dir in sorted(wandb_dir.glob("sweep-*")):
    if not sweep_dir.is_dir():
        continue
    sweep_id = sweep_dir.name.split("sweep-", 1)[-1]
    if want_wandb and sweep_id not in keep_sweep_ids:
        remove_paths.append(str(sweep_dir))

if want_wandb:
    for name in ["latest-run", "debug.log", "debug-internal.log"]:
        p = wandb_dir / name
        if not p.exists() and not p.is_symlink():
            continue
        if p.is_symlink():
            try:
                target_name = os.path.basename(os.readlink(p))
            except OSError:
                target_name = ""
            if target_name in failed_run_dir_names or not p.exists():
                remove_paths.append(str(p))

if want_outputs:
    for output_path, counts in sorted(output_status.items()):
        if counts["failed"] > 0 and counts["success"] == 0:
            remove_paths.append(output_path)

seen = set()
for path in remove_paths:
    if path not in seen:
      print(path)
      seen.add(path)
PY

  while IFS= read -r candidate; do
    [[ -z "${candidate}" ]] && continue
    queue_remove_path "${candidate}"
  done < "${tmpfile}"
  rm -f "${tmpfile}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile="${2:-}"
      if [[ -z "${profile}" ]]; then
        echo "Missing profile name after --profile" >&2
        exit 1
      fi
      case "${profile}" in
        enhancer-bashor-reset)
          CLEAN_WANDB=1
          CLEAN_BASHOR_CKPTS=1
          ;;
        full-generated-reset)
          CLEAN_WANDB=1
          CLEAN_OUTPUTS=1
          CLEAN_LEGACY_TASK_CACHE=1
          CLEAN_BASHOR_CKPTS=1
          ;;
        failed-local-state)
          CLEAN_FAILED_WANDB=1
          CLEAN_FAILED_OUTPUTS=1
          ;;
        *)
          echo "Unknown profile: ${profile}" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --clean-wandb)
      CLEAN_WANDB=1
      shift
      ;;
    --clean-outputs)
      CLEAN_OUTPUTS=1
      shift
      ;;
    --clean-failed-wandb)
      CLEAN_FAILED_WANDB=1
      shift
      ;;
    --clean-failed-outputs)
      CLEAN_FAILED_OUTPUTS=1
      shift
      ;;
    --clean-legacy-task-cache)
      CLEAN_LEGACY_TASK_CACHE=1
      shift
      ;;
    --clean-bashor-checkpoints)
      CLEAN_BASHOR_CKPTS=1
      shift
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${CLEAN_WANDB}" -eq 1 ]]; then
  queue_glob_matches "${LEARN_DIR}/wandb/run-*"
  queue_glob_matches "${LEARN_DIR}/wandb/sweep-*"
  queue_remove_path "${LEARN_DIR}/wandb/latest-run"
  queue_remove_path "${LEARN_DIR}/wandb/debug.log"
  queue_remove_path "${LEARN_DIR}/wandb/debug-internal.log"
fi

if [[ "${CLEAN_OUTPUTS}" -eq 1 ]]; then
  queue_glob_matches "${LEARN_DIR}/outputs/*"
fi

if [[ "${CLEAN_LEGACY_TASK_CACHE}" -eq 1 ]]; then
  queue_remove_path "${LEARN_DIR}/utr3_rna_activity_optimization"
  queue_remove_path "${LEARN_DIR}/utr5_hani_rna_activity"
  queue_remove_path "${LEARN_DIR}/promoter_optimization"
fi

if [[ "${CLEAN_BASHOR_CKPTS}" -eq 1 ]]; then
  queue_glob_matches "${LEARN_DIR}/bashor_lib1_scratch_basic/*"
fi

if [[ "${CLEAN_FAILED_WANDB}" -eq 1 || "${CLEAN_FAILED_OUTPUTS}" -eq 1 ]]; then
  queue_failed_local_state "${CLEAN_FAILED_WANDB}" "${CLEAN_FAILED_OUTPUTS}"
fi

if [[ "${#REMOVE_PATHS[@]}" -eq 0 ]]; then
  echo "No cleanup targets selected. Use --help for options."
  exit 0
fi

echo "Cleanup targets:"
total_bytes=0
for p in "${REMOVE_PATHS[@]}"; do
  b="$(human_size_bytes "${p}")"
  total_bytes=$((total_bytes + b))
  echo "  - ${p} ($(print_human_summary "${b}"))"
done

echo "Estimated reclaimable: $(print_human_summary "${total_bytes}")"

if [[ "${APPLY}" -ne 1 ]]; then
  echo
  echo "Dry-run only. Re-run with --apply to execute."
  exit 0
fi

echo
echo "Applying cleanup..."
for p in "${REMOVE_PATHS[@]}"; do
  rm -rf "${p}"
done

mkdir -p "${LEARN_DIR}/outputs"
if [[ ! -f "${LEARN_DIR}/outputs/.gitignore" ]]; then
  printf '%s\n' '*' > "${LEARN_DIR}/outputs/.gitignore"
fi

echo "Cleanup complete."
