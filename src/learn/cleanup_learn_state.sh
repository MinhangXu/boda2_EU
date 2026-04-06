#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="${SCRIPT_DIR}"

APPLY=0
CLEAN_WANDB=0
CLEAN_OUTPUTS=0
CLEAN_LEGACY_TASK_CACHE=0
CLEAN_BASHOR_CKPTS=0

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

  --clean-wandb
      Remove local W&B run/sweep cache under src/learn/wandb/.

  --clean-outputs
      Remove generated subdirectories under src/learn/outputs/, preserving outputs/.gitignore.

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
  REMOVE_PATHS+=("${LEARN_DIR}/wandb/run-*")
  REMOVE_PATHS+=("${LEARN_DIR}/wandb/sweep-*")
  REMOVE_PATHS+=("${LEARN_DIR}/wandb/latest-run")
  REMOVE_PATHS+=("${LEARN_DIR}/wandb/debug.log")
  REMOVE_PATHS+=("${LEARN_DIR}/wandb/debug-internal.log")
fi

if [[ "${CLEAN_OUTPUTS}" -eq 1 ]]; then
  while IFS= read -r d; do
    [[ -z "${d}" ]] && continue
    REMOVE_PATHS+=("${d}")
  done < <(compgen -G "${LEARN_DIR}/outputs/*" || true)
fi

if [[ "${CLEAN_LEGACY_TASK_CACHE}" -eq 1 ]]; then
  REMOVE_PATHS+=("${LEARN_DIR}/utr3_rna_activity_optimization")
  REMOVE_PATHS+=("${LEARN_DIR}/utr5_hani_rna_activity")
  REMOVE_PATHS+=("${LEARN_DIR}/promoter_optimization")
fi

if [[ "${CLEAN_BASHOR_CKPTS}" -eq 1 ]]; then
  while IFS= read -r d; do
    [[ -z "${d}" ]] && continue
    REMOVE_PATHS+=("${d}")
  done < <(compgen -G "${LEARN_DIR}/bashor_lib1_scratch_basic/*" || true)
fi

if [[ "${#REMOVE_PATHS[@]}" -eq 0 ]]; then
  echo "No cleanup targets selected. Use --help for options."
  exit 0
fi

echo "Cleanup targets:"
total_bytes=0
for p in "${REMOVE_PATHS[@]}"; do
  matched=0
  for m in ${p}; do
    matched=1
    b="$(human_size_bytes "${m}")"
    total_bytes=$((total_bytes + b))
    echo "  - ${m} ($(print_human_summary "${b}"))"
  done
  if [[ "${matched}" -eq 0 ]]; then
    echo "  - ${p} (no matches)"
  fi
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
  for m in ${p}; do
    rm -rf "${m}"
  done
done

mkdir -p "${LEARN_DIR}/outputs"
if [[ ! -f "${LEARN_DIR}/outputs/.gitignore" ]]; then
  printf '%s\n' '*' > "${LEARN_DIR}/outputs/.gitignore"
fi

echo "Cleanup complete."
