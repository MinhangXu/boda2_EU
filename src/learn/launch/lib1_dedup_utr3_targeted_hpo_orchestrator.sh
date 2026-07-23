#!/usr/bin/env bash
set -euo pipefail

# Thin entry point for the manifest-locked, resumable targeted 3'UTR HPO
# campaign runner. The Python runner defaults to preview mode and requires both
# --execute and --confirm-full-campaign before it starts any training process.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${BODA_PYTHON:-}" ]]; then
  PYTHON_BIN="${BODA_PYTHON}"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "boda_env" ]]; then
  PYTHON_BIN="$(command -v python)"
else
  echo "ERROR: boda_env Python was not found; activate boda_env or set BODA_PYTHON." >&2
  exit 1
fi

exec "${PYTHON_BIN}" \
  "${LEARN_DIR}/run_lib1_dedup_utr3_targeted_hpo_campaign.py" \
  "$@"
