#!/usr/bin/env bash
set -euo pipefail

# Thin entry point for the manifest-locked, resumable Stage 3 weighted-loss
# runner. Preview is the default. A one-row launch requires --execute,
# --pilot-row, and --confirm-pilot; a non-pilot launch additionally requires
# the distinct --confirm-full-campaign acknowledgement.

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

exec "${PYTHON_BIN}" "${LEARN_DIR}/run_lib1_dedup_stage3_campaign.py" "$@"
