#!/bin/bash
set -euo pipefail

# Wrapper for the Lib1 exact n=1 barcode-bin downsampling manifest.
#
# Dry run:
#   cd /home/minhang/synBio_AL/boda2_EU
#   GENERATE_MANIFEST=1 DRY_RUN=1 MAX_ROWS=5 GPU_LIST="0" \
#     bash src/learn/launch/lib1_inhouse_barcode_bin_n1_downsampling_orchestrator.sh
#
# Full run:
#   cd /home/minhang/synBio_AL/boda2_EU
#   GENERATE_MANIFEST=1 GPU_LIST="0 1 2 3 4 5 6 7" \
#     bash src/learn/launch/lib1_inhouse_barcode_bin_n1_downsampling_orchestrator.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST_TAG="${MANIFEST_TAG:-lib1_barcode_bin_n1_downsample_june2026}"
MANIFEST_JSONL="${MANIFEST_JSONL:-${LEARN_DIR}/outputs/hpo_manifests/${MANIFEST_TAG}__run_manifest.jsonl}"
STATUS_DIR="${STATUS_DIR:-${LEARN_DIR}/outputs/hpo_runs/status/${MANIFEST_TAG}}"
LAUNCH_NOTES="${LAUNCH_NOTES:-${MANIFEST_TAG}}"
GENERATE_MANIFEST="${GENERATE_MANIFEST:-0}"
INCLUDE_FULL_N1="${INCLUDE_FULL_N1:-0}"

export MANIFEST_TAG MANIFEST_JSONL STATUS_DIR LAUNCH_NOTES

if [[ "${GENERATE_MANIFEST}" == "1" || ! -f "${MANIFEST_JSONL}" ]]; then
  echo "Generating exact n=1 barcode-bin downsampling manifest: ${MANIFEST_TAG}"
  (
    cd "${LEARN_DIR}"
    args=(--manifest-tag "${MANIFEST_TAG}")
    if [[ "${INCLUDE_FULL_N1}" == "1" ]]; then
      args+=(--include-full)
    fi
    python generate_lib1_barcode_bin_n1_downsampling_manifest.py "${args[@]}"
  )
fi

if [[ ! -f "${MANIFEST_JSONL}" ]]; then
  echo "ERROR: manifest JSONL not found after generation step: ${MANIFEST_JSONL}" >&2
  exit 1
fi

exec bash "${SCRIPT_DIR}/lib1_inhouse_outer_seed_prior_orchestrator.sh"
