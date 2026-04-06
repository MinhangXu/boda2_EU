#!/bin/bash
set -euo pipefail

# This legacy helper attaches agents to an already-created sweep.
# Prefer the curated launchers in `launch/`, which create sweeps with an
# explicit W&B entity/project and record the resolved sweep path in
# `run_registry/sweep_launches.csv`.

# Required: set a full sweep path such as
# `minhangxu1998-baylor-college-of-medicine/boda2_EU-src_learn/qbj4v71s`.
SWEEP_ID="${SWEEP_ID:-}"
NUM_AGENTS="${NUM_AGENTS:-7}"
NUM_RUNS="${NUM_RUNS:-6}"
read -r -a GPU_LIST <<< "${GPU_LIST:-0 1 2 3 4 5 6 7}"

if [[ -z "${SWEEP_ID}" || "${SWEEP_ID}" != */*/* ]]; then
  echo "Set SWEEP_ID to a full sweep path: entity/project/sweep_id" >&2
  exit 1
fi

# Create output directories (using path relative to the current 'learn' dir)
mkdir -p local_artifacts/promoter/sweep/sept10_sweep/

# Run the agents
for ((i=0; i<NUM_AGENTS; i++)); do
  # Assign GPU from the list
  GPU_ID=${GPU_LIST[i % ${#GPU_LIST[@]}]}
  
  echo "Starting agent $i on GPU $GPU_ID to run $NUM_RUNS trials"
  # This command works because the agent is run from src/learn/
  # and the YAML's program path is also relative to src/learn/
  CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent --count $NUM_RUNS $SWEEP_ID &
  
  sleep 2
done

echo "All agents launched."
wait
echo "All agents completed"
