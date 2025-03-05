#!/bin/bash

# Configuration
SWEEP_ID=minhangxu1998-baylor-college-of-medicine/boda2_EU-src/8tr3pavm
NUM_AGENTS=8     # Start with 1 for initial testing
NUM_RUNS=3       # Start with 1 for initial testing
GPU_LIST=(0 1 2 3 4 5 6 7)     # Start with just GPU 0

# Create output directories
mkdir -p /home/minhang/synBio_AL/boda2/src/local_artifacts/promoter/sweep/

# Run the agents
for ((i=0; i<NUM_AGENTS; i++)); do
  # Assign GPU from the list (cycling if needed)
  GPU_ID=${GPU_LIST[i % ${#GPU_LIST[@]}]}
  
  echo "Starting agent $i on GPU $GPU_ID to run $NUM_RUNS trials"
  CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent --count $NUM_RUNS $SWEEP_ID &
  
  # Small delay between launching agents
  sleep 2
done

echo "All agents launched."
echo "Press Ctrl+C to stop all agents"

# Wait for all background processes
wait
echo "All agents completed"
