#!/bin/bash

# Number of agents to run (e.g., one per GPU)
num_agents=8  # Modify based on available GPUs
sweep_id='your-sweep-id-here'  # Replace with your actual sweep ID from Wandb

# Number of runs (trials) each agent should execute
num_runs=8

# Loop over each agent and assign a GPU.
for ((i=0; i<num_agents; i++)); do
  gpu_id=$(( i % $(nvidia-smi --list-gpus | wc -l) ))
  echo "Launching agent $i on GPU $gpu_id"
  
  # Launch the agent with the appropriate GPU.
  CUDA_VISIBLE_DEVICES=$gpu_id wandb agent --count $num_runs $sweep_id &
  
  sleep 1  # Optional delay between launching agents
done

wait
echo "All agents finished."
