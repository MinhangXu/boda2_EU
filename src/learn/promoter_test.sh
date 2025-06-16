#!/bin/bash
# Create output directories
mkdir -p /home/minhang/synBio_AL/boda2/src/local_artifacts/promoter/sweep/

# For testing: Just run one trial on one GPU
echo "Starting a single test run on GPU 0"
CUDA_VISIBLE_DEVICES=0 wandb agent --count 1 minhangxu1998-baylor-college-of-medicine/boda2_EU-src/2ymklui6

echo "Test run completed."
