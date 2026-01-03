#!/bin/bash

# Configuration
MODEL="openvlthinker"
DATASETS="mathverse,mathvision,hallusionbench,emma-math,emma-code,emma-chem,emma-physics,mmmu-pro-4,mmmu-pro-10,mmmu-pro-vision" # Comma separated list
NUM_GPUS=8

echo "=== Starting Evaluation on $NUM_GPUS GPUs ==="

# Launch shards in parallel
for ((i=0; i<NUM_GPUS; i++)); do
    echo "Launching Shard $i on GPU $i..."
    python evaluation2.0.py \
        --model $MODEL \
        --datasets $DATASETS \
        --cuda $i \
        --num_shards $NUM_GPUS \
        --shard_id $i &
done

# Wait for all evaluation processes to finish
wait
echo "=== Evaluation Completed ==="

# Merge results
echo "=== Merging Shards ==="
python merge_shards.py --dir ./evaluation/outputs