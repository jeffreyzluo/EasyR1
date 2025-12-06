#!/bin/bash

set -x

export MODEL_PATH=JeffreyZLuo/Qwen3-8B-30-Medium-Long # replace it with your local file path
export WANDB_API_KEY="b4f60e0d6c24963e5eb86302706f8ae86c9ad08d"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=ydeng9/OpenVLThinker-grpo-hard@train \
    data.val_files=ydeng9/OpenVLThinker-grpo-hard@test \
    data.test_files=JeffreyZLuo/MathVista-formatted@testmini \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=True \
    trainer.experiment_name=qwen3_grpo_hard_30_epochs_long \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=30
