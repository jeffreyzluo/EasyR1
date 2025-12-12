#!/bin/bash

set -x

export MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct # replace it with your local file path
export WANDB_API_KEY="b4f60e0d6c24963e5eb86302706f8ae86c9ad08d"

python3 -m verl.trainer.main \
    config=examples/config_ctrlg.yaml \
    data.train_files=ydeng9/OpenVLThinker-grpo-medium@train \
    data.val_files=ydeng9/OpenVLThinker-grpo-medium@test \
    data.test_files=JeffreyZLuo/MathVista-formatted@testmini \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=True \
    trainer.experiment_name=debug \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10
