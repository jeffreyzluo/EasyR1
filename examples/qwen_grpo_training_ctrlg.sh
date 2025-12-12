#!/bin/bash

set -x

export MODEL_PATH="JeffreyZLuo/OpenVLThinker-Medium-15" # replace it with your local file path
export WANDB_API_KEY="b4f60e0d6c24963e5eb86302706f8ae86c9ad08d"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "Number of GPUs: $NUM_GPU"

# ----------------- ctrlg config -----------------
export exp_r=1.0  # exploration rate for a single example rollout
export batch_exp_r=1.0  # exploration rate for a full rollout batch to explore. 0.5 means there's 50% chance to explore
export mix_constraint_types="rollout"  # ["batch", "data", "rollout"], each means that the reasoning type is mixed "under" this level.
export ctrlg_reasoning_type_list='["SymbolVerification","GeometricGrounding","VisualReinspection","Backwarding","Backtracking","Induction","Counterfactual","OverthinkingAwareness"]'  # a list for ctrlg reasoning type
export ctrlg_variant="Qwen25VLBaseCtrlgProcessorV0"  # specify the ctrlg variant here

export NCCL_TIMEOUT=12000

# Construct the ctrlg args
CTRLG_ARGS="worker.rollout.custom_rollout_flag=ctrlg \
worker.rollout.custom_rollout_args.exp_r=${exp_r} \
worker.rollout.custom_rollout_args.batch_exp_r=${batch_exp_r} \
worker.rollout.custom_rollout_args.mix_constraint_types=${mix_constraint_types} \
worker.rollout.custom_rollout_args.ctrlg_reasoning_type_list=${ctrlg_reasoning_type_list} \
worker.rollout.engine_kwargs.vllm.logits_processors=[\"ctrlg_custom_vllm:${ctrlg_variant}\"]
"

python3 -m verl.trainer.main \
    config=examples/config_ctrlg.yaml \
    data.train_files=ydeng9/OpenVLThinker-grpo-hard@train \
    data.val_files=ydeng9/OpenVLThinker-grpo-hard@test \
    data.test_files=JeffreyZLuo/MathVista-formatted@testmini \
    data.max_response_length=2048 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=True \
    ${CTRLG_ARGS} \
    trainer.experiment_name=qwen2.5_7B_medium_ctrlg_hard_V1 \
    trainer.n_gpus_per_node=$NUM_GPU \
    trainer.load_checkpoint_path=/home/jeffrey/EasyR1/checkpoints/baselines/qwen2.5_7B_medium_ctrlg_hard_V1/global_step_330\
    trainer.total_epochs=45
