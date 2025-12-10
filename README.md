# Documentation

The main entry point for running experiments is the script `examples/qwen2_5_7b_math_grpo.sh`. To conduct the different experiments, you only need to modify a few configuration parameters within this script like trainer.experiment_name, trainer.total_epochs, or data.max_response_length.

Previous experiments config:

Qwen2.5-Medium: 15 epochs, default response length (2048), medium dataset

Qwen2.5-Hard: 30 epochs, default response length (2048), hard dataset

Qwen3-Medium: 30 epochs, long response length (4096), medium dataset

Qwen3-Hard: 30 epochs, long response length (4096), hard dataset

Additionally, you need to specify a custom directory for generation logs, you should update the configuration in `verl/trainer/ray_trainer.py` at line 772.

For detailed setup and configuration instructions, refer to the main script and the trainer module.

Install the ctrlg_for_vllm
```
export ctrlg_url="https://github.com/billkunghappy/ctrlg_for_vllm.git"
export ctrlg_dir="ctrlg_for_vllm"
echo "Installing ctrlg from ${ctrlg_url}"
if [ ! -d "${ctrlg_dir}" ]; then
    git clone ${ctrlg_url}
else
    echo "Repo ${ctrlg_dir} already exists. Skipping clone."
fi
cd ${ctrlg_dir}
pip install -e .
```