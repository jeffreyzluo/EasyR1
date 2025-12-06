# Documentation

This code is part of an experiment framework. The main entry point for running experiments is the script `qwen2_5_7b_math_grpo.sh`. To conduct your own experiments, you only need to modify a few configuration parameters within this script like trainer.experiment_name, trainer.total_epochs, or data.max_response_length.

Additionally, you need to specify a custom directory for generation logs, you should update the configuration in `ray_trainer.py` at line 772.

For detailed setup and configuration instructions, refer to the main script and the trainer module.