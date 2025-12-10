from .ctrlg_logprocs import CtrlgWrappedLogitsProcessor
from .ctrlg_batch_logprocs import CtrlgBatchLogitsProcessor
from vllm.config import VllmConfig
from .keyphrases.v0 import CONSTRAINTS_DICT as CONSTRAINTS_V0
import torch

class Qwen25VLBaseCtrlgProcessorV0(CtrlgBatchLogitsProcessor):
    """Qwen2.5-VL specific configuration"""
    
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            is_pin_memory=is_pin_memory,
            hmm_model_path='billkunghappy/hmm_Qwen2.5-7B-15-Medium2Hard_openvlthinker_hard_boxed_4096-Step2000',
            tokenizer_path='Qwen/Qwen2.5-7B-Instruct',
            min_new_tokens=1,
            max_new_tokens=int(1024*1.5),
            alpha=2.0, # Change to 1.0 for v0
            constraints_dict=CONSTRAINTS_V0,
            soft_constraints=True
        )
        print("⚙️ Initialized Qwen25VLBaseCtrlgProcessorV0")
