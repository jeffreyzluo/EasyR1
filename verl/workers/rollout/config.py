# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class CustomRolloutArgs:
    exp_r: float = 0.0  # exploration rate for rollout, 0.0 means no exploration
    batch_exp_r: float = 1.0  # exploration rate for a full rollout batch to explore
    mix_constraint_types: str = "batch"  # ["batch", "data", "rollout"]
    ctrlg_reasoning_type_list: Optional[list[str]] = None  # list for ctrlg reasoning type


@dataclass
class EngineKwargs:
    logits_processors: Optional[Any] = None
    logprobs_mode: str = "processed_logprobs"


@dataclass
class VllmEngineKwargs:
    vllm: EngineKwargs = field(default_factory=EngineKwargs)


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 10240
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)
    calculate_log_probs: bool = False  # whether to calculate log probs of the rollout responses
    custom_rollout_flag: Optional[str] = None  # flag for custom rollout method
    custom_rollout_args: CustomRolloutArgs = field(default_factory=CustomRolloutArgs)
    engine_kwargs: VllmEngineKwargs = field(default_factory=VllmEngineKwargs)
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
