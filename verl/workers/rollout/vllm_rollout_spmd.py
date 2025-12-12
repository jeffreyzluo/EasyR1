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

import os
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from omegaconf import OmegaConf

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if hasattr(config, "engine_kwargs") and hasattr(config.engine_kwargs, "vllm"):
            from dataclasses import asdict
            vllm_kwargs = asdict(config.engine_kwargs.vllm)
            # Remove None values to use vllm defaults
            engine_kwargs = {k: v for k, v in vllm_kwargs.items() if v is not None}
        
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
            "logprobs": 0,
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        # Initialize custom rollout configurations
        if hasattr(config, 'custom_rollout_flag') and config.custom_rollout_flag is not None:
            print(f"🚩 =>>custom_rollout_flag: {config.custom_rollout_flag}")
            if config.custom_rollout_flag == "ctrlg":
                # Ctrl-G doesn't require loading reasoning modules
                if not hasattr(config, 'custom_rollout_args'):
                    raise ValueError("custom_rollout_args is missing in config. Please add custom_rollout_args with ctrlg_reasoning_type_list parameter.")
                if not hasattr(config.custom_rollout_args, 'ctrlg_reasoning_type_list'):
                    raise ValueError("ctrlg_reasoning_type_list is missing in custom_rollout_args. Please specify ctrlg_reasoning_type_list parameter.")
                if not config.custom_rollout_args.ctrlg_reasoning_type_list:
                    raise ValueError("ctrlg_reasoning_type_list is empty. Please provide at least one reasoning type.")
                print(f"📝 =>>ctrlg_reasoning_type_list: {config.custom_rollout_args.ctrlg_reasoning_type_list}")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        
        # Meta data
        is_validate = prompts.meta_info.get("validate", False)
        calculate_log_probs = getattr(self.config, "calculate_log_probs", False) and not is_validate
        
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]        
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
    
            # To align with verl:0.5.0 implementation, we expand the inputs when when sampling_params.n > 1.
            if self.sampling_params.n > 1:
                expanded_vllm_inputs = []
                for vllm_input in vllm_inputs:
                    for _ in range(self.sampling_params.n):
                        expanded_vllm_inputs.append(deepcopy(vllm_input))
                vllm_inputs = expanded_vllm_inputs
            # Make a copy of the original sampling params 
            use_sampling_params = deepcopy(self.sampling_params)
            use_sampling_params.n = 1  # we handle n ourselves below
            
            # Handle custom rollout configurations
            
            sampling_params_list = []
            if hasattr(self.config, 'custom_rollout_flag') and self.config.custom_rollout_flag == "ctrlg" and not is_validate:
                # Initialize local random generator to incorporate randomness in rollout across devices
                local_rng = random.Random(os.urandom(16))
                # Get basic configs for exploration
                batch_exp_r = getattr(self.config.custom_rollout_args, "batch_exp_r", None)
                exp_r = getattr(self.config.custom_rollout_args, "exp_r", 0.0)
                
                # Compute exploration parameters
                exp_n = int(self.sampling_params.n * exp_r)
                non_exp_n = self.sampling_params.n - exp_n
                print(f"🅰️ =>>self.sampling_params.n: {self.sampling_params.n}, non_exp_n: {non_exp_n}, exp_n: {exp_n}")
                
                # Set rm_type_idx for batch-level mixing
                rm_type_idx = random.randrange(0, len(self.config.custom_rollout_args.ctrlg_reasoning_type_list))
                
                # Process each data group
                for data_idx in range(0, len(vllm_inputs), self.sampling_params.n):
                    vllm_input_group = vllm_inputs[data_idx: data_idx + self.sampling_params.n]
                    assert all(x == vllm_input_group[0] for x in vllm_input_group), "Error: vllm_input_group should be the same for all items in the group"
                    
                    # Determine if this batch does exploration
                    rand_num = local_rng.random()
                    if rand_num < batch_exp_r:
                        do_exp = True
                    else:
                        do_exp = False
                    
                    # Update rm_type_idx for data-level mixing
                    if self.config.custom_rollout_args.mix_constraint_types == "data":
                        # If mix constraints type across data, sample new rm_type_idx for each data group
                        rm_type_idx = local_rng.randrange(0, len(self.config.custom_rollout_args.ctrlg_reasoning_type_list))
                    
                    for i in range(len(vllm_input_group)):
                        if i < non_exp_n or not do_exp:
                            # Non-exploration sample
                            sampling_params_list.append(deepcopy(use_sampling_params))
                        else:
                            # Exploration sample with ctrl-g
                            if self.config.custom_rollout_args.mix_constraint_types == "rollout":
                                rm_type_idx = local_rng.randrange(0, len(self.config.custom_rollout_args.ctrlg_reasoning_type_list))
                            
                            rm_type = self.config.custom_rollout_args.ctrlg_reasoning_type_list[rm_type_idx]
                            sp = deepcopy(use_sampling_params)
                            setattr(sp, 'extra_args', {"clp_id": rm_type})
                            sampling_params_list.append(sp)
            send_sampling_params = sampling_params_list if len(sampling_params_list) > 0 else use_sampling_params

            # Start generation
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=send_sampling_params, use_tqdm=self.use_tqdm
            )
            
            # Cleanup ctrlg logits processor for more memory during update_actor
            if self.config.custom_rollout_flag in ["ctrlg"] and not is_validate:
                cleanup_sp = deepcopy(use_sampling_params)
                setattr(cleanup_sp, 'extra_args', {"clean_up": True})
                setattr(cleanup_sp, 'max_tokens', 1) # only need to generate 1 token for cleanup
                _cleanup_output = self.inference_engine.generate(
                    prompts=vllm_inputs[:1],  # dummy call to warm up
                    sampling_params=cleanup_sp,
                    use_tqdm=False,
                )
                del _cleanup_output
            
            
            
            response_ids = []
            rollout_log_probs = []
            for completion in completions:
                for sample_id in range(len(completion.outputs)):
                    response_tokens = completion.outputs[sample_id].token_ids
                    response_ids.append(response_tokens)
                    if calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(completion.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_tokens[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)
            
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)
            
            if calculate_log_probs:
                rollout_log_probs = VF.pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(input_ids.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}
        
        if calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
