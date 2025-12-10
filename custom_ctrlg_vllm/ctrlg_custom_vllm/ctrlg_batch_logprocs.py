from typing import Any, Optional

import torch

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
)
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from transformers import AutoTokenizer
import ctrlg
import time


class CtrlgBatchLogitsProcessor(LogitsProcessor):
    """Batch-level logits processor for ctrl-g ConstraintLogitsProcessor"""

    def __init__(
        self, 
        vllm_config: VllmConfig, 
        device: torch.device, 
        is_pin_memory: bool,
        hmm_model_path: str,
        tokenizer_path: str,
        min_new_tokens: int,
        max_new_tokens: int,
        alpha: float = 1.0,
        hmm_batch_size:int = 1024,
        constraints_dict: dict = {
            "ConstraintsKey": [[' Keyword_1_Choice_A', ' Keyword_1_Choice_B'], [' Keyword_2_Choice_A']]
            # ConstraintsKey is the key to identify which ConstraintLogitsProcessor to use in the request
            # The value is a list of list of phrases (each inner list is a set of alternative phrases)
            # The example constraints will enforce ((Keyword_1_Choice_A || Keyword_1_Choice_B) && (Keyword_2_Choice_A))
        },
        soft_constraints: bool = False # If True, don't build the EOS builder
    ):
        # Store request info for batch processing
        self.req_info: dict[int, dict] = {}
        
        # Init hmm model and default dfa model
        self.device = device
        
        # Custom Config (now parameterized)
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens
        self.alpha = alpha
        self.hmm_batch_size = hmm_batch_size
        self.soft_constraints = soft_constraints
        
        # Init hmm and tokenizer
        self.hmm_model = ctrlg.HMM.from_pretrained(hmm_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.vocab_size = self.hmm_model.vocab_size
        self.eos_token_id = self.hmm_model.eos_token_id
        self.constraints_dict = constraints_dict
        # Init custom ConstraintLogitsProcessor
        self.custom_clp_dict = self.get_custom_clp_dict(self.vocab_size, self.eos_token_id, self.constraints_dict)
        print("Custom ConstraintLogitsProcessor: ", self.custom_clp_dict.keys())

    def construct_dfa_kw(self, phrases_list, vocab_size, eos_token_id):
        # Basic Builders
        ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)
        eos_builder = ctrlg.EOSBuilder(vocab_size, eos_token_id)
        # Build Keyphrases 
        dfa_graphs = []
        for keyphrase in phrases_list:
            patterns = [self.tokenizer.encode(x) for x in keyphrase]
            dfa_graphs.append(ac_builder.build(patterns))
        dfa_graphs = [ctrlg.DFA_concatenate(dfa_graphs)] # concatenate the patterns so they appear in
        
        if not self.soft_constraints:
            dfa_graphs.append(eos_builder.build())

        dfa_graphs = ctrlg.DFA_prod(dfa_graphs, mode='intersection')
        dfa_model = ctrlg.DFAModel(dfa_graphs, vocab_size)
        
        return dfa_model

    def get_custom_clp_dict(self, vocab_size, eos_token_id, constraints_dict):
        # Return a dictionary of ConstraintLogitsProcessor
        clp_dict = {}
        
        for clp_key, phrases_list in constraints_dict.items():
            start_time = time.time()

            dfa_model = self.construct_dfa_kw(phrases_list, vocab_size, eos_token_id).to(self.device)
            clp = ctrlg.ConstraintLogitsProcessor(
                self.hmm_model, dfa_model, self.min_new_tokens, self.max_new_tokens,
                prompt_ids=None, prefix_ids=[], suffix_ids=[], alpha=self.alpha, hmm_batch_size=self.hmm_batch_size
            )
            clp_dict[clp_key] = clp

            elapsed = time.time() - start_time
            print(
                f"⚙️ Initialized ConstraintLogitsProcessor for key: {clp_key} "
                f"(states: {dfa_model.num_states}, edges: {dfa_model.VE_mask.shape[1]}) "
                f"in {elapsed:.4f} seconds"
            )
        return clp_dict

    def clear_cache_for_all(self):
        print("🧹 Clearing cache for all ConstraintLogitsProcessors...")
        # Get GPU memory usage before clearing cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(self.device) / (1024**3)  # Convert to GB
            memory_reserved_before = torch.cuda.memory_reserved(self.device) / (1024**3)  # Convert to GB
            print(f"🔍 GPU Memory before clearing cache: {memory_before:.2f} GB allocated, {memory_reserved_before:.2f} GB reserved")
        
        # for clp in self.custom_clp_dict.values():
            # clp._cleanup_cache(prefixes = []) # Clear all cache
        del self.custom_clp_dict
        self.custom_clp_dict = None
        
        # Get GPU memory usage after clearing cache and force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear PyTorch's cache
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated(self.device) / (1024**3)  # Convert to GB
            memory_reserved_after = torch.cuda.memory_reserved(self.device) / (1024**3)  # Convert to GB
            
            memory_freed = memory_before - memory_after
            memory_reserved_freed = memory_reserved_before - memory_reserved_after
            
            print(f"🧹 GPU Memory after clearing cache: {memory_after:.2f} GB allocated, {memory_reserved_after:.2f} GB reserved")
            print(f"💾 Memory freed: {memory_freed:.2f} GB allocated, {memory_reserved_freed:.2f} GB reserved")
        else:
            print("🚫 CUDA not available, cannot monitor GPU memory")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        process_dict_updates(
            self.req_info,
            batch_update,
            # This function returns the LP's per-request state based on the
            # request details, or None if this LP does not apply to the request.
            lambda params, prompt_ids, output_ids: self._extract_request_state(params, prompt_ids, output_ids),
        )

    def _extract_request_state(self, params, prompt_ids, output_ids):
        """Extract per-request state for constraint processing"""
        clp_id = params.extra_args and params.extra_args.get("clp_id", None)
        clean_up = params.extra_args and params.extra_args.get("clean_up", False)
        if (not clean_up) and (clp_id is None or clp_id not in self.constraints_dict):
            return None
        
        # Return state needed for constraint processing
        return {
            'clp_id': clp_id,
            'clean_up': clean_up,
            'prompt_ids': prompt_ids,
            'output_ids': output_ids,
            'prefix_ids': [],  # Can be customized per request if needed
            'suffix_ids': [],  # Can be customized per request if needed
        }

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        # Process requests that need constraint application
        batch_size = logits.shape[0]
        logits = torch.log_softmax(logits, dim=-1)
        
        # Group requests by constraint type for efficient batch processing
        constraint_groups = {}
        for req_idx, req_state in self.req_info.items():
            # Clear cache signal
            if req_state.get('clean_up', False):
                self.clear_cache_for_all()
                continue

            if req_idx >= batch_size:
                continue
            
            clp_id = req_state['clp_id']
            output_ids = req_state['output_ids']
            
            # Skip if we've generated too many tokens
            if len(output_ids) >= self.max_new_tokens:
                continue
            if self.custom_clp_dict is None:
                # Being deleted previously, re-initialize
                self.custom_clp_dict = self.get_custom_clp_dict(self.vocab_size, self.eos_token_id, self.constraints_dict)
                print("Custom ConstraintLogitsProcessor: ", self.custom_clp_dict.keys())
            if clp_id not in constraint_groups:
                constraint_groups[clp_id] = []
            constraint_groups[clp_id].append((req_idx, req_state))

        # print("🐞 Constraint groups to process: ", {k: len(v) for k, v in constraint_groups.items()})
        
        # Process each constraint group
        for clp_id, req_group in constraint_groups.items():
            if not req_group:
                continue
                
            # Get the pre-initialized ConstraintLogitsProcessor for this constraint type
            constraint_processor = self.custom_clp_dict[clp_id]
            
            # Extract data for batch processing - following original code pattern
            req_indices = []
            prefixes = []
            
            for req_idx, req_state in req_group:
                prefix_ids = req_state.get('prefix_ids', [])
                output_ids = req_state['output_ids']
                
                req_indices.append(req_idx)
                prefixes.append(tuple(prefix_ids + output_ids))

            # The original implementation in ctrlg will check if the last token is eos_token_id
            # If so, it will skip applying the constraint for that request
            # However, in vllm batch processing, it already handles finished requests
            # So here we don't need to check for eos_token_id
            # Instead, we can directly process all requests in the group
            selected_idx = [i for i, _ in enumerate(prefixes)]

            if not selected_idx:
                continue

            # Get selected prefixes and token ranges following original pattern
            selected_prefixes = [prefixes[i] for i in selected_idx]
            selected_req_indices = [req_indices[i] for i in selected_idx]
            
            # Handle token_ranges following original code logic
            if len(constraint_processor.token_ranges) == 1:
                selected_token_ranges = [constraint_processor.token_ranges[0] for _ in selected_idx]
            else:
                selected_token_ranges = [constraint_processor.token_ranges[i] for i in selected_idx]

            # Compute HMM logits for this batch
            hmm_batch_size = len(selected_idx) if constraint_processor.hmm_batch_size is None else min(len(selected_idx), constraint_processor.hmm_batch_size)
            
            # print("🐞 Hmm batch size: ", hmm_batch_size, " for constraint: ", clp_id, " with ", len(selected_idx), " requests")
            # print("🐞 len(selected_prefixes[0]):", len(selected_prefixes[0]))

            hmm_logits, hmm_logits_ = constraint_processor.compute_logits(
                selected_prefixes, selected_token_ranges, hmm_batch_size
            )
            hmm_logits -= hmm_logits_

            # Handle vocabulary size mismatch
            if hmm_logits.shape[1] < logits.shape[1]:
                neginf = torch.full(
                    (hmm_logits.shape[0], logits.shape[1] - hmm_logits.shape[1]), 
                    -1e30, 
                    device=hmm_logits.device
                )
                hmm_logits = torch.cat((hmm_logits, neginf), dim=1)

            
            # Apply HMM logits to the corresponding batch positions
            for i, req_idx in enumerate(selected_req_indices):
                if req_idx < logits.shape[0]:
                    logits[req_idx, :] += self.alpha * hmm_logits[i, :]

        logits = torch.log_softmax(logits, dim=-1)
        return logits