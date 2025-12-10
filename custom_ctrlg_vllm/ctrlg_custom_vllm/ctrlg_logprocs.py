# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""This example demonstrates wrapping a request-level logits processor to be
compatible with vLLM's batch-level logits processing

For demo purposes, a dummy logits processor is employed which, if
`target_token` is passed as a keyword argument to `SamplingParams.extra_args`,
will mask out all tokens except `target_token`. This logits processor can be
applied to a vector of logits associated with a single decode step for a single
request. The logits processor cannot be applied to a request which does not
pass in a `target_token` custom argument.

The request-level dummy logits processor is wrapped to create a batch-level
logits processor, which can apply the logits processor to output logits from
all requests in the persistent batch in a given decode step. For requests which
do not provide a `target_token` argument, the corresponding row of `logits`
will not be modified.

A batch is constructed with `temperature=0.0` and 50% of requests specifying
`target_token`, and for these requests - and *only* these requests - we
expect the `target_token` to be decoded in each step, yielding an output
similar to that shown below:

Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    " ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '"
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    " not a racist. He is a racist.\nHe's a racist because he"
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' also also also also also also also also also also also also also
             also also also'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    ' in the hands of the people.\n\nThe future of AI is in the'
------------------------------------------------------------
"""

from typing import Any, Optional

import torch

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from transformers import AutoTokenizer
import ctrlg


class CtrlgPerReqLogitsProcessor(ctrlg.ConstraintLogitsProcessor):
    """The request-level logits processor for ctrl-g ConstraintLogitsProcessor"""

    def __init__(self, hmm_model, dfa_model, min_new_tokens, max_new_tokens, 
                 prompt_ids, prefix_ids=[], suffix_ids=[], temperature=1.0, 
                 alpha=1.0, token_ranges=None, hmm_batch_size=None):
        """Initialize the CtrlgPerReqLogitsProcessor.
        
        Args:
            hmm_model: The HMM model for constraint processing
            dfa_model: The DFA model for constraint processing  
            min_new_tokens: Minimum number of new tokens to generate
            max_new_tokens: Maximum number of new tokens to generate
            prompt_ids: List of prompt token IDs. Can be send as None. (Don't need it here)
            prefix_ids: List of prefix token IDs (default: [])
            suffix_ids: List of suffix token IDs (default: [])
            temperature: Sampling temperature (default: 1.0)
            alpha: HMM weight parameter (default: 1.0)
            token_ranges: Token range constraints (default: None)
            hmm_batch_size: Batch size for HMM processing (default: None)
        """
        # Pass all arguments directly to parent class without modification
        super().__init__(hmm_model, dfa_model, min_new_tokens, max_new_tokens,
                         prompt_ids, prefix_ids, suffix_ids, temperature, 
                         alpha, token_ranges, hmm_batch_size)
        # Store the token ranges
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens


    def __call__(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # When len(output_ids) > self.max_new_tokens, don't apply the logits processor
        # This is to avoid applying the logits processor when the generation is not finished
        # print(len(output_ids))
        if len(output_ids) >= self.max_new_tokens:
            return logits

        # In the original __call__, it handels the output_ids and logits in batch level.
        # For simplicity, we add a batch dimension and handle it in batch level and recover it when return the logits
        output_ids = [output_ids]
        logits = logits.unsqueeze(0)
        # original _call_
        prefixes = [tuple(self.prefix_ids + x) for x in output_ids]

        if len(prefixes[0]) > 0:
            selected_idx = [i for i, prefix in enumerate(prefixes)
                if prefix[-1] != self.hmm_model.eos_token_id]
        else:
            selected_idx = [i for i, _ in enumerate(prefixes)]

        logits = torch.log_softmax(logits, dim=-1)

        if len(selected_idx) > 0:
            selected_prefixes = [prefixes[i] for i in selected_idx]
            if len(self.token_ranges) == 1:
                selected_token_ranges = [self.token_ranges[0] for _ in selected_idx]
            else:
                selected_token_ranges = [self.token_ranges[i] for i in selected_idx]

            hmm_batch_size = len(selected_idx) if self.hmm_batch_size is None else min(len(selected_idx), self.hmm_batch_size)
            hmm_logits, hmm_logits_ = self.compute_logits(selected_prefixes, selected_token_ranges, hmm_batch_size)
            hmm_logits -= hmm_logits_

            # ban special tokens that are not in the HMM
            if hmm_logits.shape[1] < logits.shape[1]:
                neginf = torch.full((hmm_logits.shape[0], logits.shape[1]-hmm_logits.shape[1]), -1e30, device=hmm_logits.device)
                hmm_logits = torch.cat((hmm_logits, neginf), dim=1)
            logits[selected_idx, :] += self.alpha * hmm_logits
            logits = torch.log_softmax(logits, dim=-1)

        # logits = torch.log_softmax(logits / self.temperature, dim=-1)

        # Recover the original shape by removing the added batch dimension
        return logits.squeeze(0)



class CtrlgWrappedLogitsProcessor(AdapterLogitsProcessor):
    """Example of wrapping a fake request-level logit processor to create a
    batch-level logits processor"""

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
        constraints_dict: dict = {
            "ConstraintsKey": [[' Keyword_1_Choice_A', ' Keyword_1_Choice_B'], [' Keyword_2_Choice_A']]
            # ConstraintsKey is the key to identify which ConstraintLogitsProcessor to use in the request
            # The value is a list of list of phrases (each inner list is a set of alternative phrases)
            # The example constraints will enforce ((Keyword_1_Choice_A || Keyword_1_Choice_B) && (Keyword_2_Choice_A))
        }
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        
        # Init hmm model and default dfa model
        self.device = device
        print(f"👀==>> self.device: {self.device}")
        
        
        # Custom Config (now parameterized)
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens
        self.alpha = alpha

        # Init hmm and tokenizer
        self.hmm_model = ctrlg.HMM.from_pretrained(hmm_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        vocab_size = self.hmm_model.vocab_size
        eos_token_id = self.hmm_model.eos_token_id
        
        # Init custom ConstraintLogitsProcessor
        # The only per-request variable is prompt_ids, which will be set later
        self.custom_clp_dict = self.get_custom_clp_dict(vocab_size, eos_token_id, constraints_dict)
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
        dfa_graphs.append(eos_builder.build())

        dfa_graphs = ctrlg.DFA_prod(dfa_graphs, mode='intersection')
        dfa_model = ctrlg.DFAModel(dfa_graphs, vocab_size)
        
        return dfa_model

    def get_custom_clp_dict(self, vocab_size, eos_token_id, constraints_dict):
        # Return a dictionary of ConstraintLogitsProcessor
        clp_dict = {}
        
        for clp_key, phrases_list in constraints_dict.items():
            dfa_model = self.construct_dfa_kw(phrases_list, vocab_size, eos_token_id).to(self.device)
            clp_dict[clp_key] = CtrlgPerReqLogitsProcessor(
                self.hmm_model, dfa_model,
                self.min_new_tokens, self.max_new_tokens,
                prompt_ids=None, prefix_ids=[], suffix_ids=[], alpha=self.alpha)
        return clp_dict


    
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """This method returns a new request-level logits processor, customized
        to the `target_token` value associated with a particular request.

        Returns None if the logits processor should not be applied to the
        particular request. To use the logits processor the request must have
        a "target_token" custom argument with an integer value.

        Args:
          params: per-request sampling params

        Returns:
          `Callable` request logits processor, or None
        """
        # Get clp_id from extra_args. This specifies which ConstraintLogitsProcessor to use
        clp_id: Optional[Any] = params.extra_args and params.extra_args.get(
            "clp_id"
        )
        if clp_id is None:
            return None
        if not isinstance(clp_id, str):
            print(
                "clp_id value %s is not str; not applying logits"
                " processor to request.",
                clp_id,
            )
            return None
        if not clp_id in self.custom_clp_dict:
            print(
                "clp_id value %s is not in custom_clp_dict %s; not applying logits"
                " processor to request.",
                clp_id,
                str(self.custom_clp_dict.keys())
            )
            return None

        return self.custom_clp_dict[clp_id]


