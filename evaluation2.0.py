import os
import logging
import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from mathruler.grader import grade_answer, extract_boxed_content

# --- Configuration Enums & Classes (Same as before) ---

class ModelType(Enum):
    OPENVLTHINKER = "openvlthinker"
    QWEN = "qwen"

class DatasetType(Enum):
    MATHVISTA = "mathvista"
    MATHVERSE = "mathverse"
    MATHVISION = "mathvision"
    SFTSEED = "sftseed"
    HALLUSIONBENCH = "hallusionbench"
    EMMA_MATH = "emma-math"
    EMMA_CHEM = "emma-chem"
    EMMA_CODE = "emma-code"
    EMMA_PHYSICS = "emma-physics"
    MMMU_PRO_VISION = "mmmu-pro-vision"
    MMMU_PRO_4 = "mmmu-pro-4"
    MMMU_PRO_10 = "mmmu-pro-10"

@dataclass
class DatasetConfig:
    name: str
    split: str
    image_field: Union[str, List[str]]
    response_field: str
    instruction_field: Optional[str] = None
    subset: Optional[str] = None
    choices_field: Optional[str] = None
    options_field: Optional[str] = None
    source_field: Optional[str] = None

@dataclass
class ModelEvaluationConfig:
    model_name: str
    processor_name: str
    prompt_suffix: str
    max_new_tokens: int = 2048
    top_p: float = 0.001
    top_k: int = 1
    temperature: float = 0.01
    repetition_penalty: float = 1.0

def get_model_eval_config(model_type: ModelType) -> ModelEvaluationConfig:
    configs = {
        ModelType.OPENVLTHINKER: ModelEvaluationConfig(
            model_name="ydeng9/OpenVLThinker-7B-v1.2",
            processor_name="Qwen/Qwen2.5-VL-7B-Instruct",
            prompt_suffix=""
        ),
        ModelType.QWEN: ModelEvaluationConfig(
            model_name="JeffreyZLuo/Qwen2.5-45-Hard-V8",
            processor_name="Qwen/Qwen2.5-VL-7B-Instruct",
            prompt_suffix="\n\nYour final answer MUST BE put in \\boxed{}"
        )
    }
    return configs[model_type]

def get_dataset_config(dataset_type: DatasetType) -> DatasetConfig:
    # (Configuration dictionary remains the same as your original script)
    configs = {
        DatasetType.MATHVISTA: DatasetConfig(name="AI4Math/MathVista", split="testmini", image_field="decoded_image", instruction_field="query", response_field="answer", choices_field="choices"),
        DatasetType.MATHVERSE: DatasetConfig(name="AI4Math/MathVerse", subset="testmini", split="testmini", image_field="image", instruction_field="query_cot", response_field="answer"),
        DatasetType.MATHVISION: DatasetConfig(name="MathLLMs/MathVision", split="test", image_field="decoded_image", instruction_field="question", response_field="answer", options_field="options"),
        DatasetType.SFTSEED: DatasetConfig(name="ydeng9/sft_seed", split="train", image_field="decoded_image", instruction_field="problem", response_field="answer", source_field="source"),
        DatasetType.HALLUSIONBENCH: DatasetConfig(name="lmms-lab/HallusionBench", split="image", image_field="image", instruction_field="question", response_field="gt_answer"),
        DatasetType.EMMA_MATH: DatasetConfig(name="luckychao/EMMA", subset="Math", split="test", image_field="image_1", instruction_field="question", response_field="answer", options_field="options"),
        DatasetType.EMMA_CHEM: DatasetConfig(name="luckychao/EMMA", subset="Chemistry", split="test", image_field=["image_1","image_2","image_3","image_4","image_5"], instruction_field="question", response_field="answer", options_field="options"),
        DatasetType.EMMA_CODE: DatasetConfig(name="luckychao/EMMA", subset="Coding", split="test", image_field=["image_1","image_2","image_3","image_4","image_5"], instruction_field="question", response_field="answer", options_field="options"),
        DatasetType.EMMA_PHYSICS: DatasetConfig(name="luckychao/EMMA", subset="Physics", split="test", image_field=["image_1","image_2","image_3","image_4","image_5"], instruction_field="question", response_field="answer", options_field="options"),
        DatasetType.MMMU_PRO_VISION: DatasetConfig(name="MMMU/MMMU_Pro", subset="vision", split="test", image_field="image", instruction_field="question", response_field="answer", options_field="options"),
        DatasetType.MMMU_PRO_4: DatasetConfig(name="MMMU/MMMU_Pro", subset="standard (4 options)", split="test", image_field=["image_1","image_2","image_3","image_4","image_5", "image_6", "image_7"], instruction_field="question", response_field="answer", options_field="options"),
        DatasetType.MMMU_PRO_10: DatasetConfig(name="MMMU/MMMU_Pro", subset="standard (10 options)", split="test", image_field=["image_1","image_2","image_3","image_4","image_5", "image_6", "image_7"], instruction_field="question", response_field="answer", options_field="options"),
    }
    return configs[dataset_type]

# --- Helper Classes ---

class Evaluator:
    def __init__(self, model_config: ModelEvaluationConfig, device: str):
        self.device = device
        self.model_config = model_config
        self.model = self._load_model()
        self.processor = self._load_processor()

    def _load_model(self) -> Qwen2_5_VLForConditionalGeneration:
        print(f"Loading model: {self.model_config.model_name}")
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device
        )

    def _load_processor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.model_config.processor_name)

    def generate_response(self, image_urls: Union[str, List[str]], instruction: str) -> Optional[str]:
        full_instruction = instruction + self.model_config.prompt_suffix
        urls = [image_urls] if not isinstance(image_urls, list) else image_urls
        
        content = [{"type": "image", "image": url} for url in urls if url]
        content.append({"type": "text", "text": full_instruction})
        messages = [{"role": "user", "content": content}]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(self.device)
            
            generated_ids = self.model.generate(
                **inputs, do_sample=True,
                max_new_tokens=self.model_config.max_new_tokens, top_p=self.model_config.top_p,
                top_k=self.model_config.top_k, temperature=self.model_config.temperature,
                repetition_penalty=self.model_config.repetition_penalty,
            )
            
            trimmed_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, generated_ids)]
            return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

def load_dataset_items(config: DatasetConfig) -> List[Dict[str, Any]]:
    print(f"Loading dataset: {config.name}")
    dataset = load_dataset(config.name, config.subset, split=config.split) if config.subset else load_dataset(config.name, split=config.split)
    items = []
    for item in dataset:
        image_url = [img for img in (item.get(x) for x in config.image_field)] if isinstance(config.image_field, list) else item[config.image_field]
        items.append({
            'image_url': image_url,
            'instruction': item.get(config.instruction_field, ''),
            'response': item.get(config.response_field, ''),
            'choices': item.get(config.choices_field),
            'options': item.get(config.options_field, []),
            'source': item.get(config.source_field)
        })
    return items

def format_instruction(instruction: str, options: Optional[List[str]] = None, is_yes_no: bool = False, is_vision_only: bool = False) -> str:
    options = eval(options) if isinstance(options, str) else options
    hint = "Hint: Please answer the question"
    if is_vision_only:
        hint += " shown in the image."
        if options:
            hint += " Provide the correct option letter, e.g., A, B, C, D, E, at the end."
            choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
            return f"{hint}\nChoices:\n{choice_list}"
        return hint
    if is_yes_no:
        return f"{hint} requiring an answer of yes or no.\nQuestion: {instruction}"
    if options:
        hint += " and provide the correct option letter, e.g., A, B, C, D, E, at the end."
        choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return f"{hint}\nQuestion: {instruction}\nChoices:\n{choice_list}"
    return f"{hint} requiring an answer.\nQuestion: {instruction}"

def process_ground_truth(response: str, choices: Optional[List[str]], options: Optional[List[str]] = None) -> str:
    search_list, options_map = choices or options, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    if search_list:
        try: return options_map[search_list.index(response)]
        except (ValueError, IndexError): pass
    return response

def extract_answers(reasoning: str, model_type: ModelType) -> List[str]:
    if not reasoning: return ["Failed to generate."]
    if model_type == ModelType.OPENVLTHINKER:
        if "</answer>" in reasoning: return [reasoning.split("<answer>")[-1].split("</answer>")[0].strip()]
        return ["Failed to extract."]
    if model_type == ModelType.QWEN:
        candidates = []
        if (boxed := extract_boxed_content(reasoning)): candidates.append(boxed)
        if "answer:" in reasoning.lower(): candidates.append(reasoning.lower().split("answer:")[-1].strip())
        return candidates or ["Failed to extract."]
    return [reasoning]

def check_correctness(gt: str, preds: List[str]) -> bool:
    return any(gt.lower() == pred.lower() or grade_answer(gt, pred) for pred in preds if isinstance(gt, str) and isinstance(pred, str))

def save_results(results: List[Dict], output_file: str) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f: json.dump(results, f, indent=2)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='Multi-dataset sharded evaluation.')
    parser.add_argument('--model', type=str, choices=[m.value for m in ModelType], required=True)
    parser.add_argument('--datasets', type=str, required=True, help='Comma-separated list of datasets (e.g. mathvista,mathverse)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--num_shards', type=int, default=1, help='Total shards')
    parser.add_argument('--shard_id', type=int, default=0, help='Current shard ID')
    args = parser.parse_args()

    # Log setup (per shard to avoid conflict)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(f'evaluation_shard_{args.shard_id}.log'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    model_type = ModelType(args.model)
    model_config = get_model_eval_config(model_type)
    
    # Load model once for all datasets
    evaluator = Evaluator(model_config, device)

    # Parse dataset list
    dataset_names = [d.strip() for d in args.datasets.split(',')]
    
    for ds_name in dataset_names:
        try:
            dataset_type = DatasetType(ds_name)
        except ValueError:
            logger.error(f"Unknown dataset: {ds_name}. Skipping.")
            continue

        dataset_config = get_dataset_config(dataset_type)
        
        # Output filename includes shard info
        output_file = f"./evaluation/outputs/{dataset_type.value}_{model_config.model_name.split('/')[-1]}_shard_{args.shard_id}_of_{args.num_shards}.json"
        
        logger.info(f"--- Starting {ds_name} (Shard {args.shard_id}/{args.num_shards}) ---")
        
        try:
            all_items = load_dataset_items(dataset_config)
            
            # Sharding Logic
            total_items = len(all_items)
            chunk_size = total_items // args.num_shards
            start_idx = args.shard_id * chunk_size
            end_idx = start_idx + chunk_size if args.shard_id < args.num_shards - 1 else total_items
            shard_items = all_items[start_idx:end_idx]
            
            logger.info(f"Processing items {start_idx} to {end_idx} (Total: {len(shard_items)})")

            results, correct_count = [], 0
            
            for i, item in tqdm(enumerate(shard_items), total=len(shard_items), desc=f"{ds_name} Shard {args.shard_id}"):
                instruction = format_instruction(
                    item['instruction'], item.get('options'),
                    is_yes_no=(dataset_type == DatasetType.HALLUSIONBENCH),
                    is_vision_only=(dataset_type == DatasetType.MMMU_PRO_VISION)
                )

                reasoning = evaluator.generate_response(item['image_url'], instruction)
                predicted_answers = extract_answers(reasoning, model_type)
                
                if dataset_type in [DatasetType.MMMU_PRO_VISION, DatasetType.MMMU_PRO_4, DatasetType.MMMU_PRO_10]:
                    gt_answer = item['response']
                else:
                    gt_answer = process_ground_truth(item['response'], item.get('choices'), item.get('options'))
                
                if dataset_type == DatasetType.HALLUSIONBENCH: gt_answer = "Yes" if gt_answer == "1" else "No"
                    
                is_correct = check_correctness(gt_answer, predicted_answers)
                if is_correct: correct_count += 1
                
                # Use global index (start_idx + i) for ID to allow easy merging later
                results.append({
                    'id': start_idx + i, 
                    'instruction': item['instruction'], 
                    'ground_truth': gt_answer, 
                    'reasoning': reasoning, 
                    'predicted_answers': predicted_answers, 
                    'is_correct': is_correct,
                    'source': item.get('source')
                })
                
                if (i + 1) % 10 == 0: save_results(results, output_file)

            save_results(results, output_file)
            logger.info(f"Finished {ds_name} Shard {args.shard_id}. Partial Acc: {correct_count}/{len(shard_items)}")
            
        except Exception as e:
            logger.error(f"Failed processing {ds_name}: {e}")

if __name__ == "__main__":
    main()