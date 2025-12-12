import os
import logging
import argparse
import json
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

# Basic Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration

class ModelType(Enum):
    """Enum to specify the model being evaluated."""
    OPENVLTHINKER = "openvlthinker"
    QWEN = "qwen"

class DatasetType(Enum):
    """Enum for supported datasets."""
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
    """Stores configuration for loading a dataset."""
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
    """Stores model-specific evaluation parameters."""
    model_name: str
    processor_name: str
    prompt_suffix: str
    max_new_tokens: int = 2048
    top_p: float = 0.001
    top_k: int = 1
    temperature: float = 0.01
    repetition_penalty: float = 1.0

def get_model_eval_config(model_type: ModelType) -> ModelEvaluationConfig:
    """Returns the specific configuration for a given model type."""
    configs = {
        ModelType.OPENVLTHINKER: ModelEvaluationConfig(
            model_name="ydeng9/OpenVLThinker-7B-v1.2",
            processor_name="Qwen/Qwen2.5-VL-7B-Instruct",
            prompt_suffix=""  # Uses <answer> tags for extraction
        ),
        ModelType.QWEN: ModelEvaluationConfig(
            model_name="JeffreyZLuo/Qwen2.5-HMM-30-Hard",
            processor_name="Qwen/Qwen2.5-VL-7B-Instruct",
            prompt_suffix="\n\nYour final answer MUST BE put in \\boxed{}"
        )
    }
    return configs[model_type]

def get_dataset_config(dataset_type: DatasetType) -> DatasetConfig:
    """Returns the configuration for a given dataset."""
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

class Evaluator:
    """Handles model loading and response generation."""
    def __init__(self, model_config: ModelEvaluationConfig, device: str):
        self.device = device
        self.model_config = model_config
        self.model = self._load_model()
        self.processor = self._load_processor()

    def _load_model(self) -> Qwen2_5_VLForConditionalGeneration:
        logger.info(f"Loading model: {self.model_config.model_name}")
        try:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.device
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_processor(self) -> AutoProcessor:
        logger.info(f"Loading processor: {self.model_config.processor_name}")
        try:
            return AutoProcessor.from_pretrained(self.model_config.processor_name)
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise

    def generate_response(self, image_urls: Union[str, List[str]], instruction: str) -> Optional[str]:
        """Generates a model response for a given instruction and image(s)."""
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
            logger.error(f"Error during response generation: {e}")
            return None

# Data Handling & Formatting

def load_dataset_items(config: DatasetConfig) -> List[Dict[str, Any]]:
    """Loads and formats items from a Hugging Face dataset."""
    logger.info(f"Loading dataset: {config.name} ({config.subset or 'default'}) - split: {config.split}")
    try:
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
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def format_instruction(instruction: str, options: Optional[List[str]] = None, is_yes_no: bool = False, is_vision_only: bool = False) -> str:
    """Formats the instruction based on the question type and options."""
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
    """Converts a ground truth answer to a letter option if applicable."""
    search_list, options_map = choices or options, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    if search_list:
        try: return options_map[search_list.index(response)]
        except (ValueError, IndexError): pass
    return response

# Answer Extraction, Grading & Saving

def extract_answers(reasoning: str, model_type: ModelType) -> List[str]:
    """Extracts a list of candidate final answers from the model's reasoning."""
    if not reasoning: return ["Failed to generate."]
    if model_type == ModelType.OPENVLTHINKER:
        if "</answer>" in reasoning: return [reasoning.split("<answer>")[-1].split("</answer>")[0].strip()]
        logger.warning("Could not find <answer> tag in response.")
        return ["Failed to extract."]
    if model_type == ModelType.QWEN:
        candidates = []
        if (boxed := extract_boxed_content(reasoning)): candidates.append(boxed)
        if "answer:" in reasoning.lower(): candidates.append(reasoning.lower().split("answer:")[-1].strip())
        if not candidates: logger.warning("Could not find boxed content or 'Answer:' in response.")
        return candidates or ["Failed to extract."]
    return [reasoning]

def check_correctness(gt: str, preds: List[str]) -> bool:
    """Checks if any predicted answer is correct."""
    return any(gt.lower() == pred.lower() or grade_answer(gt, pred) for pred in preds if isinstance(gt, str) and isinstance(pred, str))

def save_results(results: List[Dict], output_file: str) -> None:
    """Saves evaluation results to a JSON file."""
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f: json.dump(results, f, indent=2)
        logger.info(f"Saved {len(results)} results to {output_file}")
    except Exception as e: logger.error(f"Failed to save results: {e}")

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description='Unified evaluation script for vision-language models.')
    parser.add_argument('--model', type=str, choices=[m.value for m in ModelType], required=True, help='Model to evaluate.')
    parser.add_argument('--dataset', type=str, choices=[d.value for d in DatasetType], required=True, help='Dataset to evaluate on.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number to use.')
    args = parser.parse_args()

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    model_type, dataset_type = ModelType(args.model), DatasetType(args.dataset)
    logger.info(f"Starting evaluation for model '{model_type.name}' on dataset '{dataset_type.name}' using device '{device}'")

    model_config = get_model_eval_config(model_type)
    dataset_config = get_dataset_config(dataset_type)
    output_file = f"./evaluation/outputs/{dataset_type.value}_{model_config.model_name.split('/')[-1]}.json"

    evaluator = Evaluator(model_config, device)
    dataset_items = load_dataset_items(dataset_config)
    
    results, correct_count, source_stats = [], 0, {}

    for i, item in tqdm(enumerate(dataset_items), total=len(dataset_items), desc=f"Evaluating"):
        # 1. Format instruction
        instruction = format_instruction(
            item['instruction'], item.get('options'),
            is_yes_no=(dataset_type == DatasetType.HALLUSIONBENCH),
            is_vision_only=(dataset_type == DatasetType.MMMU_PRO_VISION)
        )

        # 2. Generate and extract answers
        reasoning = evaluator.generate_response(item['image_url'], instruction)
        predicted_answers = extract_answers(reasoning, model_type)
        
        # 3. Process ground truth and grade
        if dataset_type in [DatasetType.MMMU_PRO_VISION, DatasetType.MMMU_PRO_4, DatasetType.MMMU_PRO_10]:
            gt_answer = item['response']
        else:
            gt_answer = process_ground_truth(item['response'], item.get('choices'), item.get('options'))
        
        if dataset_type == DatasetType.HALLUSIONBENCH: gt_answer = "Yes" if gt_answer == "1" else "No"
            
        is_correct = check_correctness(gt_answer, predicted_answers)
        if is_correct: correct_count += 1
        
        # 4. Store results
        result_item = {'id': i, 'instruction': item['instruction'], 'ground_truth': gt_answer, 'reasoning': reasoning, 'predicted_answers': predicted_answers, 'is_correct': is_correct}
        if item.get('source'):
            result_item['source'] = item['source']
            source_stats.setdefault(item['source'], {'correct': 0, 'total': 0})
            source_stats[item['source']]['total'] += 1
            if is_correct: source_stats[item['source']]['correct'] += 1
        results.append(result_item)
        
        if (i + 1) % 50 == 0: save_results(results, output_file)

    # Final save and report
    save_results(results, output_file)
    accuracy = (correct_count / len(dataset_items)) * 100 if dataset_items else 0
    logger.info(f"--- Evaluation Complete ---\n"
                f"Model: {model_type.name}\n"
                f"Dataset: {dataset_type.name}\n"
                f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(dataset_items)})")

    if dataset_type == DatasetType.SFTSEED and source_stats:
        logger.info("--- Accuracy by Source (SFTSEED) ---")
        for source, stats in sorted(source_stats.items()):
            acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            logger.info(f"  - {source}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    main()