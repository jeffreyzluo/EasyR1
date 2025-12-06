import os
from datasets import load_dataset, DatasetDict, Sequence, Image as ImageData

# --- CONFIGURATION ---
# 1. The source dataset on Hugging Face you want to convert
SOURCE_DATASET_ID = "ydeng9/OpenVLThinker-sft-iter3" 

# 2. Your personal Hugging Face username and desired repo name
TARGET_HUB_PATH = "JeffreyZLuo/SFT-formatted" 

def transform_example(example):
    """
    Transforms a single row from the source dataset to match the 
    target format: {'images': [PIL.Image], 'problem': str, 'answer': str}
    """
    
    # 1. Handle Image
    # Assumes source column is named 'image'. Wraps it in a list 
    # because your reference script outputs "images": [image]
    image = example["images"] 
    
    # 2. Handle Problem Text
    # Prepends <image> tag as per your script
    # Change 'annotat_text' to whatever the source text column is named
    problem_text = "<images>" + example["question"]
    
    # 3. Handle Answer Logic
    # Replicates the logic: data["choices"][MAPPING[data["answer"]]]
    # Ensure the source has 'choices' (list) and 'answer' (letter) columns
    answer_text = example["answer"]

    return {
        "images": image,
        "problem": problem_text,
        "answer": answer_text
    }

def main():
    print(f"Loading source dataset: {SOURCE_DATASET_ID}...")
    # Load the dataset (downloads it if not already cached)
    raw_dataset = load_dataset(SOURCE_DATASET_ID)
    first_split_name = next(iter(raw_dataset.keys()))
    cols_to_remove = raw_dataset[first_split_name].column_names
    
    print(f"Found splits: {list(raw_dataset.keys())}")
    # Apply the transformation to all splits (train, val, test)
    print("Transforming dataset format...")
    formatted_dataset = raw_dataset.map(
        transform_example,
        remove_columns=cols_to_remove, 
        num_proc=4 
    )

    # CRITICAL: Cast the 'images' column to Sequence(Image())
    # This matches the line: .cast_column("images", Sequence(ImageData())) in your script
    print("Casting columns to match schema...")
    final_dataset = formatted_dataset.cast_column("images", Sequence(ImageData()))

    # Push to your personal Hugging Face Hub
    print(f"Pushing to {TARGET_HUB_PATH}...")
    final_dataset.push_to_hub(TARGET_HUB_PATH)
    
    print("Done! Dataset is live.")

if __name__ == "__main__":
    main()