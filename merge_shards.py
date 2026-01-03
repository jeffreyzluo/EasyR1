import os
import json
import glob
import re
import argparse
from pathlib import Path

def merge_shards(output_dir):
    # Regex to capture the base filename excluding the shard info
    # Matches: "mathvista_modelname_shard_0_of_4.json" -> group 1: "mathvista_modelname"
    shard_pattern = re.compile(r"(.+)_shard_\d+_of_\d+\.json$")
    
    files = glob.glob(os.path.join(output_dir, "*_shard_*_of_*.json"))
    if not files:
        print(f"No shard files found in {output_dir}")
        return

    # Group files by their base name
    groups = {}
    for f_path in files:
        filename = os.path.basename(f_path)
        match = shard_pattern.match(filename)
        if match:
            base_name = match.group(1)
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(f_path)

    print(f"Found {len(groups)} partitioned datasets to merge.")

    for base_name, shard_files in groups.items():
        print(f"\nProcessing: {base_name} ({len(shard_files)} shards found)")
        
        all_results = []
        correct_count = 0
        
        # Load and merge
        for shard in sorted(shard_files):
            try:
                with open(shard, 'r') as f:
                    data = json.load(f)
                    all_results.extend(data)
            except Exception as e:
                print(f"  Error reading {shard}: {e}")

        # Sort by ID to ensure original order
        all_results.sort(key=lambda x: x.get('id', 0))
        
        # Recalculate metrics
        total = len(all_results)
        correct = sum(1 for item in all_results if item.get('is_correct'))
        accuracy = (correct / total * 100) if total > 0 else 0

        # Save merged file
        merged_filename = os.path.join(output_dir, f"{base_name}.json")
        with open(merged_filename, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"  Saved merged file: {merged_filename}")
        print(f"  Total Records: {total}")
        print(f"  Final Accuracy: {accuracy:.2f}%")

        # Optional: Clean up shards after successful merge
        # for f in shard_files: os.remove(f) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./evaluation/outputs', help='Directory containing shard files')
    args = parser.parse_args()
    merge_shards(args.dir)