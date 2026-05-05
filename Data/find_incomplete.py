#!/usr/bin/env python3
"""
Find incomplete or missing prompt results.
"""
import json
import os
import argparse

def find_incomplete(prompts_json, out_dir, min_results=10):
    """Find prompt IDs that are incomplete or missing."""
    with open(prompts_json, 'r') as f:
        prompts_data = json.load(f)
        if isinstance(prompts_data, list):
            prompt_ids = [p['prompt_id'] for p in prompts_data]
        else:
            prompt_ids = list(prompts_data.keys())
    
    incomplete = []
    for pid in prompt_ids:
        json_path = os.path.join(out_dir, f"{pid}.top10.json")
        if not os.path.exists(json_path):
            incomplete.append(pid)
        else:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if not data or len(data) < min_results:
                        incomplete.append(pid)
            except Exception:
                incomplete.append(pid)
    
    return incomplete

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_results", type=int, default=10)
    args = parser.parse_args()
    
    incomplete = find_incomplete(args.prompts_json, args.out_dir, args.min_results)
    print(",".join(incomplete))






