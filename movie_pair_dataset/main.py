#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main orchestrator for Context-Heavy Movie Reranker
"""
import argparse
import json
from tqdm import tqdm
from typing import List, Dict
from config import (
    PROMPTS_JSON,
    OUTPUT_DIR,
    DEBUG_DIR,
    FINAL_OUTPUT,
    MODEL
)
from data_loader import load_user_history, load_movies, format_candidate_for_llm
from batch_processor import process_prompts_with_retry, calculate_batch_size
from utils import count_tokens
import os


def load_prompts(prompts_path: str, prompt_ids: str = None) -> List[Dict]:
    """
    Load prompts from JSON file

    Args:
        prompts_path: Path to prompts.json
        prompt_ids: Comma-separated prompt IDs or 'all'

    Returns:
        List of prompt dicts
    """
    with open(prompts_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle list or dict format
    if isinstance(data, list):
        prompts = [
            {
                "prompt_id": str(p.get("prompt_id")),
                "text": str(p.get("text") or p.get("prompt") or p.get("query") or ""),
                "combo_type": p.get("combo_type", "unknown"),
                "category": p.get("category", "unknown"),
                "persona": p.get("persona", "unknown")
            }
            for p in data if p.get("prompt_id")
        ]
    else:
        prompts = [
            {
                "prompt_id": str(k),
                "text": (v if isinstance(v, str) else str(v.get("text") or v.get("prompt") or "")),
                "combo_type": v.get("combo_type", "unknown") if isinstance(v, dict) else "unknown",
                "category": v.get("category", "unknown") if isinstance(v, dict) else "unknown",
                "persona": v.get("persona", "unknown") if isinstance(v, dict) else "unknown"
            }
            for k, v in data.items()
        ]

    # Filter by prompt IDs if specified
    if prompt_ids and prompt_ids.lower() != "all":
        selected_ids = {x.strip() for x in prompt_ids.split(',') if x.strip()}
        prompts = [p for p in prompts if p["prompt_id"] in selected_ids]

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Context-Heavy Movie Reranker with Pairwise Comparisons"
    )
    parser.add_argument(
        "--prompts",
        default=PROMPTS_JSON,
        help="Path to prompts.json"
    )
    parser.add_argument(
        "--prompt_ids",
        default="all",
        help="Comma-separated prompt IDs or 'all'"
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--debug_dir",
        default=DEBUG_DIR,
        help="Debug directory for raw API responses"
    )

    args = parser.parse_args()

    # Update config globals for output directories
    import config
    config.OUTPUT_DIR = args.output_dir
    config.DEBUG_DIR = args.debug_dir
    config.FINAL_OUTPUT = os.path.join(args.output_dir, "all_pairwise_comparisons.json")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    # Load data
    print("[INFO] Loading User 1's rating history...")
    user_history = load_user_history()
    print(f"[INFO] Loaded {len(user_history)} ratings for User 1")

    print("[INFO] Loading all movies...")
    candidates = load_movies(user_history)
    print(f"[INFO] Loaded {len(candidates)} movies")

    # Estimate token usage
    formatted_candidates = [format_candidate_for_llm(c) for c in candidates]
    candidate_text = "\n".join(formatted_candidates)
    candidate_tokens = count_tokens(candidate_text)
    print(f"[INFO] Estimated candidate tokens: ~{candidate_tokens:,}")

    # Load prompts
    print(f"[INFO] Loading prompts from {args.prompts}...")
    prompts = load_prompts(args.prompts, args.prompt_ids)
    print(f"[INFO] Loaded {len(prompts)} prompts to process")

    if not prompts:
        print("[ERROR] No prompts to process!")
        return

    # Print configuration
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Model: {MODEL}")
    print(f"  Total movies: {len(candidates)}")
    print(f"  User 1 ratings: {len(user_history)}")
    print(f"  Prompts to process: {len(prompts)}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Debug directory: {args.debug_dir}")
    print(f"{'='*60}\n")

    # Process prompts in batches
    all_results = {}
    remaining_prompts = prompts.copy()

    with tqdm(total=len(prompts), desc="Processing prompts", ncols=100) as pbar:
        while remaining_prompts:
            # Calculate batch size
            batch_size = calculate_batch_size(remaining_prompts)
            current_batch = remaining_prompts[:batch_size]

            # Process batch
            results = process_prompts_with_retry(current_batch, candidates)

            # Update results
            all_results.update(results)

            # Update progress
            pbar.update(len(current_batch))

            # Remove processed prompts
            remaining_prompts = remaining_prompts[batch_size:]

    # Save final consolidated results
    print(f"\n[INFO] Saving final results to {config.FINAL_OUTPUT}...")
    with open(config.FINAL_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Print summary
    successful = len(all_results)
    failed = len(prompts) - successful
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {successful/len(prompts)*100:.1f}%")
    print(f"{'='*60}\n")

    if failed > 0:
        failed_ids = [p["prompt_id"] for p in prompts if p["prompt_id"] not in all_results]
        print(f"[WARN] Failed prompt IDs: {', '.join(failed_ids)}")


if __name__ == "__main__":
    main()
