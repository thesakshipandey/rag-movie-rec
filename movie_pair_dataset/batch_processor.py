"""
Batch processing with retry logic
"""
import json
from typing import List, Dict, Any
from prompt_builder import JUDGE_SYSTEM_PROMPT, build_user_payload
from api_client import call_openrouter
from utils import parse_json_robust, get_result_for_prompt, save_result, count_tokens
from config import MIN_PROMPTS_PER_BATCH, MAX_PROMPTS_PER_BATCH, MAX_TOKENS_OUTPUT


def calculate_batch_size(prompts: List[Dict[str, str]]) -> int:
    """
    Dynamically calculate batch size based on prompt lengths

    Args:
        prompts: List of remaining prompts

    Returns:
        Number of prompts to batch (3-5)
    """
    if not prompts:
        return 0

    # Sample up to 5 prompts to estimate token usage
    sample_size = min(5, len(prompts))
    sample_prompts = prompts[:sample_size]

    total_prompt_tokens = sum(count_tokens(p["text"]) for p in sample_prompts)
    avg_tokens_per_prompt = total_prompt_tokens / sample_size

    # Conservative thresholds - respect MAX_PROMPTS_PER_BATCH
    from config import MAX_PROMPTS_PER_BATCH

    if avg_tokens_per_prompt < 80:
        return min(MAX_PROMPTS_PER_BATCH, len(prompts))  # Short prompts
    elif avg_tokens_per_prompt < 100:
        return min(max(2, MAX_PROMPTS_PER_BATCH - 1), len(prompts))  # Medium prompts
    else:
        return min(max(1, MAX_PROMPTS_PER_BATCH - 2), len(prompts))  # Long prompts


def process_batch(
    prompts: List[Dict[str, str]],
    candidates: List[Dict],
    debug_tag: str = None
) -> Dict[str, List[Dict]]:
    """
    Process a batch of prompts through the Judge API

    Args:
        prompts: List of prompt dicts (prompt_id, text, combo_type)
        candidates: List of all movie candidates
        debug_tag: Tag for debug output

    Returns:
        Dict mapping prompt_id → list of pairwise comparison dicts

    Raises:
        ValueError: If JSON parsing fails
    """
    # Build messages
    user_content = build_user_payload(prompts, candidates)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # Call API
    response = call_openrouter(messages, max_tokens=MAX_TOKENS_OUTPUT, debug_tag=debug_tag)

    # Parse JSON
    parsed = parse_json_robust(response["raw"])

    # Extract results for each prompt
    results = {}
    results_obj = parsed.get("results", {})

    for prompt in prompts:
        prompt_id = prompt["prompt_id"]
        node = get_result_for_prompt(results_obj, prompt_id)

        if isinstance(node, dict):
            pairs = node.get("pairs", [])
            if pairs:
                results[prompt_id] = pairs

    return results


def process_single_prompt_fallback(
    prompt: Dict[str, str],
    candidates: List[Dict]
) -> List[Dict]:
    """
    Fallback: process a single prompt if batch fails

    Returns:
        List of pairwise comparison dicts (or empty list on failure)
    """
    try:
        debug_tag = f"single_{prompt['prompt_id']}"
        results = process_batch([prompt], candidates, debug_tag=debug_tag)
        return results.get(prompt["prompt_id"], [])
    except Exception as e:
        print(f"[ERROR] Single prompt fallback failed for {prompt['prompt_id']}: {e}")
        return []


def process_prompts_with_retry(
    prompts: List[Dict[str, str]],
    candidates: List[Dict]
) -> Dict[str, List[Dict]]:
    """
    Process prompts with automatic batching and retry logic

    Strategy:
    1. Try batch (3-5 prompts)
    2. If parse fails → retry batch once
    3. If still fails → split into individual prompts
    4. Save results as we go

    Returns:
        Dict mapping prompt_id → pairs (for successfully processed prompts)
    """
    batch_size = calculate_batch_size(prompts)
    batch = prompts[:batch_size]
    tag = f"batch_{batch[0]['prompt_id']}_to_{batch[-1]['prompt_id']}"

    all_results = {}

    # Attempt 1: Batch processing
    try:
        print(f"[INFO] Processing batch: {tag} ({len(batch)} prompts)")
        results = process_batch(batch, candidates, debug_tag=f"{tag}_attempt1")

        # Save successful results
        for prompt_id, pairs in results.items():
            save_result(prompt_id, pairs)
            all_results[prompt_id] = pairs

        # Identify missing prompts
        missing = [p for p in batch if p["prompt_id"] not in results]

        if missing:
            print(f"[WARN] Batch {tag}: missing {len(missing)} prompts")

            # Attempt 2: Retry batch once
            try:
                print(f"[INFO] Retrying batch {tag}...")
                results = process_batch(batch, candidates, debug_tag=f"{tag}_attempt2")

                for prompt_id, pairs in results.items():
                    if prompt_id not in all_results:
                        save_result(prompt_id, pairs)
                        all_results[prompt_id] = pairs

                # Re-check missing
                missing = [p for p in batch if p["prompt_id"] not in all_results]
            except Exception as e:
                print(f"[ERROR] Batch retry failed: {e}")

        # Attempt 3: Fallback to individual prompts
        if missing:
            print(f"[INFO] Falling back to individual prompts for {len(missing)} prompts")
            for prompt in missing:
                pairs = process_single_prompt_fallback(prompt, candidates)
                if pairs:
                    save_result(prompt["prompt_id"], pairs)
                    all_results[prompt["prompt_id"]] = pairs
                else:
                    print(f"[ERROR] Failed to process prompt {prompt['prompt_id']}")

        return all_results

    except Exception as e:
        print(f"[ERROR] Batch {tag} failed completely: {e}")

        # Fallback to individual prompts
        print(f"[INFO] Falling back to individual prompts for entire batch")
        for prompt in batch:
            pairs = process_single_prompt_fallback(prompt, candidates)
            if pairs:
                save_result(prompt["prompt_id"], pairs)
                all_results[prompt["prompt_id"]] = pairs

        return all_results
