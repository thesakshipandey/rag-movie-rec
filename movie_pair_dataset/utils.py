"""
Shared utility functions
"""
import json
import re
from typing import Dict, Any, List
from config import CHARS_PER_TOKEN, OUTPUT_DIR
import os
import pandas as pd


def count_tokens(text: str) -> int:
    """Estimate token count (conservative)"""
    return max(1, len(text) // CHARS_PER_TOKEN)


def strip_code_fences(s: str) -> str:
    """Remove markdown code fences from text"""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s


def parse_json_robust(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from model output
    Handles o4-mini reasoning output and various JSON formats
    """
    s = strip_code_fences(text)

    # Try direct parse first (fastest path)
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

    # Look for JSON object with "results" key
    patterns = [
        r'\{"results"\s*:\s*\{.*?\}\s*\}(?=\s*[,\]\}]|\s*$)',  # Non-greedy with lookahead
        r'\{"results"\s*:\s*\{.*\}',  # Greedy fallback
        r'\{.*"results".*\}',  # Any object with results
        r'\{.*?\}',  # Any JSON object (non-greedy)
        r'\{.*\}',  # Any JSON object (greedy)
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, s, flags=re.S)
        for m in matches:
            candidate = m.group(0)
            try:
                obj = json.loads(candidate)
                # Verify it has the expected structure
                if isinstance(obj, dict) and "results" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

    # If we found any valid JSON at all (even without results key), try it
    for pattern in patterns:
        matches = re.finditer(pattern, s, flags=re.S)
        for m in matches:
            candidate = m.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    preview = (s[:800] if len(s) > 800 else s).replace("\n", "\\n")
    raise ValueError(f"Model did not return parseable JSON. First 800 chars:\n{preview}")


def sanitize_reasoning(text: str, max_length: int = 200) -> str:
    """Clean up reasoning text"""
    s = (text or "").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_length]


def get_result_for_prompt(results_obj: Any, prompt_id: str) -> Dict[str, Any]:
    """
    Extract result for a specific prompt_id, handling key variants
    (e.g., "0001" vs "1")
    """
    # Generate variants
    variants = [prompt_id, prompt_id.strip()]
    if prompt_id.isdigit():
        variants.append(str(int(prompt_id)))  # "0001" -> "1"
    if prompt_id.lstrip("0").isdigit():
        variants.append(prompt_id.lstrip("0"))

    # Dedupe
    variants = list(dict.fromkeys([v for v in variants if v]))

    # Check dict keys
    if isinstance(results_obj, dict):
        for key in variants:
            if key in results_obj:
                return results_obj.get(key) or {}

    # Check list (fallback)
    if isinstance(results_obj, list):
        for node in results_obj:
            if str(node.get("prompt_id")) in variants:
                return node

    return {}


def save_result(prompt_id: str, pairs: List[Dict[str, Any]]):
    """
    Save pairwise comparison results for a single prompt

    Saves both JSON and CSV formats
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sanitize reasoning in all pairs
    for pair in pairs:
        if "reasoning" in pair:
            pair["reasoning"] = sanitize_reasoning(pair["reasoning"])

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, f"{prompt_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"prompt_id": prompt_id, "pairs": pairs}, f, ensure_ascii=False, indent=2)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{prompt_id}.csv")
    pd.DataFrame(pairs).to_csv(csv_path, index=False)
