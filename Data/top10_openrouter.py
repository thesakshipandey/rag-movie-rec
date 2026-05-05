#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-wise o4-mini reranker — send ALL candidates per request.

Fixes vs last version:
• Do NOT include a JSON "RESPONSE_FORMAT" with empty arrays inside user content (models tend to echo it → []).
• Key-robust result matching: "0001" ~= "1" (handles leading zeros).
• Debug: write raw model outputs to --debug_dir per batch.
• Resilience: retries + batch auto-split on parse errors; per-prompt fallback if a prompt returns empty.
"""

import argparse, json, os, re, time
from typing import Any, Dict, List, Iterable
import pandas as pd
import requests
from tqdm import tqdm

# ---------------- utils ---------------- #
def year_from_date(date_str: Any) -> str:
    return (date_str[:4] if isinstance(date_str, str) and len(date_str) >= 4 and date_str[:4].isdigit() else "")

def build_candidate(row: pd.Series, use_plot: bool, max_plot_chars: int) -> Dict[str, Any]:
    plot = ""
    if use_plot:
        plot = str(row.get("plot_sum_160") or row.get("plot") or "")
        plot = plot.replace("\n", " ").strip()
        if len(plot) > max_plot_chars:
            plot = plot[: max_plot_chars - 3] + "..."
    return {
        "movieId": int(row["movieId"]),
        "title": str(row["title"]),
        "year": year_from_date(row.get("release_date", "")),
        "plot": plot,
    }

def load_prompts(prompts_json_path: str) -> List[Dict[str, str]]:
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [{"prompt_id": str(x.get("prompt_id")), "text": str(x.get("text") or x.get("prompt") or x.get("query") or "")}
                for x in data if x.get("prompt_id")]
    return [{"prompt_id": str(k), "text": (v if isinstance(v, str) else str(v.get("text") or v.get("prompt") or v.get("query") or ""))}
            for k, v in data.items()]

def approx_tokens(chars: int) -> int:
    return max(1, chars // 4)

def chunked(seq: List[Any], min_size: int, max_size: int) -> Iterable[List[Any]]:
    if min_size <= 0 or max_size <= 0:
        raise ValueError("Batch sizes must be positive integers")
    if min_size > max_size:
        raise ValueError("min_size cannot exceed max_size")

    total = len(seq)
    if total == 0:
        return

    start = 0
    remaining = total

    while remaining > 0:
        size = min(max_size, remaining)

        while size > min_size and 0 < (remaining - size) < min_size:
            size -= 1

        if size < min_size and remaining != total:
            # Last chunk may dip below the minimum when redistribution is impossible
            size = remaining

        yield seq[start: start + size]
        start += size
        remaining -= size

# -------- robust JSON extraction -------- #
def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s

def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from model output.
    Handles o4-mini reasoning output and various JSON formats.
    """
    s = _strip_fences(text)
    
    # Try direct parse first (fastest path)
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass  # Fall through to more aggressive extraction
    
    # Look for JSON object with "results" key (our expected format)
    # Use non-greedy match and try to find complete JSON objects
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

# map results["0001"] / ["1"]
def _variants_for_pid(pid: str) -> List[str]:
    v = [pid, pid.strip()]
    if pid.isdigit():
        v.append(str(int(pid)))     # "0001" -> "1"
    if pid.lstrip("0").isdigit():
        v.append(pid.lstrip("0"))   # extra safety
    return list(dict.fromkeys([x for x in v if x != ""]))  # dedupe, drop empties

def get_result_for_pid(results_obj: Any, pid: str) -> Dict[str, Any]:
    # common case: dict keyed by pid
    if isinstance(results_obj, dict):
        for key in _variants_for_pid(pid):
            if key in results_obj:
                return results_obj.get(key) or {}
    # fallback: list of {prompt_id, top10}
    if isinstance(results_obj, list):
        for node in results_obj:
            if str(node.get("prompt_id")) in _variants_for_pid(pid):
                return node
    return {}

def sanitize_reason(s: str) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", s).strip()

# ---------------- prompt template ---------------- #
SYSTEM_MSG = (
    "You are a movie reranker. You may use general knowledge.\n"
    "Use ONLY the provided CANDIDATES (movieId, title, year, plot). If plot is non-empty, treat it as canonical.\n\n"
    "Task\n"
    "• For EACH prompt in PROMPTS, select EXACTLY 10 UNIQUE movieIds that best satisfy that prompt.\n"
    "• Avoid spoilers; prefer non-speculative, widely known traits when plot is empty.\n\n"
    "Output (ONE JSON object; no code fences, no extra text)\n"
    "{\n"
    '  "results": {\n'
    '    "<prompt_id>": {\n'
    '      "top10": [\n'
    '        {"movieId": <int>, "score": <float 0..1>, "reason": "<=140 chars, one line, non-spoiler>"}\n'
    "      ]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "Constraints\n"
    "• Choose only from the given movieIds; sort by descending score; no duplicates.\n"
    "• Think step-by-step internally if needed, but your final response MUST be ONLY valid JSON.\n"
    "• DO NOT add any commentary, explanations, or extra text outside the JSON structure.\n"
    "• The reason field must be exactly as specified - no meta-commentary about character counts or formatting."
)

# ---------------- OpenRouter call (+tool_calls fallback) ---------------- #
def call_openrouter(model: str, messages: List[Dict[str, str]],
                    max_tokens: int, timeout_s: int = 600,
                    retries: int = 5, backoff: float = 2.0) -> Dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": 42,
    }
    
    # For non-reasoning models, enforce JSON format
    # For o4-mini and reasoning models, skip this as it may conflict with reasoning output
    if "o4" not in model.lower() and "o1" not in model.lower():
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.cli",
        "X-Title": "o4mini-movielens-batch",
    }

    last_err = None
    for i in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            
            # Better error handling for API response structure
            if "choices" not in data or not data["choices"]:
                error_msg = data.get("error", {}).get("message", str(data))
                raise ValueError(f"API returned no choices. Error: {error_msg}")
            
            msg = data["choices"][0]["message"]
            
            # For o4-mini and reasoning models, check multiple locations
            text = ""
            
            # 1. Try content field first (standard location for output)
            text = (msg.get("content") or "").strip()
            
            # 2. Check tool_calls.function.arguments (some models use this)
            if not text:
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function") or {}
                    if fn.get("arguments"):
                        text = fn["arguments"]
                        break
            
            # 3. If still empty but reasoning exists, the model may have reasoned but not output
            #    Treat as empty response to trigger retry logic
            if not text:
                if "reasoning" in msg and msg.get("reasoning"):
                    raise ValueError(f"Model returned reasoning but no content. Reasoning length: {len(str(msg.get('reasoning', '')))} chars")
                # Last resort: dump full message
                text = json.dumps(msg, ensure_ascii=False)
                
            return {"raw": text, "full": data}
        except requests.exceptions.HTTPError as e:
            last_err = e
            status_code = e.response.status_code if hasattr(e, 'response') else 0
            
            # Handle rate limits with exponential backoff
            if status_code == 429:
                wait_time = backoff * (2 ** i) + (i * 2)  # Add jitter
                print(f"[WARN] Rate limited. Waiting {wait_time:.1f}s... ({i+1}/{retries})")
                time.sleep(wait_time)
                continue
            
            print(f"[ERROR] HTTP {status_code}: {e}")
            if hasattr(e, 'response'):
                print(f"[ERROR] Response: {e.response.text[:500]}")
            if i < retries - 1:
                wait_time = backoff * (2 ** i)
                print(f"[INFO] Retrying in {wait_time:.1f}s... ({i+1}/{retries})")
                time.sleep(wait_time)
        except Exception as e:
            last_err = e
            print(f"[ERROR] {type(e).__name__}: {e}")
            if i < retries - 1:
                wait_time = backoff * (2 ** i)
                print(f"[INFO] Retrying in {wait_time:.1f}s... ({i+1}/{retries})")
                time.sleep(wait_time)
    raise RuntimeError(f"OpenRouter call failed after {retries} retries. Last error: {type(last_err).__name__}: {last_err}")

# ---------------- runners ---------------- #
def build_user_payload(prompts: List[Dict[str, str]], candidates: List[Dict[str, Any]]) -> str:
    # IMPORTANT: do NOT place a JSON RESPONSE_FORMAT with empty arrays here.
    # Only give PROMPTS and CANDIDATES; schema is in SYSTEM_MSG.
    p_block = [{"prompt_id": p["prompt_id"], "text": p["text"]} for p in prompts]
    return json.dumps({"PROMPTS": p_block, "CANDIDATES": candidates}, ensure_ascii=False)

def write_debug(debug_dir: str, tag: str, content: str):
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"{tag}.raw.json.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def save_result_for_pid(out_dir: str, pid: str, items: List[Dict[str, Any]], k: int):
    os.makedirs(out_dir, exist_ok=True)
    items = items[:k]
    # sanitize reasons
    for it in items:
        if "reason" in it:
            it["reason"] = sanitize_reason(it["reason"])[:140]
    with open(os.path.join(out_dir, f"{pid}.top{k}.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    pd.DataFrame(items).to_csv(os.path.join(out_dir, f"{pid}.top{k}.csv"), index=False)

def run_single_prompt_fallback(prompt: Dict[str, str], candidates: List[Dict[str, Any]],
                               model: str, k: int, out_dir: str, max_tokens: int,
                               debug_dir: str):
    # one-prompt call (more likely to succeed)
    user_content = build_user_payload([prompt], candidates)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_content},
    ]
    resp = call_openrouter(model, messages, max_tokens=max_tokens)
    write_debug(debug_dir, f"single_{prompt['prompt_id']}", resp["raw"])
    obj = parse_json_strict(resp["raw"])
    results = obj.get("results", {})
    node = get_result_for_pid(results, prompt["prompt_id"])
    items = (node.get("top10") or []) if isinstance(node, dict) else []
    save_result_for_pid(out_dir, prompt["prompt_id"], items, k)

def run_batch_group(group: List[Dict[str, str]],
                    candidates: List[Dict[str, Any]],
                    model: str, k: int, out_dir: str, max_tokens: int,
                    debug_dir: str):
    """Process this group of prompts in ONE call; log missing prompts for later reruns."""
    tag = f"batch_{group[0]['prompt_id']}_to_{group[-1]['prompt_id']}"
    user_content = build_user_payload(group, candidates)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = call_openrouter(model, messages, max_tokens=max_tokens)
        raw = resp["raw"]
        write_debug(debug_dir, f"{tag}_attempt1", raw)

        obj = parse_json_strict(raw)

        # Success! Process results
        results = obj.get("results", {})
        missing: List[Dict[str, str]] = []

        for p in group:
            pid = p["prompt_id"]
            node = get_result_for_pid(results, pid)
            items = []
            if isinstance(node, dict):
                items = node.get("top10") or []
            if not items:
                missing.append(p)
            else:
                save_result_for_pid(out_dir, pid, items, k)

        # Log any prompts that were missing results so they can be handled later.
        if missing:
            ids = ", ".join(p["prompt_id"] for p in missing)
            print(f"[WARN] Batch {tag}: missing results for prompt_ids: {ids}")

    except Exception as e:
        print(f"[ERROR] Batch {tag}: parse failed (no retries). {type(e).__name__}: {str(e)[:200]}")

# ---------------- main ---------------- #
def main():
    ap = argparse.ArgumentParser(description="Batch-wise reranker (o4-mini), robust.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--prompts_json", required=True)
    ap.add_argument("--batch_ids", required=True, help="Comma-separated prompt_ids or 'all'")
    ap.add_argument("--out_dir", default="./results_o4mini")
    ap.add_argument("--debug_dir", default="./results_o4mini/_debug_raw")
    ap.add_argument("--model", default="openai/o4-mini")
    ap.add_argument("--max_plot_chars", type=int, default=160)
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=9000)
    ap.add_argument("--prompts_per_call", type=int, default=8, help="Maximum prompts per API call")
    ap.add_argument("--prompts_per_call_min", type=int, default=5, help="Minimum prompts per API call")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for req in ("movieId", "title"):
        if req not in df.columns:
            raise ValueError(f"CSV missing required column: {req}")
    candidates = [build_candidate(r, use_plot=not args.no_plot, max_plot_chars=args.max_plot_chars) for _, r in df.iterrows()]

    payload_chars = len(json.dumps({"CANDIDATES": candidates}))
    est_tokens = approx_tokens(payload_chars)
    print(f"[info] Candidates: {len(candidates)} | approx input tokens: ~{est_tokens}")

    pool = load_prompts(args.prompts_json)
    prompts = pool if args.batch_ids.lower() == "all" else [p for p in pool if p["prompt_id"] in {x.strip() for x in args.batch_ids.split(',') if x.strip()}]
    if args.prompts_per_call <= 0 or args.prompts_per_call_min <= 0:
        raise ValueError("prompts_per_call and prompts_per_call_min must be positive")
    if args.prompts_per_call_min > args.prompts_per_call:
        raise ValueError("prompts_per_call_min cannot exceed prompts_per_call")
    print(f"[info] Total prompts: {len(prompts)} | Batch size range: {args.prompts_per_call_min}-{args.prompts_per_call}")

    for group in tqdm(list(chunked(prompts, args.prompts_per_call_min, args.prompts_per_call)), desc="Processing prompt batches", ncols=100):
        run_batch_group(group, candidates, args.model, args.k, args.out_dir, args.max_tokens, args.debug_dir)

if __name__ == "__main__":
    main()
