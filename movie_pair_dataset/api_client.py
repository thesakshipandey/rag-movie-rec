"""
OpenRouter API client with retry logic
"""
import requests
import time
import json
from typing import Dict, Any, List
from config import (
    OPENROUTER_API_KEY,
    MODEL,
    API_BASE_URL,
    MAX_TOKENS_OUTPUT,
    TEMPERATURE,
    TOP_P,
    SEED,
    TIMEOUT_SECONDS,
    MAX_RETRIES,
    BACKOFF_MULTIPLIER,
    DEBUG_DIR
)
import os


def write_debug(tag: str, content: str):
    """Write raw response to debug directory"""
    if not DEBUG_DIR:
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, f"{tag}.raw.txt")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def call_openrouter(
    messages: List[Dict[str, str]],
    max_tokens: int = MAX_TOKENS_OUTPUT,
    debug_tag: str = None
) -> Dict[str, Any]:
    """
    Call OpenRouter API with retry logic

    Args:
        messages: List of message dicts (system, user)
        max_tokens: Max output tokens
        debug_tag: Tag for debug file (optional)

    Returns:
        Dict with 'raw' (response text) and 'full' (complete API response)

    Raises:
        RuntimeError: If all retries fail
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "seed": SEED,
    }

    # For reasoning models, skip JSON format enforcement
    if "o4" not in MODEL.lower() and "o1" not in MODEL.lower():
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.cli",
        "X-Title": "context-heavy-judge-reranker",
    }

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_BASE_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT_SECONDS
            )
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if "choices" not in data or not data["choices"]:
                error_msg = data.get("error", {}).get("message", str(data))
                raise ValueError(f"API returned no choices. Error: {error_msg}")

            msg = data["choices"][0]["message"]

            # Extract text from various possible locations
            text = (msg.get("content") or "").strip()

            # Check tool_calls (some models use this)
            if not text:
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function") or {}
                    if fn.get("arguments"):
                        text = fn["arguments"]
                        break

            # If still empty but reasoning exists, raise error
            if not text:
                if "reasoning" in msg and msg.get("reasoning"):
                    raise ValueError(
                        f"Model returned reasoning but no content. "
                        f"Reasoning length: {len(str(msg.get('reasoning', '')))} chars"
                    )
                # Last resort
                text = json.dumps(msg, ensure_ascii=False)

            # Write debug output
            if debug_tag:
                write_debug(debug_tag, text)

            return {"raw": text, "full": data}

        except requests.exceptions.HTTPError as e:
            last_error = e
            status_code = e.response.status_code if hasattr(e, 'response') else 0

            # Handle rate limiting with exponential backoff
            if status_code == 429:
                wait_time = BACKOFF_MULTIPLIER * (2 ** attempt) + attempt * 2
                print(f"[WARN] Rate limited. Waiting {wait_time:.1f}s... ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue

            # Log error
            print(f"[ERROR] HTTP {status_code}: {e}")
            if hasattr(e, 'response'):
                print(f"[ERROR] Response: {e.response.text[:500]}")

            # Retry with backoff
            if attempt < MAX_RETRIES - 1:
                wait_time = BACKOFF_MULTIPLIER * (2 ** attempt)
                print(f"[INFO] Retrying in {wait_time:.1f}s... ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait_time)

        except Exception as e:
            last_error = e
            print(f"[ERROR] {type(e).__name__}: {e}")

            if attempt < MAX_RETRIES - 1:
                wait_time = BACKOFF_MULTIPLIER * (2 ** attempt)
                print(f"[INFO] Retrying in {wait_time:.1f}s... ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait_time)

    raise RuntimeError(
        f"OpenRouter call failed after {MAX_RETRIES} retries. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )
