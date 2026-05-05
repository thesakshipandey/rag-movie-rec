# src/emotions/emotion_prompt.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

# reuse existing utilities
from src.emotions.emotion_index import (
    EMOS, load_emotion_index, parse_prompt_emotion, js_similarity_batch
)

def _resolve_device(dev_arg: str):
    try:
        import torch
        if dev_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev_arg)
    except Exception:
        # if torch not present or any error
        class _CPU:  # tiny stub
            type = "cpu"
        return _CPU()

def _load_temperature(model_dir: str) -> float:
    try:
        with open(f"{model_dir}/calibration.json","r") as f:
            return float(json.load(f).get("temperature", 1.0))
    except Exception:
        return 1.0

def _emo_vec_from_model(model_dir: str, text: str, device: str="auto", dtype: str="float16", max_len: int=128) -> np.ndarray:
    """
    Run the fine-tuned classifier on the query and return an 8-d prob vector
    aligned to EMOS = [Joy,Trust,Fear,Anticipation,Sadness,Anger,Surprise,Disgust].
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import os

    dmap = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dmap.get(dtype, torch.float16)
    dev = _resolve_device(device)
    if getattr(dev, "type", "cpu") == "cuda":
        try:
            torch.cuda.set_device(dev)
        except Exception:
            pass

    # Ensure we're loading from a local directory
    model_dir = str(Path(model_dir).resolve())
    
    # Set environment variable to disable HF Hub telemetry/validation
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        tok = AutoTokenizer.from_pretrained(
            model_dir, 
            local_files_only=True,
            trust_remote_code=False
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, 
            device_map=None, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            trust_remote_code=False
        ).to(dev).eval()
    finally:
        # Clean up environment variables
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    with open(f"{model_dir}/label_map.json","r") as f:
        lm = json.load(f)
    id2label = {int(k): v for k, v in lm["id2label"].items()}
    canon_map = {e.lower(): i for i, e in enumerate(EMOS)}
    T = _load_temperature(model_dir)

    batch = tok([text], return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    batch = {k: v.to(dev) if hasattr(v, "to") else v for k, v in batch.items()}
    with torch.inference_mode():
        logits = model(**batch).logits / T
        p = logits.softmax(dim=-1)[0].float().cpu().numpy()

    out = np.zeros(8, dtype=np.float32)
    for j in range(len(p)):
        lab = str(id2label.get(j, "")).strip().lower()
        if lab in canon_map:
            out[canon_map[lab]] = float(p[j])
        else:
            alias = {
                "happiness": "joy", "happy": "joy",
                "anticipate": "anticipation", "surprised": "surprise",
                "disgusted": "disgust", "angry": "anger",
            }.get(lab)
            if alias and alias in canon_map:
                out[canon_map[alias]] = float(p[j])

    s = out.sum()
    if s > 0:
        out /= s
    else:
        out[:] = 1.0 / 8.0
    return out

def infer_prompt_vector(
    query: str,
    emo_model_dir: Optional[str] = None,
    prompt_emotion: Optional[str] = None,
    device: str = "auto",
    dtype: str = "float16",
    max_len: int = 128,
) -> Tuple[np.ndarray, str]:
    """
    Decide the prompt emotion vector:
      1) If prompt_emotion provided -> parse & return ("prompt")
      2) Else if emo_model_dir provided -> model inference ("model")
      3) Else -> keyword lexicon from query ("lexicon")
    """
    if prompt_emotion:
        p = parse_prompt_emotion(prompt_emotion, query_text=query)
        return p, "prompt"
    if emo_model_dir:
        try:
            p = _emo_vec_from_model(emo_model_dir, query, device=device, dtype=dtype, max_len=max_len)
            return p, "model"
        except Exception as e:
            # If model loading fails, fall back to lexicon
            import warnings
            warnings.warn(f"Failed to load emotion model from {emo_model_dir}: {e}. Falling back to lexicon-based inference.")
            return parse_prompt_emotion(None, query_text=query), "lexicon_fallback"
    # fallback
    return parse_prompt_emotion(None, query_text=query), "lexicon"

def score_movies_emotion(
    movie_ids: Iterable[int],
    emotion_dir: str,
    p_vec: np.ndarray,
    use_uniform_fallback: bool = True,
) -> np.ndarray:
    """
    Return JS-similarity scores aligned to movie_ids.
    """
    mid, M = load_emotion_index(emotion_dir)
    id2row = {int(i): idx for idx, i in enumerate(mid.tolist())}
    rows = []
    for m in movie_ids:
        idx = id2row.get(int(m), -1)
        if idx < 0 and use_uniform_fallback:
            rows.append(np.full(8, 1.0/8, dtype=np.float32))
        else:
            rows.append(M[idx] if idx >= 0 else np.full(8, 1.0/8, dtype=np.float32))
    mat = np.stack(rows, axis=0).astype(np.float32)
    return js_similarity_batch(mat, p_vec)
