
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse, json, re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch

# Hugging Face auth is optional; provide HF_TOKEN or HUGGINGFACEHUB_API_TOKEN when needed.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

def pair_agreement_from_margin(s: torch.Tensor, y01: torch.Tensor, tol: float) -> torch.Tensor:
    j = torch.where(y01 > 0.5, torch.tensor(1.0, device=s.device), torch.tensor(-1.0, device=s.device))
    sign_s = torch.where(s.abs() <= tol, torch.zeros_like(s), torch.sign(s))
    return (sign_s * j).to(torch.int8)

def summarize(agree_np: np.ndarray) -> Tuple[int, int, int, float, float]:
    pos = int((agree_np == 1).sum())
    neg = int((agree_np == -1).sum())
    ties = int((agree_np == 0).sum())
    N = int(agree_np.size)
    acc_no_ties = pos / max(1, (pos + neg))
    acc_ties_half = (pos + 0.5 * ties) / max(1, N)
    return pos, neg, ties, acc_no_ties, acc_ties_half

def extract_cols(df: pd.DataFrame, id_col: str, text_col_candidates: List[str]) -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"Missing id column '{id_col}' in mapping parquet")
    for c in text_col_candidates:
        if c in df.columns:
            out = df[[id_col, c]].rename(columns={c: "text"})
            out["text"] = out["text"].astype(str).fillna("")
            return out
    raise ValueError(f"No text column found; tried {text_col_candidates}")

def safe_group_report(df_slice: pd.DataFrame, agree: np.ndarray, col: str) -> None:
    if col not in df_slice.columns:
        return
    def agg(g: pd.DataFrame) -> pd.Series:
        A = agree[g.index.to_numpy()]
        p, n, t, a1, a2 = summarize(A)
        return pd.Series({
            "+1": p, "-1": n, "0(ties)": t,
            "agree_no_ties": round(a1, 4),
            "agree_ties_0p5": round(a2, 4),
            "count": len(A),
        })
    try:
        rep = df_slice.groupby(col, dropna=False, sort=True).apply(agg, include_groups=False)
    except TypeError:
        rep = df_slice.groupby(col, dropna=False, sort=True).apply(agg)
    print(f"\nBy {col}:\n", rep)

def load_triplets(features_path: str, prompt_text_path: str, movie_text_path: str, split: str):
    df = pd.read_parquet(features_path)
    if split != "all":
        if "split" not in df.columns:
            raise ValueError("No 'split' column; run your split persister first.")
        df = df[df["split"] == split].copy()
    df = df.reset_index(drop=True)
    required = {"prompt_id", "movieA", "movieB", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"features parquet missing columns: {missing}")
    pmap = extract_cols(pd.read_parquet(prompt_text_path), "prompt_id",
                        ["prompt_text", "text", "query", "q_text"])
    mmap = extract_cols(pd.read_parquet(movie_text_path), "movie_id",
                        ["movie_text", "text", "doc", "content"])
    prompt_text_by_id: Dict[Any, str] = dict(zip(pmap["prompt_id"], pmap["text"]))
    movie_text_by_id: Dict[Any, str] = dict(zip(mmap["movie_id"], mmap["text"]))
    return df, prompt_text_by_id, movie_text_by_id

def chat_margin(model: str, query: str, docA: str, docB: str, temperature: float = 0.0, max_new_tokens: int = 64) -> float:
    try:
        from openai import OpenAI as _OpenAI
    except Exception:
        return 0.0
    client = _OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
    system = ("You are a strict relevance judge for movie recommendations. "
              "Given a user query and two movie descriptions, rate each description's relevance "
              "to the query on a continuous scale in [0,1]. Higher = more relevant. Return only JSON.")
    user = (
        "Query:\n" + str(query) + "\n\n"
        "Document A:\n" + str(docA) + "\n\n"
        "Document B:\n" + str(docB) + "\n\n"
        "Respond ONLY with JSON: {\"scoreA\": <float in [0,1]>, \"scoreB\": <float in [0,1]>}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_new_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
        data = json.loads(text)
        sA = float(data.get("scoreA", 0.5)); sB = float(data.get("scoreB", 0.5))
        sA = max(0.0, min(1.0, sA)); sB = max(0.0, min(1.0, sB))
        return sA - sB
    except Exception:
        return 0.0

def main():
    ap = argparse.ArgumentParser(description="Pairwise eval with Qwen2.5 chat-judge over HF Router (no llm4ranking).")
    ap.add_argument("--features", required=True)
    ap.add_argument("--prompt_text", required=True)
    ap.add_argument("--movie_text", required=True)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--tie_tol", type=float, default=0.05)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct:nebius")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    df, pmap, mmap = load_triplets(args.features, args.prompt_text, args.movie_text, args.split)
    margins, y_list = [], []
    for _, r in df.iterrows():
        pid = r["prompt_id"]; a = int(r["movieA"]); b = int(r["movieB"])
        prompt = str(pmap.get(pid, "")); docA = str(mmap.get(a, "")); docB = str(mmap.get(b, ""))
        m = chat_margin(args.model, prompt, docA, docB, args.temperature, args.max_new_tokens)
        margins.append(float(m)); y_list.append(float(r["y"]))

    s = torch.from_numpy(np.asarray(margins, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_list, dtype=np.float32))
    agree = pair_agreement_from_margin(s, y, args.tie_tol).cpu().numpy()
    p, n, t, a1, a2 = summarize(agree)
    print(f"\nQwen2.5 chat-judge — split={{args.split}} tol={{args.tie_tol}}")
    print(f"Model: {{args.model}}")
    print(f"Overall: +1={{p}} -1={{n}} 0(ties)={{t}} total={{len(df)}}")
    print(f"Agreement (no ties): {{a1:.4f}}")
    print(f"Agreement (ties=0.5): {{a2:.4f}}")
    for col in ("difficulty", "category"):
        safe_group_report(df, agree, col)

if __name__ == "__main__":
    main()
