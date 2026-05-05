#!/usr/bin/env python
"""
Evaluate LLM4Ranking as a PAIRWISE scorer on (prompt, movieA, movieB).

Inputs:
  - features parquet: columns ⊇ {prompt_id, movieA, movieB, y [, split]}
      y ∈ {0,1} where 1 => A preferred, 0 => B preferred
  - prompt_text parquet: has (prompt_id, <one text column>)
  - movie_text  parquet: has (movie_id, <one text column>)

Example:
  python -m src.evaluations.models.llm4rank_eval \
    --features artifacts/router/features_sum.with_splits.parquet \
    --prompt_text artifacts/prompts/prompt_text.parquet \
    --movie_text artifacts/movies/movie_text.parquet \
    --split test \
    --approach rankgpt \
    --model_type hf \
    --model_name /mnt/nas/sakshipandey/main/models/Qwen2.5-1.5B-Instruct \
    --tie_tol 0.05
"""
import argparse
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ------------------------------ helpers ------------------------------ #

def pair_agreement_from_margin(s: torch.Tensor, y01: torch.Tensor, tol: float) -> torch.Tensor:
    """
    s: margin logits (A vs B). Positive means A is better.
    y01: label in {0,1}; 1 => A preferred, 0 => B preferred.
    tol: tiny tolerance band for ties.
    Returns tensor in {-1,0,+1}: +1 correct, -1 wrong, 0 tie.
    """
    # map label to judgment in {+1,-1}
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
            # ensure text is str
            out["text"] = out["text"].astype(str).fillna("")
            return out
    raise ValueError(f"No text column found; tried {text_col_candidates}")


def safe_group_report(df_slice: pd.DataFrame, agree: np.ndarray, col: str) -> None:
    if col not in df_slice.columns:
        return

    def agg(g: pd.DataFrame) -> pd.Series:
        A = agree[g.index.to_numpy()]  # indexes aligned because we reset_index below
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
        # pandas < 2.2
        rep = df_slice.groupby(col, dropna=False, sort=True).apply(agg)
    print(f"\nBy {col}:\n", rep)


# ------------------------------ main ------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--prompt_text", required=True)
    ap.add_argument("--movie_text", required=True)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--tie_tol", type=float, default=0.05)

    # LLM4Ranking knobs
    ap.add_argument("--approach", default="rankgpt",
                    help="Approach supported by llm4ranking (e.g., rankgpt, first, prp, etc.)")
    ap.add_argument("--model_type", default="hf",
                    help="Backend for llm4ranking (e.g., hf, openai, vllm)")
    ap.add_argument("--model_name", required=True,
                    help="HF model id/local path or provider-specific name")
    # Optional generation/control knobs (best-effort pass-through)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)  # llm4ranking API is usually per-query; keep 1 by default
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load data
    df = pd.read_parquet(args.features)
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError("No 'split' column; run make_split to persist splits first.")
        df = df[df["split"] == args.split].copy()
    df = df.reset_index(drop=True)  # CRITICAL to keep slice indices aligned

    required = {"prompt_id", "movieA", "movieB", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"features parquet missing columns: {missing}")

    pmap = extract_cols(pd.read_parquet(args.prompt_text), "prompt_id",
                        ["prompt_text", "text", "query", "q_text"])
    mmap = extract_cols(pd.read_parquet(args.movie_text), "movie_id",
                        ["movie_text", "text", "doc", "content"])
    prompt_text_by_id: Dict[Any, str] = dict(zip(pmap["prompt_id"], pmap["text"]))
    movie_text_by_id: Dict[Any, str] = dict(zip(mmap["movie_id"], mmap["text"]))

    # Build reranker
    try:
        from llm4ranking import Reranker
    except Exception as e:
        raise RuntimeError("Install llm4ranking (`pip install -e .`) before running.") from e

    # Best-effort to pass generation args if supported by the library version
    init_kwargs = {
        "reranking_approach": args.approach,
        "model_type": args.model_type,
        "model_name": args.model_name,
    }
    # optional pass-throughs (some versions accept **kwargs or gen_kwargs)
    extra_gen = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # other safe defaults could go here
    }
    rr = None
    try:
        rr = Reranker(**init_kwargs, gen_kwargs=extra_gen)
    except TypeError:
        # Older versions without gen_kwargs
        rr = Reranker(**init_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Reranker: {e}")

    # Score pairs
    margins: List[float] = []
    y_list: List[float] = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="scoring"):
        pid = r["prompt_id"]
        a = int(r["movieA"])
        b = int(r["movieB"])
        prompt = prompt_text_by_id.get(pid, "")
        docA = movie_text_by_id.get(a, "")
        docB = movie_text_by_id.get(b, "")

        # Basic guardrails: avoid None, ensure str
        prompt = "" if prompt is None else str(prompt)
        docA = "" if docA is None else str(docA)
        docB = "" if docB is None else str(docB)

        # llm4ranking API: rerank(query, candidates=[...])
        try:
            res = rr.rerank(query=prompt, candidates=[docA, docB])
        except Exception as e:
            # if the call fails, mark tie to avoid crashing a full run
            margins.append(0.0)
            y_list.append(float(r["y"]))
            continue

        # Try to compute numeric margin if possible, else order-only margin
        scoreA = None
        scoreB = None
        if isinstance(res, (list, tuple)) and len(res) >= 1:
            # Typical shape: [{"text": <cand_text>, "score": float}, ...] sorted desc by score
            try:
                def _text_of(x): return x.get("text", x) if isinstance(x, dict) else str(x)
                def _score_of(x): return float(x["score"]) if isinstance(x, dict) and "score" in x else None

                # best-effort exact text match (assumes engine returns original strings)
                for x in res:
                    tx = _text_of(x)
                    sc = _score_of(x)
                    if tx == docA and scoreA is None:
                        scoreA = sc
                    if tx == docB and scoreB is None:
                        scoreB = sc
            except Exception:
                pass

            if scoreA is not None and scoreB is not None:
                margins.append(float(scoreA - scoreB))
            else:
                # order-only fallback: top candidate wins
                top = res[0]
                top_text = top["text"] if isinstance(top, dict) and "text" in top else str(top)
                margins.append(+1.0 if top_text == docA else -1.0)
        else:
            # unexpected shape → safe tie
            margins.append(0.0)

        y_list.append(float(r["y"]))

    # Aggregate & report
    s = torch.from_numpy(np.asarray(margins, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_list, dtype=np.float32))
    agree = pair_agreement_from_margin(s, y, args.tie_tol).cpu().numpy()

    pos, neg, ties, acc_nt, acc_ties = summarize(agree)
    print(f"\nLLM4Ranking (pairwise) — split={args.split}  tol={args.tie_tol}")
    print(f"Model: {args.model_type}::{args.model_name}  approach={args.approach}")
    print(f"Overall: +1={pos}  -1={neg}  0(ties)={ties}  total={len(df)}")
    print(f"Agreement (no ties): {acc_nt:.4f}")
    print(f"Agreement (ties=0.5): {acc_ties:.4f}")

    # Slices (if present)
    safe_group_report(df, agree, "difficulty")
    safe_group_report(df, agree, "category")


if __name__ == "__main__":
    # optional: make HF cache explicit if user set env var
    if "HF_HUB_CACHE" in os.environ:
        os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
    main()
