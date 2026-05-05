#!/usr/bin/env python
"""Evaluate BGE cross-encoder (bge-reranker-v2-m3) as a pairwise scorer.

Inputs:
  - features parquet (needs columns: prompt_id, movieA, movieB, y [, split])
  - prompt_text parquet with columns: {prompt_id, prompt_text}
  - movie_text  parquet with columns: {movie_id, movie_text}

Example:
  python -m src.evaluations.models.bge_reranker_eval \
    --features artifacts/router/features_sum.with_splits.parquet \
    --prompt_text artifacts/prompts/prompt_text.parquet \
    --movie_text artifacts/movies/movie_text.parquet \
    --split test --batch_size 64 --tie_tol 0.05
"""
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def pair_agreement_from_margin(s: torch.Tensor, y01: torch.Tensor, tol: float):
    j = torch.where(y01 > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
    sign_s = torch.where(s.abs() <= tol, torch.zeros_like(s), torch.sign(s))
    return (sign_s * j).to(torch.int8)

def summarize(agree_np: np.ndarray):
    pos = int((agree_np==1).sum()); neg = int((agree_np==-1).sum()); ties= int((agree_np==0).sum())
    N = int(agree_np.size)
    return pos, neg, ties, pos/max(1,pos+neg), (pos+0.5*ties)/max(1,N)

def extract_cols(df, id_col, text_col_candidates):
    if id_col not in df.columns:
        raise ValueError(f"Missing id column '{id_col}' in mapping parquet")
    for c in text_col_candidates:
        if c in df.columns:
            return df[[id_col, c]].rename(columns={c: "text"})
    raise ValueError(f"No text column found; tried {text_col_candidates}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--prompt_text", required=True)
    ap.add_argument("--movie_text", required=True)
    ap.add_argument("--split", default="all", choices=["all","train","val","test"])
    ap.add_argument("--tie_tol", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=1)  # FlagReranker API is pair-based; keep 1
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--use_fp16", action="store_true")
    args = ap.parse_args()

    # Load maps
    df = pd.read_parquet(args.features)
    if args.split!="all":
        if "split" not in df.columns:
            raise ValueError("No 'split' column; create persistent splits first.")
        df = df[df["split"]==args.split].copy().reset_index(drop=True)
        if df.empty: raise ValueError(f"No rows in split={args.split}")

    req = {"prompt_id","movieA","movieB","y"}
    missing = req - set(df.columns)
    if missing: raise ValueError(f"features missing {missing}")

    pmap = extract_cols(pd.read_parquet(args.prompt_text), "prompt_id",
                        ["prompt_text","text","query","q_text"])
    mmap = extract_cols(pd.read_parquet(args.movie_text), "movie_id",
                        ["movie_text","text","doc","content"])
    p_dict = dict(zip(pmap["prompt_id"], pmap["text"]))
    m_dict = dict(zip(mmap["movie_id"], mmap["text"]))

    # Model
    try:
        from FlagEmbedding import FlagReranker
    except Exception as e:
        raise RuntimeError("pip install FlagEmbedding first") from e

    reranker = FlagReranker(args.model, use_fp16=args.use_fp16)

    margins = []
    y_list  = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="scoring"):
        pid = r["prompt_id"]; a = int(r["movieA"]); b = int(r["movieB"])
        prompt = p_dict.get(pid, "")
        docA   = m_dict.get(a, "")
        docB   = m_dict.get(b, "")
        sA = float(reranker.compute_score([prompt, docA]))
        sB = float(reranker.compute_score([prompt, docB]))
        margins.append(sA - sB)
        y_list.append(float(r["y"]))

    s = torch.from_numpy(np.array(margins, dtype=np.float32))
    y = torch.from_numpy(np.array(y_list, dtype=np.float32))
    agree = pair_agreement_from_margin(s, y, args.tie_tol).numpy()

    pos,neg,ties,acc_nt,acc_ties = summarize(agree)
    print(f"\nBGE reranker  split={args.split}  tol={args.tie_tol}")
    print(f"Overall: +1={pos}  -1={neg}  0(ties)={ties}  total={len(df)}")
    print(f"Agreement (no ties): {acc_nt:.4f}")
    print(f"Agreement (ties=0.5): {acc_ties:.4f}")

    def slice_report(col):
        if col in df.columns:
            def agg(g):
                A = agree[g.index.to_numpy()]
                p,n,t,a1,a2 = summarize(A)
                return pd.Series({"+1":p,"-1":n,"0(ties)":t,"agree_no_ties":round(a1,4),
                                  "agree_ties_0p5":round(a2,4),"count":len(A)})
            try:
                rep = df.groupby(col, dropna=False, sort=True).apply(agg, include_groups=False)
            except TypeError:
                rep = df.groupby(col, dropna=False, sort=True).apply(agg)
            print(f"\nBy {col}:\n", rep)

    slice_report("difficulty")
    slice_report("category")

if __name__ == "__main__":
    main()

# python -m src.evaluations.models.bge_reranker_eval \
#   --features artifacts/router/features_sum.with_splits.parquet \
#   --prompt_text artifacts/prompts/prompt_text.parquet \
#   --movie_text artifacts/movies/movie_text.parquet \
#   --split test --batch_size 64 --tie_tol 0.05