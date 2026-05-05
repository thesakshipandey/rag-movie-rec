#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm4rank_qwen3_eval.py  (cross-encoder only, no llm4ranking)

Evaluate a cross-encoder reranker such as Qwen/Qwen3-Reranker-0.6B on pairwise
(prompt, movieA, movieB). Robust to 1-logit or 2-logit heads.
"""
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------ helpers ------------------------------ #
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

# ---------------------- cross-encoder backend --------------------- #
class CrossEncoderScorer:
    def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 512,
                 local_files_only: bool = False, dtype: str = "auto"):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # choose dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # tokenizer/model (allow local-only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=local_files_only,
        )
        self.model.eval()

        with torch.no_grad():
            tok = self.tokenizer([("q", "d")], padding=True, truncation=True, max_length=16, return_tensors="pt")
            tok = {k: v.to(self.model.device) for k, v in tok.items()}
            test_logits = self.model(**tok).logits
        self.logit_shape = tuple(test_logits.shape[1:])



# ------------------------------ main ------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--prompt_text", required=True)
    ap.add_argument("--movie_text", required=True)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--tie_tol", type=float, default=0.05)

    ap.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--local_files_only", action="store_true", help="Do not hit the network when loading the model/tokenizer")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--show_samples", type=int, default=0, help="Print first N per-row margins")

    args = ap.parse_args()

    # Load data
    df = pd.read_parquet(args.features)
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError("No 'split' column; run make_split to persist splits first.")
        df = df[df["split"] == args.split].copy()
    df = df.reset_index(drop=True)

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

    # Init scorer ONCE
    # 4) pass the new flags when creating the scorer
    scorer = CrossEncoderScorer(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
        dtype=args.dtype,
    )

    print(f"Qwen3-Reranker CE — split={args.split} tol={args.tie_tol}")
    print(f"Model: {args.model}  device={args.device}  head_shape={scorer.logit_shape}")

    margins: List[float] = []
    y_list: List[float] = []

    sample_shown = 0
    # 5) add a progress bar to the scoring loop
    for _, r in tqdm(df.itertuples(index=False), total=len(df), desc="scoring"):
        pid = r["prompt_id"]
        a = int(r["movieA"])
        b = int(r["movieB"])
        prompt = str(prompt_text_by_id.get(pid, ""))
        docA = str(movie_text_by_id.get(a, ""))
        docB = str(movie_text_by_id.get(b, ""))

        try:
            m, sA, sB = scorer.margin_A_minus_B(prompt, docA, docB)
        except Exception:
            m, sA, sB = 0.0, 0.0, 0.0

        if args.verbose and sample_shown < 5:
            print(f"[sample] pid={pid} A={a} B={b}  sA={sA:.4f} sB={sB:.4f}  margin={m:.4f}")
            sample_shown += 1

        margins.append(float(m))
        y_list.append(float(r["y"]))

    s = torch.from_numpy(np.asarray(margins, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_list, dtype=np.float32))
    agree = pair_agreement_from_margin(s, y, args.tie_tol).cpu().numpy()

    p, n, t, a1, a2 = summarize(agree)
    print(f"\nOverall: +1={p}  -1={n}  0(ties)={t}  total={len(df)}")
    print(f"Agreement (no ties): {a1:.4f}")
    print(f"Agreement (ties=0.5): {a2:.4f}")

    for col in ("difficulty", "category"):
        safe_group_report(df, agree, col)

if __name__ == "__main__":
    main()
