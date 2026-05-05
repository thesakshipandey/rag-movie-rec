#!/usr/bin/env python
"""Evaluate a single expert by using its Δz column as the margin: s = dz_expert.

Example:
  python -m src.evaluations.models.expert_eval \
    --features artifacts/router/features_sum.with_splits.parquet \
    --split test --expert alpha --tie_tol 0.05
"""
import argparse
import numpy as np
import pandas as pd
import torch

EXPERTS = {"alpha","beta","gamma","delta"}

def pair_agreement_from_margin(s: torch.Tensor, y01: torch.Tensor, tol: float):
    j = torch.where(y01 > 0.5, torch.tensor(1.0), torch.tensor(-1.0))  # {+1,-1}
    sign_s = torch.where(s.abs() <= tol, torch.zeros_like(s), torch.sign(s))
    agree = sign_s * j  # +1 correct, -1 wrong, 0 tie
    return agree.to(torch.int8)

def summarize(agree_np: np.ndarray):
    pos = int((agree_np==1).sum())
    neg = int((agree_np==-1).sum())
    ties= int((agree_np==0).sum())
    N   = int(agree_np.size)
    acc_nt   = pos / max(1, (pos+neg))
    acc_ties = (pos + 0.5*ties) / max(1, N)
    return pos, neg, ties, acc_nt, acc_ties

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", default="all", choices=["all","train","val","test"])
    ap.add_argument("--expert", required=True, choices=sorted(EXPERTS))
    ap.add_argument("--tie_tol", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError("No 'split' column found. Create persistent splits first.")
        df = df[df["split"]==args.split].copy()
        df = df.reset_index(drop=True)
        if df.empty: raise ValueError(f"No rows in split={args.split}")

    col = f"dz_{args.expert}"
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

    y = torch.from_numpy(df["y"].astype("float32").to_numpy())
    s = torch.from_numpy(df[col].astype("float32").to_numpy())
    agree = pair_agreement_from_margin(s, y, args.tie_tol).numpy()
    pos,neg,ties,acc_nt,acc_ties = summarize(agree)

    print(f"\nExpert={args.expert}  split={args.split}  tol={args.tie_tol}")
    print(f"Overall: +1={pos}  -1={neg}  0(ties)={ties}  total={len(df)}")
    print(f"Agreement (no ties): {acc_nt:.4f}")
    print(f"Agreement (ties=0.5): {acc_ties:.4f}")

    def slice_report(colname):
        if colname in df.columns:
            def agg(g):
                A = agree[g.index.to_numpy()]
                p, n, t, a1, a2 = summarize(A)
                return pd.Series({
                    "+1": p, "-1": n, "0(ties)": t,
                    "agree_no_ties": round(a1,4),
                    "agree_ties_0p5": round(a2,4),
                    "count": len(A)
                })
            try:
                rep = df.groupby(colname, dropna=False, sort=True).apply(agg, include_groups=False)
            except TypeError:
                rep = df.groupby(colname, dropna=False, sort=True).apply(agg)
            print(f"\nBy {colname}:\n", rep)

    slice_report("difficulty")
    slice_report("category")

if __name__ == "__main__":
    main()


# python -m src.evaluations.models.expert_eval \
#   --features artifacts/router/features_sum.with_splits.parquet \
#   --split test --expert alpha --tie_tol 0.05