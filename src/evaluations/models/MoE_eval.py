#!/usr/bin/env python
"""Evaluate the trained MoE router (RouterMLP) on a chosen split.

Example:
  python -m src.evaluations.models.MoE_eval \
    --features artifacts/router/features_sum.with_splits.parquet \
    --weights artifacts/router/router_mlp_sum.pt \
    --split test --tie_tol 0.05
"""
import argparse
import numpy as np
import pandas as pd
import torch
from src.router.mlp_router import RouterMLP
from src.cli.train_router import _prepare_feature_matrix

def pair_agreement_from_margin(s: torch.Tensor, y01: torch.Tensor, tol: float):
    j = torch.where(y01 > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
    sign_s = torch.where(s.abs() <= tol, torch.zeros_like(s), torch.sign(s))
    return (sign_s * j).to(torch.int8)

def summarize(agree_np: np.ndarray):
    pos = int((agree_np==1).sum()); neg = int((agree_np==-1).sum()); ties= int((agree_np==0).sum())
    N = int(agree_np.size)
    return pos, neg, ties, pos/max(1,pos+neg), (pos+0.5*ties)/max(1,N)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="all", choices=["all","train","val","test"])
    ap.add_argument("--tie_tol", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    if args.split!="all":
        if "split" not in df.columns:
            raise ValueError("No 'split' column; create persistent splits first.")
        df = df[df["split"]==args.split].copy().reset_index(drop=True)
        if df.empty: raise ValueError(f"No rows in split={args.split}")

    X_np, y_np, feature_names, dz_dim, mix_indices = _prepare_feature_matrix(df)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    m = RouterMLP(d_in=X.shape[1], dz_dim=dz_dim, mix_indices=mix_indices or None)
    state = torch.load(args.weights, map_location="cpu")
    m.load_state_dict(state)
    m.eval()

    with torch.no_grad():
        s, _w = m(X)
        agree = pair_agreement_from_margin(s, y, args.tie_tol).cpu().numpy()

    pos,neg,ties,acc_nt,acc_ties = summarize(agree)
    print(f"\nMoE Router  split={args.split}  tol={args.tie_tol}")
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

# python -m src.evaluations.models.MoE_eval \
#   --features artifacts/router/features_sum.with_splits.parquet \
#   --weights artifacts/router/router_mlp_sum.pt \
#   --split test --tie_tol 0.05