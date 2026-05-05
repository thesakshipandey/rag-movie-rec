#!/usr/bin/env python
"""Evaluate a RankNet baseline checkpoint (linear or MLP) on your pairwise dataset.

Example:
  python -m src.evaluations.models.ranknet_eval \
    --features artifacts/router/features_sum.with_splits.parquet \
    --weights  artifacts/router/ranknet_mlp.pt \
    --split test \
    --feature_cols dz_alpha dz_beta dz_gamma dz_delta \
    --tie_tol 0.05
"""
import argparse, numpy as np, pandas as pd, torch
from torch import nn

DEFAULT_FEATURE_COLS = ["dz_alpha","dz_beta","dz_gamma","dz_delta"]

# ---------- models (match training) ----------
class LinearRankNet(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(in_dim))
    def forward(self, dz):  # [B,D]
        return dz @ self.w

class MLPRankNet(nn.Module):
    def __init__(self, in_dim=4, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
    def forward(self, dz):
        return self.net(dz).squeeze(-1)

# ---------- helpers (match training) ----------
def build_splits(df, seed=42):
    """If no 'split', make prompt-level splits stratified by category."""
    if "split" in df.columns:
        assert df.groupby("prompt_id")["split"].nunique().max() == 1
        return df
    prompts = df[["prompt_id","category"]].drop_duplicates()
    split_map = {}
    for cat, g in prompts.groupby("category", dropna=False):
        ids = g["prompt_id"].sample(frac=1.0, random_state=seed).tolist()
        n = len(ids); n_tr = int(0.70*n); n_val = int(0.15*n)
        tr = set(ids[:n_tr]); va = set(ids[n_tr:n_tr+n_val]); te = set(ids[n_tr+n_val:])
        for pid in tr: split_map[pid] = "train"
        for pid in va: split_map[pid] = "val"
        for pid in te: split_map[pid] = "test"
    df = df.copy()
    df["split"] = df["prompt_id"].map(split_map)
    assert df.groupby("prompt_id")["split"].nunique().max() == 1
    return df

def standardize_by_train(df, cols, split_col="split"):
    """Z-score using TRAIN stats (exactly like training)."""
    mu = df[df[split_col]=="train"][cols].mean()
    sd = df[df[split_col]=="train"][cols].std().replace(0, 1.0)
    out = df.copy()
    out[cols] = (out[cols] - mu) / sd
    return out

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
    ap.add_argument("--weights",  required=True)
    ap.add_argument("--split", default="all", choices=["all","train","val","test"])
    ap.add_argument("--tie_tol", type=float, default=0.05)
    ap.add_argument("--model", choices=["linear","mlp"], default="mlp",
                    help="Set to the architecture you trained.")
    ap.add_argument("--feature_cols", nargs="+", default=None,
                    help="Columns used at train time. If omitted, defaults to Δz only.")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)

    # feature columns (must match training!)
    if args.feature_cols is None:
        feat_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
    else:
        missing = [c for c in args.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing in dataframe: {missing}")
        feat_cols = list(args.feature_cols)
    if not feat_cols:
        raise ValueError("No feature columns found/selected.")

    # make/ensure splits, then standardize using TRAIN stats (same as training script)
    df = build_splits(df, seed=42)
    df_std = standardize_by_train(df, feat_cols, split_col="split")

    # row subset for evaluation
    if args.split != "all":
        df_eval = df_std[df_std["split"]==args.split].copy().reset_index(drop=True)
    else:
        df_eval = df_std.copy().reset_index(drop=True)

    needed = {"prompt_id","pair_id","y",*feat_cols}
    missing = needed - set(df_eval.columns)
    if missing:
        raise ValueError(f"Missing columns in features parquet: {missing}")

    X = torch.from_numpy(df_eval[feat_cols].astype(np.float32).values)
    y = torch.from_numpy(df_eval["y"].astype(np.float32).values)

    # model
    in_dim = len(feat_cols)
    if args.model == "linear":
        m = LinearRankNet(in_dim=in_dim)
    else:
        m = MLPRankNet(in_dim=in_dim, hid=32)
    state = torch.load(args.weights, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()

    with torch.no_grad():
        s = m(X)  # margins
        agree = pair_agreement_from_margin(s, y, args.tie_tol).cpu().numpy()

    pos,neg,ties,acc_nt,acc_ties = summarize(agree)
    total = len(df_eval)

    print(f"\nRankNet  split={args.split}  tol={args.tie_tol}")
    print(f"Overall: +1={pos}  -1={neg}  0(ties)={ties}  total={total}")
    print(f"Agreement (no ties): {acc_nt:.4f}")
    print(f"Agreement (ties=0.5): {acc_ties:.4f}")

    def slice_report(col):
        if col in df_eval.columns:
            def agg(g):
                A = agree[g.index.to_numpy()]
                p,n,t,a1,a2 = summarize(A)
                return pd.Series({"+1":p,"-1":n,"0(ties)":t,
                                  "agree_no_ties":round(a1,4),
                                  "agree_ties_0p5":round(a2,4),
                                  "count":len(A)})
            try:
                rep = df_eval.groupby(col, dropna=False, sort=True).apply(agg, include_groups=False)
            except TypeError:
                rep = df_eval.groupby(col, dropna=False, sort=True).apply(agg)
            print(f"\nBy {col}:\n", rep)

    slice_report("difficulty")
    slice_report("category")

if __name__ == "__main__":
    main()


# python -m src.evaluations.models.ranknet_eval \
#   --features artifacts/router/features_sum.with_splits.parquet \
#   --weights  artifacts/router/ranknet_mlp.pt \
#   --model mlp \
#   --split all \
#   --feature_cols dz_alpha dz_beta dz_gamma dz_delta \
#   --tie_tol 0.05
