#!/usr/bin/env python
"""
Flip A/B with probability p to remove position bias.

Usage:
  python -m src.cli.ab_shuffle \
    --in  artifacts/router/features_sum.with_splits.parquet \
    --out artifacts/router/features_sum.with_splits.shuf.parquet \
    --ab_shuffle train --ab_prob 0.5 --only_easy

Notes:
- Requires a 'split' column if --ab_shuffle=train.
- Flips: movieA <-> movieB, dz_* *= -1, y <- 1 - y
- Leaves metadata (difficulty/category/… ) untouched.
"""
import argparse, pandas as pd, numpy as np

DZ_COLS = ["dz_alpha","dz_beta","dz_gamma","dz_delta"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="fin",  required=True)
    ap.add_argument("--out", dest="fout", required=True)
    ap.add_argument("--ab_shuffle", choices=["none","train","all"], default="train",
                    help="Where to apply A/B flipping.")
    ap.add_argument("--ab_prob", type=float, default=0.5, help="Flip probability.")
    ap.add_argument("--only_easy", action="store_true", help="Restrict flips to difficulty=='easy'.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.fin)

    if args.ab_shuffle == "none":
        df.to_parquet(args.fout, index=False)
        print("No shuffling requested. Wrote:", args.fout)
        return

    if args.ab_shuffle == "train":
        if "split" not in df.columns:
            raise ValueError("Need a 'split' column for --ab_shuffle=train.")
        mask = (df["split"] == "train")
    else:  # all
        mask = pd.Series(True, index=df.index)

    if args.only_easy and "difficulty" in df.columns:
        mask = mask & (df["difficulty"].astype(str) == "easy")

    # choose which rows to flip
    to_flip = mask & (rng.random(len(df)) < args.ab_prob)
    n_flip = int(to_flip.sum())
    print(f"Flipping {n_flip} rows (p={args.ab_prob}) | scope={args.ab_shuffle}"
          + (" | only_easy" if args.only_easy else ""))

    if n_flip > 0:
        idx = df.index[to_flip]

        # 1) swap movieA/movieB if present
        for a, b in [("movieA","movieB")]:
            if a in df.columns and b in df.columns:
                tmp = df.loc[idx, a].copy()
                df.loc[idx, a] = df.loc[idx, b]
                df.loc[idx, b] = tmp

        # 2) flip all Δz columns (A-B -> B-A)
        for c in DZ_COLS:
            if c in df.columns:
                df.loc[idx, c] = -df.loc[idx, c]

        # 3) flip label: y \in {0,1} with y=1 meaning A wins
        if "y" in df.columns:
            df.loc[idx, "y"] = 1 - df.loc[idx, "y"]

    # simple sanity
    if "split" in df.columns:
        by = df.groupby("split")["y"].mean().to_dict()
        print("p(y=+1) by split:", {k: round(v,3) for k,v in by.items()})
    else:
        print("p(y=+1):", round(float(df["y"].mean()),3))

    df.to_parquet(args.fout, index=False)
    print("Wrote:", args.fout)

if __name__ == "__main__":
    main()

# python -m src.cli.ab_shuffle \
#   --in  artifacts/router/features_sum.with_splits.parquet \
#   --out artifacts/router/features_sum.with_splits.shuf.parquet \
#   --ab_shuffle train --ab_prob 0.5 --only_easy