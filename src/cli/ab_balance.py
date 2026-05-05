#!/usr/bin/env python
"""
Balance A/B orientation per group so p(y=+1) hits a target (default 0.5).

It flips the minimal number of rows in each group (e.g., (split,difficulty)):
  - y := 1 - y
  - movieA <-> movieB
  - dz_* := -dz_*
Other columns (mix_*, context features) are left unchanged.

Usage:
  python -m src.cli.ab_balance \
    --in artifacts/router/features_sum.with_splits.parquet \
    --out artifacts/router/features_sum.with_splits.bal.parquet \
    --group_by split difficulty \
    --target 0.5 --seed 42 [--only_easy]

Tips:
- Use group_by "split difficulty" to fix each split/difficulty bucket.
- Add "category" if you want even tighter balancing.
"""
import argparse, numpy as np, pandas as pd

def pick_to_flip(idx_pos, idx_neg, n_pos_now, n_total, target_p, rng):
    desired_pos = int(round(target_p * n_total))
    delta = n_pos_now - desired_pos
    if delta > 0:
        # too many positives; flip some pos -> neg
        k = min(delta, len(idx_pos))
        return rng.choice(idx_pos, size=k, replace=False)
    elif delta < 0:
        # too few positives; flip some neg -> pos
        k = min(-delta, len(idx_neg))
        return rng.choice(idx_neg, size=k, replace=False)
    else:
        return np.array([], dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group_by", nargs="+", default=["split","difficulty"],
                    help="Columns to balance within (e.g., split difficulty [category])")
    ap.add_argument("--target", type=float, default=0.5, help="Target p(y=+1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_easy", action="store_true",
                    help="Only flip rows where difficulty=='easy'")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.inp)
    if "y" not in df.columns:
        raise ValueError("Expected column 'y' in features parquet.")
    if not set(args.group_by).issubset(df.columns):
        missing = set(args.group_by) - set(df.columns)
        raise ValueError(f"group_by columns missing: {missing}")

    # Identify swappable cols
    dz_cols = [c for c in df.columns if c.startswith("dz_")]
    for needed in ["movieA","movieB"]:
        if needed not in df.columns:
            raise ValueError(f"Missing column '{needed}' in features parquet.")

    # Pre-stats
    def stats(label):
        g = df.groupby(args.group_by)["y"].mean().round(3)
        print(f"{label} p(y=+1) by {args.group_by}:")
        print(g.to_string())
        print()
    stats("BEFORE")

    flip_mask = np.zeros(len(df), dtype=bool)

    # Iterate groups
    for keys, g in df.groupby(args.group_by, dropna=False, sort=False):
        if args.only_easy:
            g = g[g.get("difficulty") == "easy"]
            if g.empty:
                continue

        idx = g.index.to_numpy()
        y = g["y"].to_numpy().astype(int)
        idx_pos = idx[y == 1]
        idx_neg = idx[y == 0]
        n_pos = int((y == 1).sum())
        n_tot = int(len(y))
        if n_tot == 0: 
            continue

        chosen = pick_to_flip(idx_pos, idx_neg, n_pos, n_tot, args.target, rng)
        flip_mask[chosen] = True

    # Apply flips
    df2 = df.copy()
    idx_flip = np.where(flip_mask)[0]
    if idx_flip.size:
        # y
        df2.loc[idx_flip, "y"] = 1 - df2.loc[idx_flip, "y"]
        # movieA <-> movieB
        a = df2.loc[idx_flip, "movieA"].copy()
        b = df2.loc[idx_flip, "movieB"].copy()
        df2.loc[idx_flip, "movieA"] = b
        df2.loc[idx_flip, "movieB"] = a
        # dz_* -> -dz_*
        for c in dz_cols:
            df2.loc[idx_flip, c] = -df2.loc[idx_flip, c]

    # Post-stats
    stats("AFTER")

    # Save
    df2.to_parquet(args.out, index=False)
    print(f"Flipped rows: {idx_flip.size} / {len(df)}")
    print(f"Wrote balanced parquet → {args.out}")

if __name__ == "__main__":
    main()

# python -m src.cli.ab_balance \
#   --in artifacts/router/features_sum.with_splits.parquet \
#   --out artifacts/router/features_sum.with_splits.bal.parquet \
#   --group_by split difficulty \
#   --target 0.5 --seed 42 --only_easy