#src/evaluations/router/eval_router.py
#!/usr/bin/env python
"""Evaluate the trained router with +1/0/−1 agreement (tiny tie tolerance = 0.05)."""
import argparse
import pandas as pd
import torch
from src.router.mlp_router import RouterMLP
from src.router.logger_utils import setup_router_logger, log_config
from src.cli.train_router import _prepare_feature_matrix, TIE_TOL  # reuse feature engineering

def pair_agreement_from_margin(s: torch.Tensor, judgment: torch.Tensor, tol: float = TIE_TOL):
    """
    s: tensor [N] fused margin (A vs B), A better if s>0.
    judgment: tensor [N] in {+1, -1} where +1 means "A preferred", -1 means "B preferred".
    returns tensor [N] in {-1, 0, +1}:
      +1 if sign(s) matches judgment, -1 if it contradicts, 0 if |s| <= tol (tie)
    """
    sign_s = torch.where(s.abs() <= tol, torch.zeros_like(s), torch.sign(s))
    res = sign_s * judgment
    return res.to(torch.int8)

def summarize_agreement(arr):
    # arr: numpy array in {-1,0,+1}
    import numpy as np
    pos = int((arr == 1).sum())
    neg = int((arr == -1).sum())
    ties = int((arr == 0).sum())
    N = int(arr.size)
    acc_no_ties = pos / max(1, (pos + neg))
    acc_ties_half = (pos + 0.5 * ties) / max(1, N)
    return pos, neg, ties, acc_no_ties, acc_ties_half

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tol", type=float, default=TIE_TOL, help="tiny tolerance for ties on |s|")
    ap.add_argument("--logs_dir", default="logs", help="Directory for log files")
    ap.add_argument("--split", default="all", choices=["all","train","val","test"])
    args = ap.parse_args()

    # Setup logging
    logger, log_file = setup_router_logger(args.logs_dir, name="eval_router")
    logger.info("Starting Router Evaluation")
    
    config = {
        "features": args.features,
        "weights": args.weights,
        "tie_tolerance": args.tol,
    }
    log_config(logger, config, "Evaluation Configuration")

    logger.info(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError("features parquet has no 'split' column; run the split maker first.")
        df = df[df["split"] == args.split].copy()
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows for split={args.split}")

    logger.info(f"Loaded {len(df)} pairs")
    
    X_np, y_np, feature_names, dz_dim, mix_indices = _prepare_feature_matrix(df)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    logger.info(f"Loading model weights from {args.weights}")
    m = RouterMLP(d_in=X.shape[1], dz_dim=dz_dim, mix_indices=mix_indices or None)
    state = torch.load(args.weights, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    logger.info("Model loaded successfully")
    
    logger.info("Running inference...")
    with torch.no_grad():
        s, w = m(X)
        j = torch.where(y > 0.5, torch.tensor(1.0), torch.tensor(-1.0))  # {+1,-1}
        agree = pair_agreement_from_margin(s, j, tol=args.tol).cpu().numpy()

    pos, neg, ties, acc_no_ties, acc_ties_half = summarize_agreement(agree)
    
    # Log overall results
    logger.info("=" * 80)
    logger.info(f"Overall Results (tol={args.tol:.2f}):")
    logger.info(f"  Correct (+1):    {pos:6d}")
    logger.info(f"  Incorrect (-1):  {neg:6d}")
    logger.info(f"  Ties (0):        {ties:6d}")
    logger.info(f"  Total:           {pos+neg+ties:6d}")
    logger.info(f"  Agreement (no ties):    {acc_no_ties:.4f}")
    logger.info(f"  Agreement (ties=0.5):   {acc_ties_half:.4f}")
    logger.info("=" * 80)
    
    print(f"\nOverall (tol={args.tol:.2f}): +1={pos}  -1={neg}  0(ties)={ties}")
    print(f"Agreement (no ties): {acc_no_ties:.4f}")
    print(f"Agreement (ties=0.5): {acc_ties_half:.4f}")

    def slice_report(col):
        if col in df.columns:
            def agg(g):
                A = agree[g.index.values]
                p, n, t, a1, a2 = summarize_agreement(A)
                return pd.Series({
                    "+1": p, "-1": n, "0(ties)": t,
                    "agree_no_ties": round(a1, 4),
                    "agree_ties_0p5": round(a2, 4),
                    "count": len(A),
                })
            try:
                rep = df.groupby(col, dropna=False, sort=True).apply(agg, include_groups=False)
            except TypeError:
                # older pandas w/o include_groups
                rep = df.groupby(col, dropna=False, sort=True).apply(agg)

            print(f"\nBy {col} (tol={args.tol:.2f}):\n", rep)

    slice_report("difficulty")
    slice_report("category")
    
    logger.info(f"\nEvaluation complete! Log file: {log_file}")
    print(f"\nLog file: {log_file}")

if __name__ == "__main__":
    main()


# python -m src.evaluations.router.eval_router \
#   --features artifacts/router/features_sum.with_splits.parquet \
#   --weights artifacts/router/router_mlp_sum.pt \
#   --split test
