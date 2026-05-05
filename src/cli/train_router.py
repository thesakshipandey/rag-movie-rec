# src/cli/train_router.py
#!/usr/bin/env python
"""Train the MLP router (MoE + BTL) on Δz features with tiny tie tolerance (0.05).

Example:
  python -m src.cli.train_router \
    --features artifacts/router/features.parquet \
    --out artifacts/router/router_mlp.pt \
    --epochs 10 --lr 5e-4 --ent_lambda 1e-3 --ent_target 1.2 \
    --ab_shuffle_easy train --swap_prob 0.5 --seed 42
"""
import os, argparse, torch, pandas as pd, numpy as np
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from src.router.mlp_router import RouterMLP, btl_loss
from src.router.logger_utils import setup_router_logger, log_config, log_model_summary, log_epoch_metrics

TIE_TOL = 0.05  # tiny tolerance
DZ_COLS = ["dz_alpha", "dz_beta", "dz_gamma", "dz_delta"]
MIX_COLS = ["mix_alpha", "mix_beta", "mix_gamma", "mix_delta"]
NUMERIC_OPTIONAL = ["multi_intent", "cold_user", "has_genre_terms", "has_negation", "has_year", "length_words", "num_genre_terms"]
CATEGORY_COLS = ["category", "difficulty", "primary_expert", "length_bucket", "persona_style"]


def _prepare_feature_matrix(df: pd.DataFrame):
    df_proc = df.copy()
    dz_cols = [c for c in DZ_COLS if c in df_proc.columns]
    if len(dz_cols) != len(DZ_COLS):
        missing = [c for c in DZ_COLS if c not in dz_cols]
        raise ValueError(f"Missing Δz columns in features: {missing}")
    # Derived Δz signals to capture coverage/strength cues
    derived_cols = []
    for col in dz_cols:
        df_proc[f"abs_{col}"] = df_proc[col].abs()
        df_proc[f"pos_{col}"] = df_proc[col].clip(lower=0.0)
        df_proc[f"neg_{col}"] = (-df_proc[col]).clip(lower=0.0)
        df_proc[f"hit_{col}"] = (df_proc[col].abs() > 1e-9).astype("float32")
        derived_cols.extend([f"abs_{col}", f"pos_{col}", f"neg_{col}", f"hit_{col}"])

    mix_cols = [c for c in MIX_COLS if c in df_proc.columns]
    numeric_cols = dz_cols + mix_cols
    numeric_cols += [c for c in NUMERIC_OPTIONAL if c in df_proc.columns]
    numeric_cols += derived_cols
    X_num = df_proc[numeric_cols].fillna(0.0).astype("float32") if numeric_cols else pd.DataFrame(index=df_proc.index)

    cat_cols = [c for c in CATEGORY_COLS if c in df_proc.columns]
    if cat_cols:
        cat_df = pd.get_dummies(df_proc[cat_cols].fillna("UNK"), prefix=cat_cols, dtype="float32")
        X = np.concatenate([X_num.values if numeric_cols else np.zeros((len(df), 0), dtype=np.float32),
                            cat_df.values], axis=1)
        feature_names = numeric_cols + cat_df.columns.tolist()
    else:
        X = X_num.values if numeric_cols else np.zeros((len(df), 0), dtype=np.float32)
        feature_names = numeric_cols[:]

    if X.shape[1] == 0:
        raise ValueError("No feature columns constructed. Check feature preprocessing.")

    y = df_proc["y"].values.astype("float32")
    dz_dim = len(dz_cols)
    mix_indices = [feature_names.index(col) for col in MIX_COLS if col in feature_names]
    return X.astype("float32"), y, feature_names, dz_dim, mix_indices


def make_loaders(df: pd.DataFrame, batch: int = 256, seed: int = 42):
    X, y, feature_names, dz_dim, mix_indices = _prepare_feature_matrix(df)
    if "split" in df.columns:
        tr_idx = df.index[df["split"] == "train"].to_numpy()
        va_idx = df.index[df["split"] == "val"].to_numpy()
    else:
        groups = df["prompt_id"].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, va_idx = next(gss.split(X, y, groups))

    def mk(idx, shuffle):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)

    metadata = {
        "feature_names": feature_names,
        "dz_dim": dz_dim,
        "input_dim": X.shape[1],
        "mix_indices": mix_indices,
    }
    return mk(tr_idx, True), mk(va_idx, False), metadata


def _agreement_counts(s: torch.Tensor, y: torch.Tensor, tol: float = TIE_TOL):
    """
    s: margin logits (A vs B), y in {0,1}. Convert to j in {+1,-1}.
    Returns (#pos, #neg, #ties) under tiny tolerance.
    """
    j = torch.where(y > 0.5, torch.tensor(1.0, device=s.device), torch.tensor(-1.0, device=s.device))
    sign_s = torch.where(s.abs() <= tol, torch.zeros_like(s), torch.sign(s))
    agree = sign_s * j  # +1 correct, -1 wrong, 0 tie
    pos = int((agree > 0).sum().item())
    neg = int((agree < 0).sum().item())
    ties = int((agree == 0).sum().item())
    return pos, neg, ties


def _apply_ab_shuffle_easy(df: pd.DataFrame, mode: str, p: float, seed: int, logger) -> pd.DataFrame:
    """
    Randomly swap A/B *only* for rows with difficulty == 'easy'.
      - If mode == 'train': limit to df['split']=='train' (if split exists).
      - If mode == 'all':   apply on all rows regardless of split.
      - If mode == 'off':   no-op.
    For selected rows with coin flip < p:
      - dz_* *= -1
      - y = 1 - y
      - swap movieA ↔ movieB (if present)
    """
    if mode == "off":
        logger.info("A/B shuffle for easy: OFF")
        return df

    if "difficulty" not in df.columns:
        logger.warning("A/B shuffle requested but 'difficulty' column not found; skipping.")
        return df

    mask = (df["difficulty"].astype(str) == "easy")
    if mode == "train" and "split" in df.columns:
        mask = mask & (df["split"] == "train")

    n_candidates = int(mask.sum())
    if n_candidates == 0:
        logger.info("A/B shuffle: no 'easy' rows matched the criteria; skipping.")
        return df

    rng = np.random.default_rng(seed)
    flip_mask = pd.Series(False, index=df.index)
    to_flip = rng.random(n_candidates) < float(p)
    flip_mask.loc[df.index[mask]] = to_flip

    # Perform flips in-place
    dz_cols_present = [c for c in DZ_COLS if c in df.columns]
    if dz_cols_present:
        df.loc[flip_mask, dz_cols_present] = -df.loc[flip_mask, dz_cols_present].values
    if "y" in df.columns:
        df.loc[flip_mask, "y"] = 1.0 - df.loc[flip_mask, "y"].astype(float).values

    if "movieA" in df.columns and "movieB" in df.columns:
        a_tmp = df.loc[flip_mask, "movieA"].copy()
        df.loc[flip_mask, "movieA"] = df.loc[flip_mask, "movieB"].values
        df.loc[flip_mask, "movieB"] = a_tmp.values

    n_flipped = int(flip_mask.sum())
    logger.info(f"A/B shuffle for easy='{mode}' | swap_prob={p} | flipped={n_flipped}/{n_candidates} candidates "
                f"(seed={seed})")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Parquet with Δz features and labels y")
    ap.add_argument("--out", default="artifacts/router/router_mlp.pt", help="Path to save weights")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ent_lambda", type=float, default=1e-3, help="Entropy regularization weight")
    ap.add_argument("--ent_target", type=float, default=1.2, help="Target entropy (approx bits)")
    ap.add_argument("--tie_tol", type=float, default=TIE_TOL, help="tiny tolerance for ties on |s|")
    ap.add_argument("--logs_dir", default="logs", help="Directory for log files")

    # NEW: control A/B shuffling for easy
    ap.add_argument("--ab_shuffle_easy", choices=["off", "train", "all"], default="off",
                    help="Randomly swap A/B for difficulty=='easy'. 'train' restricts to train split.")
    ap.add_argument("--swap_prob", type=float, default=0.5, help="Probability of swapping a selected easy row")
    ap.add_argument("--seed", type=int, default=42, help="Seed for data shuffling & AB swap RNG")
    ap.add_argument("--focal_gamma", type=float, default=0.0,
                    help="If >0, enables focal weighting on the BTL loss to focus on hard pairs.")
    ap.add_argument("--margin_push", type=float, default=0.0,
                    help="Strength of margin push regularizer (encourages |s| >= margin_band).")
    ap.add_argument("--margin_band", type=float, default=0.2,
                    help="Target |s| band for the margin push regularizer.")

    args = ap.parse_args()

    # Setup logging
    logger, log_file = setup_router_logger(args.logs_dir, name="train_router")
    logger.info("Starting MLP Router Training")
    logger.info(f"Features file: {args.features}")
    logger.info(f"Output weights: {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Log configuration
    config = {
        "features": args.features,
        "output": args.out,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "entropy_lambda": args.ent_lambda,
        "entropy_target": args.ent_target,
        "tie_tolerance": args.tie_tol,
        "logs_dir": args.logs_dir,
        "ab_shuffle_easy": args.ab_shuffle_easy,
        "swap_prob": args.swap_prob,
        "seed": args.seed,
        "focal_gamma": args.focal_gamma,
        "margin_push": args.margin_push,
        "margin_band": args.margin_band,
    }
    log_config(logger, config, "Training Configuration")

    # Load data
    logger.info(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    logger.info(f"Loaded {len(df)} pairs")
    logger.info(f"Columns: {list(df.columns)}")
    if "category" in df.columns:
        logger.info(f"Categories: {df['category'].value_counts().to_dict()}")
    if "difficulty" in df.columns:
        logger.info(f"Difficulties: {df['difficulty'].value_counts().to_dict()}")

    # >>> A/B shuffle (position debias) for 'easy'
    df = _apply_ab_shuffle_easy(df, args.ab_shuffle_easy, args.swap_prob, args.seed, logger)

    tr, va, meta = make_loaders(df, seed=args.seed)
    feature_names = meta["feature_names"]
    dz_dim = meta["dz_dim"]
    input_dim = meta["input_dim"]
    mix_indices = meta["mix_indices"]
    logger.info(f"Train batches: {len(tr)}, Val batches: {len(va)}")
    logger.info(f"Feature dim: {input_dim} (dz_dim={dz_dim})")
    logger.debug(f"Features: {feature_names}")
    
    model = RouterMLP(d_in=input_dim, dz_dim=dz_dim, mix_indices=mix_indices or None)
    log_model_summary(logger, model, "Router MLP")
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger.info(f"Optimizer: AdamW(lr={args.lr})")

    def run(dl, train: bool):
        model.train(train)
        tot_loss, N = 0.0, 0
        pos_total = neg_total = ties_total = 0
        for X, y in dl:
            s, w = model(X)
            loss = btl_loss(s, y, focal_gamma=args.focal_gamma)  # signed BTL inside
            if args.margin_push > 0.0:
                margin_band = max(0.0, args.margin_band)
                margin_pen = F.relu(margin_band - s.abs()).mean()
                loss = loss + args.margin_push * margin_pen
            ent = -(w * (w.clamp_min(1e-8).log())).sum(dim=-1).mean()
            loss = loss + args.ent_lambda * F.relu(ent - args.ent_target)

            if train:
                opt.zero_grad(); loss.backward(); opt.step()

            tot_loss += loss.item() * len(y); N += len(y)
            p, n, t = _agreement_counts(s.detach(), y, tol=args.tie_tol)
            pos_total += p; neg_total += n; ties_total += t

        loss_avg = tot_loss / max(1, N)
        acc_no_ties = pos_total / max(1, (pos_total + neg_total))
        acc_ties_half = (pos_total + 0.5 * ties_total) / max(1, N)
        return loss_avg, acc_no_ties, acc_ties_half, pos_total, neg_total, ties_total

    logger.info("=" * 80)
    logger.info("Starting training loop")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_no_ties, tr_ties_half, tr_pos, tr_neg, tr_ties = run(tr, True)
        va_loss, va_no_ties, va_ties_half, va_pos, va_neg, va_ties = run(va, False)

        print(f"[ep {ep}] tie_tol={args.tie_tol:.2f} "
              f"| train loss {tr_loss:.4f} acc(no-ties) {tr_no_ties:.3f} acc(ties=0.5) {tr_ties_half:.3f} "
              f"| +1/{tr_pos} -1/{tr_neg} 0/{tr_ties} "
              f"|| val loss {va_loss:.4f} acc(no-ties) {va_no_ties:.3f} acc(ties=0.5) {va_ties_half:.3f} "
              f"| +1/{va_pos} -1/{va_neg} 0/{va_ties}")

        # Log epoch metrics
        train_metrics = {
            "loss": tr_loss,
            "acc_no_ties": tr_no_ties,
            "acc_ties_half": tr_ties_half,
            "pos": tr_pos,
            "neg": tr_neg,
            "ties": tr_ties,
        }
        val_metrics = {
            "loss": va_loss,
            "acc_no_ties": va_no_ties,
            "acc_ties_half": va_ties_half,
            "pos": va_pos,
            "neg": va_neg,
            "ties": va_ties,
        }
        log_epoch_metrics(logger, ep, args.epochs, train_metrics, val_metrics)
        
        # Track best model
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            logger.info(f"  → New best validation loss: {best_val_loss:.4f}")
        if va_no_ties > best_val_acc:
            best_val_acc = va_no_ties
            logger.info(f"  → New best validation accuracy (no-ties): {best_val_acc:.3f}")

    logger.info("=" * 80)
    logger.info(f"Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation accuracy (no-ties): {best_val_acc:.3f}")
    logger.info("=" * 80)

    torch.save(model.state_dict(), args.out)
    logger.info(f"Model saved to: {args.out}")
    logger.info(f"Log file: {log_file}")
    print(f"\nTraining complete! Model saved to: {args.out}")
    print(f"Log file: {log_file}")

if __name__ == "__main__":
    main()
