#!/usr/bin/env python
"""Train the MLP router with CASCADE GATING support.

This script trains router models with different cascade threshold values,
allowing the router to learn when to rely on a single dominant expert
vs. using the full mixture.

Example:
  # Train with cascade threshold 0.75 and gating enabled
  python -m src.cli.train_router_cascade \
    --features artifacts/router/features_sum.with_splits.bal.parquet \
    --out artifacts/router/router_cascade_0.75.pt \
    --cascade_threshold 0.75 \
    --gating \
    --epochs 20 --lr 5e-4 --seed 42

  # Train without gating (baseline)
  python -m src.cli.train_router_cascade \
    --features artifacts/router/features_sum.with_splits.bal.parquet \
    --out artifacts/router/router_no_gating.pt \
    --epochs 20 --lr 5e-4 --seed 42
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


def cascade_gating_loss(weights: torch.Tensor, threshold: float = 0.75, strength: float = 0.1):
    """
    Cascade gating regularizer that encourages the router to learn when to
    rely on a single dominant expert vs. using a mixture.
    
    Args:
        weights: [B, 4] expert weights from router
        threshold: minimum weight to be considered dominant
        strength: regularization strength
    
    Returns:
        Scalar loss encouraging dominant expert selection
    """
    max_weights = weights.max(dim=-1)[0]  # [B]
    
    # Encourage decisiveness: if max_weight is high, push it higher
    # If it's low, allow mixture
    gating_signal = torch.sigmoid((max_weights - threshold) * 10.0)
    
    # Entropy penalty when gating signal suggests single expert
    entropy = -(weights * torch.log(weights.clamp(min=1e-8))).sum(dim=-1)  # [B]
    max_entropy = np.log(4.0)  # entropy for uniform distribution over 4 experts
    normalized_entropy = entropy / max_entropy
    
    # When gating_signal is high (max_weight > threshold), penalize high entropy
    cascade_penalty = gating_signal * normalized_entropy
    
    return strength * cascade_penalty.mean()


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
    ap.add_argument("--out", default="artifacts/router/router_cascade.pt", help="Path to save weights")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ent_lambda", type=float, default=1e-3, help="Entropy regularization weight")
    ap.add_argument("--ent_target", type=float, default=1.2, help="Target entropy (approx bits)")
    ap.add_argument("--tie_tol", type=float, default=TIE_TOL, help="tiny tolerance for ties on |s|")
    ap.add_argument("--logs_dir", default="logs", help="Directory for log files")

    # CASCADE PARAMETERS (NEW)
    ap.add_argument("--cascade_threshold", type=float, default=0.75,
                    help="Threshold for cascade gating: max(w) >= threshold to use single expert")
    ap.add_argument("--gating", action="store_true",
                    help="Enable cascade gating during training (default: False)")
    ap.add_argument("--gating_strength", type=float, default=0.1,
                    help="Strength of cascade gating regularizer")

    # A/B shuffling
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
    logger, log_file = setup_router_logger(args.logs_dir, name="train_router_cascade")
    logger.info("=" * 80)
    logger.info("Starting CASCADE-AWARE MLP Router Training")
    logger.info("=" * 80)
    logger.info(f"Features file: {args.features}")
    logger.info(f"Output weights: {args.out}")
    logger.info(f"Cascade threshold: {args.cascade_threshold}")
    logger.info(f"Gating enabled: {args.gating}")
    if args.gating:
        logger.info(f"Gating strength: {args.gating_strength}")

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
        "cascade_threshold": args.cascade_threshold,
        "gating_enabled": args.gating,
        "gating_strength": args.gating_strength if args.gating else 0.0,
        "ab_shuffle_easy": args.ab_shuffle_easy,
        "swap_prob": args.swap_prob,
        "seed": args.seed,
        "focal_gamma": args.focal_gamma,
        "margin_push": args.margin_push,
        "margin_band": args.margin_band,
    }
    log_config(logger, config, "CASCADE Training Configuration")

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
    log_model_summary(logger, model, "Cascade-Aware Router MLP")
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger.info(f"Optimizer: AdamW(lr={args.lr})")

    def run(dl, train: bool):
        model.train(train)
        tot_loss, tot_btl, tot_cascade, tot_ent, tot_margin = 0.0, 0.0, 0.0, 0.0, 0.0
        N = 0
        pos_total = neg_total = ties_total = 0
        
        # Cascade statistics
        dominant_count = 0  # how many samples have max(w) >= threshold
        
        for X, y in dl:
            s, w = model(X)
            
            # BTL loss
            btl = btl_loss(s, y, focal_gamma=args.focal_gamma)
            loss = btl
            tot_btl += btl.item() * len(y)
            
            # Margin push regularizer
            margin_pen = 0.0
            if args.margin_push > 0.0:
                margin_band = max(0.0, args.margin_band)
                margin_pen = F.relu(margin_band - s.abs()).mean()
                loss = loss + args.margin_push * margin_pen
                tot_margin += margin_pen.item() * len(y)
            
            # Entropy regularizer
            ent = -(w * (w.clamp_min(1e-8).log())).sum(dim=-1).mean()
            ent_penalty = args.ent_lambda * F.relu(ent - args.ent_target)
            loss = loss + ent_penalty
            tot_ent += ent_penalty.item() * len(y)
            
            # CASCADE GATING LOSS (NEW)
            if args.gating:
                cascade_loss = cascade_gating_loss(w, threshold=args.cascade_threshold, 
                                                   strength=args.gating_strength)
                loss = loss + cascade_loss
                tot_cascade += cascade_loss.item() * len(y)
            
            # Count dominant samples
            with torch.no_grad():
                max_w = w.max(dim=-1)[0]
                dominant_count += int((max_w >= args.cascade_threshold).sum().item())

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            tot_loss += loss.item() * len(y)
            N += len(y)
            p, n, t = _agreement_counts(s.detach(), y, tol=args.tie_tol)
            pos_total += p
            neg_total += n
            ties_total += t

        loss_avg = tot_loss / max(1, N)
        btl_avg = tot_btl / max(1, N)
        cascade_avg = tot_cascade / max(1, N) if args.gating else 0.0
        ent_avg = tot_ent / max(1, N)
        margin_avg = tot_margin / max(1, N)
        acc_no_ties = pos_total / max(1, (pos_total + neg_total))
        acc_ties_half = (pos_total + 0.5 * ties_total) / max(1, N)
        dominant_pct = 100.0 * dominant_count / max(1, N)
        
        return {
            "loss": loss_avg,
            "btl": btl_avg,
            "cascade": cascade_avg,
            "ent": ent_avg,
            "margin": margin_avg,
            "acc_no_ties": acc_no_ties,
            "acc_ties_half": acc_ties_half,
            "pos": pos_total,
            "neg": neg_total,
            "ties": ties_total,
            "dominant_pct": dominant_pct
        }

    logger.info("=" * 80)
    logger.info("Starting training loop")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for ep in range(1, args.epochs + 1):
        tr_metrics = run(tr, True)
        va_metrics = run(va, False)

        if args.gating:
            print(f"[ep {ep}] threshold={args.cascade_threshold:.2f} "
                  f"| train loss {tr_metrics['loss']:.4f} (btl {tr_metrics['btl']:.4f} + "
                  f"cascade {tr_metrics['cascade']:.4f} + ent {tr_metrics['ent']:.4f}) "
                  f"acc {tr_metrics['acc_no_ties']:.3f} dominant {tr_metrics['dominant_pct']:.1f}% "
                  f"|| val loss {va_metrics['loss']:.4f} acc {va_metrics['acc_no_ties']:.3f} "
                  f"dominant {va_metrics['dominant_pct']:.1f}%")
        else:
            print(f"[ep {ep}] NO GATING "
                  f"| train loss {tr_metrics['loss']:.4f} acc {tr_metrics['acc_no_ties']:.3f} "
                  f"|| val loss {va_metrics['loss']:.4f} acc {va_metrics['acc_no_ties']:.3f}")

        # Log to file
        log_epoch_metrics(logger, ep, args.epochs, tr_metrics, va_metrics)
        
        # Track best model
        if va_metrics['loss'] < best_val_loss:
            best_val_loss = va_metrics['loss']
            logger.info(f"  → New best validation loss: {best_val_loss:.4f}")
        if va_metrics['acc_no_ties'] > best_val_acc:
            best_val_acc = va_metrics['acc_no_ties']
            logger.info(f"  → New best validation accuracy (no-ties): {best_val_acc:.3f}")

    logger.info("=" * 80)
    logger.info(f"Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation accuracy (no-ties): {best_val_acc:.3f}")
    logger.info("=" * 80)

    # Save model with metadata
    save_dict = {
        "state_dict": model.state_dict(),
        "config": config,
        "cascade_threshold": args.cascade_threshold,
        "gating_enabled": args.gating,
        "input_dim": input_dim,
        "dz_dim": dz_dim,
        "mix_indices": mix_indices,
    }
    torch.save(save_dict, args.out)
    logger.info(f"Model saved to: {args.out}")
    logger.info(f"Log file: {log_file}")
    print(f"\nTraining complete! Model saved to: {args.out}")
    print(f"Log file: {log_file}")

if __name__ == "__main__":
    main()

