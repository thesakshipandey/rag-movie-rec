# src/cli/train_ranknet_baseline.py
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_FEATURE_COLS = ["dz_alpha","dz_beta","dz_gamma","dz_delta"]

# -------------------------
# Datasets
# -------------------------
class PairwiseDataset(Dataset):
    def __init__(self, df, feature_cols):
        X = df[feature_cols].astype(np.float32).values
        y = df["y"].astype(np.int32).values   # 0/1 → convert
        y = np.where(y > 0, 1.0, -1.0).astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        if "difficulty" in df.columns:
            diff = df["difficulty"].fillna("UNK").astype(str)
        else:
            diff = pd.Series(["UNK"] * len(df), index=df.index)
        if "category" in df.columns:
            cat = df["category"].fillna("UNK").astype(str)
        else:
            cat = pd.Series(["UNK"] * len(df), index=df.index)
        self.diff = diff.tolist()
        self.cat  = cat.tolist()

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        meta = {"difficulty": self.diff[i], "category": self.cat[i]}
        return self.X[i], self.y[i], meta

# -------------------------
# Models
# -------------------------
class LinearRankNet(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        # small random init breaks symmetry; avoids zero-gradient at start
        w0 = 1e-2 * torch.randn(in_dim)
        self.w = nn.Parameter(w0)

    def forward(self, dz):  # dz: [B,4]
        return dz @ self.w  # margin directly since dz = zA - zB

class MLPRankNet(nn.Module):
    def __init__(self, in_dim=4, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1)
        )
        # good defaults for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, dz):         # dz: [B,4]
        return self.net(dz).squeeze(-1)  # margins

# -------------------------
# Utils
# -------------------------
def build_splits(df, seed=42):
    """Create prompt-level splits stratified by category."""
    if "split" in df.columns:
        # Trust existing split but ensure no leakage
        assert df.groupby("prompt_id")["split"].nunique().max() == 1
        return df
    prompts = df[["prompt_id","category"]].drop_duplicates()
    split_map = {}
    for cat, g in prompts.groupby("category"):
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
    """Standardize given columns using TRAIN statistics only."""
    mu = df[df[split_col]=="train"][cols].mean()
    sd = df[df[split_col]=="train"][cols].std().replace(0, 1.0)
    df = df.copy()
    df[cols] = (df[cols] - mu) / sd
    return df, mu, sd

def binary_auc_from_scores(y_true_pos1, scores):
    """
    Compute ROC AUC without sklearn.
    y_true_pos1: array-like of {0,1}
    scores: array-like of floats
    Uses rank-based formula with average ranks for ties.
    """
    y = np.asarray(y_true_pos1, dtype=np.int32)
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    ranks = s.rank(method="average").to_numpy()  # 1..N
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, tie_tol=0.05):
    model.eval()
    total = 0
    correct_nt = 0
    ties = 0
    by_diff = {}
    by_cat = {}

    def upd(bucket, hit, tie):
        a = bucket.setdefault("a", 0)
        b = bucket.setdefault("b", 0)
        t = bucket.setdefault("t", 0)
        bucket["a"] = a + (1 if hit else 0)
        bucket["b"] = b + (0 if tie else 1)  # denom for no-ties acc
        bucket["t"] = t + 1                  # denom for ties=0.5 acc

    # for AUC
    all_scores = []
    all_labels01 = []

    for X, y, meta in loader:
        X = X.to(device)
        y = y.to(device)                     # +1 / -1
        margin = model(X)                    # [B]
        pred = torch.sign(margin)            # -1/0/+1
        is_tie = (margin.abs() < tie_tol)
        ties += is_tie.sum().item()
        total += y.numel()
        correct_nt += ((pred == y) & (~is_tie)).sum().item()

        # accumulate for AUC (convert y to {0,1})
        all_scores.append(margin.detach().cpu().numpy())
        all_labels01.append(((y.detach().cpu().numpy() + 1) / 2).astype(np.int32))

        diffs = meta["difficulty"]
        cats  = meta["category"]
        for i in range(len(y)):
            diff = diffs[i]
            cat  = cats[i]
            hit = (pred[i].item() == y[i].item()) and (abs(margin[i].item()) >= tie_tol)
            tie = abs(margin[i].item()) < tie_tol
            upd(by_diff.setdefault(diff,{}), hit, tie)
            upd(by_cat.setdefault(cat,{}), hit, tie)

    acc_nt = correct_nt / max(1, (total - ties))
    tie_rate = ties / max(1, total)

    # AUC
    scores = np.concatenate(all_scores) if all_scores else np.array([])
    labels01 = np.concatenate(all_labels01) if all_labels01 else np.array([])
    auc = binary_auc_from_scores(labels01, scores) if scores.size else float("nan")

    def fmt(b):
        out = {}
        for k, v in b.items():
            acc_nt_k = v["a"] / (v["b"] + 1e-9) if v["b"]>0 else float("nan")
            out[k] = round(acc_nt_k, 3)
        return dict(sorted(out.items(), key=lambda x: x[0]))

    return acc_nt, tie_rate, float(auc), fmt(by_diff), fmt(by_cat)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="artifacts/router/ranknet_global.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--tie_tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["linear","mlp"], default="linear")
    ap.add_argument("--hidden", type=int, default=32, help="hidden size for MLP model")
    ap.add_argument("--feature_cols", nargs="+", default=None,
                    help="Columns to train on. Use 'auto' to include Δz plus available context priors.")
    ap.add_argument("--no_standardize", action="store_true",
                    help="Skip z-score standardization (use raw feature values).")
    ap.add_argument("--weight_decay", type=float, default=1e-4,
                    help="Weight decay for AdamW.")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader workers.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading:", args.features)
    df = pd.read_parquet(args.features)

    raw_feature_arg = args.feature_cols
    if raw_feature_arg is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
    else:
        if len(raw_feature_arg) == 1 and raw_feature_arg[0].lower() == "auto":
            extras = [
                "mix_alpha","mix_beta","mix_gamma","mix_delta",
                "length_words","num_genre_terms",
                "multi_intent","cold_user",
                "has_genre_terms","has_negation","has_year",
            ]
            selected = []
            for c in DEFAULT_FEATURE_COLS + extras:
                if c in df.columns and c not in selected:
                    selected.append(c)
            feature_cols = selected
        else:
            feature_cols = []
            seen = set()
            for c in raw_feature_arg:
                if c in df.columns and c not in seen:
                    feature_cols.append(c)
                    seen.add(c)
                elif c not in df.columns:
                    raise ValueError(f"Requested feature column '{c}' not found in dataframe.")
    if not feature_cols:
        raise ValueError("No feature columns selected. Check --feature_cols or dataframe columns.")
    print(f"Using feature columns ({len(feature_cols)}): {feature_cols}")

    needed = {"prompt_id","pair_id","y",*feature_cols,"difficulty","category"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Build non-leaky prompt-level splits
    df = build_splits(df, seed=args.seed)
    print("Split sizes:", df["split"].value_counts().to_dict())

    # Standardize using TRAIN stats only (unless disabled)
    if args.no_standardize:
        mu = pd.Series({c: 0.0 for c in feature_cols})
        sd = pd.Series({c: 1.0 for c in feature_cols})
        print("Skipping standardization (using raw feature values).")
    else:
        df, mu, sd = standardize_by_train(df, feature_cols, split_col="split")
        print("Applied z-score standardization using train split statistics.")

    # Optional sanity (can comment out)
    ey = (df["y"] == 1).astype(int).mean()
    print(f"Label balance p(y=+1)={ey:.3f}")
    for c in feature_cols:
        m = mu.get(c, float("nan"))
        s = sd.get(c, float("nan"))
        print(f"{c}  mean(train)={m:+.4f}  std(train)={s:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = PairwiseDataset(df[df["split"]=="train"], feature_cols)
    ds_va = PairwiseDataset(df[df["split"]=="val"], feature_cols)
    ds_te = PairwiseDataset(df[df["split"]=="test"], feature_cols)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": (device.type == "cuda"),
        "num_workers": args.num_workers,
        "drop_last": False,
    }
    tr = DataLoader(ds_tr, shuffle=True, **loader_kwargs)
    va = DataLoader(ds_va, shuffle=False, **loader_kwargs)
    te = DataLoader(ds_te, shuffle=False, **loader_kwargs)

    if args.model == "linear":
        model = LinearRankNet(in_dim=len(feature_cols)).to(device)
    else:
        model = MLPRankNet(in_dim=len(feature_cols), hid=args.hidden).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_state = None
    print("Starting training...")
    for ep in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        n = 0
        for X, y, _ in tr:
            X = X.to(device)
            y = y.to(device)  # +1 / -1
            margin = model(X)  # = score(A)-score(B)
            loss = torch.nn.functional.softplus(-y * margin).mean()  # RankNet/BTL
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item() * y.numel()
            n += y.numel()
        tr_loss = total_loss / max(1, n)

        val_acc, val_tie, val_auc, by_diff, by_cat = evaluate(model, va, device, tie_tol=args.tie_tol)
        print(f"[ep {ep:02d}] train loss {tr_loss:.4f} || val acc(no-ties) {val_acc:.3f} | ties {val_tie:.3f} | AUC {val_auc:.3f}")
        print("   by difficulty:", by_diff)
        print("   by category:  ", by_cat)

        # choose best by val acc(no-ties); you can switch to AUC if you prefer
        score_for_earlystop = val_acc
        if score_for_earlystop > best_val:
            best_val = score_for_earlystop
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.out)
    print("Saved weights to:", args.out)

    test_acc, test_tie, test_auc, by_diff, by_cat = evaluate(model, te, device, tie_tol=args.tie_tol)
    print("TEST   acc(no-ties):", round(test_acc,3), "| ties:", round(test_tie,3), "| AUC:", round(test_auc,3))
    print("TEST by difficulty:", by_diff)
    print("TEST by category:  ", by_cat)

    if isinstance(model, LinearRankNet):
        w = (model.w / (model.w.abs().sum()+1e-9)).detach().cpu().numpy()
        print("Global weights (normalized L1):", w)
    else:
        print("MLP model used—no single global weight vector to print.")
    print("Done.")

if __name__ == "__main__":
    main()
