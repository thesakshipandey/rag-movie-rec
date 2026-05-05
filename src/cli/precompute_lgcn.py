# src/cli/precompute_lgcn.py
import argparse, logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from src.retrieval.lightgcn import load_embeddings, compute_cosine_matrix, save_cosine_matrix

def _setup_logger(log_dir: Path, prefix="precompute_lgcn"):
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = log_dir / f"{prefix}_{ts}.log"
    lg = logging.getLogger(prefix); lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = RotatingFileHandler(p, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    lg.addHandler(fh)
    return lg, p

def main():
    ap = argparse.ArgumentParser("Precompute full user×item cosine matrix")
    ap.add_argument("--item_emb", default="data/LightGCN_embed/item_emb.npy")
    ap.add_argument("--user_emb", default="data/LightGCN_embed/user_emb.npy")
    ap.add_argument("--out", default="artifacts/indices/lightgcn/sim_user_item.npy")
    ap.add_argument("--batch", type=int, default=None, help="optional row-batch for big matrices")
    ap.add_argument("--logs_dir", default="logs")
    args = ap.parse_args()

    logger, log_path = _setup_logger(Path(args.logs_dir))
    store = load_embeddings(args.item_emb, args.user_emb, normalize=True)
    logger.info("Loaded LightGCN embeddings: users=%d items=%d dim_u=%d dim_i=%d",
                store.user_emb.shape[0], store.item_emb.shape[0], store.user_emb.shape[1], store.item_emb.shape[1])

    M = compute_cosine_matrix(store, batch=args.batch)
    save_cosine_matrix(args.out, M)
    logger.info("Cosine matrix saved -> %s  shape=%s", args.out, tuple(M.shape))

    print("Precomputed:", args.out, " Logs ->", log_path)

if __name__ == "__main__":
    main()
