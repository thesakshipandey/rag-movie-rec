# src/cli/build_index.py
import argparse, logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.retrieval.index_faiss import (
    build_flat_index,   # returns (index, norms_tuple)
    save_index,
    write_stats,        # write_stats(path, n, dim, metric, source_embeddings, build_args, vector_norms)
)

def _setup_logger(log_dir: Path, prefix: str = "build_index") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{prefix}_{ts}.log"
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("Log file: %s", log_path)
    return logger

def _select_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "chunkId","movieId","section_type","part_index",
        "title","original_title","release_date",
        "type","adult","original_language","production_countries","spoken_languages",
        "n_words","n_chars",
        "embedding_model","embedding_dim",
        "text",
    ]
    present = [c for c in wanted if c in df.columns]
    return df[present].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Embeddings parquet/jsonl file with column 'embedding'")
    ap.add_argument("--out_dir", default="artifacts/indices/gemma", help="Output dir for index + meta + stats")
    ap.add_argument("--metric", choices=["ip","cosine","l2"], default="ip",
                    help="Use 'cosine' or 'ip' (with --cosine_like) for cosine-style retrieval")
    ap.add_argument("--cosine_like", action="store_true",
                    help="When metric=ip, L2-normalize vectors so IP≈cosine (query should also be normalized).")
    ap.add_argument("--logs_dir", default="logs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(Path(args.logs_dir))

    # --- Load embeddings table ---
    emb_path = Path(args.emb)
    if emb_path.suffix == ".jsonl":
        df = pd.read_json(emb_path, lines=True)
    else:
        df = pd.read_parquet(emb_path)

    if "embedding" not in df.columns:
        raise KeyError("Input does not contain 'embedding' column")

    # Ensure consistent shapes
    logger.info("Converting 'embedding' column to 2D float32 array...")
    emb_list = df["embedding"].tolist()
    if not len(emb_list):
        raise RuntimeError("No embeddings found.")
    d0 = len(emb_list[0])
    if any(len(v) != d0 for v in emb_list):
        raise ValueError("Embeddings have inconsistent dimensions.")
    X = np.asarray(emb_list, dtype=np.float32)           # shape (N, d)
    n, dim = X.shape
    logger.info("Vectors: n=%d, dim=%d", n, dim)

    # --- Build FAISS index (handles normalization internally) ---
    normalize_for_ip = (args.metric == "ip" and args.cosine_like)
    index, norms_tuple = build_flat_index(
        X,
        metric=args.metric,
        normalize_for_ip_as_cosine=normalize_for_ip,
    )

    # --- Save index + meta (keep EXACT order) ---
    index_path = out_dir / "faiss.index"
    save_index(index, index_path)
    logger.info("Saved index: %s", index_path)

    # explicit FAISS row id mapping (0..N-1) for robustness
    idmap = np.arange(n, dtype=np.int32)
    np.save(out_dir / "idmap.npy", idmap)

    meta = _select_meta_columns(df)
    meta.insert(0, "faiss_id", idmap)  # positional id used by FAISS
    meta_path = out_dir / "meta.parquet"
    meta.to_parquet(meta_path, index=False)
    logger.info("Saved meta: %s (rows=%d, cols=%d)", meta_path, len(meta), len(meta.columns))

    # quick sanity: warn if text seems too short (likely wrong table)
    if "n_words" in meta.columns:
        mean_words = float(pd.to_numeric(meta["n_words"], errors="coerce").fillna(0).mean())
        if mean_words < 150:
            logger.warning("Mean n_words=%.1f looks short; check that you indexed the long chunk texts.", mean_words)

    # --- Stats ---
    stats_path = out_dir / "stats.json"
    build_args = {
        "kind": "flat",
        "metric": args.metric,
        "cosine_like_on_ip": bool(normalize_for_ip),
    }
    write_stats(
        stats_path,
        n=n, dim=dim, metric=args.metric,
        source_embeddings=str(emb_path),
        build_args=build_args,
        vector_norms=norms_tuple,
    )
    logger.info("Saved stats: %s", stats_path)

    print("Index built:")
    print("  index ->", index_path)
    print("  meta  ->", meta_path)
    print("  stats ->", stats_path)
    print("  idmap ->", (out_dir / "idmap.npy"))
    print("  Logs  ->", Path(args.logs_dir).resolve())

if __name__ == "__main__":
    main()
