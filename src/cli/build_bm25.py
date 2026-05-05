# src/cli/build_bm25.py
import argparse, logging, json, pickle
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from src.retrieval.bm25 import (
    build_corpus_tokens, save_bm25, write_stats
)

def _setup_logger(log_dir: Path, prefix: str = "build_bm25") -> logging.Logger:
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
    ch = logging.StreamHandler(); ch.setLevel(logging.WARNING); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logger.info("Log file: %s", log_path)
    return logger

def _select_meta(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "chunkId","movieId","section_type","part_index",
        "title","original_title","release_date",
        "type","adult","original_language","production_countries","spoken_languages",
        "n_words","n_chars",
        # optional mappings if present
        "iid","faiss_id",
        "text",
    ]
    return df[[c for c in wanted if c in df.columns]].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.parquet",
                    help="Input chunks parquet/jsonl (same corpus as FAISS)")
    ap.add_argument("--out_dir", default="artifacts/indices/bm25",
                    help="Output dir for BM25 artifacts")
    ap.add_argument("--text_cols", nargs="+", default=["text"],
                    help="Columns to include in BM25 (default: text)")
    ap.add_argument("--title_col", default="title", help="Optional title column")
    ap.add_argument("--title_boost", type=int, default=2, help="Repeat title tokens this many times")
    ap.add_argument("--logs_dir", default="logs")
    # BM25 params (defaults match BM25Okapi)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(Path(args.logs_dir))

    # Load corpus
    in_path = Path(args.chunks)
    if in_path.suffix == ".jsonl":
        df = pd.read_json(in_path, lines=True)
    else:
        df = pd.read_parquet(in_path)
    logger.info("Loaded corpus: %s rows=%d", in_path, len(df))

    # Tokenize
    tokens_per_doc, doc_ids = build_corpus_tokens(
        df, text_cols=args.text_cols, title_col=args.title_col, title_boost=args.title_boost
    )

    # Train BM25
    bm25 = BM25Okapi(tokens_per_doc, k1=args.k1, b=args.b)

    # Save artifacts
    bm25_path = out_dir / "bm25.pkl"
    save_bm25(bm25, tokens_per_doc, bm25_path)
    logger.info("Saved BM25 model: %s", bm25_path)

    # Persist doc id map + meta for later joins
    np.save(out_dir / "doc_ids.npy", doc_ids)
    meta = _select_meta(df)
    meta.insert(0, "bm25_id", doc_ids)
    meta_path = out_dir / "meta.parquet"
    meta.to_parquet(meta_path, index=False)
    logger.info("Saved meta: %s (cols=%d)", meta_path, len(meta.columns))

    # Stats
    stats_path = out_dir / "stats.json"
    write_stats(
        stats_path,
        n_docs=len(tokens_per_doc),
        avgdl=float(getattr(bm25, "avgdl", 0.0)),
        k1=args.k1, b=args.b,
        sources={"chunks": str(in_path)}
    )
    logger.info("Saved stats: %s", stats_path)

    print("BM25 built:")
    print("  model ->", bm25_path)
    print("  meta  ->", meta_path)
    print("  stats ->", stats_path)
    print("  Logs  ->", Path(args.logs_dir).resolve())

if __name__ == "__main__":
    main()
