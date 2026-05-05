# src/utils/hybrid_helpers.py
from __future__ import annotations
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pandas as pd

def setup_logger(log_dir: Path | str, prefix: str = "hybrid_rerank"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = log_dir / f"{prefix}_{ts}.log"
    lg = logging.getLogger(prefix)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = RotatingFileHandler(p, maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    lg.addHandler(fh)
    return lg, p

def build_filters(args) -> dict | None:
    f = {}
    if getattr(args, "language", None):
        f["language"] = args.language
    if getattr(args, "year_gte", None) is not None:
        f["year_gte"] = args.year_gte
    if getattr(args, "year_lte", None) is not None:
        f["year_lte"] = args.year_lte
    if getattr(args, "adult", None) is not None:
        f["adult"] = args.adult
    if getattr(args, "type", None):
        f["type"] = args.type
    return f or None

def build_iid_to_movie_map(meta: pd.DataFrame, logger: logging.Logger, iid_map_csv: str | None) -> dict[int, int]:
    """Prefer mapping from FAISS meta if it contains 'iid' and 'movieId'. Else fall back to CSV."""
    if "iid" in meta.columns and "movieId" in meta.columns:
        try:
            tmp = meta.dropna(subset=["iid", "movieId"]).copy()
            tmp["iid"] = tmp["iid"].astype(float).astype(int)
            tmp["movieId"] = tmp["movieId"].astype(float).astype(int)
            m = tmp.drop_duplicates("iid").set_index("iid")["movieId"].to_dict()
            if len(m):
                logger.info("Using iid→movieId map from FAISS meta (n=%d).", len(m))
                return m
        except Exception as e:
            logger.warning("Failed to read iid→movieId from FAISS meta (%s). Will try --iid_map_csv.", e)

    if not iid_map_csv:
        raise KeyError(
            "iid→movieId mapping not found in FAISS meta and no --iid_map_csv provided. "
            "Pass: --iid_map_csv data/raw/item_text.csv (must contain columns: iid,movieId)."
        )
    mdf = pd.read_csv(iid_map_csv)
    if "iid" not in mdf.columns or "movieId" not in mdf.columns:
        raise KeyError(f"--iid_map_csv must have columns 'iid' and 'movieId'. Found: {list(mdf.columns)}")
    mdf = mdf.dropna(subset=["iid", "movieId"]).copy()
    mdf["iid"] = mdf["iid"].astype(float).astype(int)
    mdf["movieId"] = mdf["movieId"].astype(float).astype(int)
    m = mdf.drop_duplicates("iid").set_index("iid")["movieId"].to_dict()
    if not len(m):
        raise RuntimeError("Empty iid→movieId mapping after reading --iid_map_csv.")
    logger.info("Using iid→movieId map from CSV %s (n=%d).", iid_map_csv, len(m))
    return m
