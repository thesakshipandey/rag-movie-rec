# src/cli/chunk.py
import argparse, json, logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from src.chunking.pipeline import (
    build_chunks_for_record,
    build_chunks,
    ChunkSpec,
)

def _to_parquet_safe(df: pd.DataFrame, path: Path) -> bool:
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path, index=False)
        return True
    except Exception as e:
        print(f"[warn] parquet write failed ({e}); falling back to CSV:", path.with_suffix(".csv"))
        df.to_csv(path.with_suffix(".csv"), index=False)
        return False

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"chunk_{ts}.log"

    logger = logging.getLogger("chunk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_path)
    return logger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/raw/item_text.csv")
    ap.add_argument("--out_jsonl", default="data/processed/chunks.jsonl")
    ap.add_argument("--out_parquet", default="data/processed/chunks.parquet")
    ap.add_argument("--summary", default="data/processed/chunks_summary.json")
    ap.add_argument("--logs_dir", default="logs")

    # one-movie-per-chunk mode only (kept flag for UX)
    ap.add_argument("--mode", choices=["full"], default="full")

    # streaming & filtering
    ap.add_argument("--stream", action="store_true", help="Iterate with a progress bar")
    ap.add_argument("--min_words", type=int, default=0, help="Drop chunk if words < min_words")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_jsonl = Path(args.out_jsonl)
    out_parquet = Path(args.out_parquet)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(Path(args.logs_dir))

    df = pd.read_csv(inp)
    for col in ("overview", "plot"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    spec = ChunkSpec(min_words=max(0, args.min_words))

    logger.info("Mode: %s | rows=%d | min_words=%d", args.mode, len(df), spec.min_words)

    if args.stream:
        rows = []
        for tup in tqdm(df.itertuples(index=False), total=len(df), desc="Building full chunks", unit="movie"):
            rec = pd.Series(tup._asdict())
            try:
                rows.extend(build_chunks_for_record(rec, spec))
            except Exception:
                logger.exception("Failed on movieId=%s", rec.get("movieId", "NA"))
        chunks = pd.DataFrame(rows)
    else:
        chunks = build_chunks(df, spec)

    # JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in chunks.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote JSONL: %s", out_jsonl)

    # Parquet (CSV fallback)
    _to_parquet_safe(chunks, out_parquet)
    logger.info("Wrote Parquet/CSV: %s", out_parquet)

    # Summary
    summary = {
        "source_rows": int(len(df)),
        "total_chunks": int(len(chunks)),
        "avg_words_per_chunk": float(chunks["n_words"].mean() if len(chunks) else 0.0),
        "median_words_per_chunk": float(chunks["n_words"].median() if len(chunks) else 0.0),
        "max_words_per_chunk": int(chunks["n_words"].max() if len(chunks) else 0),
        "min_words_per_chunk": int(chunks["n_words"].min() if len(chunks) else 0),
        "full_mode": True,
        "out_jsonl": str(out_jsonl),
        "out_parquet": str(out_parquet),
        "log_dir": str(Path(args.logs_dir).resolve()),
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", summary)

    print("Wrote:")
    print("  JSONL  ->", out_jsonl)
    print("  Parquet->", out_parquet, "(or CSV fallback alongside)")
    print("  Summary->", args.summary)
    print("  Logs   ->", Path(args.logs_dir).resolve())

if __name__ == "__main__":
    main()
