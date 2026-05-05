# src/cli/embed.py
from __future__ import annotations
import argparse, json, logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from src.embeddings.backends import load_encoder


def _setup_logger(log_dir: Path, prefix: str = "embed") -> logging.Logger:
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

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_path)
    return logger


def _to_parquet_safe(df: pd.DataFrame, path: Path) -> bool:
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path, index=False)
        return True
    except Exception as e:
        print(f"[warn] parquet write failed ({e}); falling back to JSONL:", path.with_suffix(".jsonl"))
        with open(path.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return False


def main():
    ap = argparse.ArgumentParser("Embed chunks with Qwen/Gemma/MiniLM")
    # IO
    ap.add_argument("--chunks", default="data/processed/chunks.parquet",
                    help="Input chunks (parquet or jsonl)")
    ap.add_argument("--out", default="artifacts/embeddings/embeddings.parquet",
                    help="Output with embeddings (parquet; JSONL fallback)")
    ap.add_argument("--logs_dir", default="logs", help="Directory for run logs")

    # Encoder selection
    ap.add_argument("--encoder", choices=["qwen", "gemma", "minilm"], default="qwen",
                    help="Embedding backend (default: qwen)")
    ap.add_argument("--model", default="/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B",
                    help="Model path or HF id. Prefer local path for speed.")
    ap.add_argument("--device", default=None, help="Force device, e.g., cuda, cpu (default: auto)")
    ap.add_argument("--batch_size", type=int, default=64)
    # prefer text_for_embed if present (from the new full-movie pipeline)
    ap.add_argument("--text_col", default="text_for_embed",
                    help="Column to embed; falls back to 'text' if missing.")
    ap.add_argument("--normalize", action="store_true", default=True,
                    help="L2-normalize vectors (default on)")

    # Backend-specific knobs
    ap.add_argument("--max_length", type=int, default=32000,
                    help="Max tokens/wordpieces for encoder (Qwen supports 32k)")
    ap.add_argument("--local_files_only", action="store_true", default=True,
                    help="Disallow network when loading models (default on)")
    args = ap.parse_args()

    logger = _setup_logger(Path(args.logs_dir))

    in_path = Path(args.chunks)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Load chunks
    if in_path.suffix == ".jsonl":
        df = pd.read_json(in_path, lines=True)
    else:
        df = pd.read_parquet(in_path)

    # choose text column smartly
    text_col = args.text_col if args.text_col in df.columns else ("text" if "text" in df.columns else None)
    if not text_col:
        raise KeyError(f"Missing text column: {args.text_col} (and 'text' not found either)")
    texts = df[text_col].astype(str).tolist()

    # Build encoder
    enc = load_encoder(
        encoder=args.encoder,
        model=args.model,
        device=args.device,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )
    logger.info("Encoder: %s | Model: %s | Device: %s | max_length=%s",
                args.encoder, args.model, args.device or "auto", args.max_length)

    # Embed
    vecs = []
    B = max(1, args.batch_size)
    for i in tqdm(range(0, len(texts), B), desc=f"Embedding[{args.encoder}]", unit="batch"):
        batch = texts[i:i+B]
        res = enc.encode(batch, batch_size=B, normalize=args.normalize)
        vecs.extend(res.vectors)

    # Attach columns
    df["embedding"] = vecs
    df["embedding_model"] = getattr(res, "model_id", args.model)
    df["embedding_dim"] = getattr(res, "dim", (len(vecs[0]) if vecs else 0))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _to_parquet_safe(df, out_path)

    dims = (df["embedding_dim"].iloc[0] if len(df) else "NA")
    logger.info("Wrote: %s rows=%d dim=%s (text_col=%s)", out_path, len(df), dims, text_col)
    print(f"Embeddings written -> {out_path}  (rows={len(df)}, dim={dims}, text_col={text_col})")


if __name__ == "__main__":
    main()
