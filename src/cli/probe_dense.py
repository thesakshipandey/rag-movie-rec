# src/cli/probe_dense.py
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.retrieval.search import load_index, encode_query, search_dense_chunks

def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def _mean_index_vector(li, sample_n: int | None = None, seed: int = 0) -> np.ndarray:
    n = li.index.ntotal
    if n == 0:
        raise RuntimeError("Index is empty.")
    ids = np.arange(n, dtype=np.int64)
    if sample_n and 0 < sample_n < n:
        rng = np.random.default_rng(seed)
        ids = rng.choice(ids, size=sample_n, replace=False)
    acc = None
    for i in ids:
        v = li.index.reconstruct(int(i)).astype(np.float32, copy=False)
        acc = v if acc is None else (acc + v)
    return acc / float(len(ids))

def _load_stats(indices_dir: Path) -> dict:
    p = indices_dir / "stats.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _verify_encoder_matches_index(stats: dict, meta: pd.DataFrame | None, model: str) -> str | None:
    problems = []
    src = (stats or {}).get("source_embeddings", "")
    if src and (str(src) not in model) and (model not in str(src)):
        problems.append(f"index built from {src}, but query uses '{model}'")
    if meta is not None and "embedding_model" in meta.columns and meta["embedding_model"].notna().any():
        emb_model = str(meta["embedding_model"].dropna().iloc[0])
        if emb_model and (emb_model not in model) and (model not in emb_model):
            problems.append(f"index embeddings seem from '{emb_model}', but query uses '{model}'")
    return "; ".join(problems) if problems else None

def main():
    ap = argparse.ArgumentParser("Probe dense index & query sanity")
    ap.add_argument("--indices_dir", default="artifacts/indices/gemma")
    ap.add_argument("--encoder", choices=["gemma", "minilm"], default="gemma")
    ap.add_argument("--model", default="/mnt/nas/sakshipandey/main/models/embeddinggemma-300m")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample_mean", type=int, default=0,
                    help="if >0, compute mean vector from this many random index vectors; 0 = use all")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    indices_dir = Path(args.indices_dir)

    # Load FAISS + meta
    li = load_index(indices_dir, metric="ip")  # metric here just affects query scoring path
    stats = _load_stats(indices_dir)

    # Encode query
    q = encode_query(args.query, encoder=args.encoder, model=args.model)

    # Decide if index expects unit norm (IP/cosine → yes)
    expects_unit = True
    if "metric" in stats:
        expects_unit = stats["metric"] in ("ip", "cosine")
    if expects_unit:
        q = _l2norm(q)

    print(f"Query: {args.query}")
    print(f"- query norm: {float(np.linalg.norm(q)):.6f}")

    warn = _verify_encoder_matches_index(stats, getattr(li, "meta", None), args.model)
    if warn:
        print(f"[warn] Encoder/Index mismatch? {warn}")

    # Mean of index vectors (subsample if requested)
    sample = None if args.sample_mean == 0 else int(args.sample_mean)
    mean_vec = _mean_index_vector(li, sample_n=sample)
    if expects_unit:
        mean_vec = _l2norm(mean_vec)
    cos_q_mean = _cosine(q, mean_vec)
    print(f"- cosine(query, mean_index_vector): {cos_q_mean:.6f}")

    # Dense search
    hits = search_dense_chunks(li, q, top_k=args.k, filters=None)
    show_cols = [c for c in ["score_dense", "movieId", "section_type", "part_index",
                             "title", "release_date", "original_language"] if c in hits.columns]
    print("\nTop-k dense results:")
    if len(hits):
        print(hits[show_cols].head(args.k).to_string(index=False))
    else:
        print("(no hits)")

    # Optional dump
    if args.out_csv:
        p = Path(args.out_csv); p.parent.mkdir(parents=True, exist_ok=True)
        hits.head(args.k).to_csv(p, index=False)
        print(f"\nSaved -> {p.resolve()}")

if __name__ == "__main__":
    main()
