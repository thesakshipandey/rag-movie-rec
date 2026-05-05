# src/retrieval/search.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

from src.embeddings.backends import load_encoder
from functools import lru_cache
from src.retrieval.bm25 import load_bm25 as _load_bm25_bundle, search_bm25 as _bm25_search

# --------- Loaders ---------
@dataclass
class LoadedIndex:
    index: faiss.Index
    meta: pd.DataFrame
    metric: str
    dim: int

def load_index(indices_dir: str, metric: str = "ip") -> LoadedIndex:
    p = Path(indices_dir)
    index = faiss.read_index(str(p / "faiss.index"))
    meta = pd.read_parquet(p / "meta.parquet").reset_index(drop=True)
    dim = int(meta["embedding_dim"].iloc[0]) if "embedding_dim" in meta.columns and len(meta) else index.d
    return LoadedIndex(index=index, meta=meta, metric=metric, dim=dim)

@dataclass
class LoadedBM25:
    bundle: Dict[str, object]
    meta: pd.DataFrame

def load_bm25_index(bm25_dir: str) -> LoadedBM25:
    p = Path(bm25_dir)
    return LoadedBM25(bundle=_load_bm25_bundle(p / "bm25.pkl"),
                      meta=pd.read_parquet(p / "meta.parquet").reset_index(drop=True))

# --------- Encode query ---------
@lru_cache(maxsize=8)
def _get_encoder_cached(encoder: str, model: str, max_length: int) -> object:
    # cache heavy encoder instantiation to avoid reloading large models per query
    return load_encoder(
        encoder=encoder,
        model=model,
        max_length=max_length,
        local_files_only=True,
    )


def encode_query(text: str, *, encoder: str, model: str, max_length: int = 2048, normalize: bool = True) -> np.ndarray:
    enc = _get_encoder_cached(encoder=encoder, model=model, max_length=max_length)
    res = enc.encode([text], batch_size=1, normalize=normalize)
    return np.asarray(res.vectors, dtype=np.float32)[0]

# --------- Filters ---------
def apply_filters(meta: pd.DataFrame, filters: Optional[Dict] = None) -> pd.Series:
    if not filters: return pd.Series(True, index=meta.index)
    m = pd.Series(True, index=meta.index)
    lang = filters.get("language", filters.get("original_language"))
    if lang is not None and "original_language" in meta.columns:
        s = {lang} if isinstance(lang, str) else set(lang); m &= meta["original_language"].isin(s)
    if "release_date" in meta.columns and meta["release_date"].notna().any():
        y = pd.to_datetime(meta["release_date"], errors="coerce").dt.year
        ylo = filters.get("year_gte"); yhi = filters.get("year_lte")
        if ylo is not None: m &= (y >= int(ylo))
        if yhi is not None: m &= (y <= int(yhi))
    if "adult" in meta.columns and ("adult" in filters):
        m &= (meta["adult"].astype("boolean") == bool(filters["adult"]))
    if "type" in meta.columns and ("type" in filters):
        t = filters["type"]; s = {t} if isinstance(t, str) else set(t); m &= meta["type"].isin(s)
    return m

# --------- Dense search ---------
def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True); return X / (n + eps)

def search_dense_chunks(li: LoadedIndex, qvec: np.ndarray, top_k: int = 50, filters: Optional[Dict] = None, oversample: int = 4) -> pd.DataFrame:
    q = qvec.astype(np.float32, copy=False)[None, :]
    if li.metric in ("ip","cosine"): q = _l2_normalize(q)
    K = max(top_k * max(1, int(oversample)), top_k)
    D, I = li.index.search(q, K)
    ids, scores = I[0], D[0]
    mask = ids >= 0; ids, scores = ids[mask], scores[mask]
    hits = li.meta.iloc[ids].copy(); hits.insert(0, "score_dense", scores)
    if filters: hits = hits[apply_filters(hits, filters)]
    return hits.sort_values("score_dense", ascending=False).head(top_k).reset_index(drop=True)

# --------- BM25 search ---------
def search_bm25_chunks(lb: LoadedBM25, query: str, top_k: int = 100, filters: Optional[Dict] = None, oversample: int = 5) -> pd.DataFrame:
    K = max(top_k * max(1, int(oversample)), top_k)
    idx, scores = _bm25_search(lb.bundle, query, top_k=K)
    hits = lb.meta.iloc[idx].copy(); hits.insert(0, "score_bm25", scores)
    if filters: hits = hits[apply_filters(hits, filters)]
    return hits.sort_values("score_bm25", ascending=False).head(top_k).reset_index(drop=True)

# --------- Aggregation ---------
def aggregate_by_movie(hits: pd.DataFrame, score_col: str, how: str = "sum") -> pd.DataFrame:
    if not len(hits) or "movieId" not in hits.columns or score_col not in hits.columns:
        return pd.DataFrame(columns=["movieId", f"{score_col}_movie"])
    if how == "sum": agg = hits.groupby("movieId", as_index=False)[score_col].sum()
    elif how == "max": agg = hits.groupby("movieId", as_index=False)[score_col].max()
    else: raise ValueError("how must be 'sum' or 'max'")
    return agg.rename(columns={score_col: f"{score_col}_movie"})

def zscore(x: pd.Series, clip: float = 3.0) -> pd.Series:
    mu, sd = x.mean(), x.std()
    if sd <= 1e-12: return pd.Series(0.0, index=x.index)
    z = (x - mu) / (sd + 1e-12); return z.clip(-clip, clip)
