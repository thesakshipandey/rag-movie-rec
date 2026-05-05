# src/retrieval/index_faiss.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Any
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

Metric = Literal["ip", "cosine", "l2"]

@dataclass
class IndexArtifacts:
    index_path: Path
    meta_path: Path
    stats_path: Path

def _ensure_2d_float32(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X.astype("float32", copy=False)

def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def build_flat_index(
    X: np.ndarray,
    *,
    metric: Metric = "ip",
    normalize_for_ip_as_cosine: bool = False,
) -> tuple[faiss.Index, Tuple[float, float, float]]:
    """
    Build an exact (flat) FAISS index.

    - metric="ip": uses IndexFlatIP; if normalize_for_ip_as_cosine=True, L2-normalize X so IP≈cosine.
    - metric="cosine": uses IndexFlatIP and **always** L2-normalizes X.
    - metric="l2": uses IndexFlatL2 on raw X.

    Returns (index, norms_tuple) where norms are computed on the vectors
    actually added to the index.
    """
    X = _ensure_2d_float32(X)

    if metric == "cosine":
        X = _l2_normalize(X)
        index = faiss.IndexFlatIP(X.shape[1])
    elif metric == "ip":
        if normalize_for_ip_as_cosine:
            X = _l2_normalize(X)
        index = faiss.IndexFlatIP(X.shape[1])
    elif metric == "l2":
        index = faiss.IndexFlatL2(X.shape[1])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # stats based on vectors that go into the index
    norms = np.linalg.norm(X, axis=1)
    norms_tuple = (float(norms.mean()), float(norms.std()), float(norms.max()))

    index.add(X)  # implicit ids 0..n-1; IndexFlat* has no add_with_ids
    return index, norms_tuple

def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def write_stats(
    path: Path,
    *,
    n: int,
    dim: int,
    metric: Metric,
    source_embeddings: str,
    build_args: Dict[str, Any],
    vector_norms: Tuple[float, float, float] | None,
) -> None:
    data = {
        "n": n,
        "dim": dim,
        "metric": metric,
        "index_type": "IndexFlat" + ("IP" if metric in ("ip", "cosine") else "L2"),
        "source_embeddings": source_embeddings,
        "build_args": build_args,
        "vector_norms": (
            {"mean": vector_norms[0], "std": vector_norms[1], "max": vector_norms[2]}
            if vector_norms else None
        ),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
