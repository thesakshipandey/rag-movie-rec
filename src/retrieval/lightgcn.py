# src/retrieval/lightgcn.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable
import numpy as np

@dataclass
class LGCNEmbStore:
    user_emb: np.ndarray      # (U, d) float32
    item_emb: np.ndarray      # (I, d) float32
    normed: bool              # embeddings are L2-normalized

def _l2norm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def load_embeddings(item_path: str | Path, user_path: str | Path, normalize: bool = True) -> LGCNEmbStore:
    item = np.load(item_path).astype(np.float32, copy=False)
    user = np.load(user_path).astype(np.float32, copy=False)
    if normalize:
        item = _l2norm(item)
        user = _l2norm(user)
    return LGCNEmbStore(user_emb=user, item_emb=item, normed=normalize)

def compute_cosine_matrix(store: LGCNEmbStore, batch: Optional[int] = None) -> np.ndarray:
    """
    Returns full cosine matrix (U x I). If store.normed==True, it's user @ item.T
    """
    U, I = store.user_emb, store.item_emb
    if not store.normed:
        U = _l2norm(U); I = _l2norm(I)
    if batch is None:
        return U @ I.T  # (U, I)
    # batched to reduce peak memory
    rows = []
    for s in range(0, U.shape[0], batch):
        e = min(s + batch, U.shape[0])
        rows.append(U[s:e] @ I.T)
    return np.vstack(rows)

def save_cosine_matrix(path: str | Path, M: np.ndarray) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, M.astype(np.float32, copy=False))

def load_cosine_matrix(path: str | Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")  # memmap for fast partial reads

def user_item_scores(M: np.ndarray, user_idx: int, item_idx: Optional[Iterable[int]] = None) -> np.ndarray:
    """
    Returns scores for one user: shape (I,) or (len(item_idx),)
    """
    row = M[user_idx]
    if item_idx is None:
        return np.asarray(row)
    idx = np.asarray(list(item_idx), dtype=np.int64)
    return np.asarray(row[idx])

def movie_scores_from_items(
    scores_for_items: np.ndarray,
    iid_to_movie: dict[int, int],
    agg: str = "max"
) -> dict[int, float]:
    """
    Map item scores -> movieId scores using iid_to_movie mapping.
    agg: 'max' or 'sum'
    """
    by_movie: dict[int, list[float]] = {}
    for iid, s in enumerate(scores_for_items):
        mid = iid_to_movie.get(iid)
        if mid is None: 
            continue
        by_movie.setdefault(mid, []).append(float(s))
    out: dict[int, float] = {}
    if agg == "max":
        for mid, arr in by_movie.items(): out[mid] = float(np.max(arr))
    elif agg == "sum":
        for mid, arr in by_movie.items(): out[mid] = float(np.sum(arr))
    else:
        raise ValueError("agg must be 'max' or 'sum'")
    return out
