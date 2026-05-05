# src/retrieval/bm25.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Dict, Optional, Tuple
from pathlib import Path
import re, json, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def simple_tokenize(text: str) -> List[str]:
    if not isinstance(text, str): 
        return []
    return _WORD.findall(text.lower())

def build_corpus_tokens(
    df: pd.DataFrame,
    text_cols: List[str],
    title_col: Optional[str] = None,
    title_boost: int = 2,
) -> Tuple[List[List[str]], np.ndarray]:
    """
    Returns:
      tokens_per_doc: list of token lists
      doc_ids: np.ndarray of row indices (int64)
    """
    tokens_per_doc: List[List[str]] = []
    doc_ids = np.arange(len(df), dtype=np.int64)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing", unit="doc"):
        toks: List[str] = []
        if title_col and title_col in df.columns:
            title = str(row.get(title_col, "") or "")
            if title:
                t = simple_tokenize(title)
                toks.extend(t * max(1, int(title_boost)))
        for c in text_cols:
            if c in df.columns:
                t = str(row.get(c, "") or "")
                if t:
                    toks.extend(simple_tokenize(t))
        tokens_per_doc.append(toks or [""])  # avoid empty
    return tokens_per_doc, doc_ids

@dataclass
class BM25Artifacts:
    bm25_path: Path
    doc_ids_path: Path
    meta_path: Path
    stats_path: Path

def save_bm25(
    bm25: BM25Okapi,
    tokens_per_doc: List[List[str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokens": tokens_per_doc}, f)

def load_bm25(path: str | Path) -> Dict[str, object]:
    with open(path, "rb") as f:
        return pickle.load(f)

def write_stats(path: Path, *, n_docs: int, avgdl: float, k1: float, b: float, sources: Dict[str, str]) -> None:
    data = {
        "n_docs": n_docs,
        "avgdl": avgdl,
        "k1": k1,
        "b": b,
        "sources": sources,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def search_bm25(
    bm25_bundle: Dict[str, object],
    query: str,
    top_k: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    bm25: BM25Okapi = bm25_bundle["bm25"]  # type: ignore
    toks = simple_tokenize(query)
    scores = np.array(bm25.get_scores(toks), dtype=np.float32)
    if top_k >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, top_k)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]
