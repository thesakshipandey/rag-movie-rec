# src/utils/emotion_index.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
import numpy as np
import pandas as pd

EMOS = ["Joy","Trust","Fear","Anticipation","Sadness","Anger","Surprise","Disgust"]
E2I = {e:i for i,e in enumerate(EMOS)}
_EPS = 1e-12

# very small heuristic lexicon for fallback-from-query
LEX = {
    "joy": ["joy","happy","happiness","funny","feelgood","feel-good","uplifting","heartwarming","delight","wholesome"],
    "trust": ["trust","loyal","loyalty","faith","reliable","comforting","reassuring"],
    "fear": ["fear","scary","terror","horror","creepy","tense","thriller","dread"],
    "anticipation": ["anticipation","exciting","excited","suspense","hope","hopeful","build-up","await","yearning"],
    "sadness": ["sad","tragic","melancholy","bittersweet","tearjerker","grief","somber"],
    "anger": ["anger","angry","rage","furious","vengeful","revenge","wrath"],
    "surprise": ["surprise","twist","unexpected","shocking","mind-bending","reveal"],
    "disgust": ["disgust","gross","repulsive","nasty","vile"],
}

def _norm(v: np.ndarray) -> np.ndarray:
    v = np.maximum(v, 0.0)
    s = v.sum()
    if s <= 0:  # uniform fallback
        return np.full_like(v, 1.0/len(v))
    return v / s

def load_emotion_index(emotion_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (movie_ids[int64], emo_mat[float32, n×8]) aligned by row.
    Looks for meta.parquet, emotion.parquet, then jsons.
    """
    d = Path(emotion_dir)
    if not d.exists():
        raise FileNotFoundError(f"emotion_dir not found: {emotion_dir}")

    cand = [d/"meta.parquet", d/"emotion.parquet", d/"meta.json", d/"emotion.json"]
    src = next((p for p in cand if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"No emotion index file in {emotion_dir}")

    if src.suffix.lower() in {".parquet",".pq"}:
        df = pd.read_parquet(src)
    else:
        df = pd.DataFrame(json.load(open(src, "r")))
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("movieid", "movieId" if "movieId" in df.columns else None)
    if idc is None:
        raise KeyError("movieId column missing in emotion index")

    for e in EMOS:
        if e not in df.columns:
            raise KeyError(f"Missing emotion column: {e}")

    # ensure probabilities sum to 1
    M = df[EMOS].to_numpy(dtype=np.float64)
    M = np.clip(M, 0.0, None)
    M /= M.sum(axis=1, keepdims=True)
    mid = pd.to_numeric(df[idc], errors="coerce").fillna(-1).astype(np.int64).to_numpy()

    return mid, M.astype(np.float32)

def parse_prompt_emotion(arg: str|None, query_text: str|None=None) -> np.ndarray:
    """
    If arg is:
      - "Joy" (single label) -> one-hot
      - "Joy=0.4,Trust=0.2,..." -> explicit vector
      - path to json dict -> use that
    Else build from keywords in query_text as a weak heuristic.
    Returns 8-d prob vector.
    """
    v = np.zeros(8, dtype=np.float64)
    if arg:
        p = Path(arg)
        if "=" in arg and not p.exists():
            # inline weights
            for tok in arg.split(","):
                if "=" not in tok: continue
                k, val = tok.split("=", 1)
                k = k.strip().capitalize()
                if k in E2I:
                    try: v[E2I[k]] = float(val)
                    except: pass
            return _norm(v).astype(np.float32)
        if p.exists():
            obj = json.load(open(p, "r"))
            if isinstance(obj, dict):
                for k, val in obj.items():
                    k = str(k).strip().capitalize()
                    if k in E2I:
                        try: v[E2I[k]] = float(val)
                        except: pass
                return _norm(v).astype(np.float32)
        # single label
        lab = arg.strip().capitalize()
        if lab in E2I:
            v[E2I[lab]] = 1.0
            return v.astype(np.float32)

    # fallback from query keywords
    if query_text:
        txt = query_text.lower()
        for emo, words in LEX.items():
            count = 0
            for w in words:
                # word boundary-ish; allow hyphen
                if re.search(rf"(^|[^a-z]){re.escape(w)}([^a-z]|$)", txt):
                    count += 1
            if count:
                v[E2I[emo.capitalize()]] = float(count)

    return _norm(v).astype(np.float32)

def js_similarity_batch(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Jensen–Shannon similarity = 1 - sqrt(JSdiv(P||Q)).
    M: [n,8], p: [8]
    """
    p = p.astype(np.float64)
    M = M.astype(np.float64)
    P = np.clip(M, _EPS, None)
    Q = np.clip(p, _EPS, None)
    # broadcast p to rows
    Qb = np.broadcast_to(Q, P.shape)
    Mmix = 0.5*(P + Qb)

    def _kl(A, B):
        return (A * (np.log(A) - np.log(B))).sum(axis=1)
    jsd = 0.5*_kl(P, Mmix) + 0.5*_kl(Qb, Mmix)
    jsdist = np.sqrt(np.maximum(jsd, 0.0))
    sim = 1.0 - jsdist
    # clip to [0,1]
    return np.clip(sim, 0.0, 1.0).astype(np.float32)

def score_movies_by_emotion(movie_ids: Iterable[int], mid: np.ndarray, M: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    movie_ids: iterable of ids to score
    mid, M: from load_emotion_index
    returns array aligned to movie_ids
    """
    id2row: Dict[int,int] = {int(i): idx for idx, i in enumerate(mid.tolist())}
    rows = []
    missing = []
    for m in movie_ids:
        idx = id2row.get(int(m), -1)
        if idx < 0:
            missing.append(m)
            rows.append(np.full(8, 1.0/8, dtype=np.float32))  # uniform fallback
        else:
            rows.append(M[idx])
    mat = np.stack(rows, axis=0).astype(np.float32)
    return js_similarity_batch(mat, p)
