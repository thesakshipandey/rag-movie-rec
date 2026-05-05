#src/router/features.py
from __future__ import annotations
import os, json
from typing import Dict, Iterable, List, Tuple, Any
import numpy as np
import pandas as pd

from src.retrieval.search import (
    load_index, load_bm25_index, encode_query,
    search_dense_chunks, search_bm25_chunks,
)
from src.retrieval.lightgcn import load_cosine_matrix, user_item_scores
from src.emotions.emotion_index import load_emotion_index, score_movies_by_emotion
from src.emotions.emotion_prompt import infer_prompt_vector  # fallback only

PLUTCHIK_ORDER = ["Joy","Trust","Fear","Anticipation","Sadness","Anger","Surprise","Disgust"]

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x)) if x.size else 0.0
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / (sd + 1e-8)

def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    v = np.clip(v, 0.0, None)
    s = float(v.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.full(8, 1.0/8, dtype=np.float32)
    return (v / s).astype(np.float32)

def _coerce_plutchik_vector(x) -> np.ndarray:
    # unwrap (vec, meta)
    if isinstance(x, (tuple, list)) and len(x) == 2 and not isinstance(x[0], (dict, str)):
        x = x[0]
    # JSON string -> dict/list
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return np.full(8, 1.0/8, dtype=np.float32)
    # dict with nested 'emotions' or flat keys
    if isinstance(x, dict):
        if "emotions" in x and isinstance(x["emotions"], dict):
            x = x["emotions"]
        arr = np.array([float(x.get(k, 0.0)) for k in PLUTCHIK_ORDER], dtype=np.float32)
        return _normalize_vec(arr)
    # raw vector
    if isinstance(x, (list, np.ndarray)):
        arr = np.array(x, dtype=np.float32).reshape(-1)
        if arr.size == 8:
            return _normalize_vec(arr)
    return np.full(8, 1.0/8, dtype=np.float32)

# ---------------- robust hit parsing / aggregation ----------------

def _parse_movie_from_chunk_id(s: str) -> int | None:
    try:
        return int(str(s).split(":", 1)[0])
    except Exception:
        return None

def _as_hits_list(chunk_hits: Any) -> List[Any]:
    if chunk_hits is None:
        return []
    if isinstance(chunk_hits, pd.DataFrame):
        return chunk_hits.to_dict(orient="records")
    if isinstance(chunk_hits, pd.Series):
        return list(chunk_hits.items())
    if isinstance(chunk_hits, dict):
        return list(chunk_hits.items())
    if isinstance(chunk_hits, (list, tuple)):
        return list(chunk_hits)
    if isinstance(chunk_hits, np.ndarray):
        return chunk_hits.tolist()
    return [chunk_hits]

def _extract_movie_and_score(hit: Any) -> Tuple[int | None, float | None]:
    if isinstance(hit, pd.Series):
        hit = hit.to_dict()
    if isinstance(hit, dict):
        movie = hit.get("movieId", hit.get("movie_id", hit.get("movie")))
        if movie is None and "chunkId" in hit:
            movie = _parse_movie_from_chunk_id(hit["chunkId"])
        if movie is not None:
            try:
                movie = int(movie)
            except Exception:
                movie = _parse_movie_from_chunk_id(movie)
        score = None
        for key in ("score", "sim", "value", "bm25", "score_dense", "score_bm25", "dense_score", "bm25_score"):
            if key in hit and hit[key] is not None:
                try:
                    score = float(hit[key])
                except Exception:
                    score = None
                else:
                    break
        return movie, score
    if isinstance(hit, (tuple, list)) and len(hit) >= 2:
        mid, sc = hit[0], hit[1]
        if isinstance(mid, (int, np.integer)):
            movie = int(mid)
        elif isinstance(mid, str):
            movie = _parse_movie_from_chunk_id(mid)
        else:
            movie = None
        try:
            score = float(sc)
        except Exception:
            score = None
        return movie, score
    if isinstance(hit, str):
        movie = _parse_movie_from_chunk_id(hit)
        return movie, None
    return None, None

def _aggregate_chunk_hits(chunk_hits: Any, kind: str = "sum", temperature: float = 1.0) -> Dict[int, float]:
    per_movie: Dict[int, List[float]] = {}
    for h in _as_hits_list(chunk_hits):
        movie, score = _extract_movie_and_score(h)
        if movie is None or score is None:
            continue
        per_movie.setdefault(movie, []).append(score)

    if not per_movie:
        return {}

    out: Dict[int, float] = {}
    if kind in ("sum", "max"):
        for m, lst in per_movie.items():
            if not lst:
                continue
            out[m] = float(np.sum(lst)) if kind == "sum" else float(np.max(lst))
        return out

    if kind == "attn":
        t = max(1e-3, float(temperature))  # clamp tiny taus
        for m, lst in per_movie.items():
            if not lst:
                continue
            arr = np.array(lst, dtype=np.float32)

            # numerically stable softmax: subtract max, clip range
            a = (arr - np.max(arr)) / t
            a = np.clip(a, -50.0, 50.0)           # avoid inf in exp
            w = np.exp(a)
            Z = w.sum()

            if not np.isfinite(Z) or Z <= 0.0:
                # safe fallback: unweighted mean
                out[m] = float(np.mean(arr))
                continue

            w = w / Z
            out[m] = float((w * arr).sum())
        return out


    raise ValueError(f"Unknown aggregator kind: {kind}")

# ---------------- main per-prompt table ----------------

def per_prompt_movie_table(prompt_text: str, user_idx: int | None,
                           dense_idx, bm25_idx, lgcn_sim, emo_ids, emo_mat,
                           agg_kind: str = "sum", topk: int = 200,
                           encoder: str = "qwen",
                           model: str = "/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B",
                           q_emo_vec: np.ndarray | None = None,
                           attn_tau: float = 1.0) -> pd.DataFrame:

    qvec = encode_query(prompt_text, encoder=encoder, model=model)

    dense_hits = search_dense_chunks(dense_idx, qvec, top_k=topk)
    bm25_hits  = search_bm25_chunks(bm25_idx, prompt_text, top_k=topk)

    if agg_kind == "attn":
        dense_movies = _aggregate_chunk_hits(dense_hits, kind="attn", temperature=attn_tau)
        bm25_movies  = _aggregate_chunk_hits(bm25_hits,  kind="attn", temperature=attn_tau)
    else:
        dense_movies = _aggregate_chunk_hits(dense_hits, kind=agg_kind)
        bm25_movies  = _aggregate_chunk_hits(bm25_hits,  kind=agg_kind)

    # Candidate pool (guard against empty)
    mids_set = set(dense_movies.keys()) | set(bm25_movies.keys())
    if not mids_set:
        # Return an empty frame with expected columns so callers can default to zeros
        empty = pd.DataFrame(columns=[
            "movieId","dense","bm25","emo","lgcn",
            "z_dense","z_bm25","z_lgcn","z_emo"
        ])
        return empty.set_index("movieId")

    all_mids = sorted(int(m) for m in mids_set)

    # Emotion: prefer provided q_emo_vec; else fallback (your helper normalizes)
    if q_emo_vec is None:
        raw = infer_prompt_vector(prompt_text)  # fallback only
        q_emo_vec = _coerce_plutchik_vector(raw)
    else:
        q_emo_vec = _coerce_plutchik_vector(q_emo_vec)

    # Emotion scores (guard for empties already handled)
    emo_scores_arr = score_movies_by_emotion(all_mids, emo_ids, emo_mat, q_emo_vec)
    emo_movies = {int(mid): float(score) for mid, score in zip(all_mids, emo_scores_arr)}

    # LightGCN (restrict to candidate movies and keep mapping movieId -> score)
    lgcn_movies: Dict[int, float] = {}
    if user_idx is not None and lgcn_sim is not None:
        try:
            uidx = int(user_idx)
        except Exception:
            uidx = None
        if uidx is not None and 0 <= uidx < lgcn_sim.shape[0]:
            if lgcn_sim.ndim == 2:
                max_item = int(lgcn_sim.shape[1])
                valid_mids = [int(mid) for mid in all_mids if 0 <= int(mid) < max_item]
                if valid_mids:
                    scores = user_item_scores(lgcn_sim, uidx, item_idx=valid_mids)
                    scores = np.asarray(scores).reshape(-1)
                    lgcn_movies = {mid: float(score) for mid, score in zip(valid_mids, scores)}
            # else: unexpected shape, leave empty
        # else: user_idx invalid/out of range → keep empty dict

    # Join
    mids = (
        set(dense_movies.keys())
        | set(bm25_movies.keys())
        | set(emo_movies.keys())
        | set(lgcn_movies.keys())
    )
    rows = []
    for m in mids:
        rows.append({
            "movieId": int(m),
            "dense": float(dense_movies.get(m, 0.0)),
            "bm25":  float(bm25_movies.get(m, 0.0)),
            "emo":   float(emo_movies.get(m, 0.0)),
            "lgcn":  float(lgcn_movies.get(m, 0.0)),
        })
    df = pd.DataFrame(rows)
    for c in ["dense","bm25","emo","lgcn"]:
        df[f"z_{c}"] = zscore(df[c].values)
    return df.set_index("movieId")


def load_prompt_triplets(prompts_dir: str):
    def _read(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    prompts = pd.json_normalize(_read(os.path.join(prompts_dir, "prompts.json")))
    pairs   = pd.json_normalize(_read(os.path.join(prompts_dir, "pairs.json")))
    judg    = pd.json_normalize(_read(os.path.join(prompts_dir, "judgments.json")))

    # Build y ∈ {0,1}
    if "m1_gt_m2" in judg.columns:
        judg["y"] = (judg["m1_gt_m2"] == 1).astype(int)
    elif "winner" in judg.columns:
        judg["y"] = (judg["winner"].astype(str).str.lower().isin(["m1","movie1","a"])).astype(int)
    else:
        raise ValueError("judgments.json missing preference columns (m1_gt_m2 or winner)")

    # ensure int ids
    for col in ["movie1_id", "movie2_id"]:
        def _to_int(v):
            if isinstance(v, dict) and "id" in v: return int(v["id"])
            try: return int(v)
            except: 
                try:
                    d = json.loads(v)
                    if isinstance(d, dict) and "id" in d: return int(d["id"])
                except: pass
            try: return int(str(v).split(":", 1)[0])
            except: raise ValueError(f"Cannot coerce movie id from value: {v!r}")
        pairs[col] = pairs[col].apply(_to_int)

    # include plutchik_dist for free (so we can skip model inference)
    # include plutchik_dist only if it exists; category is optional too
    keep_cols_all = [
        "prompt_id",
        "prompt_text",
        "category",
        "plutchik_dist",
        "primary_expert",
        "mix_weights.alpha",
        "mix_weights.beta",
        "mix_weights.gamma",
        "mix_weights.delta",
        "context_features.length_bucket",
        "context_features.persona_style",
        "context_features.multi_intent",
        "context_features.cold_user",
        "context_features.has_genre_terms",
        "context_features.has_negation",
        "context_features.has_year",
        "context_features.length_words",
        "context_features.num_genre_terms",
    ]
    present_cols = [c for c in keep_cols_all if c in prompts.columns]

    merged = pairs.merge(
        judg[["prompt_id", "pair_id", "y"]],
        on=["prompt_id", "pair_id"]
    ).merge(
        prompts[present_cols],  # robust: select only existing cols
        on="prompt_id",
        how="left"
    )
    return merged
