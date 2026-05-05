# src/cli/build_emotion_index.py
#!/usr/bin/env python
"""
Build a Plutchik-8 emotion index for movies.

Inputs (JSON or Parquet) may be one of:
  A) List[ row ] where each row may contain:
     - Flat emotion cols: Joy, Trust, Fear, Anticipation, Sadness, Anger, Surprise, Disgust
     - or nested: emotions/scores: { "Joy": 0.7, "Anger": 0.1, ... }
     - or list of dicts: emotions/scores: [ {"label":"Joy","score":0.7}, ... ]
     - or single label: label/top_emotion/emotion (+ optional confidence)
     - optional identity/title fields: movieId/id/movie_id, title/name/original_title
  B) Dict with a list under {"movies"} (or {"data"|"items"|"results"})
  C) Dict-of-dicts: { "<movieId>": {<emotions> ...}, ... }

Outputs (in --out_dir):
  - meta.parquet : columns [movieId, title?] + Plutchik-8 emotion columns
  - stats.json   : basic ingestion/normalization stats
  - meta.json    : optional (when --write_json)
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

EMOS = ["Joy","Trust","Fear","Anticipation","Sadness","Anger","Surprise","Disgust"]
EMOS_SET = {e.lower(): e for e in EMOS}

# reasonable synonyms / loose keys (lowercased) -> canonical
EMO_SYNONYMS = {
    "joy": "Joy", "happiness": "Joy", "happy": "Joy", "amusement": "Joy", "delight": "Joy",
    "trust": "Trust", "loyalty": "Trust", "faith": "Trust",
    "fear": "Fear", "terror": "Fear", "scare": "Fear",
    "anticipation": "Anticipation", "expectation": "Anticipation", "eagerness": "Anticipation",
    "sadness": "Sadness", "grief": "Sadness", "sorrow": "Sadness",
    "anger": "Anger", "rage": "Anger", "annoyance": "Anger",
    "surprise": "Surprise", "astonishment": "Surprise", "shock": "Surprise",
    "disgust": "Disgust", "revulsion": "Disgust", "aversion": "Disgust",
}

ID_KEYS = ["movieId","movie_id","id","tmdb_id","imdb_id"]
TITLE_KEYS = ["title","name","original_title","originalName","movie_title"]

def _canon_col(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def _unwrap_container(data: Any) -> Any:
    """If the JSON is a dict with a list under common keys, return that list."""
    if isinstance(data, dict):
        for k in ["movies","data","items","results"]:
            if k in data and isinstance(data[k], list):
                return data[k]
        # dict-of-dicts -> list of rows with implied movieId from key
        if all(isinstance(v, dict) for v in data.values()):
            rows = []
            for k, v in data.items():
                row = dict(v)
                if not any(c in row for c in ID_KEYS):
                    row["movieId"] = k
                rows.append(row)
            return rows
    return data

def _pick_first(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    # also try case-insensitive
    low = {k.lower(): k for k in d}
    for k in keys:
        if k.lower() in low:
            v = d[low[k.lower()]]
            if v is not None: return v
    return None

def _to_int_or_str(x: Any) -> Any:
    if x is None: return None
    # try int
    try:
        # handle "123", 123.0
        if isinstance(x, (int, np.integer)): return int(x)
        if isinstance(x, float) and float(x).is_integer(): return int(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs.isdigit(): return int(xs)
            # allow numeric-ish like "123.0"
            f = float(xs)
            if f.is_integer(): return int(f)
            return xs  # keep as string id
        return x
    except Exception:
        return str(x)

def _label_to_canon(label: str) -> str | None:
    if not isinstance(label, str): return None
    key = label.strip().lower()
    if key in EMOS_SET:
        return EMOS_SET[key]
    if key in EMO_SYNONYMS:
        return EMO_SYNONYMS[key]
    return None

def _extract_emotions_from_row(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Return a dict for the eight emotions. Missing ones are filled later.
    Accepts:
      - flat columns (Joy..Disgust)
      - 'emotions' / 'scores' dict
      - 'emotions' / 'scores' list of {"label","score"}
      - single label: 'label'/'top_emotion'/'emotion' (+ optional 'confidence')
    """
    out: Dict[str, float] = {}

    # 1) flat columns
    for k, v in row.items():
        can = _label_to_canon(str(k))
        if can is not None:
            try:
                out[can] = float(v)
            except Exception:
                pass

    # 2) nested dict/list under 'emotions' or 'scores'
    for key in ["emotions","scores"]:
        if key in row and row[key] is not None:
            val = row[key]
            if isinstance(val, dict):
                for k, v in val.items():
                    can = _label_to_canon(k)
                    if can is not None:
                        try:
                            out[can] = float(v)
                        except Exception:
                            pass
            elif isinstance(val, list):
                for it in val:
                    if isinstance(it, dict):
                        lab = _pick_first(it, ["label","name","emotion"])
                        sc = _pick_first(it, ["score","value","prob","p"])
                        can = _label_to_canon(lab) if lab is not None else None
                        if can is not None and sc is not None:
                            try:
                                out[can] = float(sc)
                            except Exception:
                                pass

    # 3) single top label (+ optional confidence)
    if not out:
        lab = _pick_first(row, ["label","top_emotion","emotion"])
        can = _label_to_canon(lab) if lab is not None else None
        if can is not None:
            conf = _pick_first(row, ["confidence","score","prob","p"])
            try:
                conf = float(conf) if conf is not None else 1.0
            except Exception:
                conf = 1.0
            out[can] = conf

    return out

def _row_to_record(row: Dict[str, Any], epsilon: float) -> Dict[str, Any] | None:
    movie_id = _pick_first(row, ID_KEYS)
    movie_id = _to_int_or_str(movie_id)
    if movie_id is None:
        # tolerate missing id, but skip (index must be keyed)
        return None

    title = _pick_first(row, TITLE_KEYS)
    if isinstance(title, (int, float)): title = str(title)

    emos = _extract_emotions_from_row(row)

    # Build vector in EMOS order with smoothing
    vec = np.array([float(max(0.0, emos.get(e, 0.0))) for e in EMOS], dtype=float)
    s = float(vec.sum())
    if s <= 0.0:
        # fallback: uniform tiny mass, then renormalize
        vec = np.full(len(EMOS), epsilon, dtype=float)
    else:
        # clip negatives and add epsilon to avoid exact zeros, then renorm
        vec = np.clip(vec, 0.0, None) + epsilon

    vec = vec / vec.sum()

    out = {"movieId": movie_id}
    if title is not None:
        out["title"] = str(title)
    for e, v in zip(EMOS, vec.tolist()):
        out[e] = float(v)
    return out

def _read_any(path: str, epsilon: float) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
        # normalize possible capitalizations of columns later in _records_from_df
        return _records_from_df(df, epsilon)

    # else JSON
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = _unwrap_container(data)

    # list of rows
    if isinstance(data, list):
        rows: List[Dict[str, Any]] = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        # a generic dict → single row
        rows = [data]
    else:
        raise ValueError("Unsupported JSON structure. Expected list or dict.")

    recs: List[Dict[str, Any]] = []
    for r in rows:
        rec = _row_to_record(r, epsilon=epsilon)
        if rec is not None:
            recs.append(rec)

    if not recs:
        raise RuntimeError("No valid movie rows found.")

    return pd.DataFrame.from_records(recs)

def _records_from_df(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    # Canonicalize columns by case-insensitive lookup when possible
    cols = {c.lower(): c for c in df.columns}
    def col_any(cands: Iterable[str]) -> str | None:
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    id_col = col_any(ID_KEYS)
    title_col = col_any(TITLE_KEYS)

    # If already has emotion cols (any case), copy; else try nested JSON strings in 'emotions'/'scores'
    emo_cols = {e: col_any([e, e.lower()]) for e in EMOS}
    has_flat = any(v is not None for v in emo_cols.values())

    recs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        r = {}
        if id_col:
            r["movieId"] = row[id_col]
        if title_col:
            r["title"] = row[title_col]
        if has_flat:
            for e, c in emo_cols.items():
                if c is not None:
                    r[e] = row[c]
        else:
            # look for nested
            for nest in ["emotions","scores"]:
                nc = col_any([nest])
                if nc and isinstance(row[nc], dict):
                    for k, v in row[nc].items():
                        can = _label_to_canon(k)
                        if can: r[can] = v
                elif nc and isinstance(row[nc], str):
                    # maybe JSON-encoded
                    try:
                        obj = json.loads(row[nc])
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                can = _label_to_canon(k)
                                if can: r[can] = v
                    except Exception:
                        pass
        rec = _row_to_record(r, epsilon=epsilon)
        if rec is not None:
            recs.append(rec)

    if not recs:
        raise RuntimeError("No valid movie rows found in Parquet.")

    return pd.DataFrame.from_records(recs)

def _aggregate_duplicates(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if "movieId" not in df.columns:
        raise KeyError("movieId column missing after parse.")
    # Keep the most common (non-null) title, then aggregate emotions
    emo_only = df[["movieId"] + EMOS].copy()
    agg = emo_only.groupby("movieId", as_index=False).agg(how)
    if "title" in df.columns:
        tt = (
            df[["movieId","title"]]
            .dropna()
            .astype({"title":"string"})
            .groupby("movieId")["title"]
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0])
            .reset_index()
        )
        agg = agg.merge(tt, on="movieId", how="left")
        # Place title second
        cols = ["movieId","title"] + [c for c in agg.columns if c not in {"movieId","title"}]
        agg = agg[cols]
    return agg

def main():
    ap = argparse.ArgumentParser("Build Plutchik-8 emotion index for movies")
    g_in = ap.add_argument_group("Input")
    g_in.add_argument("--in_json", type=str, default=None, help="Path to JSON file")
    g_in.add_argument("--in_parquet", type=str, default=None, help="Path to Parquet file")

    g_out = ap.add_argument_group("Output")
    g_out.add_argument("--out_dir", type=str, required=True, help="Directory to write the index")
    g_out.add_argument("--write_json", action="store_true", help="Also emit meta.json next to meta.parquet")

    g_opts = ap.add_argument_group("Options")
    g_opts.add_argument("--agg", choices=["mean","max"], default="max", help="How to merge duplicate movieIds")
    g_opts.add_argument("--epsilon", type=float, default=1e-9, help="Smoothing added before normalization")

    args = ap.parse_args()

    if not args.in_json and not args.in_parquet:
        raise SystemExit("Provide one of --in_json or --in_parquet")

    inp = args.in_json or args.in_parquet
    df = _read_any(inp, epsilon=args.epsilon)

    before = len(df)
    df = _aggregate_duplicates(df, how=args.agg)
    after = len(df)

    # Sanity: re-normalize rows to sum 1 exactly
    emo_mat = df[EMOS].to_numpy(dtype=float)
    row_sum = emo_mat.sum(axis=1, keepdims=True)
    bad = (row_sum <= 0).ravel()
    if bad.any():
        emo_mat[bad, :] = 1.0 / len(EMOS)
        row_sum = emo_mat.sum(axis=1, keepdims=True)
    emo_mat = emo_mat / row_sum
    df.loc[:, EMOS] = emo_mat

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_parquet = out_dir / "meta.parquet"
    df.to_parquet(meta_parquet, index=False)

    stats = {
        "rows_in": int(before),
        "rows_out": int(after),
        "duplicates_collapsed": int(before - after),
        "epsilon": float(args.epsilon),
        "agg": args.agg,
        "columns": list(df.columns),
        "emotion_columns": EMOS,
        "movie_count": int(len(df)),
        "example": df.head(3).to_dict(orient="records"),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if args.write_json:
        meta_json = out_dir / "meta.json"
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False)

    print(f"[ok] wrote: {meta_parquet}")
    print(f"[ok] wrote: {out_dir/'stats.json'}")
    if args.write_json:
        print(f"[ok] wrote: {out_dir/'meta.json'}")

if __name__ == "__main__":
    main()
