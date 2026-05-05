# src/app/bm25_lgcn_app.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

# Project helpers
from src.retrieval.search import (
    load_bm25_index,
    search_bm25_chunks,
    aggregate_by_movie,
    zscore,
    apply_filters,
    load_index, encode_query, search_dense_chunks,  # dense optional
)
from src.retrieval.lightgcn import (
    load_cosine_matrix, user_item_scores, movie_scores_from_items
)

# ------------------ Paths / ENV ------------------
DEFAULTS = dict(
    BM25_DIR="artifacts/indices/bm25",
    LGCN_SIM="artifacts/indices/lightgcn/sim_user_item.npy",
    IID_MAP_CSV="data/raw/item_text.csv",       # must contain iid,movieId, poster_path, title etc.
    DENSE_DIR="artifacts/indices/gemma",        # used only if Dense is enabled
    ENCODER="gemma",
    MODEL="/mnt/nas/sakshipandey/main/models/embeddinggemma-300m",
)

# ------------------ Caches ------------------
@st.cache_data(show_spinner=False)
def load_item_meta(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize types we commonly use
    if "adult" in df.columns:
        df["adult"] = df["adult"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df

@st.cache_resource(show_spinner=False)
def cache_bm25(bm25_dir: str):
    return load_bm25_index(bm25_dir)

@st.cache_resource(show_spinner=False)
def cache_dense(indices_dir: str):
    return load_index(indices_dir, metric="ip")

@st.cache_resource(show_spinner=False)
def cache_lgcn(sim_path: str) -> np.ndarray:
    return load_cosine_matrix(sim_path)

@st.cache_data(show_spinner=False)
def build_iid_to_movie_map(meta_csv: str) -> Dict[int, int]:
    df = pd.read_csv(meta_csv)
    if "iid" not in df.columns or "movieId" not in df.columns:
        return {}
    m = (
        df.dropna(subset=["iid","movieId"])
          .assign(iid=lambda x: x["iid"].astype(float).astype(int),
                  movieId=lambda x: x["movieId"].astype(float).astype(int))
          .drop_duplicates("iid")
          .set_index("iid")["movieId"]
          .to_dict()
    )
    return m

# ------------------ Helpers ------------------
def poster_url(row: pd.Series) -> Optional[str]:
    """Use poster_path column from item_text.csv if available."""
    pp = row.get("poster_path")
    if isinstance(pp, str) and pp.strip():
        return f"https://image.tmdb.org/t/p/w342{pp}"
    return None

def normalize_weights(alpha: float, beta: float, gamma: float, use_dense: bool, use_lgcn: bool) -> Dict[str, float]:
    w = {
        "dense": alpha if use_dense else 0.0,
        "bm25":  beta,
        "lgcn":  gamma if use_lgcn else 0.0
    }
    s = sum(w.values())
    if s <= 0:
        return {"dense": 0.0, "bm25": 1.0, "lgcn": 0.0}
    return {k: v / s for k, v in w.items()}

def run_bm25(lb, query: str, top_k_chunks: int, filters: Optional[dict], agg: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ch = search_bm25_chunks(lb, query, top_k=top_k_chunks, filters=filters)
    mv = aggregate_by_movie(ch, "score_bm25", how=agg) if len(ch) else pd.DataFrame(columns=["movieId","score_bm25_movie"])
    return ch, mv

def run_dense(li, query: str, encoder: str, model: str, top_k_chunks: int, filters: Optional[dict], agg: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = encode_query(query, encoder=encoder, model=model)
    ch = search_dense_chunks(li, q, top_k=top_k_chunks, filters=filters)
    mv = aggregate_by_movie(ch, "score_dense", how=agg) if len(ch) else pd.DataFrame(columns=["movieId","score_dense_movie"])
    return ch, mv

def run_lgcn(sim: np.ndarray, user_idx: int, iid_to_movie: Dict[int,int], movie_meta: pd.DataFrame, filters: Optional[dict]) -> pd.DataFrame:
    if not (0 <= user_idx < sim.shape[0]):
        st.warning(f"user_idx {user_idx} out of range (0..{sim.shape[0]-1})")
        return pd.DataFrame(columns=["movieId","score_lgcn"])
    item_scores = user_item_scores(sim, user_idx=user_idx)
    mv_scores = movie_scores_from_items(item_scores, iid_to_movie, agg="max")
    lgcn_df = pd.DataFrame({"movieId": list(mv_scores.keys()), "score_lgcn": list(mv_scores.values())})

    if filters and not movie_meta.empty:
        nice = ["movieId","release_date","original_language","adult","type"]
        avail = [c for c in nice if c in movie_meta.columns]
        if not avail:
            return lgcn_df
        mx = movie_meta.drop_duplicates("movieId")[avail]
        lgcn_df = lgcn_df.merge(mx, on="movieId", how="left")
        try:
            lgcn_df = lgcn_df[apply_filters(lgcn_df, filters)][["movieId","score_lgcn"]]
        except Exception:
            pass
    return lgcn_df

def combine_and_rank(
    bm25_m: pd.DataFrame,
    dense_m: Optional[pd.DataFrame],
    lgcn_m: Optional[pd.DataFrame],
    weights: Dict[str,float],
    meta_for_join: pd.DataFrame,
    top_m: int
) -> pd.DataFrame:
    # join sources
    frames = []
    if dense_m is not None and len(dense_m): frames.append(dense_m)
    if bm25_m  is not None and len(bm25_m) : frames.append(bm25_m)
    if lgcn_m  is not None and len(lgcn_m) : frames.append(lgcn_m)

    base = None
    for fr in frames:
        base = fr if base is None else base.merge(fr, on="movieId", how="outer")
    if base is None or not len(base):
        return pd.DataFrame()

    # fill missing
    for col in ("score_dense_movie","score_bm25_movie","score_lgcn"):
        if col not in base.columns: base[col] = 0.0
        base[col] = base[col].fillna(0.0)

    # z AFTER aggregation (movie-level)
    zd = zscore(base["score_dense_movie"]) if "score_dense_movie" in base.columns else pd.Series(np.zeros(len(base)), index=base.index)
    zb = zscore(base["score_bm25_movie"]) if "score_bm25_movie" in base.columns else pd.Series(np.zeros(len(base)), index=base.index)
    zc = zscore(base["score_lgcn"])      if "score_lgcn" in base.columns      else pd.Series(np.zeros(len(base)), index=base.index)

    base["z_dense"] = zd
    base["z_bm25"]  = zb
    base["z_lgcn"]  = zc

    # contributions & final
    base["contrib_dense"] = weights.get("dense",0.0) * base["z_dense"]
    base["contrib_bm25"]  = weights.get("bm25",0.0)  * base["z_bm25"]
    base["contrib_lgcn"]  = weights.get("lgcn",0.0)  * base["z_lgcn"]
    base["score_final"]   = base["contrib_dense"] + base["contrib_bm25"] + base["contrib_lgcn"]

    # attach presentable metadata
    nice = ["movieId","title","original_title","release_date","original_language","type","poster_path","TMDbID"]
    nice_avail = ["movieId"] + [c for c in nice if c in meta_for_join.columns and c != "movieId"]
    meta_unique = meta_for_join.drop_duplicates("movieId")[nice_avail]

    out = (
        base.merge(meta_unique, on="movieId", how="left")
            .sort_values("score_final", ascending=False)
            .head(top_m)
            .reset_index(drop=True)
    )
    return out

# ------------------ UI ------------------
st.set_page_config(page_title="Movie Search — BM25 + LightGCN", layout="wide")
st.markdown("## 🎬 Movie Search: BM25 + LightGCN ")
# st.caption("BM25 over chunks + LightGCN user prior; combine *z*-scored signals with normalized weights. Dense is optional (off by default).")

# Sidebar controls (sources & weights, paths, filters)
with st.sidebar:
    st.header("⚙️ Settings")

    # Data paths
    bm25_dir   = st.text_input("BM25 dir", DEFAULTS["BM25_DIR"])
    lgcn_sim   = st.text_input("LightGCN sim (U×I .npy)", DEFAULTS["LGCN_SIM"])
    iid_map_csv= st.text_input("iid→movieId CSV", DEFAULTS["IID_MAP_CSV"])
    dense_dir  = st.text_input("Dense (FAISS) dir", DEFAULTS["DENSE_DIR"])

    # Retrieval controls
    top_k_chunks = st.number_input("Top-K chunks (per retriever)", min_value=20, max_value=2000, value=200, step=10)
    top_m_movies = st.number_input("Top-M movies", min_value=5, max_value=100, value=10, step=1)
    agg = st.selectbox("Aggregate chunks → movie", options=["sum","max"], index=0)

    # Sources
    st.subheader("Retrievers")
    use_bm25  = True
    use_dense = st.checkbox("Use DENSE (FAISS)", value=False)
    use_lgcn  = st.checkbox("Use LightGCN", value=True)

    # Weights
    st.subheader("Weights (α dense, β bm25, γ lgcn)")
    choices = [round(x,2) for x in np.linspace(0.1, 1.0, 10)]
    alpha = st.select_slider("α (dense)", options=choices, value=0.10)
    beta  = st.select_slider("β (bm25)",  options=choices, value=0.60)
    gamma = st.select_slider("γ (lgcn)",  options=choices, value=0.30)

    preset = st.selectbox("Presets",
        ["(custom)", "[0.34, 0.33, 0.33]", "[0.6, 0.2, 0.2]", "[0.5, 0.0, 0.5]", "[0.3, 0.5, 0.2]"], index=0)
    if preset != "(custom)":
        pa = [float(x) for x in preset.strip("[]").split(",")]
        alpha, beta, gamma = pa

    user_idx = st.number_input("LightGCN user index", min_value=0, value=0, step=1)

    st.subheader("Filters")
    lang = st.text_input("original_language (e.g., 'en')", value="")
    year_gte = st.number_input("year >= (0 ignore)", min_value=0, max_value=2100, value=0)
    year_lte = st.number_input("year <= (0 ignore)", min_value=0, max_value=2100, value=0)
    adult_opt= st.selectbox("Adult content", options=["ignore","only non-adult","only adult"], index=0)
    typ = st.text_input("type (e.g., 'movie')", value="")

# Query in the MAIN area (center)
with st.form("search_form", clear_on_submit=False):
    q = st.text_input("Query", value=st.session_state.get("last_query", "animated toys that come to life"),
                      placeholder="e.g., animated toys that come to life")
    c1, c2 = st.columns([1,1])
    with c1:
        submit = st.form_submit_button("🔎 Search / Rerank", use_container_width=True)
    with c2:
        st.write("")  # spacer

# Filters dict
filters = {}
if lang: filters["language"] = lang
if year_gte > 0: filters["year_gte"] = int(year_gte)
if year_lte > 0: filters["year_lte"] = int(year_lte)
if typ: filters["type"] = typ
if adult_opt != "ignore":
    filters["adult"] = (adult_opt == "only adult")

# Load artifacts
bm25 = cache_bm25(bm25_dir)
meta_df = load_item_meta(iid_map_csv)
iid_to_movie = build_iid_to_movie_map(iid_map_csv)
lgcn = cache_lgcn(lgcn_sim)
li = cache_dense(dense_dir) if use_dense else None

# Normalized weights only across enabled sources
W = normalize_weights(alpha, beta, gamma, use_dense=use_dense, use_lgcn=use_lgcn)
# st.caption(f"**Normalized weights in use** → {W}")

if submit:
    st.session_state["last_query"] = q
    if not q.strip():
        st.warning("Enter a query to search.")
        st.stop()

    with st.spinner("Running retrieval…"):
        # BM25 (always on)
        bm25_chunks, bm25_movies = run_bm25(bm25, q, top_k_chunks, filters or None, agg)

        # Dense (optional)
        dense_chunks, dense_movies = None, None
        if use_dense and li is not None:
            dense_chunks, dense_movies = run_dense(li, q, DEFAULTS["ENCODER"], DEFAULTS["MODEL"], top_k_chunks, filters or None, agg)

        # LightGCN (optional)
        lgcn_movies = run_lgcn(lgcn, user_idx, iid_to_movie, meta_df, filters or None) if use_lgcn else None

        # Combine
        final = combine_and_rank(
            bm25_m=bm25_movies,
            dense_m=dense_movies,
            lgcn_m=lgcn_movies,
            weights=W,
            meta_for_join=meta_df,
            top_m=top_m_movies
        )

    if final.empty:
        st.info("No results.")
        st.stop()

    st.subheader("Results")
    for _, row in final.iterrows():
        cL, cR = st.columns([1, 2.2])
        with cL:
            pu = poster_url(row)
            if pu: st.image(pu, use_container_width=True)
            else:  st.write("No poster")
        with cR:
            title = row.get("title") or row.get("original_title") or f"movieId {row['movieId']}"
            rd = row.get("release_date", "")
            st.markdown(f"### {title}")
            if isinstance(rd, str) and rd:
                st.caption(rd)
            st.caption(f"movieId: {int(row['movieId'])}")

            # Show math: raw, z, contribution
            bm25_raw = row.get("score_bm25_movie", 0.0); zb = row.get("z_bm25", 0.0);  cb = row.get("contrib_bm25", 0.0)
            dense_raw= row.get("score_dense_movie", 0.0); zd = row.get("z_dense", 0.0); cd = row.get("contrib_dense", 0.0)
            lgcn_raw = row.get("score_lgcn", 0.0);       zc = row.get("z_lgcn", 0.0);  cc = row.get("contrib_lgcn", 0.0)

            st.metric("score_final", f"{row['score_final']:.3f}")
            st.caption(f"bm25: {bm25_raw:.3f}  (z={zb:+.3f}, contrib={cb:+.3f})")
            if use_dense:
                st.caption(f"dense: {dense_raw:.3f} (z={zd:+.3f}, contrib={cd:+.3f})")
            if use_lgcn:
                st.caption(f"lgcn: {lgcn_raw:.3f}  (z={zc:+.3f}, contrib={cc:+.3f})")

        # Chunk previews
        with st.expander("BM25 chunks"):
            if bm25_chunks is not None and len(bm25_chunks):
                subset = bm25_chunks[bm25_chunks["movieId"] == row["movieId"]]
                show_cols = [c for c in ["section_type","part_index","score_bm25","text"] if c in subset.columns]
                st.dataframe(subset[show_cols].head(10), use_container_width=True)
            else:
                st.write("—")

        if use_dense and dense_chunks is not None and len(dense_chunks):
            with st.expander("Dense chunks"):
                subset = dense_chunks[dense_chunks["movieId"] == row["movieId"]]
                show_cols = [c for c in ["section_type","part_index","score_dense","text"] if c in subset.columns]
                st.dataframe(subset[show_cols].head(10), use_container_width=True)

        st.divider()

    with st.expander("Debug: combined table"):
        st.dataframe(final, use_container_width=True)

# st.caption("Tip: adjust α/β/γ, toggle LightGCN/Dense, and try sum/max aggregation.")
