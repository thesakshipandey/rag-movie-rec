# src/app/router_app.py
"""
Streamlit UI for 4-Expert RAG Movie Recommender with Trained Router
Experts: Dense (FAISS), BM25, LightGCN, Emotion
Router: Trained MLP that predicts expert weights from query characteristics
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

# Project imports
from src.retrieval.search import (
    load_index, load_bm25_index, encode_query,
    search_dense_chunks, search_bm25_chunks,
    aggregate_by_movie, zscore, apply_filters,
)
from src.retrieval.lightgcn import (
    load_cosine_matrix, user_item_scores, movie_scores_from_items
)
from src.emotions.emotion_index import (
    EMOS, load_emotion_index, score_movies_by_emotion
)
from src.emotions.emotion_prompt import infer_prompt_vector
from src.router.mlp_router import RouterMLP

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DEFAULTS = {
    "DENSE_DIR": "artifacts/indices/qwen_fullmovie",
    "BM25_DIR": "artifacts/indices/bm25",
    "LGCN_SIM": "artifacts/indices/lightgcn/sim_user_item.npy",
    "EMOTION_DIR": "artifacts/indices/emotion",
    "ROUTER_MODEL": "artifacts/router/router_mlp_sum.pt",
    "EMOTION_MODEL": "/mnt/nas/sakshipandey/main/models/roberta-plutchik-query_noKD/final",
    "QWEN_MODEL": "/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B",
    "METADATA_CSV": "data/raw/item_text.csv",
    "ENCODER": "qwen",
}

# ==================== Caching Functions ====================
@st.cache_data(show_spinner=False)
def load_item_meta(csv_path: str) -> pd.DataFrame:
    """Load movie metadata CSV with normalization"""
    df = pd.read_csv(csv_path)
    if "adult" in df.columns:
        df["adult"] = df["adult"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df

@st.cache_resource(show_spinner=False)
def cache_dense(indices_dir: str):
    """Load FAISS index"""
    return load_index(indices_dir, metric="ip")

@st.cache_resource(show_spinner=False)
def cache_bm25(bm25_dir: str):
    """Load BM25 index"""
    return load_bm25_index(bm25_dir)

@st.cache_resource(show_spinner=False)
def cache_lgcn(sim_path: str) -> np.ndarray:
    """Load LightGCN similarity matrix"""
    return load_cosine_matrix(sim_path)

@st.cache_resource(show_spinner=False)
def cache_emotion_index(emotion_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load emotion index (movie_ids, emotion_matrix)"""
    return load_emotion_index(emotion_dir)

@st.cache_resource(show_spinner=False)
def cache_router_model(model_path: str, device: str = "cpu") -> RouterMLP:
    """Load trained router MLP"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Router model not found: {model_path}")
    
    # Load state dict (saved as just state_dict(), not a checkpoint dict)
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer architecture from state_dict keys
    # The first layer is 'net.0.weight' with shape [d_hidden, d_in]
    d_hidden, d_in = state_dict["net.0.weight"].shape
    
    # dz_dim is stored in 'a' parameter shape (per-expert calibration)
    dz_dim = state_dict["a"].shape[0]
    
    # Check if mix_indices are used (optional parameters)
    mix_indices = None
    if "mix_scale" in state_dict and state_dict["mix_scale"] is not None:
        # mix_margin_weight tells us how many mix indices
        if "mix_margin_weight" in state_dict:
            n_mix = state_dict["mix_margin_weight"].shape[0]
            # Assume mix indices are the columns after dz_dim
            mix_indices = list(range(dz_dim, dz_dim + n_mix))
    
    # Initialize router with inferred parameters
    router = RouterMLP(
        d_in=d_in,
        dz_dim=dz_dim,
        d_hidden=d_hidden,
        temperature=1.0,  # Will be loaded from state_dict
        dropout=0.1,      # Doesn't matter for inference (eval mode)
        mix_indices=mix_indices,
    )
    
    # Load weights
    router.load_state_dict(state_dict)
    router.eval()
    router.to(device)
    
    return router

@st.cache_data(show_spinner=False)
def build_iid_to_movie_map(meta_csv: str) -> Dict[int, int]:
    """Build mapping from LightGCN item IDs to movie IDs"""
    df = pd.read_csv(meta_csv)
    if "iid" not in df.columns or "movieId" not in df.columns:
        return {}
    m = (
        df.dropna(subset=["iid", "movieId"])
          .assign(iid=lambda x: x["iid"].astype(float).astype(int),
                  movieId=lambda x: x["movieId"].astype(float).astype(int))
          .drop_duplicates("iid")
          .set_index("iid")["movieId"]
          .to_dict()
    )
    return m

# ==================== Helper Functions ====================
def poster_url(row: pd.Series) -> Optional[str]:
    """Generate TMDb poster URL from poster_path"""
    pp = row.get("poster_path")
    if isinstance(pp, str) and pp.strip():
        return f"https://image.tmdb.org/t/p/w342{pp}"
    return None

def run_dense(li, query: str, encoder: str, model: str, top_k: int, filters: Optional[dict], agg: str):
    """Run dense retrieval"""
    qvec = encode_query(query, encoder=encoder, model=model)
    chunks = search_dense_chunks(li, qvec, top_k=top_k, filters=filters)
    movies = aggregate_by_movie(chunks, "score_dense", how=agg) if len(chunks) else pd.DataFrame(columns=["movieId", "score_dense_movie"])
    return chunks, movies

def run_bm25(lb, query: str, top_k: int, filters: Optional[dict], agg: str):
    """Run BM25 retrieval"""
    chunks = search_bm25_chunks(lb, query, top_k=top_k, filters=filters)
    movies = aggregate_by_movie(chunks, "score_bm25", how=agg) if len(chunks) else pd.DataFrame(columns=["movieId", "score_bm25_movie"])
    return chunks, movies

def run_lgcn(sim: np.ndarray, user_idx: int, iid_to_movie: Dict[int, int], 
             movie_meta: pd.DataFrame, filters: Optional[dict]) -> pd.DataFrame:
    """Run LightGCN retrieval"""
    if not (0 <= user_idx < sim.shape[0]):
        return pd.DataFrame(columns=["movieId", "score_lgcn"])
    
    item_scores = user_item_scores(sim, user_idx=user_idx)
    mv_scores = movie_scores_from_items(item_scores, iid_to_movie, agg="max")
    lgcn_df = pd.DataFrame({"movieId": list(mv_scores.keys()), "score_lgcn": list(mv_scores.values())})
    
    # Apply filters if provided
    if filters and not movie_meta.empty:
        cols = ["movieId", "release_date", "original_language", "adult", "type"]
        avail = [c for c in cols if c in movie_meta.columns]
        if avail:
            mx = movie_meta.drop_duplicates("movieId")[avail]
            lgcn_df = lgcn_df.merge(mx, on="movieId", how="left")
            try:
                lgcn_df = lgcn_df[apply_filters(lgcn_df, filters)][["movieId", "score_lgcn"]]
            except Exception:
                pass
    
    return lgcn_df

def run_emotion(query: str, all_movie_ids: list, emo_ids: np.ndarray, emo_mat: np.ndarray,
                emo_model_dir: Optional[str], device: str = "cpu", dtype: str = "float16") -> Tuple[pd.DataFrame, np.ndarray, str]:
    """Run emotion-based retrieval"""
    # Infer query emotion
    query_emo, source = infer_prompt_vector(
        query=query,
        emo_model_dir=emo_model_dir,
        prompt_emotion=None,
        device=device,
        dtype=dtype,
        max_len=128,
    )
    
    # Score movies by JS divergence
    emo_scores = score_movies_by_emotion(all_movie_ids, emo_ids, emo_mat, query_emo)
    emo_df = pd.DataFrame({"movieId": all_movie_ids, "score_emo": emo_scores})
    
    return emo_df, query_emo, source

def compute_router_features(movie_scores_df: pd.DataFrame, expected_dim: int = 4) -> np.ndarray:
    """
    Compute Δz summary features for router input
    For inference, we use max - median as a proxy for discriminative power
    
    Args:
        movie_scores_df: DataFrame with z-score columns
        expected_dim: Expected input dimension from router model
    
    Returns:
        Feature vector of shape [expected_dim]
    """
    # Compute basic Δz features (4 experts)
    dz_cols = ["z_dense", "z_bm25", "z_lgcn", "z_emo"]
    dz_features = []
    for col in dz_cols:
        if col in movie_scores_df.columns:
            z_vals = movie_scores_df[col].values
            if len(z_vals) > 0:
                dz = float(np.max(z_vals) - np.median(z_vals))
            else:
                dz = 0.0
        else:
            dz = 0.0
        dz_features.append(dz)
    
    dz_array = np.array(dz_features, dtype=np.float32)
    
    # If model expects more features (derived features from training), compute them
    if expected_dim > 4:
        # Add derived features: abs, pos, neg, hit for each Δz
        derived = []
        for dz in dz_features:
            derived.append(abs(dz))              # abs_dz
            derived.append(max(0.0, dz))         # pos_dz
            derived.append(max(0.0, -dz))        # neg_dz
            derived.append(1.0 if abs(dz) > 1e-9 else 0.0)  # hit_dz
        
        # Concatenate basic + derived
        all_features = np.concatenate([dz_array, np.array(derived, dtype=np.float32)])
        
        # If still need more features, pad with zeros (for mix weights and categorical features)
        if len(all_features) < expected_dim:
            padding = np.zeros(expected_dim - len(all_features), dtype=np.float32)
            all_features = np.concatenate([all_features, padding])
        
        return all_features[:expected_dim]
    
    return dz_array

def fuse_with_router(movie_scores_df: pd.DataFrame, router: RouterMLP, device: str = "cpu") -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Use router to predict weights and fuse scores
    Returns: (updated dataframe with final scores, weights [α, β, γ, δ])
    """
    # Get expected input dimension from router
    expected_dim = router.net[0].in_features
    
    # Compute router input features
    dz = compute_router_features(movie_scores_df, expected_dim=expected_dim)
    
    # Router inference
    dz_tensor = torch.from_numpy(dz).unsqueeze(0).to(device)  # [1, expected_dim]
    with torch.no_grad():
        _, weights = router(dz_tensor)  # weights: [1, 4]
    
    weights = weights.cpu().numpy()[0]  # [4]
    
    # Ensure z-score columns exist
    for col in ["z_dense", "z_bm25", "z_lgcn", "z_emo"]:
        if col not in movie_scores_df.columns:
            movie_scores_df[col] = 0.0
    
    # Compute contributions
    movie_scores_df["contrib_dense"] = weights[0] * movie_scores_df["z_dense"]
    movie_scores_df["contrib_bm25"] = weights[1] * movie_scores_df["z_bm25"]
    movie_scores_df["contrib_lgcn"] = weights[2] * movie_scores_df["z_lgcn"]
    movie_scores_df["contrib_emo"] = weights[3] * movie_scores_df["z_emo"]
    
    # Final score
    movie_scores_df["score_final"] = (
        movie_scores_df["contrib_dense"] +
        movie_scores_df["contrib_bm25"] +
        movie_scores_df["contrib_lgcn"] +
        movie_scores_df["contrib_emo"]
    )
    
    return movie_scores_df, weights

def combine_expert_scores(dense_m: pd.DataFrame, bm25_m: pd.DataFrame, 
                         lgcn_m: pd.DataFrame, emo_m: pd.DataFrame) -> pd.DataFrame:
    """Combine scores from all experts into single dataframe"""
    # Outer join all sources
    frames = []
    if len(dense_m): frames.append(dense_m)
    if len(bm25_m): frames.append(bm25_m)
    if len(lgcn_m): frames.append(lgcn_m)
    if len(emo_m): frames.append(emo_m)
    
    if not frames:
        return pd.DataFrame()
    
    base = frames[0]
    for fr in frames[1:]:
        base = base.merge(fr, on="movieId", how="outer")
    
    # Fill missing scores
    for col in ["score_dense_movie", "score_bm25_movie", "score_lgcn", "score_emo"]:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = base[col].fillna(0.0)
    
    # Compute z-scores
    base["z_dense"] = zscore(base["score_dense_movie"])
    base["z_bm25"] = zscore(base["score_bm25_movie"])
    base["z_lgcn"] = zscore(base["score_lgcn"])
    base["z_emo"] = zscore(base["score_emo"])
    
    return base

def plot_emotion_distribution(emotion_vec: np.ndarray, title: str = ""):
    """Plot emotion distribution as bar chart"""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Bar(x=EMOS, y=emotion_vec, marker_color='lightblue')
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Emotion",
        yaxis_title="Probability",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# ==================== Main App ====================
st.set_page_config(page_title="RAG Movie Recommender (Router)", layout="wide")
st.markdown("## 🎬 RAG Movie Recommender with Router")
st.caption("4-Expert system: Dense (FAISS) + BM25 + LightGCN + Emotion, fused with trained MLP router")

# ==================== Sidebar Configuration ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Paths")
    dense_dir = st.text_input("Dense (FAISS) dir", DEFAULTS["DENSE_DIR"])
    bm25_dir = st.text_input("BM25 dir", DEFAULTS["BM25_DIR"])
    lgcn_sim = st.text_input("LightGCN sim matrix", DEFAULTS["LGCN_SIM"])
    emotion_dir = st.text_input("Emotion index dir", DEFAULTS["EMOTION_DIR"])
    metadata_csv = st.text_input("Metadata CSV", DEFAULTS["METADATA_CSV"])
    
    # Router model selection
    st.subheader("Router Model")
    router_dir = Path("artifacts/router")
    available_routers = sorted([f.name for f in router_dir.glob("router_mlp*.pt")]) if router_dir.exists() else []
    if available_routers:
        default_idx = available_routers.index("router_mlp_sum.pt") if "router_mlp_sum.pt" in available_routers else 0
        router_file = st.selectbox("Select router model", available_routers, index=default_idx)
        router_model_path = str(router_dir / router_file)
    else:
        router_model_path = st.text_input("Router model path", DEFAULTS["ROUTER_MODEL"])
    
    use_emotion_model = st.checkbox("Use emotion model (RoBERTa)", value=False, 
                                      help="If unchecked, uses keyword-based emotion inference")
    emotion_model_dir = st.text_input("Emotion classifier dir", DEFAULTS["EMOTION_MODEL"]) if use_emotion_model else None
    
    st.subheader("Retrieval Settings")
    top_k_chunks = st.number_input("Top-K chunks (per retriever)", min_value=20, max_value=2000, value=200, step=10)
    top_m_movies = st.number_input("Top-M movies", min_value=5, max_value=100, value=10, step=1)
    agg = st.selectbox("Chunk aggregation", options=["sum", "max"], index=0)
    
    st.subheader("User Context")
    user_idx = st.number_input("LightGCN user index", min_value=0, max_value=942, value=42, step=1)
    
    st.subheader("Filters")
    lang = st.text_input("Language (e.g., 'en')", value="")
    year_gte = st.number_input("Year >= (0 to ignore)", min_value=0, max_value=2100, value=0)
    year_lte = st.number_input("Year <= (0 to ignore)", min_value=0, max_value=2100, value=0)
    typ = st.text_input("Type (e.g., 'movie')", value="")
    adult_opt = st.selectbox("Adult content", options=["ignore", "only non-adult", "only adult"], index=0)

# Build filters dict
filters = {}
if lang: filters["language"] = lang
if year_gte > 0: filters["year_gte"] = int(year_gte)
if year_lte > 0: filters["year_lte"] = int(year_lte)
if typ: filters["type"] = typ
if adult_opt != "ignore":
    filters["adult"] = (adult_opt == "only adult")

# ==================== Query Input ====================
with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Query",
        value=st.session_state.get("last_query", "animated toys that come to life"),
        placeholder="e.g., dark thriller with unexpected twists"
    )
    submit = st.form_submit_button("🔎 Search", use_container_width=True)

# ==================== Load Resources ====================
try:
    with st.spinner("Loading indices and models..."):
        dense_idx = cache_dense(dense_dir)
        bm25_idx = cache_bm25(bm25_dir)
        lgcn_matrix = cache_lgcn(lgcn_sim)
        emo_ids, emo_mat = cache_emotion_index(emotion_dir)
        router = cache_router_model(router_model_path, device="cpu")
        metadata = load_item_meta(metadata_csv)
        iid_to_movie = build_iid_to_movie_map(metadata_csv)
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# ==================== Search Execution ====================
if submit and query.strip():
    st.session_state["last_query"] = query
    
    with st.spinner("Running 4-expert retrieval + router..."):
        # Run all 4 experts
        dense_chunks, dense_movies = run_dense(
            dense_idx, query, DEFAULTS["ENCODER"], DEFAULTS["QWEN_MODEL"],
            top_k_chunks, filters or None, agg
        )
        
        bm25_chunks, bm25_movies = run_bm25(
            bm25_idx, query, top_k_chunks, filters or None, agg
        )
        
        lgcn_movies = run_lgcn(
            lgcn_matrix, user_idx, iid_to_movie, metadata, filters or None
        )
        
        # Get all candidate movie IDs for emotion scoring
        all_movie_ids = set()
        for df in [dense_movies, bm25_movies, lgcn_movies]:
            if len(df):
                all_movie_ids.update(df["movieId"].astype(int).tolist())
        
        emo_movies, query_emotion, query_emotion_source = run_emotion(
            query, sorted(all_movie_ids), emo_ids, emo_mat,
            emotion_model_dir, device="cpu", dtype="float16"
        )
        
        # Combine all expert scores
        combined = combine_expert_scores(dense_movies, bm25_movies, lgcn_movies, emo_movies)
        
        if combined.empty:
            st.warning("No results found.")
            st.stop()
        
        # Apply router to get weights and final scores
        combined, router_weights = fuse_with_router(combined, router, device="cpu")
        
        # Attach metadata and rank
        nice_cols = ["movieId", "title", "original_title", "release_date", "original_language", "type", "poster_path", "TMDbID"]
        nice_avail = ["movieId"] + [c for c in nice_cols if c in metadata.columns and c != "movieId"]
        meta_unique = metadata.drop_duplicates("movieId")[nice_avail]
        
        final = (
            combined.merge(meta_unique, on="movieId", how="left")
                    .sort_values("score_final", ascending=False)
                    .head(top_m_movies)
                    .reset_index(drop=True)
        )
    
    # ==================== Display Results ====================
    
    # Show query emotion
    st.subheader("Query Analysis")
    
    # Show emotion source info
    if query_emotion_source == "lexicon_fallback":
        st.warning("⚠️ Emotion model failed to load. Using keyword-based inference as fallback.")
    elif query_emotion_source == "lexicon":
        st.info("ℹ️ Using keyword-based emotion inference (model disabled).")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Query Emotion Distribution** ({query_emotion_source})")
        emo_df = pd.DataFrame([query_emotion], columns=EMOS)
        st.dataframe(emo_df.round(3), use_container_width=True)
    with col2:
        fig = plot_emotion_distribution(query_emotion, "Query Emotion Profile")
        st.plotly_chart(fig, use_container_width=True)
    
    # Show router weights
    st.subheader("Router-Predicted Weights")
    weights_df = pd.DataFrame([{
        "α (Dense)": f"{router_weights[0]:.3f}",
        "β (BM25)": f"{router_weights[1]:.3f}",
        "γ (LightGCN)": f"{router_weights[2]:.3f}",
        "δ (Emotion)": f"{router_weights[3]:.3f}",
    }])
    st.dataframe(weights_df, use_container_width=True)
    
    # Display top-k movies
    st.subheader(f"Top {len(final)} Recommendations")
    
    for idx, row in final.iterrows():
        with st.container():
            col_poster, col_info = st.columns([1, 3])
            
            with col_poster:
                pu = poster_url(row)
                if pu:
                    st.image(pu, use_container_width=True)
                else:
                    st.write("No poster")
            
            with col_info:
                title = row.get("title") or row.get("original_title") or f"movieId {row['movieId']}"
                rd = row.get("release_date", "")
                st.markdown(f"### {title}")
                if isinstance(rd, str) and rd:
                    st.caption(f"Released: {rd}")
                st.caption(f"movieId: {int(row['movieId'])}")
                
                # Final score
                st.metric("Final Score", f"{row['score_final']:.4f}")
                
                # Score breakdown
                with st.expander("📊 Score Breakdown", expanded=True):
                    breakdown_data = {
                        "Expert": ["Dense", "BM25", "LightGCN", "Emotion"],
                        "Raw Score": [
                            f"{row.get('score_dense_movie', 0.0):.3f}",
                            f"{row.get('score_bm25_movie', 0.0):.3f}",
                            f"{row.get('score_lgcn', 0.0):.3f}",
                            f"{row.get('score_emo', 0.0):.3f}",
                        ],
                        "Z-Score": [
                            f"{row.get('z_dense', 0.0):+.3f}",
                            f"{row.get('z_bm25', 0.0):+.3f}",
                            f"{row.get('z_lgcn', 0.0):+.3f}",
                            f"{row.get('z_emo', 0.0):+.3f}",
                        ],
                        "Weight": [
                            f"{router_weights[0]:.3f}",
                            f"{router_weights[1]:.3f}",
                            f"{router_weights[2]:.3f}",
                            f"{router_weights[3]:.3f}",
                        ],
                        "Contribution": [
                            f"{row.get('contrib_dense', 0.0):+.3f}",
                            f"{row.get('contrib_bm25', 0.0):+.3f}",
                            f"{row.get('contrib_lgcn', 0.0):+.3f}",
                            f"{row.get('contrib_emo', 0.0):+.3f}",
                        ],
                    }
                    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)
                
                # Emotion distribution
                with st.expander("😊 Emotion Profile"):
                    mid = int(row["movieId"])
                    id2row_map = {int(i): idx for idx, i in enumerate(emo_ids.tolist())}
                    if mid in id2row_map:
                        movie_emo = emo_mat[id2row_map[mid]]
                    else:
                        movie_emo = np.full(8, 1.0/8, dtype=np.float32)
                    
                    fig_emo = plot_emotion_distribution(movie_emo, f"Emotion Profile: {title}")
                    st.plotly_chart(fig_emo, use_container_width=True)
                
                # Chunk evidence
                with st.expander("📄 Dense Chunks"):
                    if len(dense_chunks):
                        subset = dense_chunks[dense_chunks["movieId"] == row["movieId"]]
                        if len(subset):
                            show_cols = [c for c in ["section_type", "part_index", "score_dense", "text"] if c in subset.columns]
                            st.dataframe(subset[show_cols].head(5), use_container_width=True)
                        else:
                            st.write("—")
                    else:
                        st.write("—")
                
                with st.expander("📝 BM25 Chunks"):
                    if len(bm25_chunks):
                        subset = bm25_chunks[bm25_chunks["movieId"] == row["movieId"]]
                        if len(subset):
                            show_cols = [c for c in ["section_type", "part_index", "score_bm25", "text"] if c in subset.columns]
                            st.dataframe(subset[show_cols].head(5), use_container_width=True)
                        else:
                            st.write("—")
                    else:
                        st.write("—")
            
            st.divider()
    
    # Debug section
    with st.expander("🔧 Debug: Full Results Table"):
        st.dataframe(final, use_container_width=True)
    
    with st.expander("🔧 Debug: Router Features"):
        expected_dim = router.net[0].in_features
        dz = compute_router_features(combined, expected_dim=expected_dim)
        
        # Show basic Δz features
        router_features = pd.DataFrame([{
            "dz_dense": f"{dz[0]:.4f}",
            "dz_bm25": f"{dz[1]:.4f}",
            "dz_lgcn": f"{dz[2]:.4f}",
            "dz_emo": f"{dz[3]:.4f}",
        }])
        st.dataframe(router_features, use_container_width=True)
        st.caption(f"Router input: Δz = max(z) - median(z) per expert")
        st.caption(f"Total features: {len(dz)} (including derived features and padding)")

elif submit:
    st.warning("Please enter a query.")

# Footer
st.markdown("---")
st.caption("RAG Movie Recommender — 4-Expert System with Learned Router")


