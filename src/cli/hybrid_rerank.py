# src/cli/hybrid_rerank.py
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.retrieval.search import (
    load_index, load_bm25_index, encode_query,
    search_dense_chunks, search_bm25_chunks,
    aggregate_by_movie, zscore, apply_filters,
)

# Emotion index + constants
from src.emotions.emotion_index import (
    EMOS, load_emotion_index, score_movies_by_emotion,
)
# Prompt vector (explicit, model, or lexicon)
from src.emotions.emotion_prompt import infer_prompt_vector

# LightGCN
from src.retrieval.lightgcn import (
    load_cosine_matrix, user_item_scores, movie_scores_from_items
)

# Utilities
from src.utils.hybrid_helpers import (
    setup_logger, build_filters, build_iid_to_movie_map
)


def main():
    ap = argparse.ArgumentParser("Hybrid re-ranker: Dense + BM25 + LightGCN + (optional) Emotion")
    # Dense / FAISS
    ap.add_argument("--indices_dir", default="artifacts/indices/qwen_fullmovie",
                    help="FAISS index dir (built from Qwen full-movie embeddings)")
    ap.add_argument("--encoder", choices=["qwen", "gemma", "minilm"], default="qwen")
    ap.add_argument("--model", default="/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k_chunks", type=int, default=200)
    ap.add_argument("--top_m_movies", type=int, default=10)
    ap.add_argument("--agg", choices=["sum", "max"], default="sum",
                    help="How to aggregate chunk scores to movie scores (default: sum)")

    # BM25
    ap.add_argument("--bm25_dir", default="artifacts/indices/bm25",
                    help="BM25 index dir (built on the same 'text' you search over)")

    # LightGCN
    ap.add_argument("--lgcn_sim", default="artifacts/indices/lightgcn/sim_user_item.npy")
    ap.add_argument("--user_idx", type=int, required=True)
    ap.add_argument("--iid_map_csv", default="data/raw/item_text.csv",
                    help="CSV with columns iid,movieId used to map LightGCN items to movies (fallback if meta lacks iid).")

    # Emotion index (movie vectors) + prompt (query vector)
    ap.add_argument("--emotion_dir", type=str, default="artifacts/indices/emotion",
                    help="Dir with emotion meta.parquet|emotion.parquet (or meta.json|emotion.json).")
    ap.add_argument("--prompt_emotion", type=str, default=None,
                    help="Single label 'Joy' | inline 'Joy=0.4,Trust=0.2,...' | path to JSON dict.")
    ap.add_argument("--prompt_emotion_model_dir", type=str, default=None,
                    help="Dir to finetuned RoBERTa classifier (.../final). Used when --prompt_emotion is not provided.")
    ap.add_argument("--prompt_emotion_device", type=str, default="auto")
    ap.add_argument("--prompt_emotion_dtype", type=str, choices=["float32","float16","bfloat16"], default="float16")
    ap.add_argument("--prompt_emotion_max_len", type=int, default=128)

    # Weights
    ap.add_argument("--alpha", type=float, default=0.0, help="weight for dense (set 0 to disable dense entirely)")
    ap.add_argument("--beta",  type=float, default=0.65, help="weight for bm25")
    ap.add_argument("--gamma", type=float, default=0.35, help="weight for lightgcn")
    ap.add_argument("--delta", type=float, default=0.0,  help="weight for emotion (JS similarity)")

    # Filters
    ap.add_argument("--language", default=None)
    ap.add_argument("--year_gte", type=int, default=None)
    ap.add_argument("--year_lte", type=int, default=None)
    ap.add_argument("--adult", type=lambda x: x.lower() in {"1", "true", "yes"}, default=None)
    ap.add_argument("--type", default=None)

    # Output & logs
    ap.add_argument("--show_chunks", action="store_true")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--logs_dir", default="logs")
    args = ap.parse_args()

    logger, log_path = setup_logger(Path(args.logs_dir))
    filters = build_filters(args)

    # ---- Always infer prompt emotion vector (so we can display it) ----
    pvec, emo_src = infer_prompt_vector(
        query=args.query,
        emo_model_dir=args.prompt_emotion_model_dir,
        prompt_emotion=args.prompt_emotion,
        device=args.prompt_emotion_device,
        dtype=args.prompt_emotion_dtype,
        max_len=args.prompt_emotion_max_len,
    )
    logger.info("prompt_emotion_source=%s", emo_src)

    # Try to load emotion index (so we can display movie distributions)
    mid = None
    M_emo = None
    try:
        if args.emotion_dir:
            mid, M_emo = load_emotion_index(args.emotion_dir)
    except Exception as e:
        logger.warning("Emotion index not available: %s", e)

    # ---- Load FAISS index/meta (always; we use meta for joins/filters) ----
    li = load_index(args.indices_dir, metric="ip")

    # ---- Dense retrieval (optional; only if alpha>0) ----
    dense_hits = pd.DataFrame()
    use_dense = (args.alpha or 0.0) > 0.0
    if use_dense:
        qvec = encode_query(args.query, encoder=args.encoder, model=args.model)
        dense_hits = search_dense_chunks(li, qvec, top_k=args.top_k_chunks, filters=filters)

    # ---- BM25 retrieval (expected ON) ----
    bm25_hits = pd.DataFrame()
    if args.bm25_dir:
        try:
            lb = load_bm25_index(args.bm25_dir)
            bm25_hits = search_bm25_chunks(lb, args.query, top_k=args.top_k_chunks, filters=filters)
        except Exception as e:
            logger.warning("BM25 failed: %s; continuing without BM25.", e)

    # ---- LightGCN (movie prior) ----
    map_iid_to_movie = build_iid_to_movie_map(li.meta, logger, args.iid_map_csv)
    M = load_cosine_matrix(args.lgcn_sim)  # (U, I)
    if not (0 <= args.user_idx < M.shape[0]):
        raise IndexError(f"user_idx {args.user_idx} out of range 0..{M.shape[0]-1}")
    item_scores = user_item_scores(M, user_idx=args.user_idx)
    movie_prior = movie_scores_from_items(item_scores, map_iid_to_movie, agg="max")
    lgcn_df = pd.DataFrame({"movieId": list(movie_prior.keys()), "score_lgcn": list(movie_prior.values())})

    # Optional: apply filters at movie level to LightGCN scores
    if len(lgcn_df) and filters:
        needed = ["movieId", "release_date", "original_language", "adult", "type"]
        avail = [c for c in needed if c in li.meta.columns]
        meta_unique = li.meta.drop_duplicates("movieId")[avail] if avail else li.meta.drop_duplicates("movieId")[["movieId"]]
        lgcn_df = lgcn_df.merge(meta_unique, on="movieId", how="left")
        try:
            lgcn_df = lgcn_df[apply_filters(lgcn_df, filters)][["movieId", "score_lgcn"]]
        except KeyError:
            pass

    # ---- Aggregate chunks → movie ----
    md = aggregate_by_movie(dense_hits, "score_dense", how=args.agg) if len(dense_hits) else pd.DataFrame(columns=["movieId", "score_dense_movie"])
    mb = aggregate_by_movie(bm25_hits,  "score_bm25",  how=args.agg) if len(bm25_hits)  else pd.DataFrame(columns=["movieId", "score_bm25_movie"])

    # ---- Join sources ----
    base = None
    for fr in [md, mb]:
        if len(fr):
            base = fr if base is None else base.merge(fr, on="movieId", how="outer")
    base = lgcn_df if base is None else base.merge(lgcn_df, on="movieId", how="outer")

    if base is None or not len(base):
        print("No candidates.")
        print("\nLogs ->", log_path)
        return

    # Ensure all score columns exist
    for col in ("score_dense_movie", "score_bm25_movie", "score_lgcn", "score_emo"):
        if col not in base.columns:
            base[col] = 0.0
        base[col] = base[col].fillna(0.0)

    # Z-normalize per source
    dz = zscore(base["score_dense_movie"]) if "score_dense_movie" in base.columns else pd.Series(np.zeros(len(base)), index=base.index)
    bz = zscore(base["score_bm25_movie"]) if "score_bm25_movie" in base.columns else pd.Series(np.zeros(len(base)), index=base.index)
    cz = zscore(base["score_lgcn"])       if "score_lgcn"       in base.columns else pd.Series(np.zeros(len(base)), index=base.index)

    # ---- Emotion scoring (optional for blend; we still display distributions even if delta=0) ----
    ez = pd.Series(np.zeros(len(base)), index=base.index)
    emo_used = False
    if M_emo is not None:
        try:
            emo_scores = score_movies_by_emotion(
                base["movieId"].fillna(-1).astype(float).astype(int).tolist(),
                mid, M_emo, pvec
            )
            base["score_emo"] = emo_scores
            if (args.delta or 0.0) > 0.0:
                ez = zscore(base["score_emo"])
                emo_used = True
        except Exception as e:
            logging.warning("Emotion scoring failed: %s", e)

    # ---- Blend ----
    w_raw = {
        "dense": max(0.0, args.alpha or 0.0),
        "bm25":  max(0.0, args.beta  or 0.0),
        "lgcn":  max(0.0, args.gamma or 0.0),
        "emo":   max(0.0, args.delta or 0.0),
    }
    s = sum(w_raw.values()) or 1.0
    w = {k: v / s for k, v in w_raw.items()}

    base["score_final"] = (
        w.get("dense", 0.0) * dz +
        w.get("bm25",  0.0) * bz +
        w.get("lgcn",  0.0) * cz +
        w.get("emo",   0.0) * ez
    )

    # ---- Attach presentation cols & rank ----
    nice = ["movieId", "title"]
    nice_avail = [c for c in nice if c in li.meta.columns]
    if "movieId" not in nice_avail:
        nice_avail = ["movieId"] + nice_avail
    meta_unique = li.meta.drop_duplicates("movieId")[nice_avail]

    out = (
        base.merge(meta_unique, on="movieId", how="left")
            .sort_values("score_final", ascending=False)
            .head(args.top_m_movies)
            .reset_index(drop=True)
    )

    # ------- Pretty print -------
    pd.set_option("display.max_colwidth", 100)

    # Prompt emotion distribution
    print("\nPrompt emotion distribution (source: %s):" % emo_src)
    prompt_df = pd.DataFrame([pvec], columns=EMOS)
    print(prompt_df.round(3).to_string(index=False))

    # Main ranking
    cols = [c for c in ["movieId", "score_final", "score_dense_movie", "score_bm25_movie", "score_lgcn", "score_emo", "title"]
            if c in out.columns]
    print("\nQuery:", args.query)
    print("\nHybrid results (final)  weights:", w)
    print(out[cols].to_string(index=False))

    # Top movies' emotion distributions (rows sum to 1)
    if M_emo is not None and len(out):
        id2row = {int(i): idx for idx, i in enumerate(mid.tolist())}
        rows = []
        for _, r in out.iterrows():
            mid_i = int(r["movieId"])
            if mid_i in id2row:
                vec = M_emo[id2row[mid_i]]
            else:
                vec = np.full(8, 1.0/8, dtype=np.float32)
            rows.append(vec)

        emo_df = pd.DataFrame(rows, columns=EMOS)
        emo_df.insert(0, "movieId", out["movieId"].astype(int).tolist())
        if "title" in out.columns:
            emo_df.insert(1, "title", out["title"].tolist())
        print("\nTop movies: emotion distributions (per-movie probs):")
        print(emo_df.round(3).to_string(index=False))

    # --- Dump chunks to logs (full, non-truncated) when requested ---
    if args.show_chunks:
        log_dir = Path(args.logs_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(dense_hits):
            dense_csv = log_dir / f"dense_chunks_{ts}.csv"
            dense_hits.to_csv(dense_csv, index=False)
            print(f"\n[written] all dense chunks -> {dense_csv}")
            print("\nTop dense chunks (preview):")
            preview_cols = ["movieId","section_type","part_index","score_dense","title","text"]
            preview_cols = [c for c in preview_cols if c in dense_hits.columns]
            print(dense_hits[preview_cols].head(15).to_string(index=False))
        if len(bm25_hits):
            bm25_csv = log_dir / f"bm25_chunks_{ts}.csv"
            bm25_hits.to_csv(bm25_csv, index=False)
            print(f"\n[written] all BM25 chunks  -> {bm25_csv}")
            print("\nTop BM25 chunks (preview):")
            preview_cols = ["movieId","section_type","part_index","score_bm25","title","text"]
            preview_cols = [c for c in preview_cols if c in bm25_hits.columns]
            print(bm25_hits[preview_cols].head(15).to_string(index=False))

    # --- CSV export (include emotion columns if available) ---
    if args.out_csv:
        out_export = out.copy()
        if M_emo is not None and len(out_export):
            id2row = {int(i): idx for idx, i in enumerate(mid.tolist())}
            rows = []
            for _, r in out_export.iterrows():
                mid_i = int(r["movieId"])
                vec = M_emo[id2row[mid_i]] if mid_i in id2row else np.full(8, 1.0/8, dtype=np.float32)
                rows.append(vec)
            emo_cols_df = pd.DataFrame(rows, columns=[f"emo_{e}" for e in EMOS])
            out_export = pd.concat([out_export.reset_index(drop=True), emo_cols_df], axis=1)

        out_p = Path(args.out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_export.to_csv(out_p, index=False)
        print("\nSaved ->", out_p.resolve())

    logging.getLogger().info(
        "query=%s | kd=%d | tm=%d | agg=%s | weights=%s | bm25=%s | lgcn=%s | emo_used=%s | emo_src=%s",
        args.query, args.top_k_chunks, args.top_m_movies, args.agg, w, bool(args.bm25_dir), True, emo_used, emo_src
    )
    print("\nLogs ->", log_path)


if __name__ == "__main__":
    main()
