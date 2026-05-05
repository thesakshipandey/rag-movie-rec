# src/router/build_router_features.py
#!/usr/bin/env python
"""
CLI: build Δz features parquet for the router.

Reads the unzipped prompts dataset (prompts.json, pairs.json, judgments.json),
builds per-prompt movie tables using current expert indices, and emits a
training-ready features parquet with Δz = z(A) - z(B) for each pair.

Assumes `src/router/features.py` provides:
  - load_prompt_triplets(prompts_dir)  # includes plutchik_dist if present
  - per_prompt_movie_table(..., q_emo_vec=None)  # optional prompt emotion vec
"""
from __future__ import annotations

import os
import argparse
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.router.features import per_prompt_movie_table, load_prompt_triplets
from src.router.logger_utils import setup_router_logger, log_config

# Retrieval backends
from src.retrieval.search import load_index, load_bm25_index
from src.emotions.emotion_index import load_emotion_index
# LightGCN optional
# (we load the npy directly to avoid importing torch-heavy deps if not needed)


def _maybe_float(x):
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _maybe_int(x):
    if x is None or pd.isna(x):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return _maybe_float(x)


def _maybe_bool(x):
    if x is None or pd.isna(x):
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"", "na", "nan", "none"}:
            return None
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n"}:
            return False
    return bool(x)


def _maybe_str(x):
    if x is None or pd.isna(x):
        return None
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_dir", required=True, help="Path to folder containing prompts.json, pairs.json, judgments.json")
    ap.add_argument("--indices_dir", default="artifacts/indices", help="Root directory of built indices")
    ap.add_argument("--out", default="artifacts/router/features.parquet", help="Output parquet path")
    ap.add_argument("--agg_kind", default="sum", choices=["sum", "max", "attn"], help="Aggregation over chunks → movie")
    ap.add_argument("--topk", type=int, default=200, help="Top-K chunks for dense/BM25 retrieval")
    ap.add_argument("--user_idx_default", type=int, default=1, help="Fallback user index if none in dataset (default: 1)")
    ap.add_argument("--encoder", default="qwen", choices=["qwen", "gemma", "minilm"], help="Encoder key for query embedding")
    ap.add_argument("--model", default="/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B", help="Encoder model path/name")
    ap.add_argument("--logs_dir", default="logs", help="Directory for log files")
    ap.add_argument("--attn_tau", type=float, default=1.0,
                help="Softmax temperature for --agg_kind=attn (ignored for sum/max)")
    args = ap.parse_args()

    # Logging
    logger, log_file = setup_router_logger(args.logs_dir, name="build_features")
    logger.info("Starting Router Feature Building")

    cfg = {
        "prompts_dir": args.prompts_dir,
        "indices_dir": args.indices_dir,
        "output": args.out,
        "aggregation_kind": args.agg_kind,
        "topk": args.topk,
        "user_idx_default": args.user_idx_default,
        "encoder": args.encoder,
        "model": args.model,
    }
    log_config(logger, cfg, "Feature Building Configuration")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load triplets (includes plutchik_dist from dataset if present)
    logger.info(f"Loading prompt triplets from {args.prompts_dir}")
    df = load_prompt_triplets(args.prompts_dir)
    logger.info(f"Loaded {len(df)} pairs from {df['prompt_id'].nunique()} prompts")

    if "user_idx" not in df.columns:
        df["user_idx"] = args.user_idx_default if args.user_idx_default is not None else 1
        logger.info(f"Added default user_idx: {df['user_idx'].iloc[0]}")
    elif df["user_idx"].isna().all() or (df["user_idx"] == 0).all():
        # If user_idx exists but is all NaN or all 0, set to default
        df["user_idx"] = args.user_idx_default if args.user_idx_default is not None else 1
        logger.info(f"Replaced null/zero user_idx with default: {df['user_idx'].iloc[0]}")

    # Load indices
    logger.info("Loading indices...")
    dense_idx = load_index(os.path.join(args.indices_dir, "qwen_fullmovie"), metric="ip")
    logger.info(f"  Dense index loaded: {len(dense_idx.meta)} entries")

    bm25_idx = load_bm25_index(os.path.join(args.indices_dir, "bm25"))
    logger.info(f"  BM25 index loaded: {len(bm25_idx.meta)} entries")

    emo_ids, emo_mat = load_emotion_index(os.path.join(args.indices_dir, "emotion"))
    logger.info(f"  Emotion index loaded: {len(emo_ids)} movies")

    lgcn_sim = None
    lgcn_path = os.path.join(args.indices_dir, "lightgcn", "sim_user_item.npy")
    if os.path.exists(lgcn_path):
        lgcn_sim = np.load(lgcn_path)
        logger.info(f"  LightGCN matrix loaded: shape {lgcn_sim.shape}")
    else:
        logger.warning(f"  LightGCN matrix not found at {lgcn_path}")

    logger.info("=" * 80)
    logger.info("Building features for each prompt...")
    logger.info("=" * 80)

    rows = []
    # Group once per prompt to reuse per-prompt computations
    for pid, g in tqdm(df.groupby("prompt_id"), desc="prompts"):
        ptext = g["prompt_text"].iloc[0]
        uidx  = g["user_idx"].iloc[0]  # may be None
        mix_alpha = _maybe_float(g["mix_weights.alpha"].iloc[0]) if "mix_weights.alpha" in g.columns else None
        mix_beta  = _maybe_float(g["mix_weights.beta"].iloc[0]) if "mix_weights.beta" in g.columns else None
        mix_gamma = _maybe_float(g["mix_weights.gamma"].iloc[0]) if "mix_weights.gamma" in g.columns else None
        mix_delta = _maybe_float(g["mix_weights.delta"].iloc[0]) if "mix_weights.delta" in g.columns else None
        primary_expert = _maybe_str(g["primary_expert"].iloc[0]) if "primary_expert" in g.columns else None
        len_bucket = _maybe_str(g["context_features.length_bucket"].iloc[0]) if "context_features.length_bucket" in g.columns else None
        persona_style = _maybe_str(g["context_features.persona_style"].iloc[0]) if "context_features.persona_style" in g.columns else None
        multi_intent = _maybe_bool(g["context_features.multi_intent"].iloc[0]) if "context_features.multi_intent" in g.columns else None
        cold_user = _maybe_bool(g["context_features.cold_user"].iloc[0]) if "context_features.cold_user" in g.columns else None
        has_genre_terms = _maybe_bool(g["context_features.has_genre_terms"].iloc[0]) if "context_features.has_genre_terms" in g.columns else None
        has_negation = _maybe_bool(g["context_features.has_negation"].iloc[0]) if "context_features.has_negation" in g.columns else None
        has_year = _maybe_bool(g["context_features.has_year"].iloc[0]) if "context_features.has_year" in g.columns else None
        length_words = _maybe_float(g["context_features.length_words"].iloc[0]) if "context_features.length_words" in g.columns else None
        num_genre_terms = _maybe_float(g["context_features.num_genre_terms"].iloc[0]) if "context_features.num_genre_terms" in g.columns else None
        # Emotion vector: always infer from prompt text using lexicon-based inference
        # This is fast and avoids issues with plutchik_dist column format variations
        from src.emotions.emotion_prompt import infer_prompt_vector
        pemo_vec, emo_src = infer_prompt_vector(
            query=ptext,
            emo_model_dir=None,  # Use lexicon (fast, no model loading needed)
            prompt_emotion=None,
        )
        # infer_prompt_vector returns (np.ndarray, str), we just need the vector
        pemo = pemo_vec  # This is a numpy array of shape (8,)


        try:
            per_movie = per_prompt_movie_table(
                prompt_text=ptext,
                user_idx=uidx,
                dense_idx=dense_idx,
                bm25_idx=bm25_idx,
                lgcn_sim=lgcn_sim,
                emo_ids=emo_ids,
                emo_mat=emo_mat,
                agg_kind=args.agg_kind,
                topk=args.topk,
                encoder=args.encoder,
                model=args.model,
                q_emo_vec=pemo,  # _coerce_plutchik_vector in features.py will handle all types
                attn_tau=args.attn_tau,
            )
            logger.debug(f"Prompt {pid}: computed {len(per_movie)} movie scores")
        except Exception as e:
            logger.error(f"Failed to process prompt {pid}: {e}")
            logger.debug("Traceback:\n" + traceback.format_exc())
            continue

        # Build Δz rows for all pairs under this prompt
        for _, r in g.iterrows():
            try:
                a, b, y = int(r["movie1_id"]), int(r["movie2_id"]), int(r["y"])
                
                # Skip pairs where movies aren't in retrieval results
                if a not in per_movie.index or b not in per_movie.index:
                    logger.debug(f"Skipping pair {r['pair_id']}: movies not in retrieval results (movieA={a}, movieB={b})")
                    continue
                
                # Get scores for both movies
                za = per_movie.loc[a][["z_dense", "z_bm25", "z_lgcn", "z_emo"]].to_numpy(dtype=np.float32)
                zb = per_movie.loc[b][["z_dense", "z_bm25", "z_lgcn", "z_emo"]].to_numpy(dtype=np.float32)
                dz = (za - zb).astype("float32")
                
                # Skip pairs where all deltas are zero (no learning signal)
                if np.allclose(dz, 0, atol=1e-8):
                    logger.debug(f"Skipping pair {r['pair_id']}: all deltas are zero (movies have identical scores)")
                    continue

                rows.append({
                    "prompt_id": pid,
                    "pair_id": r["pair_id"],
                    "difficulty": r.get("difficulty", None),
                    "category": (r.get("category", None) or g["category"].iloc[0]) if "category" in g.columns else None,
                    "movieA": a,
                    "movieB": b,
                    "y": y,  # 1 => A preferred, 0 => B preferred
                    "dz_alpha": float(dz[0]),
                    "dz_beta":  float(dz[1]),
                    "dz_gamma": float(dz[2]),
                    "dz_delta": float(dz[3]),
                    "agg_kind": args.agg_kind,
                    "mix_alpha": mix_alpha,
                    "mix_beta": mix_beta,
                    "mix_gamma": mix_gamma,
                    "mix_delta": mix_delta,
                    "primary_expert": primary_expert,
                    "length_bucket": len_bucket,
                    "persona_style": persona_style,
                    "multi_intent": float(multi_intent) if multi_intent is not None else None,
                    "cold_user": float(cold_user) if cold_user is not None else None,
                    "has_genre_terms": float(has_genre_terms) if has_genre_terms is not None else None,
                    "has_negation": float(has_negation) if has_negation is not None else None,
                    "has_year": float(has_year) if has_year is not None else None,
                    "length_words": length_words,
                    "num_genre_terms": num_genre_terms,
                })
            except Exception as e:
                logger.error(f"Failed to build pair row for prompt {pid}: {e}")
                logger.debug("Traceback:\n" + traceback.format_exc())
                continue

    logger.info("=" * 80)
    logger.info(f"Feature building complete! Generated {len(rows)} pairs")

    out = pd.DataFrame(rows)
    
    # Log delta feature statistics
    dz_cols = ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta']
    if all(c in out.columns for c in dz_cols):
        logger.info("Delta feature statistics:")
        for col in dz_cols:
            nonzero = (out[col].abs() > 1e-8).sum()
            logger.info(f"  {col}: {nonzero}/{len(out)} non-zero ({100*nonzero/len(out):.1f}%)")
        
        all_zero_pairs = (out[dz_cols].abs() <= 1e-8).all(axis=1).sum()
        logger.info(f"  Pairs with ALL zeros: {all_zero_pairs}/{len(out)} ({100*all_zero_pairs/len(out):.1f}%)")
    
    out.to_parquet(args.out, index=False)

    logger.info(f"Saved features to: {args.out}")
    logger.info(f"Features shape: {out.shape}")
    logger.info(f"Columns: {list(out.columns)}")
    if "category" in out.columns:
        try:
            logger.info(f"Categories: {out['category'].value_counts().to_dict()}")
        except Exception:
            pass
    if "difficulty" in out.columns:
        try:
            logger.info(f"Difficulties: {out['difficulty'].value_counts().to_dict()}")
        except Exception:
            pass

    logger.info(f"Log file: {log_file}")
    print(f"\nWrote {len(out)} rows to {args.out}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
