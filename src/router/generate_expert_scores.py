#!/usr/bin/env python
"""
Generate expert predictions for all prompt-movie pairs in the listwise dataset.

For each prompt in merged_all.json:
1. Run all 4 experts (alpha/dense, beta/BM25, gamma/LGCN, delta/emotion)
2. Get raw scores for each movie
3. Apply z-score normalization per expert per prompt
4. Apply softmax to get probability distributions
5. Save to parquet with ground truth scores

Usage:
    python -m src.router.generate_expert_scores \
        --data_dir projects/Data \
        --indices_dir projects/rag-movie-rec/artifacts/indices \
        --out artifacts/router/listwise_expert_scores.parquet \
        --encoder qwen \
        --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.search import load_index, load_bm25_index, search_dense_chunks, search_bm25_chunks, encode_query
from src.retrieval.lightgcn import user_item_scores, movie_scores_from_items
from src.emotions.emotion_index import load_emotion_index, score_movies_by_emotion


def zscore_normalize(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalization."""
    mean = scores.mean()
    std = scores.std()
    if std < eps:
        return np.zeros_like(scores)
    return (scores - mean) / std


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x_scaled = x / temperature
    x_max = x_scaled.max()
    exp_x = np.exp(x_scaled - x_max)
    return exp_x / exp_x.sum()


def load_prompts_and_rankings(data_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load prompts.json and merged_all.json.
    
    Returns:
        prompts_df: DataFrame with prompt metadata
        rankings_dict: {prompt_id -> list of {movieId, score, reason}}
    """
    prompts_path = Path(data_dir) / "prompts.json"
    rankings_path = Path(data_dir) / "merged_all.json"
    
    # Load prompts
    with open(prompts_path, 'r') as f:
        prompts_list = json.load(f)
    
    # Extract prompt_id mapping (assuming prompts have numeric IDs like "0001", "0002", ...)
    # The prompts.json uses UUID, but merged_all.json uses numeric IDs
    # We need to map between them
    
    # For now, assume merged_all.json keys are like "0001", "0002" and we need prompt metadata
    prompts_df = pd.DataFrame(prompts_list)
    
    # Load rankings
    with open(rankings_path, 'r') as f:
        rankings_dict = json.load(f)
    
    print(f"Loaded {len(prompts_df)} prompts and {len(rankings_dict)} rankings")
    
    return prompts_df, rankings_dict


def get_expert_scores_for_prompt(
    prompt_id: str,
    prompt_text: str,
    movie_ids: List[int],
    encoder: str,
    model_path: str,
    dense_index,
    bm25_index,
    lgcn_scores: Optional[np.ndarray],
    movie_emotions: Optional[Dict[int, np.ndarray]],
    emotion_tokenizer,
    emotion_model,
    user_idx: int = 0,
    topk_retrieval: int = 1000,
    agg_kind: str = "sum"
) -> Dict[str, np.ndarray]:
    """
    Get scores from all 4 experts for a specific prompt and list of movies.
    
    Returns:
        Dict with keys: 'alpha', 'beta', 'gamma', 'delta'
        Each value is np.ndarray of shape [len(movie_ids)]
    """
    movie_ids_set = set(movie_ids)
    
    # ===== ALPHA: Dense/Semantic Retrieval =====
    q_emb = encode_query(prompt_text, encoder=encoder, model=model_path)
    dense_hits = search_dense_chunks(dense_index, q_emb, top_k=topk_retrieval)
    
    # Aggregate chunk scores to movie scores
    alpha_scores_dict = {}
    for _, row in dense_hits.iterrows():
        movie_id = int(row['movieId'])
        chunk_score = float(row['score_dense'])
        if movie_id in movie_ids_set:
            if movie_id not in alpha_scores_dict:
                alpha_scores_dict[movie_id] = []
            alpha_scores_dict[movie_id].append(chunk_score)
    
    # Aggregate (sum or max)
    if agg_kind == "sum":
        alpha_scores_dict = {mid: sum(scores) for mid, scores in alpha_scores_dict.items()}
    else:  # max
        alpha_scores_dict = {mid: max(scores) for mid, scores in alpha_scores_dict.items()}
    
    alpha_scores = np.array([alpha_scores_dict.get(mid, 0.0) for mid in movie_ids], dtype=np.float32)
    
    # ===== BETA: BM25 Retrieval =====
    bm25_hits = search_bm25_chunks(bm25_index, prompt_text, top_k=topk_retrieval)
    
    beta_scores_dict = {}
    for _, row in bm25_hits.iterrows():
        movie_id = int(row['movieId'])
        chunk_score = float(row['score_bm25'])
        if movie_id in movie_ids_set:
            if movie_id not in beta_scores_dict:
                beta_scores_dict[movie_id] = []
            beta_scores_dict[movie_id].append(chunk_score)
    
    if agg_kind == "sum":
        beta_scores_dict = {mid: sum(scores) for mid, scores in beta_scores_dict.items()}
    else:
        beta_scores_dict = {mid: max(scores) for mid, scores in beta_scores_dict.items()}
    
    beta_scores = np.array([beta_scores_dict.get(mid, 0.0) for mid in movie_ids], dtype=np.float32)
    
    # ===== GAMMA: LightGCN (Collaborative Filtering) =====
    if lgcn_scores is not None:
        # lgcn_scores is [num_users, num_items] similarity matrix
        # Use user_idx (default 0, or from prompt metadata if available)
        if user_idx < lgcn_scores.shape[0]:
            user_item_sim = lgcn_scores[user_idx, :]  # [num_items]
            # Assuming item indices correspond to movieIds
            # You may need to load a mapping if they don't match
            gamma_scores = np.array([
                user_item_sim[mid] if mid < len(user_item_sim) else 0.0 
                for mid in movie_ids
            ], dtype=np.float32)
        else:
            gamma_scores = np.zeros(len(movie_ids), dtype=np.float32)
    else:
        gamma_scores = np.zeros(len(movie_ids), dtype=np.float32)
    
    # ===== DELTA: Emotion-based (Jensen-Shannon Divergence) =====
    if movie_emotions is not None and emotion_tokenizer is not None and emotion_model is not None:
        # Get prompt emotion distribution
        with torch.no_grad():
            inputs = emotion_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = emotion_model(**inputs)
            # Assuming model outputs logits for 8 Plutchik emotions
            prompt_emotion = torch.softmax(outputs.logits[0], dim=0).cpu().numpy()
        
        # Compute Jensen-Shannon divergence for each movie
        delta_scores = np.zeros(len(movie_ids), dtype=np.float32)
        for i, mid in enumerate(movie_ids):
            if mid in movie_emotions:
                movie_emo = movie_emotions[mid]
                # Jensen-Shannon divergence
                # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
                M = 0.5 * (prompt_emotion + movie_emo)
                js_div = 0.5 * np.sum(prompt_emotion * np.log((prompt_emotion + 1e-10) / (M + 1e-10))) + \
                         0.5 * np.sum(movie_emo * np.log((movie_emo + 1e-10) / (M + 1e-10)))
                # Convert divergence to similarity (lower divergence = higher similarity)
                # Use 1 - JS since JS is in [0, log(2)]
                delta_scores[i] = 1.0 - (js_div / np.log(2))
            else:
                delta_scores[i] = 0.0
    else:
        delta_scores = np.zeros(len(movie_ids), dtype=np.float32)
    
    return {
        'alpha': alpha_scores,
        'beta': beta_scores,
        'gamma': gamma_scores,
        'delta': delta_scores
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="projects/Data", help="Directory containing prompts.json and merged_all.json")
    ap.add_argument("--indices_dir", default="projects/rag-movie-rec/artifacts/indices", help="Indices directory")
    ap.add_argument("--out", default="projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet")
    ap.add_argument("--encoder", default="qwen", choices=["qwen", "gemma", "minilm"])
    ap.add_argument("--model", default="/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B")
    ap.add_argument("--emotion_model_path", default="/mnt/nas/sakshipandey/main/models/roberta-plutchik-query_noKD/final", 
                    help="Path to RoBERTa emotion model")
    ap.add_argument("--topk_retrieval", type=int, default=1000, help="Top-K chunks to retrieve")
    ap.add_argument("--agg_kind", default="sum", choices=["sum", "max"])
    ap.add_argument("--user_idx_default", type=int, default=0, help="Default user index for LGCN")
    ap.add_argument("--apply_softmax", action="store_true", help="Apply softmax to expert scores")
    args = ap.parse_args()
    
    print("=" * 80)
    print("Expert Score Generation")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading prompts and rankings...")
    prompts_df, rankings_dict = load_prompts_and_rankings(args.data_dir)
    
    # Load indices
    print("\n[2/6] Loading dense index...")
    # Dense index is in a subdirectory based on encoder
    dense_index_dir = os.path.join(args.indices_dir, "qwen_fullmovie" if args.encoder == "qwen" else "gemma")
    if not os.path.exists(os.path.join(dense_index_dir, "faiss.index")):
        print(f"  Warning: Dense index not found at {dense_index_dir}")
        print(f"  Available indices: {os.listdir(args.indices_dir)}")
        raise FileNotFoundError(f"FAISS index not found at {dense_index_dir}/faiss.index")
    dense_index = load_index(dense_index_dir, metric="ip")
    print(f"  Loaded from {dense_index_dir}")
    
    print("\n[3/6] Loading BM25 index...")
    bm25_index_dir = os.path.join(args.indices_dir, "bm25")
    if not os.path.exists(os.path.join(bm25_index_dir, "bm25.pkl")):
        print(f"  Warning: BM25 index not found at {bm25_index_dir}")
        raise FileNotFoundError(f"BM25 index not found at {bm25_index_dir}/bm25.pkl")
    bm25_index = load_bm25_index(bm25_index_dir)
    print(f"  Loaded from {bm25_index_dir}")
    
    print("\n[4/6] Loading LightGCN scores...")
    try:
        lgcn_dir = os.path.join(args.indices_dir, "lightgcn")
        lgcn_path = os.path.join(lgcn_dir, "sim_user_item.npy")
        
        if os.path.exists(lgcn_path):
            lgcn_scores = np.load(lgcn_path)
            print(f"  Loaded LightGCN: {lgcn_scores.shape} from {lgcn_path}")
            # Assuming rows=users, cols=items, we'll use user 0 or average
            # In actual use, you'd match user_idx from prompt metadata
            iid_to_movie = None  # Will use item indices directly as movie IDs
        else:
            print(f"  LightGCN not found at {lgcn_path}, will use zeros for gamma scores")
            lgcn_scores = None
            iid_to_movie = None
    except Exception as e:
        print(f"  Warning: Failed to load LightGCN: {e}")
        lgcn_scores = None
        iid_to_movie = None
    
    print("\n[5/6] Loading emotion data...")
    try:
        # Movie emotions
        emotion_json_path = os.path.join(args.indices_dir, "emotion", "emotion.json")
        if os.path.exists(emotion_json_path):
            with open(emotion_json_path, 'r') as f:
                emotion_data = json.load(f)
            # Convert to dict {movieId: emotion_vector}
            movie_emotions = {}
            for item in emotion_data:
                if 'movieId' in item and 'plutchik_dist' in item:
                    movie_emotions[int(item['movieId'])] = np.array([
                        item['plutchik_dist'].get('joy', 0),
                        item['plutchik_dist'].get('trust', 0),
                        item['plutchik_dist'].get('fear', 0),
                        item['plutchik_dist'].get('surprise', 0),
                        item['plutchik_dist'].get('sadness', 0),
                        item['plutchik_dist'].get('disgust', 0),
                        item['plutchik_dist'].get('anger', 0),
                        item['plutchik_dist'].get('anticipation', 0)
                    ], dtype=np.float32)
            print(f"  Loaded emotions for {len(movie_emotions)} movies from {emotion_json_path}")
        else:
            print(f"  Emotion data not found at {emotion_json_path}, will use zeros for delta scores")
            movie_emotions = None
        
        # Prompt emotion model
        emotion_model_path = args.emotion_model_path
        if emotion_model_path and os.path.exists(emotion_model_path):
            print(f"  Loading emotion model from {emotion_model_path}")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
            emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
            emotion_model.eval()
            if torch.cuda.is_available():
                emotion_model = emotion_model.cuda()
            print(f"  Emotion model loaded")
        else:
            print(f"  Emotion model not found at {emotion_model_path}, will use zeros for delta scores")
            emotion_tokenizer = None
            emotion_model = None
            
    except Exception as e:
        print(f"  Warning: Failed to load emotion data: {e}")
        print(f"  Will use zeros for delta scores")
        movie_emotions = None
        emotion_tokenizer = None
        emotion_model = None
    
    print("\n[6/6] Encoder ready...")
    print(f"  Using {args.encoder} with model {args.model}")
    
    # Process all prompts
    print("\n" + "=" * 80)
    print("Processing prompts...")
    print("=" * 80)
    
    all_rows = []
    
    for prompt_id, ranking_list in tqdm(rankings_dict.items(), desc="Generating expert scores"):
        # Get movie IDs and ground truth scores
        # Skip items that don't have movieId (data quality issue)
        valid_items = [item for item in ranking_list if 'movieId' in item and 'score' in item]
        if len(valid_items) == 0:
            print(f"\n  Warning: Prompt {prompt_id} has no valid items, skipping")
            continue
        if len(valid_items) < len(ranking_list):
            print(f"\n  Warning: Prompt {prompt_id} has {len(ranking_list) - len(valid_items)} items without movieId, skipping them")
        
        movie_ids = [item['movieId'] for item in valid_items]
        gt_scores = [item['score'] for item in valid_items]
        
        # Find corresponding prompt metadata
        # Since prompts.json uses UUID and merged_all.json uses numeric IDs,
        # we need to match by some other field or assume ordering
        # For now, we'll just use the prompt_id as is
        
        # Try to find prompt text from prompts_df
        # This is tricky - we need to establish the mapping
        # For simplicity, let's assume the merged_all.json was generated from the same prompts
        # and we can match by index or some identifier
        
        # TEMPORARY: Use a generic prompt text if we can't match
        # In production, you'd need proper ID mapping
        prompt_text = f"Recommend movies for query {prompt_id}"
        prompt_emotion = None
        
        # Try to match by some logic (this depends on your data structure)
        # For now, skip emotion and use placeholder
        
        # Get expert scores
        expert_scores = get_expert_scores_for_prompt(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            movie_ids=movie_ids,
            encoder=args.encoder,
            model_path=args.model,
            dense_index=dense_index,
            bm25_index=bm25_index,
            lgcn_scores=lgcn_scores,
            movie_emotions=movie_emotions,
            emotion_tokenizer=emotion_tokenizer,
            emotion_model=emotion_model,
            user_idx=args.user_idx_default,
            topk_retrieval=args.topk_retrieval,
            agg_kind=args.agg_kind
        )
        
        # Apply z-score normalization per expert
        z_alpha = zscore_normalize(expert_scores['alpha'])
        z_beta = zscore_normalize(expert_scores['beta'])
        z_gamma = zscore_normalize(expert_scores['gamma'])
        z_delta = zscore_normalize(expert_scores['delta'])
        
        # Optionally apply softmax
        if args.apply_softmax:
            prob_alpha = softmax(z_alpha)
            prob_beta = softmax(z_beta)
            prob_gamma = softmax(z_gamma)
            prob_delta = softmax(z_delta)
        else:
            prob_alpha = z_alpha
            prob_beta = z_beta
            prob_gamma = z_gamma
            prob_delta = z_delta
        
        # Create rows for each movie
        for i, movie_id in enumerate(movie_ids):
            row = {
                'prompt_id': prompt_id,
                'movieId': movie_id,
                'ground_truth_score': gt_scores[i],
                'rank': i,  # 0-indexed rank
                
                # Raw scores
                'score_alpha': expert_scores['alpha'][i],
                'score_beta': expert_scores['beta'][i],
                'score_gamma': expert_scores['gamma'][i],
                'score_delta': expert_scores['delta'][i],
                
                # Z-scores
                'z_alpha': z_alpha[i],
                'z_beta': z_beta[i],
                'z_gamma': z_gamma[i],
                'z_delta': z_delta[i],
                
                # Probabilities (softmax or same as z-scores)
                'prob_alpha': prob_alpha[i],
                'prob_beta': prob_beta[i],
                'prob_gamma': prob_gamma[i],
                'prob_delta': prob_delta[i],
            }
            all_rows.append(row)
    
    # Create DataFrame
    print("\n" + "=" * 80)
    print("Creating DataFrame...")
    df = pd.DataFrame(all_rows)
    
    print(f"\nGenerated {len(df)} rows for {len(rankings_dict)} prompts")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample rows:")
    print(df.head(20))
    
    # Save to parquet
    print(f"\nSaving to {args.out}...")
    df.to_parquet(args.out, index=False)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"\nOutput saved to: {args.out}")
    print(f"Total rows: {len(df)}")
    print(f"Unique prompts: {df['prompt_id'].nunique()}")
    print(f"Unique movies: {df['movieId'].nunique()}")


if __name__ == "__main__":
    main()

