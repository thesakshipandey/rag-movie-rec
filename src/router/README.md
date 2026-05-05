# Router Module - Mixture of Experts (MoE) for RAG Movie Recommender

## Overview

This module implements a **learned query-aware router** that dynamically assigns weights to 4 retrieval experts based on query characteristics. The router uses a **Bradley-Terry-Luce (BTL) pairwise loss** to learn from human preference judgments.

## Architecture

### Four Expert Retrievers

1. **α (Dense/FAISS)**: Semantic search via neural embeddings
   - Encoder: Qwen3-Embedding-8B or EmbeddingGemma-300M
   - Captures conceptual similarity and paraphrases

2. **β (BM25)**: Sparse lexical search
   - TF-IDF based keyword matching
   - Excellent for exact term matches and titles

3. **γ (LightGCN)**: Collaborative filtering
   - Graph convolutional networks on user-item interactions
   - Provides personalized user priors

4. **δ (Emotion)**: Affective matching
   - Plutchik-8 emotion distributions
   - Matches emotional tone of query and movies

### Router MLP

```python
class RouterMLP(nn.Module):
    """
    Input:  Δz ∈ R^{B×4} (per-expert score differences)
    Output: 
      - s ∈ R^{B} (fused margin logit favoring A over B)
      - w ∈ R^{B×4} (softmax weights over [α,β,γ,δ])
    """
```

**Architecture:**
- 3-layer MLP: `d_in=4 → d_hidden=64 → d_hidden=64 → d_out=4`
- Learnable temperature parameter for weight calibration
- Per-expert affine calibration: `dz_tilde = a * dz + b`
- Fused margin: `s = Σ_e w_e * (a_e * Δz_e + b_e)`

### Loss Function

**Bradley-Terry-Luce (BTL) Loss:**
```python
def btl_loss(s, y):
    """
    s: margin logits (A better if s > 0)
    y ∈ {0,1}: 1 means A preferred, 0 means B preferred
    
    Loss = y * softplus(-s) + (1-y) * softplus(s)
    """
```

**Entropy Regularization:**
```python
ent = -Σ w_i * log(w_i)
loss_total = btl_loss + λ * ReLU(ent - target_entropy)
```
- Encourages peaked distributions when appropriate
- Prevents over-diffuse weights

### 3-Layer Cascade with Dominance Gating

**Layer 1: Full Expert Set [α, β, γ, δ]**
- Compute z-scores for all experts
- Router assigns weights `w`
- Check dominance: `max(w) ≥ 0.75 AND entropy ≤ threshold AND gap ≥ margin`

**If δ (Emotion) dominates:**
- Filter to Top-K by `z_δ`
- Drop δ in subsequent layers (already filtered)

**If γ (LightGCN) dominates:**
- Filter to Top-K by `z_γ`
- Use [α, β] in next layer (γ served as prior)

**If α/β dominates:**
- Union of Top-K from α and β
- Use [α, β, γ] in next layer (drop δ)

**Layer 2 & 3:**
- Reroute on filtered candidates with reduced expert set
- Layer 3: final rerank, no further filtering

## File Structure

```
src/router/
├── __init__.py                  # Module exports
├── README.md                    # This file
├── logger_utils.py              # Logging utilities
├── mlp_router.py                # Router model & BTL loss
├── features.py                  # Feature extraction utilities
├── gating.py                    # Dominance gating logic
├── cascade.py                   # 3-layer cascade implementation
└── build_router_features.py     # CLI: build Δz features
```

## Data Format

### Input: Prompt Dataset

Located at: `/mnt/nas/sakshipandey/main/projects/rag-movie-rec/data/prompts/`

**prompts.json:**
```json
[
  {
    "prompt_id": 1,
    "prompt_text": "animated toys that come to life",
    "category": "plot_based",
    "user_idx": null
  }
]
```

**pairs.json:**
```json
[
  {
    "prompt_id": 1,
    "pair_id": 1,
    "movie1_id": 1,
    "movie2_id": 862,
    "difficulty": "easy"
  }
]
```

**judgments.json:**
```json
[
  {
    "prompt_id": 1,
    "pair_id": 1,
    "m1_gt_m2": true  // or "winner": "m1"
  }
]
```

### Output: Δz Features

**features.parquet:**
```
Columns:
- prompt_id: int
- pair_id: int
- category: str (plot_based, mood_based, etc.)
- difficulty: str (easy, medium, hard)
- movieA: int (movie1_id)
- movieB: int (movie2_id)
- dz_alpha: float (z_dense[A] - z_dense[B])
- dz_beta: float (z_bm25[A] - z_bm25[B])
- dz_gamma: float (z_lgcn[A] - z_lgcn[B])
- dz_delta: float (z_emo[A] - z_emo[B])
- y: int (1 if A preferred, 0 if B preferred)
- agg_kind: str (sum, max, or attn)
```

## Usage

### Step 1: Build Features

**With Sum Aggregation:**
```bash
python -m src.router.build_router_features \
  --prompts_dir /mnt/nas/sakshipandey/main/projects/rag-movie-rec/data/prompts \
  --indices_dir artifacts/indices \
  --out artifacts/router/features_sum.parquet \
  --agg_kind sum \
  --topk 200 \
  --encoder qwen \
  --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B \
  --logs_dir logs
```

**With Attention Aggregation:**
```bash
python -m src.router.build_router_features \
  --prompts_dir /mnt/nas/sakshipandey/main/projects/rag-movie-rec/data/prompts \
  --indices_dir artifacts/indices \
  --out artifacts/router/features_attn.parquet \
  --agg_kind attn \
  --topk 200 \
  --encoder qwen \
  --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B \
  --logs_dir logs
```

**Parameters:**
- `--prompts_dir`: Directory with prompts.json, pairs.json, judgments.json
- `--indices_dir`: Root directory with expert indices (qwen_fullmovie/, bm25/, emotion/, lightgcn/)
- `--out`: Output parquet file path
- `--agg_kind`: Chunk→movie aggregation (sum, max, or attn)
- `--topk`: Number of chunks to retrieve per expert (default: 200)
- `--encoder`: Dense encoder type (qwen, gemma, minilm)
- `--model`: Path to encoder model
- `--user_idx_default`: Default user index if not in prompts (optional)
- `--logs_dir`: Directory for log files (default: logs)

### Step 2: Train Router

```bash
python -m src.cli.train_router \
  --features artifacts/router/features_sum.parquet \
  --out artifacts/router/router_mlp_sum.pt \
  --epochs 10 \
  --lr 5e-4 \
  --ent_lambda 1e-3 \
  --ent_target 1.2 \
  --tie_tol 0.05 \
  --logs_dir logs
```

**Parameters:**
- `--features`: Input features parquet from Step 1
- `--out`: Output PyTorch weights file
- `--epochs`: Number of training epochs (default: 8)
- `--lr`: Learning rate (default: 5e-4)
- `--ent_lambda`: Entropy regularization weight (default: 1e-3)
- `--ent_target`: Target entropy in nats (default: 1.2)
- `--tie_tol`: Tie tolerance for agreement metrics (default: 0.05)
- `--logs_dir`: Directory for log files (default: logs)

**Training Output:**
```
Epoch [  1/ 10]
  Train: loss=0.6234 | acc(no-ties)=0.723 | acc(ties=0.5)=0.698 | +1= 3245 -1= 1243 0= 512
  Val:   loss=0.6187 | acc(no-ties)=0.731 | acc(ties=0.5)=0.705 | +1=  812 -1=  298 0= 140
  → New best validation loss: 0.6187
  → New best validation accuracy (no-ties): 0.731
```

### Step 3: Evaluate Router

```bash
python -m src.evaluations.router.eval_router \
  --features artifacts/router/features_sum.parquet \
  --weights artifacts/router/router_mlp_sum.pt \
  --tol 0.05 \
  --logs_dir logs
```

**Parameters:**
- `--features`: Features parquet (same as training)
- `--weights`: Trained model weights
- `--tol`: Tie tolerance (default: 0.05)
- `--logs_dir`: Directory for log files (default: logs)

**Evaluation Output:**
```
Overall (tol=0.05): +1=4057  -1=1541  0=652
Agreement (no ties): 0.7248
Agreement (ties=0.5): 0.7033

By difficulty (tol=0.05):
           +1   -1  0(ties)  agree_no_ties  agree_ties_0p5  count
easy      1523  312      121         0.8299          0.7812   1956
medium    1834  721      298         0.7180          0.6734   2853
hard       700  508      233         0.5794          0.5391   1441

By category (tol=0.05):
              +1   -1  0(ties)  agree_no_ties  agree_ties_0p5  count
plot_based   2134  723      312         0.7471          0.7103   3169
mood_based    912  421      189         0.6842          0.6532   1522
title_based   501  201       81         0.7137          0.6852    783
multi_genre   510  196       70         0.7224          0.6954    776
```

## Logging

All scripts automatically log to `/mnt/nas/sakshipandey/main/projects/rag-movie-rec/logs/` with timestamped files:

- `build_features_YYYYMMDD_HHMMSS.log`: Feature building logs
- `train_router_YYYYMMDD_HHMMSS.log`: Training logs (epoch metrics, best model tracking)
- `eval_router_YYYYMMDD_HHMMSS.log`: Evaluation logs (overall and sliced metrics)

**Log Format:**
```
2025-10-24 14:32:15 | INFO     | train_router | Starting MLP Router Training
2025-10-24 14:32:15 | INFO     | train_router | Features file: artifacts/router/features_sum.parquet
2025-10-24 14:32:16 | INFO     | train_router | Loaded 5000 pairs
2025-10-24 14:32:16 | INFO     | train_router | Categories: {'plot_based': 3169, 'mood_based': 1522, ...}
```

## Aggregators

### Sum Aggregator
```python
class SumAggregator:
    def __init__(self, mode: str = "sum"):  # "sum" or "max"
```
- **sum**: `score_movie = Σ score_chunk` for all chunks of that movie
- **max**: `score_movie = max(score_chunk)` for all chunks of that movie

### Attention Aggregator
```python
class AttentionAggregator(nn.Module):
    def __init__(self, temperature: float = 1.0):
```
- Softmax attention over chunk scores: `w_i = softmax(score_i / τ)`
- Weighted sum: `score_movie = Σ w_i * score_i`
- Learnable log-temperature parameter

**Use Cases:**
- **sum**: Favors movies with multiple relevant chunks (comprehensive match)
- **max**: Favors movies with at least one highly relevant chunk (precision)
- **attn**: Soft weighting (middle ground, differentiable)

## Experimental Workflow

### A/B Testing Aggregators

1. Build features with both aggregators:
```bash
# Sum
python -m src.router.build_router_features \
  --agg_kind sum --out artifacts/router/features_sum.parquet

# Attention
python -m src.router.build_router_features \
  --agg_kind attn --out artifacts/router/features_attn.parquet
```

2. Train routers independently:
```bash
# Sum router
python -m src.cli.train_router \
  --features artifacts/router/features_sum.parquet \
  --out artifacts/router/router_mlp_sum.pt

# Attention router
python -m src.cli.train_router \
  --features artifacts/router/features_attn.parquet \
  --out artifacts/router/router_mlp_attn.pt
```

3. Evaluate and compare:
```bash
python -m src.evaluations.router.eval_router \
  --features artifacts/router/features_sum.parquet \
  --weights artifacts/router/router_mlp_sum.pt

python -m src.evaluations.router.eval_router \
  --features artifacts/router/features_attn.parquet \
  --weights artifacts/router/router_mlp_attn.pt
```

### Hyperparameter Tuning

**Entropy Regularization:**
- Low `ent_lambda` (1e-4): Allows more diffuse weights
- High `ent_lambda` (1e-2): Forces peaked distributions

**Target Entropy:**
- Low `ent_target` (0.5): Very peaked (1-2 dominant experts)
- High `ent_target` (2.0): More uniform (all experts contribute)

**Learning Rate:**
- Start with `5e-4` for stable convergence
- Reduce to `1e-4` for fine-tuning

**Example Grid Search:**
```bash
for ent_lambda in 1e-4 5e-4 1e-3 5e-3; do
  for ent_target in 0.8 1.2 1.6; do
    python -m src.cli.train_router \
      --features artifacts/router/features_sum.parquet \
      --out artifacts/router/router_ent${ent_lambda}_tgt${ent_target}.pt \
      --ent_lambda $ent_lambda \
      --ent_target $ent_target
  done
done
```

## Integration with Retrieval Pipeline

### Inference Example

```python
import torch
import pandas as pd
from src.router.mlp_router import RouterMLP
from src.router.features import per_prompt_movie_table

# Load trained router
router = RouterMLP()
router.load_state_dict(torch.load("artifacts/router/router_mlp_sum.pt"))
router.eval()

# Get per-expert scores for a query
scores_df = per_prompt_movie_table(
    prompt_text="animated toys that come to life",
    user_idx=42,
    dense_idx=...,
    bm25_idx=...,
    lgcn_sim=...,
    emo_ids=...,
    emo_mat=...,
    agg_kind="sum",
    topk=200
)

# For each candidate pair, compute Δz and get router weights
# (In production, you'd rank all candidates directly)
dz = torch.tensor([
    [scores_df.loc[1]["z_dense"] - scores_df.loc[2]["z_dense"],
     scores_df.loc[1]["z_bm25"] - scores_df.loc[2]["z_bm25"],
     scores_df.loc[1]["z_lgcn"] - scores_df.loc[2]["z_lgcn"],
     scores_df.loc[1]["z_emo"] - scores_df.loc[2]["z_emo"]]
], dtype=torch.float32)

with torch.no_grad():
    margin, weights = router(dz)
    
print(f"Weights: α={weights[0,0]:.3f} β={weights[0,1]:.3f} γ={weights[0,2]:.3f} δ={weights[0,3]:.3f}")
# Output: Weights: α=0.123 β=0.587 γ=0.245 δ=0.045
```

### Cascade Routing

```python
from src.router.cascade import cascade_route

# Define scoring function
def get_scores(actives, pool):
    # Return {movieId: {"z_alpha":..., "z_beta":..., ...}}
    ...

# Define router function
def router_fn(dz_summary):
    # Return weights [4]
    with torch.no_grad():
        _, w = router(dz_summary.unsqueeze(0))
    return w.squeeze(0)

# Run cascade
final_scores, route_log = cascade_route(
    get_scores_fn=get_scores,
    router_fn=router_fn,
    prompt_text="animated toys",
    user_idx=42,
    max_layers=3,
    tau=0.75,        # Dominance threshold
    entropy_bits=1.3,
    margin=0.2,
    K_delta=200,
    K_gamma=200,
    K_alphabeta=250
)

# route_log contains:
# [
#   {"layer": 1, "weights": [0.1, 0.6, 0.3, 0.0], "gated": True, "expert": "beta"},
#   {"layer": 2, "weights_active": {"alpha": 0.25, "beta": 0.75}, ...},
#   {"layer": 3, "note": "final rerank"}
# ]
```

## Performance Benchmarks

**Feature Building (1000 prompts, 5000 pairs):**
- Sum aggregator: ~15 minutes (GPU)
- Attention aggregator: ~18 minutes (GPU)

**Training (5000 pairs, 10 epochs):**
- Single epoch: ~5 seconds (CPU)
- Total training: ~50 seconds

**Inference (single query):**
- Router forward pass: <1ms
- Full cascade (3 layers): ~150ms (including retrieval)

## Troubleshooting

### Common Errors

**1. `ImportError: cannot import name 'infer_prompt_emotion'`**
- **Fix:** Updated to `infer_prompt_vector` in latest version

**2. `TypeError: encode_query() missing required argument: 'encoder'`**
- **Fix:** Pass `encoder` and `model` parameters to `per_prompt_movie_table`

**3. `KeyError: 'z_lgcn'` during feature building**
- **Cause:** LightGCN matrix not found or user_idx out of range
- **Fix:** Set `--user_idx_default None` if no user data available

**4. `RuntimeError: CUDA out of memory`**
- **Fix:** Reduce `--topk` or `--batch_size` in embedding step

### Debug Tips

1. **Enable debug logging:**
```python
logger.setLevel(logging.DEBUG)
```

2. **Inspect features:**
```python
df = pd.read_parquet("artifacts/router/features_sum.parquet")
print(df.describe())
print(df[["dz_alpha", "dz_beta", "dz_gamma", "dz_delta"]].describe())
```

3. **Check for NaN/Inf:**
```python
assert not df[["dz_alpha", "dz_beta", "dz_gamma", "dz_delta"]].isna().any().any()
assert not df[["dz_alpha", "dz_beta", "dz_gamma", "dz_delta"]].isin([float('inf'), float('-inf')]).any().any()
```

## References

1. **Bradley-Terry Models:**
   - Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs.

2. **Mixture of Experts:**
   - Jacobs, R. A., et al. (1991). Adaptive mixtures of local experts.

3. **LightGCN:**
   - He, X., et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.

4. **Attention Aggregation:**
   - Vaswani, A., et al. (2017). Attention is all you need.

## License

Part of the RAG Movie Recommender project. See main README for license details.

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the maintainers.

---

**Last Updated:** October 24, 2025  
**Version:** 1.0.0  
**Author:** RAG Movie Rec Team

