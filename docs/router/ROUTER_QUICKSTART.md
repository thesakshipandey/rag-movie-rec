# Router Quick Start Guide

This guide will help you train and evaluate the MoE (Mixture of Experts) router for the RAG Movie Recommender system.

## Prerequisites

1. **Indices must be built:**
   - Dense (FAISS): `artifacts/indices/qwen_fullmovie/`
   - BM25: `artifacts/indices/bm25/`
   - Emotion: `artifacts/indices/emotion/`
   - LightGCN (optional): `artifacts/indices/lightgcn/sim_user_item.npy`

2. **Prompt dataset must be available:**
   - Location: `/mnt/nas/sakshipandey/main/projects/rag-movie-rec/data/prompts/`
   - Required files: `prompts.json`, `pairs.json`, `judgments.json`

## 3-Step Workflow

### Step 1: Build Router Features

This step extracts Δz features (score differences between movie pairs) from all four experts.

**Option A: Sum Aggregation (Recommended)**
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

python -m src.router.build_router_features \
  --prompts_dir data/prompts \
  --indices_dir artifacts/indices \
  --out artifacts/router/features_sum.parquet \
  --agg_kind sum \
  --topk 200 \
  --encoder qwen \
  --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B \
  --logs_dir logs
```

**Option B: Attention Aggregation (Experimental)**
```bash
python -m src.router.build_router_features \
  --prompts_dir data/prompts \
  --indices_dir artifacts/indices \
  --out artifacts/router/features_attn.parquet \
  --agg_kind attn \
  --topk 200 \
  --encoder qwen \
  --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B \
  --logs_dir logs
```

**Expected Output:**
```
Loading prompt triplets from data/prompts
Loaded 5000 pairs from 1000 prompts
Loading indices...
  Dense index loaded: 1682 entries
  BM25 index loaded: 1682 entries
  Emotion index loaded: 1682 movies
  LightGCN matrix loaded: shape (943, 1682)
Building features for each prompt...
prompts: 100%|████████████| 1000/1000 [12:34<00:00, 1.33it/s]
Feature building complete! Generated 5000 pairs

Wrote 5000 rows to artifacts/router/features_sum.parquet
Log file: logs/build_features_20251024_143215.log
```

**Time Estimate:** 10-20 minutes (depends on dataset size and GPU)

---

### Step 2: Train Router

Train the MLP router using Bradley-Terry-Luce (BTL) pairwise loss.

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

**Expected Output:**
```
Starting MLP Router Training
Loaded 5000 pairs
Train batches: 16, Val batches: 4
Router MLP
  Total parameters:      8,708
  Trainable parameters:  8,708
Starting training loop
================================================================================
Epoch [  1/ 10]
  Train: loss=0.6234 | acc(no-ties)=0.723 | acc(ties=0.5)=0.698 | +1= 3245 -1= 1243 0= 512
  Val:   loss=0.6187 | acc(no-ties)=0.731 | acc(ties=0.5)=0.705 | +1=  812 -1=  298 0= 140
  → New best validation loss: 0.6187
  → New best validation accuracy (no-ties): 0.731
Epoch [  2/ 10]
  Train: loss=0.5987 | acc(no-ties)=0.745 | acc(ties=0.5)=0.721 | +1= 3387 -1= 1167 0= 446
  Val:   loss=0.5912 | acc(no-ties)=0.752 | acc(ties=0.5)=0.728 | +1=  829 -1=  273 0= 148
  → New best validation loss: 0.5912
  → New best validation accuracy (no-ties): 0.752
...
================================================================================
Training complete!
Best validation loss: 0.5512
Best validation accuracy (no-ties): 0.784
================================================================================

Training complete! Model saved to: artifacts/router/router_mlp_sum.pt
Log file: logs/train_router_20251024_143830.log
```

**Time Estimate:** 1-2 minutes on CPU, <30 seconds on GPU

---

### Step 3: Evaluate Router

Evaluate the trained router on the full dataset (or held-out test set).

```bash
python -m src.evaluations.router.eval_router \
  --features artifacts/router/features_sum.parquet \
  --weights artifacts/router/router_mlp_sum.pt \
  --tol 0.05 \
  --logs_dir logs
```

**Expected Output:**
```
Starting Router Evaluation
Loaded 5000 pairs
Model loaded successfully
Running inference...
================================================================================
Overall Results (tol=0.05):
  Correct (+1):       4057
  Incorrect (-1):     1541
  Ties (0):            652
  Total:              6250
  Agreement (no ties):    0.7248
  Agreement (ties=0.5):   0.7033
================================================================================

Overall (tol=0.05): +1=4057  -1=1541  0(ties)=652
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

Log file: logs/eval_router_20251024_144523.log
```

**Time Estimate:** <10 seconds

---

## Advanced: A/B Testing Aggregators

Compare Sum vs Attention aggregation:

```bash
# Build features for both
python -m src.router.build_router_features \
  --agg_kind sum --out artifacts/router/features_sum.parquet --logs_dir logs

python -m src.router.build_router_features \
  --agg_kind attn --out artifacts/router/features_attn.parquet --logs_dir logs

# Train routers independently
python -m src.cli.train_router \
  --features artifacts/router/features_sum.parquet \
  --out artifacts/router/router_mlp_sum.pt --logs_dir logs

python -m src.cli.train_router \
  --features artifacts/router/features_attn.parquet \
  --out artifacts/router/router_mlp_attn.pt --logs_dir logs

# Evaluate both
python -m src.evaluations.router.eval_router \
  --features artifacts/router/features_sum.parquet \
  --weights artifacts/router/router_mlp_sum.pt --logs_dir logs

python -m src.evaluations.router.eval_router \
  --features artifacts/router/features_attn.parquet \
  --weights artifacts/router/router_mlp_attn.pt --logs_dir logs
```

Compare validation accuracies and choose the better aggregator.

---

## Hyperparameter Tuning

### Grid Search Example

```bash
#!/bin/bash
# grid_search.sh

ENT_LAMBDAS=(1e-4 5e-4 1e-3 5e-3)
ENT_TARGETS=(0.8 1.2 1.6 2.0)

for ent_lambda in "${ENT_LAMBDAS[@]}"; do
  for ent_target in "${ENT_TARGETS[@]}"; do
    echo "Training with ent_lambda=$ent_lambda ent_target=$ent_target"
    
    python -m src.cli.train_router \
      --features artifacts/router/features_sum.parquet \
      --out artifacts/router/router_ent${ent_lambda}_tgt${ent_target}.pt \
      --epochs 10 \
      --lr 5e-4 \
      --ent_lambda $ent_lambda \
      --ent_target $ent_target \
      --logs_dir logs
    
    python -m src.evaluations.router.eval_router \
      --features artifacts/router/features_sum.parquet \
      --weights artifacts/router/router_ent${ent_lambda}_tgt${ent_target}.pt \
      --logs_dir logs > results/eval_ent${ent_lambda}_tgt${ent_target}.txt
  done
done

# Find best configuration
echo "Best configurations by validation accuracy:"
grep "Agreement (no ties)" results/*.txt | sort -k4 -nr | head -5
```

**Run:**
```bash
chmod +x grid_search.sh
./grid_search.sh
```

---

## Integration with Inference Pipeline

### Python API Example

```python
import torch
import pandas as pd
from src.router.mlp_router import RouterMLP
from src.retrieval.search import load_index, load_bm25_index
from src.retrieval.lightgcn import load_cosine_matrix
from src.emotions.emotion_index import load_emotion_index
from src.router.features import per_prompt_movie_table

# Load indices (once at startup)
dense_idx = load_index("artifacts/indices/qwen_fullmovie", metric="ip")
bm25_idx = load_bm25_index("artifacts/indices/bm25")
lgcn_sim = load_cosine_matrix("artifacts/indices/lightgcn/sim_user_item.npy")
emo_ids, emo_mat = load_emotion_index("artifacts/indices/emotion")

# Load router
router = RouterMLP()
router.load_state_dict(torch.load("artifacts/router/router_mlp_sum.pt"))
router.eval()

# Inference function
def get_recommendations(query: str, user_idx: int, top_k: int = 10):
    # Get per-movie z-scores from all experts
    scores_df = per_prompt_movie_table(
        prompt_text=query,
        user_idx=user_idx,
        dense_idx=dense_idx,
        bm25_idx=bm25_idx,
        lgcn_sim=lgcn_sim,
        emo_ids=emo_ids,
        emo_mat=emo_mat,
        agg_kind="sum",
        topk=200,
        encoder="qwen",
        model="/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B"
    )
    
    # Compute router weights (from distribution summary)
    z_vals = scores_df[["z_dense", "z_bm25", "z_lgcn", "z_emo"]].values
    dz_summary = torch.tensor([
        z_vals[:, 0].max() - z_vals[:, 0].mean(),  # alpha
        z_vals[:, 1].max() - z_vals[:, 1].mean(),  # beta
        z_vals[:, 2].max() - z_vals[:, 2].mean(),  # gamma
        z_vals[:, 3].max() - z_vals[:, 3].mean(),  # delta
    ], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        _, weights = router(dz_summary)
    
    w = weights.squeeze(0).numpy()  # [4]
    
    # Fuse scores with router weights
    scores_df["score_final"] = (
        w[0] * scores_df["z_dense"] +
        w[1] * scores_df["z_bm25"] +
        w[2] * scores_df["z_lgcn"] +
        w[3] * scores_df["z_emo"]
    )
    
    # Rank and return top-K
    top_movies = scores_df.nlargest(top_k, "score_final")
    
    return {
        "weights": {"alpha": float(w[0]), "beta": float(w[1]), 
                   "gamma": float(w[2]), "delta": float(w[3])},
        "recommendations": top_movies[["movieId", "score_final"]].to_dict("records")
    }

# Example usage
result = get_recommendations("animated toys that come to life", user_idx=42, top_k=10)
print(f"Router weights: {result['weights']}")
print(f"Top recommendations: {result['recommendations']}")
```

---

## Monitoring & Logs

All operations write detailed logs to `logs/` directory:

```bash
# View latest training log
tail -f logs/train_router_*.log | grep -E "Epoch|loss|acc"

# View all router logs
ls -lht logs/*router*.log | head -5

# Extract validation accuracies
grep "Val:.*acc(no-ties)" logs/train_router_*.log
```

**Log Contents:**
- Configuration parameters
- Data loading statistics
- Model architecture summary
- Per-epoch training/validation metrics
- Best model checkpoints
- Final evaluation results

---

## Troubleshooting

### Error: `KeyError: 'z_lgcn'`
**Cause:** LightGCN matrix not available or user_idx out of range  
**Fix:** Set `--user_idx_default None` in feature building

### Error: `CUDA out of memory`
**Cause:** Too many chunks retrieved or large batch size  
**Fix:** Reduce `--topk` from 200 to 100

### Error: `ImportError: cannot import name 'infer_prompt_emotion'`
**Cause:** Outdated code version  
**Fix:** Use `infer_prompt_vector` instead (already updated)

### Warning: `Mean n_words=45 looks short`
**Cause:** Indexed wrong chunk file (overview-only instead of full-movie)  
**Fix:** Rebuild dense index with full movie chunks

---

## Performance Benchmarks

**System:** NVIDIA A100 40GB, AMD EPYC 7742, 256GB RAM

| Task                    | Time     | Notes                      |
|-------------------------|----------|----------------------------|
| Build features (1K prompts) | 12 min | Qwen-8B encoder on GPU |
| Train router (10 epochs)    | 45 sec | CPU sufficient         |
| Evaluate router             | 8 sec  | Full dataset           |
| Single query inference      | 150 ms | Including retrieval    |

**Memory Usage:**
- Feature building: ~8GB GPU VRAM
- Training: ~500MB RAM
- Inference: ~100MB RAM

---

## Next Steps

1. **Integrate with Web App:**
   - Modify `src/app/bm25_lgcn_app.py` to use router weights
   - Add "Auto (Router)" preset in UI

2. **Online Learning:**
   - Collect user click data
   - Fine-tune router with online feedback

3. **Multi-Stage Cascade:**
   - Implement 3-layer cascade with dominance gating
   - See `src/router/cascade.py` for implementation

4. **Cross-Domain Transfer:**
   - Train router on MovieLens, apply to other datasets
   - Evaluate zero-shot performance

---

## References

- **Router Module:** `src/router/README.md`
- **Full Workflow:** `data/workflow/workflow.md`
- **Main README:** `README.md`

For detailed documentation, see `src/router/README.md`.

---

**Last Updated:** October 24, 2025  
**Author:** RAG Movie Rec Team

