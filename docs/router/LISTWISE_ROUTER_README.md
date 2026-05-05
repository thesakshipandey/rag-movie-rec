# Listwise Router Training with Contextual Hedge

This document describes the listwise router training pipeline for expert selection in the movie recommendation system.

## Overview

The system trains a **contextual hedge router** that learns to combine 4 expert models:
- **Alpha (α)**: Dense/semantic retrieval (FAISS)
- **Beta (β)**: Lexical retrieval (BM25)
- **Gamma (γ)**: Collaborative filtering (LightGCN)
- **Delta (δ)**: Emotion-based ranking

The router uses contextual features (prompt metadata, emotion distributions, etc.) to output a softmax distribution over experts, and combines their z-scored predictions using **ListMLE** loss.

## Pipeline Steps

### 1. Generate Expert Scores

First, run all 4 experts on your listwise data to generate predictions:

```bash
python -m src.router.generate_expert_scores \
    --data_dir projects/Data \
    --indices_dir projects/rag-movie-rec/artifacts/indices \
    --out projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --encoder qwen \
    --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B \
    --topk_retrieval 1000 \
    --agg_kind sum \
    --apply_softmax
```

**Output**: `listwise_expert_scores.parquet` with columns:
- `prompt_id`, `movieId`, `rank`, `ground_truth_score`
- `score_alpha`, `score_beta`, `score_gamma`, `score_delta` (raw scores)
- `z_alpha`, `z_beta`, `z_gamma`, `z_delta` (z-normalized)
- `prob_alpha`, `prob_beta`, `prob_gamma`, `prob_delta` (softmax)

### 2. Train Router

Train the contextual hedge router with ListMLE loss:

```bash
python -m src.cli.train_router_listwise \
    --expert_scores projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path projects/Data/prompts.json \
    --out projects/rag-movie-rec/artifacts/router/router_listwise.pt \
    --loss listmle \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 32 \
    --d_context 128 \
    --d_hidden 256 \
    --dropout 0.2 \
    --temperature 1.0 \
    --entropy_weight 0.001 \
    --entropy_target 1.2 \
    --seed 42 \
    --device cuda
```

**Key hyperparameters**:
- `--loss`: Loss function (`listmle`, `listnet`, `approx_ndcg`)
- `--entropy_weight`: Regularization to prevent collapsing to single expert
- `--entropy_target`: Target entropy (1.2 ≈ log2(4) = 2 bits, slightly below uniform)
- Train/val/test split: 70/15/15 (random shuffle)

**Output**: `router_listwise.pt` checkpoint with model weights and training metadata

### 3. Evaluate Single Experts

Evaluate each expert independently as baselines:

```bash
python -m src.evaluations.eval_single_experts \
    --expert_scores projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path projects/Data/prompts.json \
    --out projects/rag-movie-rec/artifacts/evaluation_results/listwise/single_expert_metrics.json \
    --k 10 \
    --relevance_threshold 0.9 \
    --seed 42
```

**Metrics computed**:
- nDCG@10
- MRR (Mean Reciprocal Rank)
- Hit@10

**Baselines**:
- Single experts (α, β, γ, δ)
- Uniform mixture (0.25 each)
- Oracle mixture (ground truth mix_weights)
- Random baseline

**Output**: `single_expert_metrics.json`

### 4. Evaluate Router

Evaluate trained router on test set:

```bash
python -m src.evaluations.eval_router_listwise \
    --expert_scores projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path projects/Data/prompts.json \
    --router_checkpoint projects/rag-movie-rec/artifacts/router/router_listwise.pt \
    --out projects/rag-movie-rec/artifacts/evaluation_results/listwise/router_metrics.json \
    --k 10 \
    --relevance_threshold 0.9 \
    --device cuda
```

**Outputs**:
- `router_metrics.json`: Aggregate metrics and expert selection analysis
- `router_metrics_per_prompt.json`: Per-prompt detailed results

### 5. Compare All Methods

Generate comprehensive comparison tables and plots:

```bash
python -m src.evaluations.compare_methods \
    --single_expert_metrics projects/rag-movie-rec/artifacts/evaluation_results/listwise/single_expert_metrics.json \
    --router_metrics projects/rag-movie-rec/artifacts/evaluation_results/listwise/router_metrics.json \
    --out_dir projects/rag-movie-rec/artifacts/evaluation_results/listwise/comparison \
    --k 10
```

**Outputs**:
- `comparison_table.csv`: Comparison table
- `comparison_table.tex`: LaTeX table for papers
- `comparison_bar_chart.png`: Bar chart comparing all methods
- `expert_usage.png`: Average expert weights
- `expert_weight_distribution.png`: Weight statistics per expert
- `relative_improvement.png`: Router improvement over baselines
- `summary.json`: Summary statistics

## Architecture Details

### Contextual Hedge Router

The router consists of two components:

1. **Context Feature Encoder**:
   - Input: Prompt features (emotion, category, difficulty, etc.)
   - Architecture: Embeddings + MLP
   - Output: Context encoding (128D)

2. **Router MLP**:
   - Input: Context encoding
   - Architecture: 3-layer MLP with LayerNorm and Dropout
   - Output: Logits → Softmax → Expert weights [α, β, γ, δ]
   - Learnable temperature for calibration

### Loss Functions

#### ListMLE (Primary)
Plackett-Luce likelihood for listwise ranking:
```
Loss = -Σᵢ [s_πᵢ - log(Σⱼ≥ᵢ exp(s_πⱼ))]
```

#### ListNet (Alternative)
Cross-entropy on top-1 distributions:
```
P(i is top-1) = softmax(scores)
Loss = -Σᵢ P_gt(i) * log P_pred(i)
```

#### ApproxNDCG (Alternative)
Differentiable approximation of nDCG using soft ranking.

### Prediction

For a prompt with N movies:
1. Encode context → router weights [w_α, w_β, w_γ, w_δ]
2. Get expert z-scores: Z ∈ ℝ^(N×4)
3. Combine: s_final = Z @ w (weighted sum)
4. Rank movies by s_final

## Evaluation Metrics

### nDCG@K
Normalized Discounted Cumulative Gain at position K:
```
DCG@K = Σᵢ (2^relᵢ - 1) / log₂(i + 1)
nDCG@K = DCG@K / IDCG@K
```

### MRR
Mean Reciprocal Rank (of first relevant item):
```
MRR = 1 / rank_of_first_relevant
```

### Hit@K
Fraction of relevant items in top-K:
```
Hit@K = |relevant items in top-K| / |total relevant items|
```

## Data Format

### Input: `merged_all.json`
```json
{
  "0001": [
    {"movieId": 1127, "score": 1.0, "reason": "..."},
    {"movieId": 89, "score": 0.95, "reason": "..."},
    ...
  ],
  "0002": [...]
}
```

### Input: `prompts.json`
```json
[
  {
    "prompt_id": "uuid",
    "prompt_text": "...",
    "category": "mood",
    "plutchik_dist": {...},
    "mix_weights": {"alpha": 0.2, "beta": 0.3, "gamma": 0.1, "delta": 0.4},
    "context_features": {...}
  }
]
```

### Output: Expert Scores Parquet
| prompt_id | movieId | rank | ground_truth_score | z_alpha | z_beta | z_gamma | z_delta |
|-----------|---------|------|-------------------|---------|---------|---------|---------|
| 0001      | 1127    | 0    | 1.0               | 2.1     | 0.5     | -0.3    | 1.8     |
| 0001      | 89      | 1    | 0.95              | 1.9     | 0.3     | -0.1    | 1.6     |

## Ablation Studies

The pipeline supports several ablations:

1. **Loss function comparison**: `--loss listmle|listnet|approx_ndcg`
2. **Context features**: Modify encoder to include/exclude features
3. **Expert combinations**: Evaluate subsets of experts
4. **Temperature**: Adjust softmax temperature for calibration
5. **Entropy regularization**: Vary `--entropy_weight` and `--entropy_target`

## Tips & Best Practices

1. **Data splits**: Use 70/15/15 train/val/test with random shuffle (data is category-sorted)
2. **Hyperparameter tuning**: Start with defaults, tune learning rate and entropy weight
3. **Expert balance**: Monitor expert usage to ensure no single expert dominates
4. **Convergence**: Track validation loss and expert usage during training
5. **Evaluation**: Always evaluate on held-out test set, not validation set

## Troubleshooting

### Issue: Expert scores all zero
- Check that indices are properly loaded
- Verify embedder is working
- Ensure movie IDs match between datasets

### Issue: Router collapses to single expert
- Increase `--entropy_weight` (try 0.01)
- Lower `--entropy_target` (try 1.0)
- Check if one expert dominates on validation set

### Issue: Poor nDCG scores
- Verify ground truth scores are in correct format (1.0, 0.95, ...)
- Check z-score normalization is applied correctly
- Try different loss functions (ListNet may work better for some data)

### Issue: Out of memory
- Reduce `--batch_size`
- Reduce `--d_context` or `--d_hidden`
- Use CPU with `--device cpu` (slower but uses less memory)

## Citation

If you use this code, please cite:

```bibtex
@misc{listwise_router_2025,
  title={Listwise Router Training with Contextual Hedge for Movie Recommendation},
  author={Your Name},
  year={2025}
}
```

## License

[Your License Here]

