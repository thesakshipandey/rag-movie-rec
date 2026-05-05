# Why Do We Generate Expert Predictions?

## Your Question

> "Why are you generating expert predictions already because of softmax?"

## The Answer

We generate expert predictions because **the router needs to learn which expert to trust for each query**. Here's the full pipeline:

## Pipeline Flow

```
Query → [4 Experts] → Raw Scores → Z-score → Router → Final Ranking
```

### Step-by-Step Explanation

#### 1. **Generate Expert Predictions** (Current Step)
For each prompt, we run **all 4 experts independently**:

| Expert | What It Does | Output |
|--------|--------------|---------|
| **Alpha** | Semantic search (FAISS) | Raw similarity scores for each movie |
| **Beta** | Keyword search (BM25) | Raw BM25 scores for each movie |
| **Gamma** | Collaborative filtering (LGCN) | Raw CF scores for each movie |
| **Delta** | Emotion matching | Raw JS divergence scores for each movie |

**Result**: Each expert gives different scores because they look at different aspects:
- Alpha likes movies semantically similar to the query
- Beta likes movies with matching keywords
- Gamma likes movies similar users liked
- Delta likes movies with matching emotional tone

#### 2. **Z-Score Normalization**
Each expert's raw scores are on different scales:
- Alpha: [0.2, 0.8, 0.1, ...]
- Beta: [10.5, 3.2, 15.8, ...]
- Gamma: [0.001, 0.005, 0.002, ...]

We **normalize** them per prompt so they're comparable:
```python
z_alpha = (scores_alpha - mean_alpha) / std_alpha
z_beta = (scores_beta - mean_beta) / std_beta
# Now all on same scale: [-2, 0, 1.5, ...]
```

#### 3. **Optional: Softmax**
The `--apply_softmax` flag converts z-scores to probabilities:
```python
prob_alpha = softmax(z_alpha)  # [0.1, 0.3, 0.6, ...]
```

This is OPTIONAL and mainly for interpretability. The router can work with either z-scores or probabilities.

#### 4. **Router Training** (Next Step)
The router learns **which expert to trust for each query type**:

```
Context features (emotion, category, etc.)
    ↓
  Router
    ↓
Expert weights: [w_α=0.7, w_β=0.1, w_γ=0.1, w_δ=0.1]
    ↓
Final score = 0.7 * z_alpha + 0.1 * z_beta + 0.1 * z_gamma + 0.1 * z_delta
```

**Example**:
- For query "emotional romantic drama": Router learns to trust Delta (emotion) more
- For query "sci-fi action thriller": Router learns to trust Alpha (semantic) more
- For query with user history: Router learns to trust Gamma (CF) more

## Why Not Just Use One Expert?

Because different queries need different experts:

| Query Type | Best Expert | Why |
|------------|-------------|-----|
| "Find movies like Inception" | Alpha (semantic) | Needs deep semantic understanding |
| "Movies about time travel" | Beta (keywords) | Simple keyword matching works |
| "Movies I'll like" | Gamma (CF) | User history matters |
| "Uplifting feel-good movie" | Delta (emotion) | Emotional tone is key |

## Why Not Just Combine All Equally?

A **uniform combination** (0.25 each) is a baseline, but suboptimal because:
- Some experts are irrelevant for some queries
- Some experts might be wrong for some queries
- Context determines which expert to trust

## The Router's Job

Learn a function:
```
f(query_context) → [w_α, w_β, w_γ, w_δ]
```

Such that the weighted combination:
```
score_final = Σ(w_k * expert_k_scores)
```

Produces the best ranking according to ListMLE loss.

## Why Pre-Generate Expert Scores?

1. **Efficiency**: Running 4 experts on 1000 prompts is expensive (~30 min)
   - We do it once and cache the results
   - Training can then iterate quickly on the cached scores

2. **Reproducibility**: Same expert scores for all experiments
   - Compare different router architectures
   - Try different loss functions
   - Tune hyperparameters

3. **Debugging**: Can inspect what each expert predicted
   - See which expert was right/wrong
   - Understand router's decisions

## Summary

**Without expert predictions**: No data for router to learn from  
**With expert predictions**: Router learns adaptive combination  

The router is learning **when to trust which expert**, not replacing the experts themselves!

Think of it like a smart traffic router that:
- Knows multiple routes (experts)
- Looks at current conditions (context)
- Chooses the best route mix for each trip (query)

## Your Specific Data

For your 1000 prompts:
- Each prompt has ~10 movies ranked by score
- We generate predictions from all 4 experts for these movies
- Router learns to weight experts to match the ground truth ranking
- **Goal**: Router's combined ranking matches the ground truth better than any single expert

Does this clarify why we need this step?

