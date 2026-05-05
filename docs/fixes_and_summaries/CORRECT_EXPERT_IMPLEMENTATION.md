# Correct Expert Implementation

## Expert Sources (As Specified)

### Alpha: Dense/Semantic Retrieval ✅
- **Model**: Qwen3-Embedding-8B (`/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B`)
- **Method**: Embed prompt + movie chunks, compute cosine similarity
- **Index**: FAISS at `indices/qwen_fullmovie/faiss.index`
- **Scores**: Cosine similarity between prompt embedding and chunk embeddings
- **Aggregation**: Sum or max over chunks per movie

### Beta: BM25 Lexical Retrieval ✅
- **Method**: BM25 keyword matching
- **Index**: BM25 at `indices/bm25/bm25.pkl`
- **Scores**: BM25 relevance scores for prompt keywords vs movie chunks
- **Aggregation**: Sum or max over chunks per movie

### Gamma: LightGCN Collaborative Filtering ✅
- **File**: `indices/lightgcn/sim_user_item.npy`
- **Format**: [num_users, num_items] similarity matrix
- **Method**: Use user-item similarity scores
- **User**: Default user_idx=0 (or from prompt metadata if available)
- **Scores**: Pre-computed user-item similarities from graph convolution

### Delta: Emotion-Based Matching ✅
- **Movie Emotions**: `indices/emotion/emotion.json`
  - Contains Plutchik 8-emotion distribution per movie
  - Format: `{movieId: int, plutchik_dist: {joy, trust, fear, ...}}`
- **Prompt Emotion Model**: RoBERTa at `models/roberta-plutchik-query_noKD/final`
  - Predicts 8 Plutchik emotions for prompt
- **Method**: Jensen-Shannon Divergence
  ```python
  M = 0.5 * (prompt_emo + movie_emo)
  JS = 0.5 * KL(prompt_emo || M) + 0.5 * KL(movie_emo || M)
  similarity = 1 - (JS / log(2))
  ```
- **Range**: [0, 1] where 1 = perfect emotional match

## Why Not Just Pick Highest-Scoring Expert?

### The Problem

```python
# WRONG approach:
best_expert = argmax([alpha_score, beta_score, gamma_score, delta_score])
final_score = expert_scores[best_expert]
```

### Why This Fails

1. **Scale Mismatch**: Raw scores are on different scales
   - Alpha: [0.2, 0.8] (cosine similarity)
   - Beta: [10, 50] (BM25 scores)
   - Gamma: [0.001, 0.01] (CF similarities)
   - Delta: [0.5, 0.95] (JS similarity)
   
   **Problem**: Beta will always "win" numerically but might be wrong!

2. **Confidence ≠ Correctness**
   - An expert giving high scores doesn't mean it's reliable
   - Example: BM25 might give high scores for keyword matches but miss semantic meaning
   - Example: CF might confidently recommend popular movies that don't match the query

3. **Context Matters**
   - Different queries need different experts:
     - "Emotional romantic drama" → Delta (emotion) is most reliable
     - "Movies like Inception with time travel" → Alpha (semantic) is most reliable
     - "What do people who liked Pulp Fiction enjoy?" → Gamma (CF) is most reliable
     - "Films about quantum physics" → Beta (keywords) is most reliable

4. **Complementary Information**
   - Experts look at different aspects
   - Best ranking often needs **multiple** experts combined
   - Example: For "heartwarming sci-fi adventure":
     - Delta provides emotional tone
     - Alpha provides genre semantics
     - Beta provides keyword matching
     - Optimal: 0.4*delta + 0.4*alpha + 0.1*beta + 0.1*gamma

## The Correct Approach: Contextual Learning

### Step 1: Z-Score Normalization
```python
z_alpha = (alpha_scores - mean(alpha_scores)) / std(alpha_scores)
z_beta = (beta_scores - mean(beta_scores)) / std(beta_scores)
z_gamma = (gamma_scores - mean(gamma_scores)) / std(gamma_scores)
z_delta = (delta_scores - mean(delta_scores)) / std(delta_scores)
```

**Result**: All experts now on same scale (mean=0, std=1)

### Step 2: Contextual Router
```python
# Extract context features from prompt
context = encode_context(prompt.emotion, prompt.category, prompt.features)

# Router predicts expert weights based on context
weights = softmax(router_mlp(context))  # [w_α, w_β, w_γ, w_δ]

# Weighted combination
final_scores = w_α * z_alpha + w_β * z_beta + w_γ * z_gamma + w_δ * z_delta
```

**The router learns**:
- Which expert(s) to trust for each query type
- Adaptive combination based on context
- Not just highest score, but most **reliable** score

### Step 3: ListMLE Training
```python
# Ground truth: ranked list [movie1, movie2, ...]
# Loss: How well does weighted combination match ground truth ranking?
loss = -log P(ground_truth_ranking | weighted_scores)
```

**The router optimizes for**:
- Best ranking according to ground truth
- Not necessarily highest individual expert score
- Contextual reliability, not raw confidence

## Example Scenarios

### Scenario 1: Semantic Query
```
Query: "Mind-bending sci-fi like Inception"

Raw Scores (unnormalized):
- Alpha (semantic): [0.85, 0.72, 0.68, ...]  ← High, reliable
- Beta (keywords): [25, 18, 30, ...]          ← High but noisy
- Gamma (CF): [0.005, 0.008, 0.003, ...]     ← Low
- Delta (emotion): [0.65, 0.58, 0.71, ...]   ← Medium

Router learns: w_α=0.7, w_β=0.2, w_γ=0.05, w_δ=0.05
Why? Semantic understanding is key, keywords help, CF/emotion less relevant
```

### Scenario 2: Emotional Query
```
Query: "Uplifting feel-good movie to cheer me up"

Raw Scores:
- Alpha (semantic): [0.62, 0.58, 0.71, ...]  ← Medium
- Beta (keywords): [8, 12, 5, ...]            ← Low
- Gamma (CF): [0.006, 0.007, 0.004, ...]     ← Low
- Delta (emotion): [0.92, 0.88, 0.95, ...]   ← High, reliable

Router learns: w_α=0.1, w_β=0.05, w_γ=0.1, w_δ=0.75
Why? Emotional tone is primary signal, semantic helps a bit
```

### Scenario 3: Popular Recommendation
```
Query: "Movies everyone loves"

Raw Scores:
- Alpha (semantic): [0.55, 0.61, 0.48, ...]  ← Low
- Beta (keywords): [10, 8, 12, ...]           ← Low
- Gamma (CF): [0.012, 0.015, 0.011, ...]     ← High, reliable
- Delta (emotion): [0.68, 0.72, 0.65, ...]   ← Medium

Router learns: w_α=0.1, w_β=0.05, w_γ=0.7, w_δ=0.15
Why? Collaborative filtering knows what's popular, emotion helps
```

## Key Insight

**Highest score ≠ Best expert for this query**

The router learns:
1. Which expert's **pattern of scores** (not just magnitude) is informative
2. Which expert is **reliably correct** for which query types
3. How to **combine** complementary signals

This is why we:
- Generate predictions from ALL experts (even if some score low)
- Z-normalize to make them comparable
- Train router to learn contextual reliability
- Use weighted combination, not winner-take-all

## Implementation Summary

✅ Alpha: Qwen3 embeddings + FAISS cosine similarity  
✅ Beta: BM25 keyword matching  
✅ Gamma: LightGCN user-item similarities from `sim_user_item.npy`  
✅ Delta: RoBERTa emotion + Jensen-Shannon divergence  
✅ Z-score normalization per expert per prompt  
✅ Router learns contextual weights via ListMLE loss  
✅ Final ranking = weighted combination of all experts  

The router will learn when to trust each expert, not just pick the highest scorer!

