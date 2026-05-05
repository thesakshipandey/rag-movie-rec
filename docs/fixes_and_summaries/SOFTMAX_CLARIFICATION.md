# Softmax Usage Clarification

## What Gets Softmaxed and When?

### Current Implementation

There are **TWO different softmax operations** in different places:

## 1. Expert Scores (Optional, Per-Prompt)

```python
# In generate_expert_scores.py

# Step 1: Get raw scores from each expert
alpha_raw = [0.82, 0.75, 0.68, ...]  # Cosine similarities
beta_raw = [25.3, 18.7, 30.2, ...]   # BM25 scores
gamma_raw = [0.005, 0.008, 0.003, ...] # CF similarities  
delta_raw = [0.85, 0.78, 0.92, ...]  # JS similarities

# Step 2: Z-score normalize (ALWAYS done)
z_alpha = (alpha_raw - mean) / std  # [-1.2, 0.3, 1.5, ...]
z_beta = (beta_raw - mean) / std    # [-0.8, 1.1, 0.2, ...]
z_gamma = (gamma_raw - mean) / std  # [-1.0, 0.5, 0.8, ...]
z_delta = (delta_raw - mean) / std  # [-0.5, 1.2, 0.3, ...]

# Step 3: Optional softmax (if --apply_softmax flag)
if args.apply_softmax:
    prob_alpha = softmax(z_alpha)  # [0.05, 0.15, 0.80, ...]
    prob_beta = softmax(z_beta)    # [0.10, 0.60, 0.30, ...]
    # ... etc
```

**Current Status**: `--apply_softmax` IS used in the pipeline script

**Purpose**: Convert z-scores to probabilities (interpretable as "how likely is this movie for this expert")

**Question**: Should we use this? Let's discuss below.

## 2. Router Weights (Always Softmaxed)

```python
# In contextual_hedge_router.py

# Router ALWAYS applies softmax to expert weights
context_features = encode(prompt_emotion, category, ...)
logits = mlp(context_features)  # [2.5, -0.8, 1.2, 0.3]
expert_weights = softmax(logits / temperature)  # [0.65, 0.02, 0.18, 0.15]
```

**Status**: ALWAYS done (not optional)

**Purpose**: Ensure expert weights sum to 1 and represent a probability distribution over experts

## The Question: Should We Softmax Expert Scores?

### Option A: Use Z-Scores (Current Default)

```python
# Per-prompt z-normalization
z_alpha = [-1.2, 0.3, 1.5, -0.5, ...]  # Can be negative
z_beta = [-0.8, 1.1, 0.2, 0.9, ...]

# Router combines
final = w_α * z_alpha + w_β * z_beta + w_γ * z_gamma + w_δ * z_delta
```

**Pros**:
- Preserves relative magnitudes
- Negative scores indicate "below average"
- More expressive for ranking

**Cons**:
- Not interpretable as probabilities
- Can have negative values

### Option B: Use Softmax Scores

```python
# Per-prompt softmax
prob_alpha = softmax(z_alpha)  # [0.05, 0.15, 0.80, ...]  # Sums to 1
prob_beta = softmax(z_beta)    # [0.10, 0.60, 0.30, ...]  # Sums to 1

# Router combines
final = w_α * prob_alpha + w_β * prob_beta + w_γ * prob_gamma + w_δ * prob_delta
```

**Pros**:
- Interpretable as probabilities
- All positive, bounded [0, 1]
- Each expert's scores sum to 1

**Cons**:
- Loses relative magnitude information
- Overly confident (always sums to 1 even for uncertain expert)
- May hurt ranking performance

## Recommendation

### For Your Use Case: **Don't Use Softmax on Expert Scores**

**Reasoning**:

1. **You're doing RANKING, not probability estimation**
   - Goal: Order movies correctly
   - Don't need probabilities, need relative scores

2. **Z-scores preserve discriminative power**
   - Movie with z=2.5 is much better than z=0.5
   - After softmax, this difference gets compressed

3. **Softmax per prompt loses cross-prompt comparability**
   - If Alpha gives all movies low scores → after softmax, still sums to 1
   - Router can't tell "this expert is uncertain"

4. **Router can learn reliability without probabilities**
   - Router learns: "trust Alpha when z_alpha values are high"
   - Doesn't need them to be probabilities

### What You Should Do

**Remove the `--apply_softmax` flag:**

```bash
# In run_listwise_pipeline.sh, change:
--apply_softmax

# To: (remove the flag)
# (just delete that line)
```

**Keep**:
- ✅ Z-score normalization (essential for combining experts)
- ✅ Softmax on router weights (essential for expert selection)
- ❌ Softmax on expert scores (not helpful for ranking)

## Updated Pipeline

```python
# 1. Expert scores (raw)
alpha_raw, beta_raw, gamma_raw, delta_raw = get_expert_scores(prompt, movies)

# 2. Z-normalize per expert per prompt (KEEP THIS)
z_alpha = zscore(alpha_raw)
z_beta = zscore(beta_raw)
z_gamma = zscore(gamma_raw)
z_delta = zscore(delta_raw)

# 3. NO softmax on scores (REMOVE THIS)
# prob_alpha = softmax(z_alpha)  ❌ Don't do this

# 4. Router predicts weights (KEEP THIS)
context = encode_context(prompt)
logits = router_mlp(context)
weights = softmax(logits / temperature)  # [w_α, w_β, w_γ, w_δ]

# 5. Weighted combination (KEEP THIS)
final_scores = w_α * z_alpha + w_β * z_beta + w_γ * z_gamma + w_δ * z_delta

# 6. Rank by final scores (KEEP THIS)
ranking = argsort(-final_scores)
```

## Summary

| Component | Softmax? | Why? |
|-----------|----------|------|
| **Expert raw scores** | ❌ No | Different scales, need z-norm |
| **Expert z-scores** | ❌ No | Preserve discriminative power for ranking |
| **Router logits** | ✅ Yes | Need probability distribution over experts |
| **Final combined scores** | ❌ No | Ranking only needs relative order |

## Action Required

1. **Re-run training without `--apply_softmax`**:
   ```bash
   # Edit run_listwise_pipeline.sh
   # Remove the --apply_softmax line from Step 1
   ```

2. **Or keep it if you want to experiment**:
   - Current implementation works either way
   - Router will learn to adapt
   - But z-scores (no softmax) is theoretically better for ranking

## Bottom Line

**Softmax on expert scores is optional but NOT recommended for ranking tasks.**

The router learns contextual reliability from the pattern of z-scores, which is more informative than probabilities that always sum to 1.

