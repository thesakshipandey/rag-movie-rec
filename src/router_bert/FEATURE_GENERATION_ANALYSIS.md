# Feature Generation Problem Analysis

## 🔍 Root Cause Identified

### Problem Summary
- **61% of pairs have ALL ZERO deltas** (5,497 out of 9,000)
- **dz_gamma (LightGCN) is ALL ZEROS** for every single pair

### Expert Mapping
Looking at `build_router_features.py` lines 195-212:

```python
za = per_movie.loc[a][["z_dense", "z_bm25", "z_lgcn", "z_emo"]].to_numpy()
```

The mapping is:
- **dz_alpha** = `z_dense` (Dense/Qwen embedding retrieval)
- **dz_beta** = `z_bm25` (BM25 lexical retrieval)  
- **dz_gamma** = `z_lgcn` (LightGCN collaborative filtering) ← **PROBLEM!**
- **dz_delta** = `z_emo` (Emotion-based retrieval)

## 🐛 Why dz_gamma is All Zeros

### The Bug in `features.py` Line 210:

```python
lgcn_movies = user_item_scores(lgcn_sim, user_idx) if (user_idx is not None and lgcn_sim is not None) else {}
```

**If `user_idx` is `None`:**
1. `lgcn_movies = {}` (empty dict)
2. All movies get `lgcn: 0.0` (line 221)
3. After z-scoring: `z_lgcn = 0.0` for ALL movies (line 225)
4. Result: `dz_gamma = z_lgcn(A) - z_lgcn(B) = 0 - 0 = 0` ✗

**Verification:**
- All 9,000 pairs have `cold_user = 0.0` (users have history)
- But `user_idx` was likely `None` during feature generation
- This caused LightGCN to be skipped entirely

## 🔍 Why 61% Have All-Zero Deltas

### Scenario 1: Movies Not in Retrieval Results

From `build_router_features.py` lines 194-198:

```python
# If a movie is missing, default to zeros
za = per_movie.loc[a][...] if a in per_movie.index else np.zeros(4, dtype=np.float32)
zb = per_movie.loc[b][...] if b in per_movie.index else np.zeros(4, dtype=np.float32)
```

**When this happens:**
- Movie not in top-K results from dense/BM25 retrieval
- Movie not in emotion index
- Movie not scored by LightGCN (because user_idx=None)
- **Result:** All four scores are 0

### Scenario 2: Both Movies Have Same Scores

Even if movies are scored, if they have identical scores:
- `dz = za - zb = [s, s, s, s] - [s, s, s, s] = [0, 0, 0, 0]`

## 📊 Data Analysis

```
Total pairs: 9,000
- Pairs with ANY non-zero delta: 3,503 (39%)
- Pairs with ALL ZERO deltas: 5,497 (61%)

Per-expert non-zero counts:
- dz_alpha (dense): 3,431 (38%)
- dz_beta (BM25):   3,436 (38%)
- dz_gamma (LGCN):      0 (0%)  ← BROKEN!
- dz_delta (emotion): 3,496 (39%)
```

## ✅ Solutions

### Option 1: Fix Feature Generation (Recommended)

**Fix 1: Handle Missing user_idx**

In `build_router_features.py`, ensure `user_idx` is set:

```python
# Line 109-111, change to:
if "user_idx" not in df.columns or df["user_idx"].isna().all():
    # Use a default user or derive from data
    df["user_idx"] = args.user_idx_default if args.user_idx_default is not None else 0
    logger.info(f"Added default user_idx: {df['user_idx'].iloc[0]}")
```

**Fix 2: Don't Default to Zeros for Missing Movies**

Instead of defaulting to zeros, **skip pairs where movies aren't scored**:

```python
# In build_router_features.py, replace lines 194-199:
if a not in per_movie.index or b not in per_movie.index:
    logger.debug(f"Skipping pair {r['pair_id']}: movies not in retrieval results")
    continue  # Skip this pair instead of using zeros

za = per_movie.loc[a][["z_dense", "z_bm25", "z_lgcn", "z_emo"]].to_numpy(dtype=np.float32)
zb = per_movie.loc[b][["z_dense", "z_bm25", "z_lgcn", "z_emo"]].to_numpy(dtype=np.float32)
dz = (za - zb).astype("float32")

# Also skip if all deltas are zero (movies have identical scores)
if np.allclose(dz, 0, atol=1e-8):
    logger.debug(f"Skipping pair {r['pair_id']}: all deltas are zero")
    continue
```

**Fix 3: Increase topk**

If movies aren't in retrieval results, increase `--topk`:

```bash
# Current: --topk 200
# Try: --topk 500 or --topk 1000
```

### Option 2: Use Only 3 Experts (Quick Fix)

Since LightGCN is broken, train with only 3 experts:

**Modify the router to use 3 experts instead of 4:**

1. Change `FourHeadRouter` to `ThreeHeadRouter`
2. Use only `[dz_alpha, dz_beta, dz_delta]`
3. Skip `dz_gamma` entirely

**Pros:**
- ✅ Works immediately with existing data
- ✅ No need to regenerate features
- ✅ Still have 3 diverse experts (dense, lexical, emotion)

**Cons:**
- ❌ Lose collaborative filtering signal
- ❌ Less powerful than 4 experts

### Option 3: Regenerate Features (Best Long-term)

**Step 1: Fix the code** (apply Fix 1 and Fix 2 above)

**Step 2: Regenerate features:**

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Find the original command used
# Likely something like:
python -m src.router.build_router_features \
    --prompts_dir <path_to_prompts> \
    --indices_dir artifacts/indices \
    --out artifacts/router/features_sum.parquet \
    --topk 500 \
    --user_idx_default 0 \
    --agg_kind sum
```

**Step 3: Add splits and balance** (if needed)

**Step 4: Retrain router** with new features

## 🎯 Recommended Approach

### Short-term (Immediate):
1. ✅ **Use current filtering** (already implemented in loader.py)
   - Filters out all-zero pairs
   - Trains on 3,503 informative pairs
   - Should work reasonably well

2. ⏳ **Train and evaluate** to see if 3 experts are sufficient

### Medium-term (If performance is poor):
1. ❌ **Switch to 3-expert model** explicitly
   - Modify architecture to use only 3 heads
   - Use `[dz_alpha, dz_beta, dz_delta]`
   - Document that LightGCN is excluded

### Long-term (For best performance):
1. ❌ **Fix feature generation** (apply all fixes above)
2. ❌ **Regenerate features** with proper user_idx
3. ❌ **Retrain with all 4 experts** working correctly

## 🔬 Diagnostic Commands

### Check if LightGCN works for a specific user:

```python
import numpy as np
from src.retrieval.lightgcn import user_item_scores

lgcn_sim = np.load('artifacts/indices/lightgcn/sim_user_item.npy')
print(f"LightGCN matrix shape: {lgcn_sim.shape}")

# Test with user 0
scores = user_item_scores(lgcn_sim, user_idx=0)
print(f"User 0 has {len(scores)} movie scores")
print(f"Sample scores: {list(scores.items())[:5]}")
```

### Check per_movie table for a sample prompt:

```python
from src.router.features import per_prompt_movie_table
from src.retrieval.search import load_index, load_bm25_index
from src.emotions.emotion_index import load_emotion_index
import numpy as np

# Load indices
dense_idx = load_index("artifacts/indices/qwen_fullmovie", metric="ip")
bm25_idx = load_bm25_index("artifacts/indices/bm25")
emo_ids, emo_mat = load_emotion_index("artifacts/indices/emotion")
lgcn_sim = np.load("artifacts/indices/lightgcn/sim_user_item.npy")

# Test with a prompt
prompt = "I want a thrilling action movie"
per_movie = per_prompt_movie_table(
    prompt_text=prompt,
    user_idx=0,  # ← Try with 0
    dense_idx=dense_idx,
    bm25_idx=bm25_idx,
    lgcn_sim=lgcn_sim,
    emo_ids=emo_ids,
    emo_mat=emo_mat,
)

print(f"Movies scored: {len(per_movie)}")
print(f"\nSample scores:")
print(per_movie[["dense", "bm25", "lgcn", "emo", "z_lgcn"]].head(10))
print(f"\nLGCN non-zero: {(per_movie['lgcn'] != 0).sum()}")
print(f"z_lgcn non-zero: {(per_movie['z_lgcn'].abs() > 1e-6).sum()}")
```

## 📝 Summary

**Problem:** LightGCN (gamma expert) wasn't used during feature generation, causing:
1. All `dz_gamma` values to be zero
2. 61% of pairs to have all-zero deltas (when movies weren't in retrieval results)

**Current Solution:** Filter out uninformative pairs (already implemented)

**Better Solution:** Fix feature generation and regenerate with proper user_idx

**Best Solution:** Regenerate features + skip pairs where movies aren't scored + increase topk

**Next Steps:**
1. Train with current filtered data to see if 3 experts work
2. If performance is poor, regenerate features with fixes
3. Consider explicit 3-expert model if LightGCN can't be fixed

