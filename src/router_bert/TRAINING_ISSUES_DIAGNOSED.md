# Training Issues - Diagnosis and Fix

## 🔴 Problem Identified

Your training showed **NO LEARNING** across 10 epochs:
- Loss stuck at **0.6990** (log(2) ≈ 0.693)
- Validation agreement at **0.2007** (terrible, random would be ~0.50)
- **Zero improvement** across all epochs

## 🔍 Root Cause Analysis

### Issue 1: Majority of Pairs Have No Learning Signal

**Data Analysis:**
```
Total pairs: 9,000
Pairs with ALL ZERO deltas: 5,497 (61%)
Pairs with ANY non-zero delta: 3,503 (39%)
```

**Why this breaks training:**
- When all deltas are zero: `score = w · dz = 0` for ANY weights
- The model can't distinguish between positive and negative examples
- Loss becomes constant regardless of what the model learns
- These pairs contribute only noise to training

### Issue 2: Gamma Expert is Completely Unused

**Delta feature statistics:**
```
dz_alpha: 3,431 non-zero values (38%)
dz_beta:  3,436 non-zero values (38%)
dz_gamma:     0 non-zero values (0%)  ← PROBLEM!
dz_delta: 3,496 non-zero values (39%)
```

**Impact:**
- We're training a 4-expert router but only 3 experts have signal
- Gamma weight is meaningless (always multiplied by zero)
- Model effectively has only 3 degrees of freedom

### Issue 3: Why Agreement is 0.20 Instead of 0.50

For the 61% of pairs with all-zero deltas:
- Score is always 0 regardless of weights
- sign(0) = 0, which doesn't match y' ∈ {-1, +1}
- These all count as "wrong" in agreement_no_ties
- This drags down the metric significantly

**Expected with random model:**
- 39% informative pairs: ~50% agreement = 0.195
- 61% uninformative pairs: 0% agreement = 0.000
- **Overall: 0.195 ≈ 0.20** ✓ (matches what we see!)

## ✅ Solution Implemented

### Fix 1: Filter Out Uninformative Pairs

**Modified:** `src/router_bert/data/loader.py`

```python
# Drop rows where ALL delta features are zero (no learning signal)
has_signal = (df[dz_cols].abs() > 1e-8).any(axis=1)
n_before = len(df)
df = df[has_signal].copy()
n_after = len(df)
if n_before > n_after:
    print(f"Filtered out {n_before - n_after} pairs with all-zero deltas")
```

**Expected impact:**
- Training set: 6,300 → ~2,457 pairs (39%)
- Validation set: 1,350 → ~527 pairs (39%)
- Test set: 1,350 → ~527 pairs (39%)

### Fix 2: Added Diagnostics

**Modified:** `src/router_bert/train_router.py`

Added tracking of:
- `nonzero_score_ratio`: What fraction of scores are non-zero
- This helps verify the fix is working

## 📊 Expected Results After Fix

### Before Fix (Current)
```
Train Loss: 0.6990 (stuck at log(2))
Val Agree (no ties): 0.2007 (terrible)
Val Agree (ties 0.5): 0.5019 (random)
```

### After Fix (Expected)
```
Train Loss: 0.65 → 0.55 → 0.50 (decreasing)
Val Agree (no ties): 0.55 → 0.65 → 0.70 (improving)
Val Agree (ties 0.5): 0.60 → 0.70 → 0.75 (improving)
Nonzero scores: ~100% (all pairs have signal)
```

## 🚀 How to Test the Fix

### Step 1: Retrain with Fix

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate

# Quick test (1 epoch)
python -m src.router_bert.train_router \
    --epochs 1 \
    --batch_prompts 8 \
    --out_dir artifacts/router/bert_router/fixed_test

# Look for this output:
# "Filtered out 5497 pairs with all-zero deltas (no learning signal)"
# "Loaded XXX prompts with 3503 total pairs"  (instead of 6300)
```

### Step 2: Verify Learning is Happening

Watch for:
- ✅ Loss **decreasing** over epochs (not stuck at 0.699)
- ✅ Val agreement **increasing** over epochs (not stuck at 0.20)
- ✅ Nonzero score ratio near **100%** (not 39%)

### Step 3: Full Training

If the quick test shows improvement:

```bash
python -m src.router_bert.train_router \
    --epochs 10 \
    --batch_prompts 16 \
    --out_dir artifacts/router/bert_router/fixed_run
```

## 🔬 Why This Happened

### Likely Cause: Feature Generation Issue

The all-zero deltas suggest a problem in how features were computed:

1. **Possible scenario:** Some movie pairs were not scored by all experts
2. **When expert didn't score:** Delta set to 0 instead of being excluded
3. **Result:** Many "fake" pairs with no information

### Why Gamma is All Zeros

This needs investigation in the feature generation code:
- Check `src/router/build_router_features.py` or similar
- Look for where `dz_gamma` is computed
- Possible issues:
  - Gamma expert failed to run
  - Gamma scores were all identical (delta = 0)
  - Bug in feature computation

## 📝 Recommendations

### Short-term (Implemented)
1. ✅ Filter out all-zero pairs during training
2. ✅ Add diagnostics to track nonzero scores
3. ⏳ Retrain and verify learning happens

### Medium-term (TODO)
1. ❌ Investigate why gamma is all zeros
2. ❌ Fix feature generation to avoid all-zero pairs
3. ❌ Consider using only 3 experts (alpha, beta, delta) if gamma is broken

### Long-term (TODO)
1. ❌ Add data validation in feature generation pipeline
2. ❌ Add assertions to catch all-zero deltas early
3. ❌ Document expected feature distributions

## 🎯 Success Criteria

After retraining with the fix, you should see:

1. **Data loading:**
   ```
   Filtered out 5497 pairs with all-zero deltas (no learning signal)
   Loaded 700 prompts with 3503 total pairs  (train)
   Loaded 150 prompts with 527 total pairs   (val)
   ```

2. **Training progress:**
   ```
   Epoch 1: Loss 0.650, Val Agree 0.550
   Epoch 2: Loss 0.620, Val Agree 0.580
   Epoch 3: Loss 0.590, Val Agree 0.620
   Epoch 5: Loss 0.550, Val Agree 0.680
   Epoch 10: Loss 0.520, Val Agree 0.720
   ```

3. **Diagnostics:**
   ```
   Train nonzero scores: 95-100%
   (Some might still be near-zero due to model predictions)
   ```

## ⚠️ Important Notes

### This is NOT a Model Problem

The model architecture is fine. The problem was:
- ❌ NOT the BERT encoder
- ❌ NOT the attention mechanism
- ❌ NOT the Bradley-Terry loss
- ✅ **BAD DATA** (61% uninformative pairs)

### Filtering is the Right Solution

Some might worry about "throwing away data", but:
- ✅ All-zero pairs provide **zero information**
- ✅ They only add noise and slow training
- ✅ Filtering them makes training **more efficient**
- ✅ Evaluation on filtered data is more meaningful

### Gamma Expert Investigation

After training works, investigate:
```bash
# Check feature generation code
grep -r "dz_gamma" src/router/

# Check if gamma scores exist
python -c "
import pandas as pd
df = pd.read_parquet('artifacts/router/features_sum.with_splits.bal.parquet')
print('Columns:', [c for c in df.columns if 'gamma' in c.lower()])
"
```

## 📞 Next Steps

1. **Run the fixed training** (see commands above)
2. **Verify loss decreases** and agreement improves
3. **Report results** - compare before/after metrics
4. **Investigate gamma** if training now works
5. **Consider 3-expert model** if gamma can't be fixed

## Summary

**Problem:** 61% of training pairs had all-zero delta features, providing no learning signal.

**Solution:** Filter out uninformative pairs during data loading.

**Expected outcome:** Model will now learn properly with loss decreasing and agreement improving.

**Action required:** Retrain with the fixed code!

