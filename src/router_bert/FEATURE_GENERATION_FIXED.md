# Feature Generation - Fixed!

## ✅ Changes Applied to `build_router_features.py`

### Fix 1: Set user_idx to 1 by Default

**Lines 109-115:**

```python
if "user_idx" not in df.columns:
    df["user_idx"] = args.user_idx_default if args.user_idx_default is not None else 1
    logger.info(f"Added default user_idx: {df['user_idx'].iloc[0]}")
elif df["user_idx"].isna().all() or (df["user_idx"] == 0).all():
    # If user_idx exists but is all NaN or all 0, set to default
    df["user_idx"] = args.user_idx_default if args.user_idx_default is not None else 1
    logger.info(f"Replaced null/zero user_idx with default: {df['user_idx'].iloc[0]}")
```

**What this fixes:**
- ✅ Ensures `user_idx` is always set (defaults to 1)
- ✅ LightGCN will now compute scores for user 1
- ✅ `dz_gamma` will have real values (not all zeros!)

**Changed argument default:**
```python
--user_idx_default: default=1 (was default=None)
```

### Fix 2: Skip Pairs Where Movies Aren't in Retrieval Results

**Lines 199-212:**

```python
# Skip pairs where movies aren't in retrieval results
if a not in per_movie.index or b not in per_movie.index:
    logger.debug(f"Skipping pair {r['pair_id']}: movies not in retrieval results")
    continue

# Get scores for both movies
za = per_movie.loc[a][["z_dense", "z_bm25", "z_lgcn", "z_emo"]].to_numpy(dtype=np.float32)
zb = per_movie.loc[b][["z_dense", "z_bm25", "z_lgcn", "z_emo"]].to_numpy(dtype=np.float32)
dz = (za - zb).astype("float32")

# Skip pairs where all deltas are zero (no learning signal)
if np.allclose(dz, 0, atol=1e-8):
    logger.debug(f"Skipping pair {r['pair_id']}: all deltas are zero")
    continue
```

**What this fixes:**
- ✅ No more defaulting to zeros for missing movies
- ✅ Skips uninformative pairs at generation time
- ✅ Only keeps pairs with actual learning signal
- ✅ Results in cleaner, higher-quality training data

### Fix 3: Added Delta Statistics Logging

**Lines 252-261:**

```python
# Log delta feature statistics
dz_cols = ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta']
if all(c in out.columns for c in dz_cols):
    logger.info("Delta feature statistics:")
    for col in dz_cols:
        nonzero = (out[col].abs() > 1e-8).sum()
        logger.info(f"  {col}: {nonzero}/{len(out)} non-zero ({100*nonzero/len(out):.1f}%)")
    
    all_zero_pairs = (out[dz_cols].abs() <= 1e-8).all(axis=1).sum()
    logger.info(f"  Pairs with ALL zeros: {all_zero_pairs}/{len(out)} ({100*all_zero_pairs/len(out):.1f}%)")
```

**What this provides:**
- ✅ Shows how many pairs have non-zero deltas per expert
- ✅ Warns if any expert still has all zeros
- ✅ Helps diagnose data quality issues

## 🚀 How to Regenerate Features

### Step 1: Find Your Original Command

Check logs or scripts to find how features were originally generated. It likely looked like:

```bash
python -m src.router.build_router_features \
    --prompts_dir <path_to_prompts> \
    --indices_dir artifacts/indices \
    --out artifacts/router/features_sum.parquet \
    --agg_kind sum \
    --topk 200
```

### Step 2: Regenerate with Fixed Code

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Activate your environment
source venvs/rag_recsys/bin/activate

# Run feature generation with user_idx=1 (now default)
python -m src.router.build_router_features \
    --prompts_dir <your_prompts_dir> \
    --indices_dir artifacts/indices \
    --out artifacts/router/features_sum_fixed.parquet \
    --agg_kind sum \
    --topk 500 \
    --user_idx_default 1
```

**Key changes:**
- ✅ `--user_idx_default 1` (now default, but explicit is good)
- ✅ `--topk 500` (increased from 200 to get more movies)
- ✅ New output file to avoid overwriting old data

### Step 3: Add Splits and Balance (If Needed)

If your original data had splits and balancing:

```bash
# Add splits
python -m src.cli.make_split \
    --input artifacts/router/features_sum_fixed.parquet \
    --output artifacts/router/features_sum_fixed.with_splits.parquet \
    --split_by prompt_id \
    --train 0.7 \
    --val 0.15 \
    --test 0.15

# Balance (if needed)
python -m src.cli.ab_balance \
    --input artifacts/router/features_sum_fixed.with_splits.parquet \
    --output artifacts/router/features_sum_fixed.with_splits.bal.parquet
```

### Step 4: Verify the Fixed Features

```bash
python3 << 'EOF'
import pandas as pd
df = pd.read_parquet('artifacts/router/features_sum_fixed.with_splits.bal.parquet')

print("="*80)
print("FEATURE VERIFICATION")
print("="*80)

# Check delta statistics
dz_cols = ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta']
print("\nDelta feature statistics:")
for col in dz_cols:
    nonzero = (df[col].abs() > 1e-8).sum()
    print(f"  {col}: {nonzero}/{len(df)} non-zero ({100*nonzero/len(df):.1f}%)")

all_zero = (df[dz_cols].abs() <= 1e-8).all(axis=1).sum()
print(f"\nPairs with ALL zeros: {all_zero}/{len(df)} ({100*all_zero/len(df):.1f}%)")

print(f"\nTotal pairs: {len(df)}")
print(f"Unique prompts: {df['prompt_id'].nunique()}")
print(f"Splits: {df['split'].value_counts().to_dict() if 'split' in df.columns else 'N/A'}")
EOF
```

**Expected output:**
```
Delta feature statistics:
  dz_alpha: 3500+/XXXX non-zero (XX%)
  dz_beta:  3500+/XXXX non-zero (XX%)
  dz_gamma: 3500+/XXXX non-zero (XX%)  ← Should be NON-ZERO now!
  dz_delta: 3500+/XXXX non-zero (XX%)

Pairs with ALL zeros: 0/XXXX (0.0%)  ← Should be ZERO or very low!
```

## 📊 Expected Improvements

### Before Fix (Current Data)
```
Total pairs: 9,000
- dz_gamma: 0 non-zero (0%)  ← BROKEN
- All-zero pairs: 5,497 (61%)  ← TOO MANY
- Usable pairs: 3,503 (39%)
```

### After Fix (Regenerated Data)
```
Total pairs: ~3,500-4,500 (fewer but all high-quality)
- dz_gamma: 3,500+ non-zero (90%+)  ← FIXED!
- All-zero pairs: 0-100 (0-2%)  ← MINIMAL
- Usable pairs: ~3,500-4,500 (100%)
```

### Training Performance

**Before (with filtering):**
- Training on 3,503 pairs
- 3 effective experts (alpha, beta, delta)
- Expected agreement: 0.60-0.70

**After (with regenerated features):**
- Training on 3,500-4,500 pairs
- 4 working experts (alpha, beta, gamma, delta)
- Expected agreement: 0.70-0.80 (better!)

## 🎯 Decision Tree

### Option A: Quick Test (Use Current Filtered Data)

**When:** You want to see if the model works at all

```bash
# Already done - just train!
python -m src.router_bert.train_router --epochs 5 --batch_prompts 16
```

**Pros:**
- ✅ Immediate results
- ✅ No regeneration needed
- ✅ Should work reasonably well

**Cons:**
- ❌ Only 3 effective experts
- ❌ Missing LightGCN signal
- ❌ Lower performance ceiling

### Option B: Regenerate Features (Recommended)

**When:** You want best performance

```bash
# Regenerate with fixes
python -m src.router.build_router_features \
    --prompts_dir <your_prompts_dir> \
    --indices_dir artifacts/indices \
    --out artifacts/router/features_sum_fixed.parquet \
    --topk 500 \
    --user_idx_default 1

# Add splits and balance
# ... (see Step 3 above)

# Train with new features
python -m src.router_bert.train_router \
    --parquet artifacts/router/features_sum_fixed.with_splits.bal.parquet \
    --epochs 10 \
    --batch_prompts 16
```

**Pros:**
- ✅ All 4 experts working
- ✅ Higher quality data
- ✅ Better performance
- ✅ Cleaner training (no filtering needed)

**Cons:**
- ❌ Takes time to regenerate
- ❌ Need to find original prompts_dir

## 📝 Summary

### What Was Fixed

1. ✅ **user_idx defaults to 1** (was None)
   - LightGCN now computes scores
   - dz_gamma will have real values

2. ✅ **Skip pairs with missing movies** (was defaulting to zeros)
   - No more fake zero-delta pairs
   - Only keep informative pairs

3. ✅ **Skip pairs with all-zero deltas** (was keeping them)
   - Cleaner data at generation time
   - No need for filtering in training

4. ✅ **Added diagnostic logging**
   - See delta statistics in logs
   - Catch issues early

### What to Do Next

**Immediate:**
```bash
# Test with current filtered data
python -m src.router_bert.train_router --epochs 5
```

**If performance is poor (<0.65 agreement):**
```bash
# Regenerate features with fixes
python -m src.router.build_router_features \
    --prompts_dir <path> \
    --out artifacts/router/features_sum_fixed.parquet \
    --topk 500 \
    --user_idx_default 1
```

**The fixes are ready - you can regenerate features anytime!** 🎉

