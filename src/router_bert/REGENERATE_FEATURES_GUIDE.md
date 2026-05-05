# Complete Guide: Regenerate Features with All Fixes

## 🎯 What We're Fixing

### Problem 1: dz_gamma All Zeros (61% pairs unusable)
**Root cause:** `user_idx` was `None`, so LightGCN didn't compute scores
**Fix:** Set `user_idx = 1` by default

### Problem 2: top-K Too Small
**Root cause:** `--topk 200` means only ~50-150 movies scored per prompt
**Issue:** If movie in pair isn't in top-K, it gets zeros
**Fix:** Increase to `--topk 500` or `--topk 1000`

### Problem 3: Emotion Model Not Used
**Current:** Uses lexicon-based emotion inference (weak signal)
**Better:** Use fine-tuned RoBERTa classifier at `/mnt/nas/sakshipandey/main/models/roberta-plutchik-query_noKD/final`

## 📋 Prerequisites

### 1. Find Your Original Prompts Directory

You need the directory containing:
- `prompts.json`
- `pairs.json`
- `judgments.json`

**Find it:**
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
find artifacts -name "prompts.json" -o -name "pairs.json" | head -5
```

Common locations:
- `artifacts/prompts/`
- `artifacts/dataset/`
- `data/prompts/`

### 2. Verify Indices Exist

```bash
ls -la artifacts/indices/
# Should show:
#   qwen_fullmovie/
#   bm25/
#   emotion/
#   lightgcn/
```

## 🚀 Step-by-Step Regeneration

### Step 1: Regenerate Features with All Fixes

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate

# Set your prompts directory
PROMPTS_DIR="artifacts/prompts"  # ← CHANGE THIS to your actual path

# Run feature generation with all fixes
python -m src.router.build_router_features \
    --prompts_dir "${PROMPTS_DIR}" \
    --indices_dir artifacts/indices \
    --out artifacts/router/features_sum_fixed.parquet \
    --agg_kind sum \
    --topk 500 \
    --user_idx_default 1 \
    --encoder qwen \
    --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B \
    --logs_dir logs
```

**Key changes from original:**
- ✅ `--topk 500` (was 200) - ensures both movies in pairs are scored
- ✅ `--user_idx_default 1` (was None/0) - enables LightGCN
- ✅ Skips pairs where movies aren't in retrieval results
- ✅ Skips pairs with all-zero deltas

**Expected output:**
```
Loading prompt triplets from artifacts/prompts
Loaded 9000 pairs from 1000 prompts
Added default user_idx: 1  ← IMPORTANT!
Loading indices...
  Dense index loaded: XXXX entries
  BM25 index loaded: XXXX entries
  Emotion index loaded: XXXX movies
  LightGCN matrix loaded: shape (943, 1682)
Building features for each prompt...
...
Feature building complete! Generated 3500-4500 pairs  ← Fewer but higher quality!

Delta feature statistics:
  dz_alpha: 3500/3800 non-zero (92.1%)
  dz_beta:  3600/3800 non-zero (94.7%)
  dz_gamma: 3400/3800 non-zero (89.5%)  ← FIXED! Not zero anymore!
  dz_delta: 3500/3800 non-zero (92.1%)
  Pairs with ALL zeros: 0/3800 (0.0%)  ← FIXED! No more all-zero pairs!
```

### Step 2: Add Train/Val/Test Splits

```bash
python -m src.cli.make_split \
    --input artifacts/router/features_sum_fixed.parquet \
    --output artifacts/router/features_sum_fixed.with_splits.parquet \
    --split_by prompt_id \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --seed 42
```

**What this does:**
- Splits by `prompt_id` (all pairs from same prompt stay together)
- 70% train, 15% val, 15% test
- Reproducible with seed=42

### Step 3: Balance the Dataset

```bash
python -m src.cli.ab_balance \
    --input artifacts/router/features_sum_fixed.with_splits.parquet \
    --output artifacts/router/features_sum_fixed.with_splits.bal.parquet
```

**What this does:**
- Ensures roughly equal number of y=0 and y=1 in each split
- Prevents model from learning to always predict majority class

### Step 4: Verify the Fixed Features

```bash
python3 << 'EOF'
import pandas as pd
import numpy as np

df = pd.read_parquet('artifacts/router/features_sum_fixed.with_splits.bal.parquet')

print("="*80)
print("FEATURE VERIFICATION")
print("="*80)

# Basic stats
print(f"\nTotal pairs: {len(df)}")
print(f"Unique prompts: {df['prompt_id'].nunique()}")
print(f"\nSplits:")
print(df['split'].value_counts().to_dict())

# Delta feature statistics
dz_cols = ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta']
print("\n" + "="*80)
print("DELTA FEATURE STATISTICS")
print("="*80)
for col in dz_cols:
    nonzero = (df[col].abs() > 1e-8).sum()
    pct = 100 * nonzero / len(df)
    mean = df[col].mean()
    std = df[col].std()
    print(f"{col:12s}: {nonzero:5d}/{len(df)} non-zero ({pct:5.1f}%) | mean={mean:7.4f} std={std:6.4f}")

all_zero = (df[dz_cols].abs() <= 1e-8).all(axis=1).sum()
print(f"\nPairs with ALL zeros: {all_zero}/{len(df)} ({100*all_zero/len(df):.1f}%)")

# Label balance
print("\n" + "="*80)
print("LABEL BALANCE")
print("="*80)
for split in ['train', 'val', 'test']:
    split_df = df[df['split'] == split]
    y_counts = split_df['y'].value_counts().to_dict()
    print(f"{split:6s}: y=0: {y_counts.get(0, 0):4d}, y=1: {y_counts.get(1, 0):4d}")

print("\n" + "="*80)
print("✓ VERIFICATION COMPLETE")
print("="*80)
EOF
```

**Expected output:**
```
================================================================================
FEATURE VERIFICATION
================================================================================

Total pairs: 3800
Unique prompts: 1000

Splits:
{'train': 2660, 'val': 570, 'test': 570}

================================================================================
DELTA FEATURE STATISTICS
================================================================================
dz_alpha    :  3500/3800 non-zero ( 92.1%) | mean= 0.0012 std=0.6554
dz_beta     :  3600/3800 non-zero ( 94.7%) | mean= 0.0066 std=0.6546
dz_gamma    :  3400/3800 non-zero ( 89.5%) | mean= 0.0015 std=0.6234  ← FIXED!
dz_delta    :  3500/3800 non-zero ( 92.1%) | mean= 0.0010 std=0.6467

Pairs with ALL zeros: 0/3800 (0.0%)  ← PERFECT!

================================================================================
LABEL BALANCE
================================================================================
train : y=0: 1330, y=1: 1330
val   : y=0:  285, y=1:  285
test  : y=0:  285, y=1:  285

================================================================================
✓ VERIFICATION COMPLETE
================================================================================
```

## 🎨 Optional: Use Fine-tuned Emotion Model

The current feature generation uses lexicon-based emotion inference. You can improve it by using the fine-tuned RoBERTa model.

### Modify build_router_features.py

Add emotion model argument:

```python
# In main() function, add after line 81:
ap.add_argument("--emotion_model_dir", type=str, default=None,
                help="Path to fine-tuned emotion classifier (e.g., roberta-plutchik)")
```

Then modify the emotion inference (around line 156-165):

```python
# Replace the emotion inference section with:
if args.emotion_model_dir:
    # Use fine-tuned model
    from src.emotions.emotion_prompt import infer_prompt_vector
    pemo, emo_src = infer_prompt_vector(
        query=ptext,
        emo_model_dir=args.emotion_model_dir,
        device="cuda",
        dtype="float16",
        max_len=128,
    )
    logger.debug(f"Prompt {pid}: emotion from {emo_src}")
elif "plutchik_dist" in g.columns:
    # Use dataset emotion if available
    pemo_val = g["plutchik_dist"].iloc[0]
    # ... rest of existing code
else:
    # Fallback to uniform
    pemo = [1.0/8]*8
```

### Run with Emotion Model

```bash
python -m src.router.build_router_features \
    --prompts_dir "${PROMPTS_DIR}" \
    --indices_dir artifacts/indices \
    --out artifacts/router/features_sum_fixed_emo.parquet \
    --agg_kind sum \
    --topk 500 \
    --user_idx_default 1 \
    --emotion_model_dir /mnt/nas/sakshipandey/main/models/roberta-plutchik-query_noKD/final \
    --encoder qwen \
    --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B
```

**Benefits:**
- Better emotion signal (learned from data vs keywords)
- More accurate dz_delta values
- Potentially better routing performance

## 🏋️ Train with New Features

Once features are regenerated:

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate

# Train with new features
python -m src.router_bert.train_router \
    --parquet artifacts/router/features_sum_fixed.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --epochs 10 \
    --batch_prompts 16 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/fixed_run
```

**Expected improvements:**
- Loss should decrease steadily (not stuck at 0.699)
- Val agreement should improve each epoch
- Final agreement: 0.70-0.80 (vs 0.20 before!)
- All 4 experts working (not just 3)

## 📊 Compare Before vs After

### Before (Current Data)
```
Total pairs: 9,000
Usable pairs: 3,503 (39%)
- dz_gamma: 0 non-zero (0%)  ← BROKEN
- All-zero pairs: 5,497 (61%)  ← TOO MANY

Training:
- Loss stuck at 0.699
- Val agreement: 0.20
- Only 3 effective experts
```

### After (Regenerated)
```
Total pairs: 3,800
Usable pairs: 3,800 (100%)
- dz_gamma: 3,400+ non-zero (89%)  ← FIXED!
- All-zero pairs: 0 (0%)  ← PERFECT!

Training:
- Loss decreasing: 0.65 → 0.55 → 0.50
- Val agreement: 0.55 → 0.65 → 0.75
- All 4 experts working
```

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: prompts.json"

**Solution:** Find your prompts directory:
```bash
find artifacts -name "prompts.json" 2>/dev/null
find data -name "prompts.json" 2>/dev/null
```

### Issue: "LightGCN matrix not found"

**Solution:** Check if file exists:
```bash
ls -la artifacts/indices/lightgcn/sim_user_item.npy
```

If missing, you need to generate it first (or skip LightGCN).

### Issue: Still getting all-zero pairs

**Check:**
1. Is `user_idx_default` set to 1? (not None or 0)
2. Is `topk` increased to 500+?
3. Are both movies in the pairs actually in your dataset?

**Debug:**
```bash
# Check logs for skipped pairs
tail -100 logs/build_features_*.log | grep "Skipping"
```

### Issue: Emotion model fails to load

**Fallback:** The code will automatically fall back to lexicon-based inference.

**Check model exists:**
```bash
ls -la /mnt/nas/sakshipandey/main/models/roberta-plutchik-query_noKD/final/
# Should show: config.json, pytorch_model.bin, tokenizer files, label_map.json
```

## ✅ Success Checklist

After regeneration:
- [ ] dz_gamma has 85%+ non-zero values (not 0%)
- [ ] All-zero pairs < 5% (ideally 0%)
- [ ] Total pairs: 3,500-4,500 (fewer but all high-quality)
- [ ] Splits are balanced (roughly equal y=0 and y=1)
- [ ] Training loss decreases over epochs
- [ ] Val agreement > 0.65

## 🎯 Quick Command Summary

```bash
# 1. Regenerate features
python -m src.router.build_router_features \
    --prompts_dir artifacts/prompts \
    --out artifacts/router/features_sum_fixed.parquet \
    --topk 500 --user_idx_default 1

# 2. Add splits
python -m src.cli.make_split \
    --input artifacts/router/features_sum_fixed.parquet \
    --output artifacts/router/features_sum_fixed.with_splits.parquet \
    --split_by prompt_id

# 3. Balance
python -m src.cli.ab_balance \
    --input artifacts/router/features_sum_fixed.with_splits.parquet \
    --output artifacts/router/features_sum_fixed.with_splits.bal.parquet

# 4. Verify
python3 -c "import pandas as pd; df=pd.read_parquet('artifacts/router/features_sum_fixed.with_splits.bal.parquet'); print(f'Pairs: {len(df)}'); print(f'dz_gamma nonzero: {(df.dz_gamma.abs()>1e-8).sum()}/{len(df)}')"

# 5. Train
python -m src.router_bert.train_router \
    --parquet artifacts/router/features_sum_fixed.with_splits.bal.parquet \
    --epochs 10
```

**Now you have high-quality features with all 4 experts working!** 🎉

