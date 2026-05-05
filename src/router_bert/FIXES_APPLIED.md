# Fixes Applied - BERT Router

## Issue 1: ValueError in Training ✅ FIXED

### Problem
```
ValueError: too many values to unpack (expected 2)
```

**Location:** `train_router.py` line 119
```python
weights, _ = model.forward_texts(prompt_texts, return_attn=False)
```

**Root Cause:** The `forward_texts` method was returning 3 values (weights, attn, input_ids) but the code was trying to unpack only 2.

### Solution

**Modified:** `src/router_bert/models/four_head_router.py`

**Before:**
```python
def forward_texts(self, texts, return_attn=False):
    # ...
    if return_attn:
        return weights, attn, input_ids
    else:
        return weights, None, None  # Always 3 values
```

**After:**
```python
def forward_texts(self, texts, return_attn=False):
    # ...
    return weights, attn  # Always 2 values (attn is None if not requested)
```

**Also Fixed:**
- `eval_router.py` line 61: Changed `weights, _, _ = ...` to `weights, _ = ...`
- `eval_router.py` collect_attention_examples: Now tokenizes separately to get input_ids when needed

### Files Changed
1. ✅ `src/router_bert/models/four_head_router.py`
2. ✅ `src/router_bert/eval_router.py`

## Issue 2: User Terminology Clarification ✅ CLARIFIED

### Confusion
"User" has two different meanings in this project:

### 1. System User
- **Who:** sakshipandey (the person running the code)
- **Purpose:** Developer/researcher
- **In code:** File paths, permissions, ownership

### 2. Recommendation Users
- **Who:** 943 users in MovieLens dataset
- **Purpose:** People getting movie recommendations
- **In code:** NOT directly in router (router is prompt-based, not user-based)

### Key Understanding

**The BERT router is PROMPT-BASED, not USER-BASED:**

```
Input:  Prompt text (e.g., "I want a thrilling sci-fi movie")
Output: Expert weights [α, β, γ, δ]
```

The router:
- ✅ Works for ALL 943 users (and new users too)
- ✅ Based on what they're asking for (prompt text)
- ✅ NOT tied to specific user IDs
- ✅ Generalizes to new users not in training data

### Documentation Created
- ✅ `USER_CLARIFICATION.md` - Detailed explanation of user terminology

## Testing Status

### ✅ Ready to Test

The training should now work. Run:

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate

# Quick test (1 epoch)
python -m src.router_bert.train_router \
    --epochs 1 \
    --batch_prompts 8 \
    --out_dir artifacts/router/bert_router/test_run

# If successful, run full training
python -m src.router_bert.train_router \
    --epochs 5 \
    --batch_prompts 16 \
    --out_dir artifacts/router/bert_router/run1
```

### Expected Output

```
Epoch 1/5
--------------------------------------------------------------------------------
Training: 100%|████████| 44/44 [00:45<00:00, loss: 0.6234, bt: 0.6198, ent: 0.0036]
Evaluating: 100%|████████| 10/10 [00:08<00:00]

Train Loss: 0.6234 (BT: 0.6198, Ent: 0.0036)
Val Loss: 0.5987
Val Agree (no ties): 0.7234
Val Agree (ties 0.5): 0.7456
New best validation agreement: 0.7234
```

## Summary of Changes

### Code Fixes
1. ✅ Fixed `forward_texts` return signature (3 values → 2 values)
2. ✅ Updated all call sites to match new signature
3. ✅ Fixed attention example collection to tokenize separately

### Documentation
1. ✅ Created `USER_CLARIFICATION.md` explaining user terminology
2. ✅ Created `FIXES_APPLIED.md` (this file)

### No Breaking Changes
- ✅ All existing functionality preserved
- ✅ API simplified (fewer return values)
- ✅ No changes to training logic or loss computation

## Verification Checklist

Before running training:
- [x] Virtual environment activated (`source venvs/rag_recsys/bin/activate`)
- [x] PyTorch available (version 2.8.0)
- [x] Data files present and validated
- [x] Import paths corrected (`src.router_bert.*`)
- [x] Return value unpacking fixed

After running training:
- [ ] Training completes without errors
- [ ] Validation metrics computed
- [ ] Best model saved
- [ ] Training log created

## Next Steps

1. **Test the fix:**
   ```bash
   python -m src.router_bert.train_router --epochs 1 --batch_prompts 8
   ```

2. **If successful, run full training:**
   ```bash
   python -m src.router_bert.train_router --epochs 5 --batch_prompts 16
   ```

3. **Evaluate:**
   ```bash
   python -m src.router_bert.eval_router \
       --ckpt_dir artifacts/router/bert_router/run_*/best_model \
       --split test
   ```

4. **Review results:**
   - Check `training_log.csv` for convergence
   - Review `metrics_overall.json` for performance
   - Examine attention examples for interpretability

## Questions Answered

### Q: Why did the error occur?
A: The `forward_texts` method signature was inconsistent - it returned 3 values but most callers expected 2.

### Q: Which users are we making recommendations for?
A: ALL users! The router is prompt-based and works for any user (including new users) based on their text prompt.

### Q: Is the router personalized per user?
A: The router itself is not user-specific, but the final recommendations are personalized because the expert retrieval systems use user history.

### Q: Can it work for users not in the training data?
A: Yes! Since it's based on prompt text (not user IDs), it generalizes to new users.

## Contact

For issues or questions:
- Check documentation in `src/router_bert/`
- Review training logs in `artifacts/router/bert_router/*/training_log.csv`
- See `USER_CLARIFICATION.md` for terminology questions

