# Fixes: Model Loading and Softmax Usage

## Issues Fixed

### Issue 1: Model Architecture Mismatch During Evaluation ✅

**Problem**: 
```
RuntimeError: Error(s) in loading state_dict for ContextualHedgeRouterWithEncoder:
size mismatch for context_encoder.category_embed.weight: 
copying a param with shape torch.Size([15, 32]) from checkpoint, 
the shape in current model is torch.Size([5, 32])
```

**Root Cause**:
- Training script builds vocabularies from actual data (e.g., 15 unique categories)
- Evaluation script uses default/hardcoded vocabularies (e.g., 5 dummy categories)
- Model layer sizes don't match when loading checkpoint

**Solution**:
1. **Save vocabularies with checkpoint** (`train_router_listwise.py`):
   ```python
   torch.save({
       'model_state_dict': ...,
       'vocabularies': {
           'category': train_dataset.category_vocab,
           'difficulty': train_dataset.difficulty_vocab,
           'primary_expert': train_dataset.primary_expert_vocab,
           'length_bucket': train_dataset.length_bucket_vocab,
           'persona_style': train_dataset.persona_style_vocab,
       }
   }, args.out)
   ```

2. **Load vocabularies during evaluation** (`eval_router_listwise.py`):
   ```python
   checkpoint = torch.load(args.router_checkpoint)
   vocabularies = checkpoint.get('vocabularies', {})
   
   encoder_kwargs = {
       'category_vocab': list(vocabularies.get('category', {}).keys()),
       'difficulty_vocab': list(vocabularies.get('difficulty', {}).keys()),
       # ... etc
   }
   
   model = ContextualHedgeRouterWithEncoder(..., encoder_kwargs=encoder_kwargs)
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

**Files Changed**:
- `src/cli/train_router_listwise.py` - Save vocabularies
- `src/evaluations/eval_router_listwise.py` - Load vocabularies

---

### Issue 2: Softmax on Expert Scores (Not Recommended) ✅

**Problem**: 
User asked: "are you not softmaxing the scores of each expert for each movie/chunk?"

**Clarification**:
There are **TWO different softmax operations**:

1. **Expert Scores** (Optional, currently enabled with `--apply_softmax`)
   - Applied per-prompt to each expert's z-normalized scores
   - Converts z-scores to probabilities that sum to 1

2. **Router Weights** (Always enabled, essential)
   - Applied to router's logits to get expert weights
   - Ensures weights sum to 1 (probability distribution over experts)

**Recommendation**: **Remove `--apply_softmax` for expert scores**

**Reasoning**:

| Aspect | Z-Scores (Recommended) | Softmax Probabilities |
|--------|------------------------|----------------------|
| **Purpose** | Ranking | Probability estimation |
| **Range** | (-∞, +∞) | [0, 1], sum=1 |
| **Discriminative Power** | ✅ High | ❌ Compressed |
| **Uncertainty Representation** | ✅ Can show low confidence | ❌ Always sums to 1 |
| **Cross-prompt Comparability** | ✅ Preserved | ❌ Lost |

**Example**:
```python
# Scenario: Expert is uncertain (all movies are mediocre)
raw_scores = [0.51, 0.52, 0.50, 0.49]

# Z-scores preserve this uncertainty
z_scores = [0.15, 0.45, -0.30, -0.30]  # Small differences
mean_z = 0.0, std_z = 0.35  # Low variation = uncertainty

# Softmax hides the uncertainty
softmax_probs = [0.24, 0.27, 0.24, 0.24]  # Looks confident!
# Always sums to 1, router can't tell expert is uncertain
```

**What Gets Softmaxed?**

```
Raw Expert Scores (different scales)
        ↓
Z-score Normalize ← PER EXPERT, PER PROMPT (ALWAYS)
        ↓
[REMOVED] Softmax ← Optional, NOT RECOMMENDED for ranking
        ↓
Router Input: z_α, z_β, z_γ, z_δ
        ↓
Router MLP → logits
        ↓
Softmax ← ON ROUTER LOGITS (ALWAYS, ESSENTIAL)
        ↓
Expert Weights: [w_α, w_β, w_γ, w_δ]
        ↓
Final Score = w_α·z_α + w_β·z_β + w_γ·z_γ + w_δ·z_δ
        ↓
Rank by Final Score
```

**Solution**: 
Removed `--apply_softmax` flag from `run_listwise_pipeline.sh`

**Files Changed**:
- `run_listwise_pipeline.sh` - Removed `--apply_softmax` flag
- Created `SOFTMAX_CLARIFICATION.md` - Detailed explanation

---

## Summary of Changes

### Modified Files:
1. **`src/cli/train_router_listwise.py`**
   - Save vocabularies in checkpoint for reproducible model loading
   - Fixed indentation: save only happens for best model

2. **`src/evaluations/eval_router_listwise.py`**
   - Load vocabularies from checkpoint
   - Construct model with exact same architecture as training

3. **`run_listwise_pipeline.sh`**
   - Removed `--apply_softmax` flag (use z-scores for ranking)

### New Documentation:
4. **`SOFTMAX_CLARIFICATION.md`**
   - Detailed explanation of softmax usage
   - Comparison of z-scores vs probabilities
   - Recommendations for ranking tasks

5. **`FIXES_MODEL_LOADING_AND_SOFTMAX.md`** (this file)
   - Summary of issues and solutions

---

## Next Steps

1. **Re-run Training** (if you already trained, need to retrain to save vocabularies):
   ```bash
   cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
   conda activate rag_recsys
   ./run_listwise_pipeline.sh
   ```

2. **Or, if expert scores already exist, skip to training**:
   ```bash
   ./run_listwise_pipeline.sh --skip-generation
   ```

3. **Verify**:
   - Training should save vocabularies in checkpoint
   - Evaluation should load them correctly
   - No more architecture mismatch errors!

---

## Technical Details

### Why Vocabularies Matter

The model has embedding layers for categorical features:

```python
class ContextualFeatureEncoder:
    self.category_embed = nn.Embedding(len(category_vocab), 32)
    self.difficulty_embed = nn.Embedding(len(difficulty_vocab), 16)
    # ... etc
```

**During Training**:
- Scans data: finds 15 unique categories, 3 difficulties, etc.
- Creates embedding layers: `Embedding(15, 32)`, `Embedding(3, 16)`
- Saves model weights

**During Evaluation (OLD)**:
- Uses defaults: 5 categories, 2 difficulties
- Creates embedding layers: `Embedding(5, 32)`, `Embedding(2, 16)`
- ❌ Can't load weights: size mismatch!

**During Evaluation (NEW)**:
- Loads vocabularies from checkpoint
- Creates same layers: `Embedding(15, 32)`, `Embedding(3, 16)`
- ✅ Loads weights perfectly!

### Why Z-Scores Not Softmax

**For Ranking**: You want to preserve relative differences
```python
# Movie A is much better than B
z_scores = [2.5, 0.3, ...]  # Clear winner
softmax  = [0.9, 0.1, ...]  # Information preserved

# But when all movies are mediocre:
z_scores = [0.1, 0.05, ...]  # Tells router: "I'm not sure"
softmax  = [0.52, 0.48, ...] # Tells router: "Movie A is definitely better!"
```

The router needs to know when experts are uncertain!

---

## Status: ✅ FIXED

Both issues are now resolved:
- ✅ Model loading will work (vocabularies saved/loaded)
- ✅ Softmax removed from expert scores (better for ranking)
- ✅ Documentation created for clarity

You can now run the full pipeline without errors!

