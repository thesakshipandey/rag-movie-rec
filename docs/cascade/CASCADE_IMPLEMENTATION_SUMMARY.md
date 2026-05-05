# CASCADE-AWARE TRAINING IMPLEMENTATION SUMMARY

## What Was Implemented

I've created a complete cascade-aware training and evaluation system that addresses your concern: **"Why is gamma always dominating? Are you not retraining the MoE with different values?"**

The answer is: **Yes, now we are!** The new system trains multiple router models with cascade gating incorporated into the training process.

## Key Files Created

### 1. Training Script: `src/cli/train_router_cascade.py`

**New training script** that supports cascade-aware training (does not modify the original `train_router.py`).

**Key Features**:
- `--cascade_threshold <float>`: Set dominance threshold (e.g., 0.75)
- `--gating`: Enable cascade gating during training
- `--gating_strength <float>`: Control cascade regularization strength

**What It Does**:
When `--gating` is enabled, the router learns:
- To produce **decisive weights** when one expert should dominate
- To use **mixture behavior** when multiple experts are needed
- To optimize for both accuracy AND cascade efficiency

**Training Loss**:
```
Total Loss = BTL Loss + Entropy Loss + Cascade Gating Loss
```

The cascade gating loss:
- Encourages low entropy (decisive) when max(w) ≥ threshold
- Allows high entropy (mixture) when max(w) < threshold
- Helps the router learn WHEN to rely on a single expert

### 2. Evaluation Script: `src/evaluations/models/MoE_cascade_eval.py`

**New evaluation script** for cascade-trained models (does not modify the original `MoE_eval.py`).

**What It Evaluates**:
- Overall performance (accuracy, agreement)
- Expert weight distributions
- Cascade behavior at multiple thresholds
- Per-category and per-difficulty breakdown
- Expert selection patterns

**Output**:
- Individual JSON results for each model
- Comparison CSV table
- Comprehensive markdown summary report

### 3. Batch Training Script: `train_all_cascade_models.sh`

**Convenience script** to train all 6 configurations in one go:

1. **Baseline** (`router_no_gating.pt`): Standard MoE, no cascade gating
2. **Threshold 0.70** (`router_cascade_0.70.pt`): Low threshold, more gating
3. **Threshold 0.75** (`router_cascade_0.75.pt`): Medium threshold
4. **Threshold 0.80** (`router_cascade_0.80.pt`): High threshold
5. **Threshold 0.85** (`router_cascade_0.85.pt`): Very high threshold
6. **Threshold 0.90** (`router_cascade_0.90.pt`): Extreme threshold, rare gating

**Usage**:
```bash
./train_all_cascade_models.sh
```

### 4. Complete Pipeline: `run_cascade_pipeline.sh`

**End-to-end script** that:
1. Trains all 6 cascade models
2. Evaluates all models on test set
3. Generates comparison tables and reports
4. Displays results summary

**Usage**:
```bash
./run_cascade_pipeline.sh
```

### 5. Documentation: `CASCADE_TRAINING_GUIDE.md`

Comprehensive guide covering:
- Cascade gating concepts
- Training procedures
- Evaluation methods
- Result interpretation
- Troubleshooting

## Difference from Previous Approach

### Previous Approach (Inference-Only Cascade)

**File**: `src/evaluations/cascade_threshold_eval.py`

**What it did**:
- Loaded ONE pre-trained router
- Applied different thresholds at INFERENCE time
- Router was NOT trained with cascade awareness

**Problem**:
- Router never learned to optimize for cascade
- Always showed gamma dominance (router wasn't trained to adapt)
- No real difference between thresholds

### New Approach (Cascade-Aware Training)

**Files**: 
- Training: `src/cli/train_router_cascade.py`
- Evaluation: `src/evaluations/models/MoE_cascade_eval.py`

**What it does**:
- Trains MULTIPLE routers with different configurations
- Incorporates cascade gating INTO the training loss
- Router LEARNS when to be decisive vs. when to blend

**Expected outcome**:
- More adaptive expert selection
- Meaningful differences between thresholds
- Better understanding of when each expert is truly needed

## How to Use

### Option 1: Complete Pipeline (Recommended)

Run everything at once:

```bash
./run_cascade_pipeline.sh
```

This will:
1. Train all 6 models (~15-30 minutes)
2. Evaluate all models (~1-2 minutes)
3. Generate reports and comparisons
4. Display results summary

### Option 2: Step-by-Step

**Step 1: Train models**
```bash
./train_all_cascade_models.sh
```

**Step 2: Evaluate**
```bash
python -m src.evaluations.models.MoE_cascade_eval \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --output_dir artifacts/evaluation_results/cascade_training \
  --split test
```

**Step 3: Review results**
```bash
cat artifacts/evaluation_results/cascade_training/CASCADE_TRAINING_SUMMARY.md
cat artifacts/evaluation_results/cascade_training/cascade_training_comparison.csv
```

### Option 3: Train Individual Models

Train specific configurations:

```bash
# Baseline (no gating)
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/router_no_gating.pt \
  --epochs 20 --lr 5e-4 --seed 42

# With cascade threshold 0.75
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/router_cascade_0.75.pt \
  --cascade_threshold 0.75 \
  --gating \
  --epochs 20 --lr 5e-4 --seed 42
```

## Expected Results

### Performance Comparison

The evaluation will show:
- **Overall accuracy** for each model
- **Expert weight distributions** (mean alpha/beta/gamma/delta)
- **Cascade trigger rates** at different thresholds
- **Expert selection patterns** when gating activates

### Key Questions Answered

1. **Does cascade-aware training reduce gamma dominance?**
   - Compare `mean_gamma` across models
   - Check if other experts get more weight

2. **What's the optimal threshold?**
   - Compare accuracy across thresholds
   - Find balance between performance and efficiency

3. **Does gating help or hurt?**
   - Compare baseline (no gating) vs. cascade models
   - Check if there's a performance trade-off

4. **Are decisions more interpretable?**
   - Check `gated_expert_distribution`
   - See if different categories prefer different experts

### Output Files

After running the pipeline:

```
artifacts/
├── router/
│   ├── router_no_gating.pt              # Baseline model
│   ├── router_cascade_0.70.pt           # Threshold 0.70
│   ├── router_cascade_0.75.pt           # Threshold 0.75
│   ├── router_cascade_0.80.pt           # Threshold 0.80
│   ├── router_cascade_0.85.pt           # Threshold 0.85
│   └── router_cascade_0.90.pt           # Threshold 0.90
│
└── evaluation_results/
    └── cascade_training/
        ├── router_no_gating_results.json
        ├── router_cascade_0.70_results.json
        ├── router_cascade_0.75_results.json
        ├── router_cascade_0.80_results.json
        ├── router_cascade_0.85_results.json
        ├── router_cascade_0.90_results.json
        ├── cascade_training_comparison.csv      # Summary table
        └── CASCADE_TRAINING_SUMMARY.md          # Detailed report
```

## Technical Details

### Cascade Gating Loss Function

Located in `train_router_cascade.py`:

```python
def cascade_gating_loss(weights, threshold, strength):
    """
    Regularizer that encourages decisive expert selection.
    
    When max(w) ≥ threshold:
      - Penalize high entropy (uniform distribution)
      - Encourage single expert dominance
    
    When max(w) < threshold:
      - Allow mixture behavior
      - Don't penalize entropy
    """
    max_weights = weights.max(dim=-1)[0]
    gating_signal = sigmoid((max_weights - threshold) * 10.0)
    entropy = -(weights * log(weights)).sum(dim=-1)
    cascade_penalty = gating_signal * normalized_entropy
    return strength * cascade_penalty.mean()
```

### Model Checkpoint Format

Each model saves:
```python
{
    "state_dict": model.state_dict(),
    "config": training_config,
    "cascade_threshold": threshold,
    "gating_enabled": True/False,
    "input_dim": 66,
    "dz_dim": 4,
    "mix_indices": [...]
}
```

This allows the evaluation script to load and identify models correctly.

## Troubleshooting

### Issue: Models take too long to train

**Solution**: The batch script trains 6 models sequentially. On a GPU, this should take 15-30 minutes total. If you want faster training:
- Train fewer models (edit `train_all_cascade_models.sh`)
- Reduce epochs (`--epochs 10` instead of 20)
- Train in parallel (manual approach)

### Issue: Gamma still dominates everything

**Possible reasons**:
1. **Data-driven**: Gamma (LightGCN) genuinely performs best across most queries
2. **Gating strength too low**: Increase `--gating_strength` (try 0.2)
3. **Threshold mismatch**: Try lower thresholds (0.6-0.7)

**Analysis**:
- Check baseline (no gating) first: Does gamma dominate there too?
- Look at per-category breakdown: Maybe gamma is best for most categories
- Compare with oracle weights (if available)

### Issue: Performance drops with gating

**Possible causes**:
- Over-regularization: Lower `--gating_strength`
- Threshold too extreme: Use moderate values (0.7-0.8)
- Insufficient training: Increase `--epochs`

**Solution**:
```bash
# Try gentler cascade training
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/router_gentle_cascade.pt \
  --cascade_threshold 0.75 \
  --gating \
  --gating_strength 0.05 \
  --epochs 30 \
  --lr 5e-4
```

## Integration with Existing Code

### Compatibility

- **Does NOT modify** existing training script (`train_router.py`)
- **Does NOT modify** existing evaluation script (`MoE_eval.py`)
- **Fully compatible** with existing models and features
- **Can be used alongside** existing evaluation system

### Using Cascade Models in Production

After training and evaluation, you can use cascade models just like regular routers:

```python
# Load cascade-trained model
model = RouterMLP(d_in=66, dz_dim=4, mix_indices=mix_indices)
checkpoint = torch.load("artifacts/router/router_cascade_0.75.pt")
model.load_state_dict(checkpoint["state_dict"])

# Use at inference
weights = model(features)

# Apply cascade gating (optional)
threshold = 0.75
max_weight = weights.max()
if max_weight >= threshold:
    dominant_expert = weights.argmax()
    # Use only dominant expert
else:
    # Use full mixture
```

## Next Steps

1. **Run the pipeline**:
   ```bash
   ./run_cascade_pipeline.sh
   ```

2. **Review results**:
   - Check `artifacts/evaluation_results/cascade_training/CASCADE_TRAINING_SUMMARY.md`
   - Compare models in CSV table

3. **Analyze findings**:
   - Does cascade training improve expert diversity?
   - What's the optimal threshold?
   - Does gating help performance?

4. **Select best model**:
   - Based on accuracy and interpretability
   - Consider cascade efficiency if deploying

5. **For presentation**:
   - Use comparison table
   - Show expert weight distributions
   - Highlight cascade behavior differences

## Summary

You now have:
- ✅ **New training script** with cascade gating support
- ✅ **New evaluation script** for cascade models
- ✅ **Batch training** to train all configurations
- ✅ **Complete pipeline** for end-to-end execution
- ✅ **Comprehensive documentation** and guides
- ✅ **Backward compatibility** with existing code

**The key difference**: Now the router is TRAINED with cascade awareness, not just evaluated with it. This should provide much more meaningful results about when and why different experts are selected.

