# Cascade Training and Evaluation Guide

## Overview

This guide explains the **Cascade-Aware Training** system for the MoE router. Unlike standard MoE training, cascade-aware training incorporates a gating mechanism that teaches the router when to rely on a single dominant expert versus using the full mixture.

## Key Concepts

### What is Cascade Gating?

Cascade gating is a 3-layer routing architecture that enables early filtering based on dominant expert signals:

- **Layer 1**: Check if any expert has a dominant signal (weight ≥ threshold)
- **Layer 2**: If yes, use that expert's ranking; if no, proceed to full MoE
- **Layer 3**: Combine all expert signals using learned weights

### Training vs. Inference Cascade

**Previous Approach (Inference-only)**:
- Train router normally
- Apply gating at inference time only
- Result: Router never learned to optimize for cascade behavior

**New Approach (Cascade-Aware Training)**:
- Incorporate cascade gating into the training loss
- Router learns to produce decisive weights when appropriate
- Result: Better expert selection and more efficient inference

## Architecture

### New Training Script: `train_router_cascade.py`

Key features:
- `--cascade_threshold`: Dominance threshold (e.g., 0.75)
- `--gating`: Enable/disable cascade gating during training
- `--gating_strength`: Regularization strength for cascade loss

### Cascade Gating Loss

The training loss includes a cascade regularizer:

```python
cascade_loss = cascade_gating_loss(weights, threshold, strength)
total_loss = btl_loss + entropy_loss + cascade_loss
```

The cascade loss:
- Encourages **decisive** weight distributions when max(w) ≥ threshold
- Penalizes high entropy (uniform distributions) when gating should activate
- Allows mixture behavior when no expert is dominant

## Training Multiple Configurations

### Quick Start: Batch Training

Train all 6 configurations at once:

```bash
./train_all_cascade_models.sh
```

This trains:
1. **Baseline** (no gating): Standard MoE router
2. **Threshold 0.70**: Cascade with low threshold (more gating)
3. **Threshold 0.75**: Cascade with medium threshold
4. **Threshold 0.80**: Cascade with high threshold
5. **Threshold 0.85**: Cascade with very high threshold
6. **Threshold 0.90**: Cascade with extreme threshold (rare gating)

### Manual Training

Train individual models:

```bash
# Baseline (no gating)
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/router_no_gating.pt \
  --epochs 20 --lr 5e-4 --seed 42

# Cascade with threshold 0.75
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/router_cascade_0.75.pt \
  --cascade_threshold 0.75 \
  --gating \
  --gating_strength 0.1 \
  --epochs 20 --lr 5e-4 --seed 42
```

## Evaluation

### Comprehensive Cascade Evaluation

Evaluate all trained cascade models:

```bash
python -m src.evaluations.models.MoE_cascade_eval \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --output_dir artifacts/evaluation_results/cascade_training \
  --split test
```

This automatically:
- Finds all cascade models in `artifacts/router/`
- Evaluates each on the test set
- Analyzes expert weight distributions
- Compares performance across thresholds
- Generates comprehensive reports

### What Gets Evaluated

For each model:
1. **Overall Performance**: Accuracy, agreement rates
2. **Weight Statistics**: Mean/std of expert weights
3. **Cascade Behavior**: How often each threshold triggers gating
4. **Expert Selection**: Which expert dominates at each threshold
5. **Category Performance**: Breakdown by query category
6. **Difficulty Analysis**: Performance on easy/medium/hard queries

### Output Files

```
artifacts/evaluation_results/cascade_training/
├── router_no_gating_results.json          # Baseline results
├── router_cascade_0.70_results.json       # Threshold 0.70 results
├── router_cascade_0.75_results.json       # Threshold 0.75 results
├── router_cascade_0.80_results.json       # Threshold 0.80 results
├── router_cascade_0.85_results.json       # Threshold 0.85 results
├── router_cascade_0.90_results.json       # Threshold 0.90 results
├── cascade_training_comparison.csv        # Comparison table
└── CASCADE_TRAINING_SUMMARY.md            # Summary report
```

## Expected Outcomes

### Hypothesis

With cascade-aware training:
- Models learn when to be decisive (high max weight) vs. when to blend
- Expert selection becomes more interpretable
- Inference efficiency improves (early filtering)
- Performance remains competitive or improves

### Questions Answered

1. **Does cascade-aware training change expert distributions?**
   - Compare mean weights across models
   - Check if gamma still dominates

2. **What's the optimal threshold?**
   - Compare accuracy across thresholds
   - Analyze cascade trigger rates

3. **Does gating help or hurt performance?**
   - Compare baseline (no gating) vs. cascade models
   - Check if there's a performance-efficiency trade-off

4. **Are decisions more decisive?**
   - Compare max_weight distributions
   - Check entropy of weight distributions

## Interpreting Results

### Key Metrics

**Performance**:
- `accuracy_no_ties`: Agreement rate excluding ties
- `accuracy_ties_half`: Agreement rate counting ties as 0.5
- `correct`, `incorrect`, `ties`: Raw counts

**Weight Distribution**:
- `mean_alpha/beta/gamma/delta`: Average weight per expert
- `mean_max_weight`: Average of max weight across samples
- `std_weights`: Weight variability

**Cascade Behavior** (at each threshold):
- `dominant_percentage`: % of samples where max(w) ≥ threshold
- `dominant_count`: Number of samples that would use single expert
- `gated_expert_distribution`: Which expert gets selected when gating

### What to Look For

**Good Cascade Model**:
- Maintains or improves accuracy vs. baseline
- Shows meaningful variation in `dominant_percentage` across thresholds
- Has interpretable `gated_expert_distribution` (not always the same expert)
- Balanced `mean_weights` (not over-reliant on one expert)

**Poor Cascade Model**:
- Significantly lower accuracy than baseline
- Always gates (100% dominant) or never gates (0% dominant)
- Always selects the same expert regardless of query
- Extreme weight distributions (e.g., gamma=0.9, others=0.03)

## Advanced Usage

### Custom Thresholds

Evaluate with custom threshold range:

```bash
python -m src.evaluations.models.MoE_cascade_eval \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --thresholds 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 \
  --split test
```

### Specific Models

Evaluate only specific models:

```bash
python -m src.evaluations.models.MoE_cascade_eval \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --models artifacts/router/router_cascade_0.75.pt \
           artifacts/router/router_no_gating.pt \
  --split test
```

### Validation Split

Evaluate on validation set instead:

```bash
python -m src.evaluations.models.MoE_cascade_eval \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --split val
```

## Comparison with Previous Approach

### Old Cascade Evaluation (Inference-only)

Located in: `src/evaluations/cascade_threshold_eval.py`

- Loads a **single** pre-trained router
- Applies different thresholds at **inference time only**
- Router was **not trained** with cascade awareness
- Result: Always showed gamma dominance (router never learned to adapt)

### New Cascade Training (Training-aware)

Located in:
- Training: `src/cli/train_router_cascade.py`
- Evaluation: `src/evaluations/models/MoE_cascade_eval.py`

- Trains **multiple** routers with different cascade configurations
- Incorporates cascade gating **during training**
- Router **learns** when to be decisive
- Result: Should show more adaptive expert selection

## Troubleshooting

### Models Not Found

**Error**: "No cascade models found"

**Solution**: Train models first:
```bash
./train_all_cascade_models.sh
```

### State Dict Mismatch

**Error**: "Error(s) in loading state_dict"

**Solution**: The evaluation script handles this automatically with `strict=False` loading. If issues persist, check that model architecture matches (input_dim, dz_dim).

### Low Accuracy

**Issue**: Cascade models perform worse than baseline

**Possible causes**:
- `--gating_strength` too high (over-regularization)
- Threshold too extreme (0.9+ or <0.6)
- Insufficient training epochs

**Solution**: Try:
- Lower gating_strength (0.05 instead of 0.1)
- Use moderate thresholds (0.7-0.8)
- Increase epochs (30 instead of 20)

### Always One Expert

**Issue**: One expert (e.g., gamma) dominates all queries

**Analysis needed**:
- Check if this happens in baseline too (might be data-driven)
- Look at per-category breakdown (maybe gamma is genuinely best)
- Check training logs (did weights diverge early?)

**Next steps**:
- Try stronger entropy regularization (`--ent_lambda 1e-2`)
- Add expert-specific constraints
- Analyze which queries benefit from mixture vs. single expert

## Next Steps

After training and evaluation:

1. **Review Summary Report**: `CASCADE_TRAINING_SUMMARY.md`
2. **Compare Metrics**: Check CSV comparison table
3. **Select Best Model**: Based on accuracy and cascade behavior
4. **Analyze Patterns**: Which categories benefit from cascade?
5. **Production Deployment**: Use best model with optimal threshold

## References

- Original Router Training: `src/cli/train_router.py`
- Original MoE Evaluation: `src/evaluations/models/MoE_eval.py`
- Old Cascade Analysis: `src/evaluations/cascade_threshold_eval.py`
- Architecture Documentation: `architecture.md`

