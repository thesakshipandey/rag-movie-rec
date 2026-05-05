# Cascade Training - Quick Start

## TL;DR

Train and evaluate cascade-aware MoE routers in one command:

```bash
./run_cascade_pipeline.sh
```

That's it! This will:
1. Train 6 models (baseline + 5 cascade thresholds)
2. Evaluate all models on test set
3. Generate comparison reports

Time required: ~20-35 minutes

## What You Get

After the pipeline completes:

### Models (in `artifacts/router/`)
- `router_no_gating.pt` - Baseline
- `router_cascade_0.70.pt` - Low threshold
- `router_cascade_0.75.pt` - Medium threshold
- `router_cascade_0.80.pt` - High threshold
- `router_cascade_0.85.pt` - Very high threshold
- `router_cascade_0.90.pt` - Extreme threshold

### Results (in `artifacts/evaluation_results/cascade_training/`)
- `CASCADE_TRAINING_SUMMARY.md` - **Read this first!**
- `cascade_training_comparison.csv` - Quick comparison table
- Individual JSON results for each model

## What to Look For

### 1. Overall Performance
Check the comparison CSV:
```bash
cat artifacts/evaluation_results/cascade_training/cascade_training_comparison.csv
```

Look for:
- Which model has highest `accuracy_no_ties`
- How accuracy changes with threshold

### 2. Expert Distribution
In the CSV, check columns:
- `mean_alpha`, `mean_beta`, `mean_gamma`, `mean_delta`

Question: Does cascade training reduce gamma dominance?

### 3. Cascade Behavior
In the summary report, look at "Cascade Behavior at Inference" tables.

Question: How often does each threshold trigger single-expert gating?

### 4. Category Performance
In the summary report, check "Performance by Category" sections.

Question: Do different categories benefit from different thresholds?

## Common Scenarios

### Scenario 1: Just show me the results!
```bash
./run_cascade_pipeline.sh
# Wait 20-30 minutes
cat artifacts/evaluation_results/cascade_training/CASCADE_TRAINING_SUMMARY.md
```

### Scenario 2: I only want to train specific thresholds
Edit `train_all_cascade_models.sh` and comment out unwanted models, or train manually:
```bash
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/router_cascade_0.75.pt \
  --cascade_threshold 0.75 --gating --epochs 20 --lr 5e-4 --seed 42
```

### Scenario 3: I want to try different gating strengths
```bash
# Weak gating
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/router_cascade_weak.pt \
  --cascade_threshold 0.75 --gating --gating_strength 0.05 \
  --epochs 20 --lr 5e-4 --seed 42

# Strong gating
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/router_cascade_strong.pt \
  --cascade_threshold 0.75 --gating --gating_strength 0.2 \
  --epochs 20 --lr 5e-4 --seed 42
```

### Scenario 4: I want to evaluate on validation set instead
```bash
python -m src.evaluations.models.MoE_cascade_eval \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --output_dir artifacts/evaluation_results/cascade_training_val \
  --split val
```

## Interpreting Results

### Good Sign ✅
- Accuracy comparable to or better than baseline
- Cascade trigger rates vary meaningfully (not 0% or 100%)
- Different experts selected for different query types
- Higher thresholds → lower cascade rates (as expected)

### Warning Sign ⚠️
- Accuracy significantly worse than baseline
  → Try lower `--gating_strength`
- Always cascades (100%) or never cascades (0%)
  → Adjust threshold range
- One expert always dominates
  → Check if this happens in baseline too (might be data-driven)

### Red Flag 🚩
- Training fails or diverges
  → Lower learning rate
- Models can't load during evaluation
  → Check for version mismatches
- All models perform identically
  → Something went wrong, check logs

## For Presentation

Best tables/figures to show:

1. **Comparison CSV**: Shows all metrics side-by-side
2. **Expert weight distributions**: Show mean_alpha/beta/gamma/delta bars
3. **Cascade trigger rates**: Show how threshold affects gating frequency
4. **Per-category performance**: Show which queries benefit from cascade

Recommendation: Start with the comparison CSV, then dive into the summary report for insights.

## Need More Details?

- **Complete guide**: `CASCADE_TRAINING_GUIDE.md`
- **Implementation details**: `CASCADE_IMPLEMENTATION_SUMMARY.md`
- **Original evaluation framework**: `EVALUATION_SYSTEM_GUIDE.md`

## Troubleshooting

**Q: Script says "python not found"**
A: Activate your Python environment first, or edit the script to use your python path

**Q: Features file not found**
A: Make sure `artifacts/router/features_sum.with_splits.bal.parquet` exists

**Q: Training takes too long**
A: Normal! 6 models × 20 epochs each. Reduce epochs or train fewer models.

**Q: Gamma still dominates everything**
A: Check the baseline first. If gamma dominates there too, it might be genuinely the best expert for your data.

**Q: Models trained but evaluation fails**
A: Check that model files exist in `artifacts/router/` and are valid .pt files

## Time Estimates

- Training one model: ~3-5 minutes (depending on hardware)
- Training all 6 models: ~15-30 minutes
- Evaluation: ~1-2 minutes
- **Total pipeline**: ~20-35 minutes

Hardware matters:
- With GPU: Lower end of estimates
- CPU only: Higher end of estimates

## Next Steps After Results

1. **Identify best model**: Based on accuracy and cascade behavior
2. **Analyze patterns**: Which categories/difficulties benefit from cascade?
3. **Select threshold**: Balance performance vs. efficiency
4. **Deploy**: Use selected model with chosen threshold
5. **Present**: Use comparison table and key insights

---

**Ready?** Run the pipeline:
```bash
./run_cascade_pipeline.sh
```

