# Cascade Training Files Index

This document lists all files created for the cascade-aware training system.

## Core Implementation Files

### Training
| File | Purpose | Usage |
|------|---------|-------|
| `src/cli/train_router_cascade.py` | New training script with cascade support | `python -m src.cli.train_router_cascade --features ... --cascade_threshold 0.75 --gating` |

### Evaluation
| File | Purpose | Usage |
|------|---------|-------|
| `src/evaluations/models/MoE_cascade_eval.py` | Evaluation script for cascade models | `python -m src.evaluations.models.MoE_cascade_eval --features ... --output_dir ...` |

## Automation Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `train_all_cascade_models.sh` | Batch train all 6 configurations | `./train_all_cascade_models.sh` |
| `run_cascade_pipeline.sh` | Complete train + eval pipeline | `./run_cascade_pipeline.sh` |

## Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `CASCADE_QUICK_START.md` | **Start here!** Quick reference | Users who want to run immediately |
| `CASCADE_TRAINING_GUIDE.md` | Comprehensive guide | Users who want to understand details |
| `CASCADE_IMPLEMENTATION_SUMMARY.md` | Technical implementation details | Developers and researchers |
| `CASCADE_FILES_INDEX.md` | This file - index of all files | Reference |

## What Each File Does

### `train_router_cascade.py`
- **NEW** training script (doesn't modify original)
- Adds `--cascade_threshold` and `--gating` parameters
- Implements cascade gating loss during training
- Saves models with metadata

### `MoE_cascade_eval.py`
- **NEW** evaluation script (doesn't modify original)
- Evaluates multiple cascade models
- Analyzes expert weights and cascade behavior
- Generates comparison tables and reports

### `train_all_cascade_models.sh`
- Trains 6 models automatically:
  1. Baseline (no gating)
  2. Cascade 0.70
  3. Cascade 0.75
  4. Cascade 0.80
  5. Cascade 0.85
  6. Cascade 0.90
- ~15-30 minutes total

### `run_cascade_pipeline.sh`
- Runs complete workflow:
  1. Calls `train_all_cascade_models.sh`
  2. Runs `MoE_cascade_eval.py`
  3. Displays results
- ~20-35 minutes total

### Documentation Files

**CASCADE_QUICK_START.md**: 
- Single command to run everything
- What to look for in results
- Common scenarios
- Quick troubleshooting

**CASCADE_TRAINING_GUIDE.md**:
- Detailed explanation of cascade concepts
- Training and evaluation procedures
- Result interpretation guide
- Advanced usage examples
- Troubleshooting section

**CASCADE_IMPLEMENTATION_SUMMARY.md**:
- What was implemented and why
- Difference from previous approach
- Technical details
- Integration with existing code
- Next steps

## Files Modified

**None!** All new functionality is in new files. Original files remain untouched:
- ✅ `src/cli/train_router.py` - Original, unchanged
- ✅ `src/evaluations/models/MoE_eval.py` - Original, unchanged
- ✅ Existing evaluation system - Fully compatible

## Output Files (Generated)

After running the pipeline, you'll get:

### Models
```
artifacts/router/
├── router_no_gating.pt
├── router_cascade_0.70.pt
├── router_cascade_0.75.pt
├── router_cascade_0.80.pt
├── router_cascade_0.85.pt
└── router_cascade_0.90.pt
```

### Evaluation Results
```
artifacts/evaluation_results/cascade_training/
├── CASCADE_TRAINING_SUMMARY.md          # Main report - READ THIS!
├── cascade_training_comparison.csv       # Quick comparison table
├── router_no_gating_results.json
├── router_cascade_0.70_results.json
├── router_cascade_0.75_results.json
├── router_cascade_0.80_results.json
├── router_cascade_0.85_results.json
└── router_cascade_0.90_results.json
```

## Quick Navigation

**Want to run immediately?**
→ `CASCADE_QUICK_START.md`

**Want to understand cascade training?**
→ `CASCADE_TRAINING_GUIDE.md`

**Want implementation details?**
→ `CASCADE_IMPLEMENTATION_SUMMARY.md`

**Ready to execute?**
→ `./run_cascade_pipeline.sh`

## Relationship to Original Evaluation System

The cascade training system is **complementary** to the original comprehensive evaluation system:

| System | Files | Purpose |
|--------|-------|---------|
| **Original Eval** | `src/evaluations/comprehensive_eval.py`, etc. | Evaluate existing models comprehensively |
| **Cascade Training** | `train_router_cascade.py`, `MoE_cascade_eval.py` | Train and evaluate cascade-aware models |

Both can be used together:
1. Use cascade training to create better models
2. Use comprehensive eval to deeply analyze all aspects

## Command Cheat Sheet

```bash
# Quick start - run everything
./run_cascade_pipeline.sh

# Just training
./train_all_cascade_models.sh

# Just evaluation
python -m src.evaluations.models.MoE_cascade_eval \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --output_dir artifacts/evaluation_results/cascade_training \
  --split test

# Train single model
python -m src.cli.train_router_cascade \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --out artifacts/router/my_cascade.pt \
  --cascade_threshold 0.75 --gating \
  --epochs 20 --lr 5e-4 --seed 42

# View results
cat artifacts/evaluation_results/cascade_training/CASCADE_TRAINING_SUMMARY.md
cat artifacts/evaluation_results/cascade_training/cascade_training_comparison.csv
```

## Support

For questions or issues:
1. Check the relevant documentation file
2. Look at the troubleshooting sections
3. Examine the code comments in the Python files
4. Check training logs in `logs/` directory

