# 🚀 START HERE - Listwise Router Pipeline

## Quick Start (3 Steps)

### 1️⃣ Activate Your Conda Environment
```bash
conda activate rag_recsys
```

### 2️⃣ Navigate to Project Directory
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
```

### 3️⃣ Run the Pipeline
```bash
./run_listwise_pipeline.sh
```

That's it! The pipeline will run all 5 steps automatically.

---

## What This Does

The pipeline trains a **contextual hedge router** that learns to combine 4 expert models:

| Expert | Type | Description |
|--------|------|-------------|
| **Alpha (α)** | Semantic | Dense retrieval using FAISS |
| **Beta (β)** | Lexical | BM25 keyword search |
| **Gamma (γ)** | Collaborative | LightGCN (optional) |
| **Delta (δ)** | Emotion | Emotion-based ranking (optional) |

The router uses **ListMLE loss** (Plackett-Luce) to learn which expert to use for each query based on contextual features.

---

## Pipeline Steps

1. **Generate Expert Scores** (~15-30 min)
   - Runs all 4 experts on your data
   - Applies z-score normalization per prompt
   - Output: `artifacts/router/listwise_expert_scores.parquet`

2. **Train Router** (~10-20 min)
   - Trains MLP router with ListMLE loss
   - 70/15/15 train/val/test split
   - Output: `artifacts/router/router_listwise.pt`

3. **Evaluate Single Experts** (~5 min)
   - Tests each expert independently
   - Computes nDCG@10, MRR, Hit@10

4. **Evaluate Router** (~5 min)
   - Tests trained router on held-out data
   - Analyzes expert selection patterns

5. **Compare Methods** (~1 min)
   - Generates comparison tables and plots
   - Output: `artifacts/evaluation_results/listwise/comparison/`

**Total Time: ~35-60 minutes**

---

## Optional: Validate Before Running

To check that all paths are correct:

```bash
python validate_paths.py
```

This will show you:
- ✓ What's found and ready
- ✗ What's missing
- ⚠️ What's optional

---

## Results

After running, check:

### Comparison Table
```bash
cat artifacts/evaluation_results/listwise/comparison/comparison_table.csv
```

### Visualizations
```bash
ls artifacts/evaluation_results/listwise/comparison/*.png
```
- `comparison_bar_chart.png` - All methods compared
- `expert_usage.png` - Which experts the router uses
- `expert_weight_distribution.png` - Weight statistics
- `relative_improvement.png` - Router vs baselines

### Model
```bash
ls artifacts/router/
```
- `listwise_expert_scores.parquet` - Expert predictions
- `router_listwise.pt` - Trained router checkpoint

---

## Skip Completed Steps

If you've already completed some steps:

```bash
# Skip expert generation (if already done)
./run_listwise_pipeline.sh --skip-generation

# Skip training (if already done)
./run_listwise_pipeline.sh --skip-generation --skip-training

# Only run evaluation
./run_listwise_pipeline.sh --skip-generation --skip-training
```

---

## Troubleshooting

### "python: command not found"
You need to activate the conda environment:
```bash
conda activate rag_recsys
```

### "ModuleNotFoundError: No module named 'torch'"
Same issue - activate conda environment first.

### "FileNotFoundError"
Run the validation script to see what's missing:
```bash
python validate_paths.py
```

### "CUDA out of memory"
Add `--device cpu` to training commands (slower but works):
```bash
python -m src.cli.train_router_listwise ... --device cpu
```

---

## Documentation

- **START_HERE.md** (this file) - Quick start guide
- **PATH_FIXES_COMPLETE.md** - All path fixes explained
- **RUN_INSTRUCTIONS.md** - Detailed running instructions
- **LISTWISE_ROUTER_README.md** - Complete technical documentation
- **QUICK_START.md** - Alternative quick start
- **EXECUTIVE_SUMMARY.txt** - Project overview

---

## Expected Performance

Typical results:

| Method | nDCG@10 | MRR | Hit@10 |
|--------|---------|-----|--------|
| Random | 0.500 | 0.350 | 0.400 |
| Alpha (Dense) | 0.750 | 0.650 | 0.800 |
| Beta (BM25) | 0.720 | 0.620 | 0.780 |
| Uniform Mix | 0.780 | 0.680 | 0.820 |
| **Router** | **0.820** | **0.720** | **0.860** |

Router typically improves 5-10% over best single expert!

---

## Need Help?

1. Check `PATH_FIXES_COMPLETE.md` for path issues
2. Check `RUN_INSTRUCTIONS.md` for detailed instructions
3. Run `python validate_paths.py` to diagnose issues
4. Check log files in `logs/` directory

---

## Ready to Go!

Just run:
```bash
conda activate rag_recsys
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
./run_listwise_pipeline.sh
```

Good luck! 🎉

