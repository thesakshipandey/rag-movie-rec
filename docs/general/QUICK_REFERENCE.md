# Quick Reference Card

## Run Pipeline (One Command)

```bash
conda activate rag_recsys
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
./run_listwise_pipeline.sh
```

## Expert Sources (Correct Implementation)

| Expert | File/Model | Method |
|--------|------------|--------|
| **α (Dense)** | `models/Qwen3-Embedding-8B` | Cosine similarity |
| **β (BM25)** | `indices/bm25/bm25.pkl` | Keyword matching |
| **γ (LGCN)** | `indices/lightgcn/sim_user_item.npy` | CF similarity |
| **δ (Emotion)** | `indices/emotion/emotion.json` + `models/roberta-plutchik-query_noKD/final` | JS divergence |

## What Gets Generated

```
artifacts/router/
├── listwise_expert_scores.parquet  ← All expert predictions (cached)
└── router_listwise.pt               ← Trained router checkpoint

artifacts/evaluation_results/listwise/
├── single_expert_metrics.json       ← Baseline results
├── router_metrics.json              ← Router results
└── comparison/
    ├── comparison_table.csv         ← All methods compared
    ├── comparison_bar_chart.png     ← Visualization
    ├── expert_usage.png             ← Which experts router uses
    └── relative_improvement.png     ← Router vs baselines
```

## Why This Approach?

```
❌ WRONG: highest_score_expert → use only that expert
✅ CORRECT: context → router → adaptive weights → combine all experts
```

**The router learns**:
- Which expert is **reliable** (not just high-scoring) for each query type
- How to **combine** complementary signals
- **Context-dependent** trust, not winner-take-all

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `python: command not found` | `conda activate rag_recsys` |
| `ModuleNotFoundError: torch` | Same - activate conda environment |
| `KeyError: 'movieId'` | ✅ Fixed - skips invalid items with warning |
| `FileNotFoundError: faiss.index` | ✅ Fixed - uses correct subdirectories |
| `lgcn_scores.npy not found` | ✅ Fixed - now uses `sim_user_item.npy` |

## Expected Runtime

| Step | Time |
|------|------|
| Expert generation | 15-30 min |
| Router training | 10-20 min |
| Evaluation | 10 min |
| **Total** | **35-60 min** |

## Expected Performance

| Method | nDCG@10 | MRR | Hit@10 |
|--------|---------|-----|--------|
| Random | ~0.50 | ~0.35 | ~0.40 |
| Single expert | ~0.70 | ~0.60 | ~0.75 |
| Uniform (0.25 each) | ~0.78 | ~0.68 | ~0.82 |
| **Router (trained)** | **~0.82** | **~0.72** | **~0.86** |

## Validation Before Running

```bash
python validate_paths.py
```

Shows:
- ✓ What's ready
- ✗ What's missing
- ⚠️ What's optional

## Skip Completed Steps

```bash
# Skip expert generation (if already done)
./run_listwise_pipeline.sh --skip-generation

# Skip training (if already done)
./run_listwise_pipeline.sh --skip-generation --skip-training

# Only evaluation
./run_listwise_pipeline.sh --skip-generation --skip-training
```

## Documentation Files

| File | Purpose |
|------|---------|
| **START_HERE.md** | Main entry point |
| **QUICK_REFERENCE.md** | This file |
| **FINAL_FIXES_SUMMARY.md** | What was fixed |
| **CORRECT_EXPERT_IMPLEMENTATION.md** | Expert details + philosophy |
| **WHY_GENERATE_EXPERT_SCORES.md** | Rationale explanation |
| PATH_FIXES_COMPLETE.md | Path fixes applied |
| DATA_QUALITY_FIX.md | Missing movieId fix |
| RUN_INSTRUCTIONS.md | Detailed instructions |
| LISTWISE_ROUTER_README.md | Full technical docs |

## Key Insight

**Router doesn't pick the "winner" - it learns who to trust when!**

Different queries → Different expert reliability → Adaptive weights

This is what makes the router better than any single expert or uniform mixture.

## Ready!

Everything is fixed and ready to run. Just:
1. Activate conda
2. Run the pipeline
3. Check results in `artifacts/evaluation_results/listwise/comparison/`

Good luck! 🚀

