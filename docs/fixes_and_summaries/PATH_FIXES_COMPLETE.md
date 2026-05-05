# Path Fixes Complete ✅

## Summary

All path issues have been resolved! The validation shows:

### ✅ Working Paths
- Data files: `merged_all.json`, `prompts.json` ✓
- FAISS index (qwen): Found in `indices/qwen_fullmovie/` ✓
- BM25 index: Found in `indices/bm25/` ✓
- Emotion index: Found in `indices/emotion/` ✓
- Model: Qwen3-Embedding-8B ✓

### ⚠️ Optional (will use zeros if not found)
- LightGCN: Not found in `indices/lightgcn/` (gamma scores will be zero)

## What Was Fixed

### 1. **Index Subdirectory Paths**
The script now correctly looks for indices in their subdirectories:
- **FAISS**: `indices/qwen_fullmovie/` (or `gemma/` based on encoder)
- **BM25**: `indices/bm25/`
- **LightGCN**: `indices/lightgcn/`
- **Emotion**: `indices/emotion/`

### 2. **Absolute Paths**
All paths are now absolute to avoid directory confusion:
```bash
PROJECT_ROOT="/mnt/nas/sakshipandey/main"
DATA_DIR="${PROJECT_ROOT}/projects/Data"
INDICES_DIR="${PROJECT_ROOT}/projects/rag-movie-rec/artifacts/indices"
```

### 3. **Python Command Detection**
Script auto-detects whether to use `python` or `python3`

### 4. **Error Messages**
Better error messages that show:
- What path was expected
- What's actually available
- Clear indication of what's missing

## How to Run

### Step 1: Activate Conda Environment
```bash
conda activate rag_recsys
```

### Step 2: Validate Paths (Optional but Recommended)
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
python validate_paths.py
```

### Step 3: Run the Pipeline
```bash
./run_listwise_pipeline.sh
```

Or run steps manually:
```bash
# Generate expert scores
python -m src.router.generate_expert_scores \
    --data_dir /mnt/nas/sakshipandey/main/projects/Data \
    --indices_dir /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/indices \
    --out artifacts/router/listwise_expert_scores.parquet

# Train router
python -m src.cli.train_router_listwise \
    --expert_scores artifacts/router/listwise_expert_scores.parquet \
    --prompts_path /mnt/nas/sakshipandey/main/projects/Data/prompts.json \
    --out artifacts/router/router_listwise.pt

# Evaluate
python -m src.evaluations.eval_single_experts \
    --expert_scores artifacts/router/listwise_expert_scores.parquet \
    --prompts_path /mnt/nas/sakshipandey/main/projects/Data/prompts.json \
    --out artifacts/evaluation_results/listwise/single_expert_metrics.json

python -m src.evaluations.eval_router_listwise \
    --expert_scores artifacts/router/listwise_expert_scores.parquet \
    --prompts_path /mnt/nas/sakshipandey/main/projects/Data/prompts.json \
    --router_checkpoint artifacts/router/router_listwise.pt \
    --out artifacts/evaluation_results/listwise/router_metrics.json

python -m src.evaluations.compare_methods \
    --single_expert_metrics artifacts/evaluation_results/listwise/single_expert_metrics.json \
    --router_metrics artifacts/evaluation_results/listwise/router_metrics.json \
    --out_dir artifacts/evaluation_results/listwise/comparison
```

## File Structure

Your project structure:
```
/mnt/nas/sakshipandey/main/
├── projects/
│   ├── Data/
│   │   ├── merged_all.json         ✓ Found
│   │   └── prompts.json            ✓ Found
│   └── rag-movie-rec/
│       ├── artifacts/
│       │   ├── indices/
│       │   │   ├── qwen_fullmovie/  ✓ Found
│       │   │   │   ├── faiss.index
│       │   │   │   └── meta.parquet
│       │   │   ├── bm25/            ✓ Found
│       │   │   │   ├── bm25.pkl
│       │   │   │   └── meta.parquet
│       │   │   ├── emotion/         ✓ Found
│       │   │   └── lightgcn/        ⚠️ Optional
│       │   └── router/              ✓ Created
│       └── src/
└── models/
    └── Qwen3-Embedding-8B/          ✓ Found
```

## Expected Behavior

### With All Indices
If you have all 4 experts (alpha/beta/gamma/delta), you'll get:
- Alpha scores from FAISS semantic search
- Beta scores from BM25 lexical search  
- Gamma scores from LightGCN collaborative filtering
- Delta scores from emotion-based ranking

### Without Optional Indices
If LightGCN or Emotion indices are missing:
- The script will print a warning
- Missing expert scores will be set to zero
- The router will learn to ignore that expert
- All other experts will work normally

## Troubleshooting

### "python: command not found"
→ Activate conda environment: `conda activate rag_recsys`

### "ModuleNotFoundError: No module named 'torch'"
→ You're not in the conda environment

### "FileNotFoundError: faiss.index"
→ Run `python validate_paths.py` to check what's missing

### Expert scores are all zero
→ Check that indices loaded correctly (look for "Loaded from..." messages)

## Next Steps

You're ready to run! The pipeline will:
1. Generate expert scores (~15-30 min)
2. Train router (~10-20 min)
3. Evaluate all methods (~10 min)
4. Generate comparison plots (~1 min)

**Total time: ~35-60 minutes**

Results will be in:
- `artifacts/router/listwise_expert_scores.parquet`
- `artifacts/router/router_listwise.pt`
- `artifacts/evaluation_results/listwise/comparison/`

Good luck! 🚀

