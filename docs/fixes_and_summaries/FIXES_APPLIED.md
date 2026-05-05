# Fixes Applied to Listwise Router Implementation

## Issue 1: Import Errors in generate_expert_scores.py

**Problem**: Script tried to import non-existent functions `search_dense` and `search_bm25`.

**Root Cause**: The actual function names in `src/retrieval/search.py` are:
- `search_dense_chunks` (not `search_dense`)
- `search_bm25_chunks` (not `search_bm25`)

**Fix Applied**:
1. Updated imports to use correct function names
2. Changed function calls to match actual API:
   - `search_dense_chunks(dense_index, q_emb, top_k=topk_retrieval)`
   - `search_bm25_chunks(bm25_index, prompt_text, top_k=topk_retrieval)`
3. Updated data access pattern - functions return DataFrames, not dicts:
   - Changed from `dense_hits['indices'][0]` to `dense_hits.iterrows()`
   - Access scores via `row['score_dense']` and `row['score_bm25']`
4. Replaced custom embedder with built-in `encode_query()` function

## Issue 2: Python Command Not Found

**Problem**: Shell script used `python` which wasn't available in the environment.

**Fix Applied**:
1. Added `PYTHON_CMD` variable that defaults to `python3`
2. Can be overridden by setting `PYTHON` environment variable
3. All `python -m` calls now use `$PYTHON_CMD -m`

**Usage**:
```bash
# Default (uses python3)
./run_listwise_pipeline.sh

# Or specify Python command
PYTHON=python ./run_listwise_pipeline.sh

# Or if in conda environment
conda activate rag_recsys
PYTHON=python ./run_listwise_pipeline.sh
```

## Files Modified

1. **src/router/generate_expert_scores.py**
   - Fixed imports
   - Updated function calls
   - Changed data access patterns
   
2. **run_listwise_pipeline.sh**
   - Added PYTHON_CMD variable
   - Updated all python calls

## Verification

To verify the fixes work:

```bash
# Test imports
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
python3 -c "from src.retrieval.search import search_dense_chunks, search_bm25_chunks, encode_query; print('OK')"

# Test pipeline (skip training/eval for quick test)
./run_listwise_pipeline.sh --skip-training --skip-eval
```

## Next Steps

The implementation is now compatible with your existing codebase. You can run the full pipeline:

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
./run_listwise_pipeline.sh
```

Or run steps individually as documented in QUICK_START.md.

## Additional Notes

- The implementation integrates cleanly with your existing retrieval infrastructure
- All expert score generation uses your existing indices (FAISS, BM25, LGCN, emotion)
- Z-score normalization and softmax are applied correctly per prompt
- No changes needed to other modules (router, training, evaluation)

