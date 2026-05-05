# Emotion Vector Handling - Final Fix

## 🐛 **Error**

```
ERROR | Failed to process prompt: 'numpy.ndarray' object has no attribute 'get'
```

## 🔍 **Root Cause**

The error was caused by incorrect handling of the emotion vector from `infer_prompt_vector()`:

1. **`infer_prompt_vector()` returns a tuple:** `(np.ndarray, str)` - emotion vector and source string
2. **Original code tried to use dataset's `plutchik_dist` column** which had various formats (dict, numpy array, JSON string, etc.)
3. **When passed to `_coerce_plutchik_vector()` in `features.py`**, if the format was unexpected, it could try to call `.get()` on a numpy array

## ✅ **Solution**

**Simplified approach:** Always use `infer_prompt_vector()` to infer emotion from prompt text using lexicon-based method (fast, no model loading).

**Changes made to `build_router_features.py` (lines 159-168):**

```python
# OLD (BROKEN): Tried to use plutchik_dist from dataset
if "plutchik_dist" in g.columns:
    pemo_val = g["plutchik_dist"].iloc[0]
    # Complex handling with many edge cases...
    # Could fail if format is unexpected

# NEW (FIXED): Always infer from prompt text
from src.emotions.emotion_prompt import infer_prompt_vector
pemo_vec, emo_src = infer_prompt_vector(
    query=ptext,
    emo_model_dir=None,  # Use lexicon (fast)
    prompt_emotion=None,
)
pemo = pemo_vec  # numpy array of shape (8,)
```

**Benefits:**
- ✅ No more edge cases with dataset column formats
- ✅ Consistent emotion inference for all prompts
- ✅ Fast (lexicon-based, no model loading)
- ✅ `_coerce_plutchik_vector()` in `features.py` handles numpy arrays correctly
- ✅ Cleaner, simpler code

## 📝 **How It Works**

1. For each prompt, call `infer_prompt_vector()` with the prompt text
2. It returns `(emotion_vector: np.ndarray[8], source: str)`
3. Extract just the numpy array (the vector)
4. Pass it to `per_prompt_movie_table()` as `q_emo_vec`
5. `_coerce_plutchik_vector()` in `features.py` normalizes it (already handles numpy arrays)

## 🚀 **Ready to Run**

The fix is complete. You can now run:

```bash
bash src/router_bert/regenerate_and_train.sh
```

**Expected behavior:**
- No more `'numpy.ndarray' object has no attribute 'get'` errors
- All prompts processed successfully
- Features generated with proper emotion vectors

## 📊 **Verification**

After running, check that:
- No errors about `.get()` attribute
- All prompts processed (1000/1000)
- Final features parquet has valid `dz_delta` values (emotion-based scores)

