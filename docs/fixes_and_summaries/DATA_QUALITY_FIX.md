# Data Quality Fix - Missing movieId

## Issue

At prompt 649, the script crashed with:
```
KeyError: 'movieId'
```

## Root Cause

Some items in `merged_all.json` are **missing the `movieId` field**.

Example (Prompt 0650, item 4):
```json
{
  "score": 0.85,
  "reason": "Robotic mutants and twisted humor in a gritty animated thriller"
  // ❌ Missing "movieId"
}
```

## Impact

Out of 1000 prompts, at least one has items without `movieId`. This causes the script to crash.

## Fix Applied

Added robust error handling in `generate_expert_scores.py`:

```python
# Filter out invalid items
valid_items = [item for item in ranking_list 
               if 'movieId' in item and 'score' in item]

if len(valid_items) == 0:
    print(f"Warning: Prompt {prompt_id} has no valid items, skipping")
    continue

if len(valid_items) < len(ranking_list):
    print(f"Warning: Prompt {prompt_id} has {len(ranking_list) - len(valid_items)} items without movieId, skipping them")

# Use only valid items
movie_ids = [item['movieId'] for item in valid_items]
gt_scores = [item['score'] for item in valid_items]
```

## Behavior

**Before**: Crash on first invalid item  
**After**: 
- Skip invalid items with a warning
- Continue processing valid items
- Only skip entire prompt if ALL items are invalid

## Re-Run

Now you can re-run the pipeline:

```bash
conda activate rag_recsys
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
./run_listwise_pipeline.sh
```

It will:
1. Show warnings for prompts with missing movieIds
2. Process all valid items
3. Complete successfully

## Data Quality Check

To see how many prompts are affected:

```bash
python3 -c "
import json
data = json.load(open('/mnt/nas/sakshipandey/main/projects/Data/merged_all.json'))
count = 0
for pid, items in data.items():
    if any('movieId' not in item for item in items):
        count += 1
print(f'Prompts with missing movieIds: {count}/{len(data)}')
"
```

## Recommendation

If you want to fix the source data:
1. Identify which prompts have missing movieIds
2. Either remove those items or add the missing movieIds
3. Regenerate `merged_all.json`

For now, the script will work around this issue automatically.

