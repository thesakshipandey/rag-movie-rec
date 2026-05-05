# Movie Pair Dataset Generator - Quick Start

## Location
```
/mnt/nas/sakshipandey/main/projects/rag-movie-rec/movie_pair_dataset/
```

## Quick Test (3 prompts)

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec/movie_pair_dataset
python3 main.py --prompt_ids "0001,0002,0003"
```

Expected output:
- **Time**: ~70 seconds
- **Success Rate**: 100%
- **Output**: 27 pairwise comparisons (9 per prompt)
- **Files**: `results/0001.json`, `results/0002.json`, `results/0003.json`, `results/all_pairwise_comparisons.json`

## Run All 1,000 Prompts

```bash
python3 main.py
```

Expected:
- **Time**: ~2.5 hours
- **Cost**: ~$5.00
- **Output**: 9,000 pairwise comparisons

## Check Results

```bash
# View consolidated output
cat results/all_pairwise_comparisons.json | python3 -m json.tool | head -50

# View specific prompt
cat results/0001.json | python3 -m json.tool

# Count successful prompts
ls results/*.json | grep -v "all_pairwise" | wc -l
```

## Troubleshooting

### Missing Dependencies
```bash
pip3 install --break-system-packages pandas requests tqdm
```

### Check Progress (while running)
```bash
# Count completed prompts
ls results/*.json | grep -v "all_pairwise" | wc -l

# View recent debug output
ls -lt debug/ | head -5
tail debug/batch_*.raw.txt
```

## Output Structure

Each prompt generates 9 pairs:

### SET 1: HARD (Difficult Trade-offs)
1. Top 2 Best Matches
2. Relevance vs. Quality
3. Opposite Vibes

### SET 2: MEDIUM (Moderate Trade-offs)
4. Exact vs. Metaphor
5. Good vs. Median
6. History Logic (conditional on combo_type)

### SET 3: EASY (Clear Winners)
7. Perfect Match vs. Mismatch
8. Masterpiece vs. Flop
9. Specific vs. Cliche

## Key Files

- **[config.py](config.py)** - API keys, paths, batch sizes
- **[main.py](main.py)** - CLI entry point
- **[README.md](README.md)** - Complete documentation
