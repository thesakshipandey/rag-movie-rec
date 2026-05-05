to run the chunking

```bash
python -m src.cli.chunk \
  --inp data/raw/item_text.csv \
  --out_jsonl data/processed/chunks.jsonl \
  --out_parquet data/processed/chunks.parquet \
  --summary data/processed/chunks_summary.json \
  --target_words 350 --overlap_words 45 \
  --cap_overview_parts -1 --cap_plot_parts -1 \
  --stream
```

```bash
python -m src.cli.chunk \
  --inp data/raw/item_text.csv \
  --out_parquet data/processed/chunks_full_movie.parquet \
  --out_jsonl data/processed/chunks_full_movie.jsonl \
  --summary data/processed/chunks_summary.json \
  --mode full \
  --stream \
  --min_words 0
```