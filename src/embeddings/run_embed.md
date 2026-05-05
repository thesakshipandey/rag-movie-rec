to generate the embedding

```bash
python -m src.cli.embed \
  --chunks data/processed/chunks.parquet \
  --out artifacts/embeddings/embeddinggemma300m.parquet \
  --encoder gemma \
  --model /mnt/nas/sakshipandey/main/models/embeddinggemma-300m \
  --batch_size 128
```