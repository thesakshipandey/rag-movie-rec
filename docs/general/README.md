# RAG Movie Recommender (Skeleton)

This repository contains a **clean scaffold** for building a Retrieval-Augmented Generation (RAG) movie recommender from a single dataset (`item_text.csv`). 

## Goals
- Ingest `item_text.csv` (overview + plot + metadata)
- Produce ~2–3 **semantic chunks per movie**
- Add embeddings and a fast retrieval index
- Later: re-ranking + grounded rationales (RAG)

## Project Structure
- `data/` 
- `artifacts/` 
- `docs/`
- `src/` 
- `notebooks/` 

## Dataset
`data/raw/item_text.csv` with columns:
- `movieId,iid,title,TMDbID,original_title,release_date,type,adult,original_language,production_countries,spoken_languages,overview,plot`

### Chunking plan
- **2–3 chunks per movie**:
  - `overview` → 1 chunk
  - `plot` → 1–2 chunks (depending on length)
- Target ~**512 tokens** per chunk with ~**64 tokens** overlap.
- Chunk IDs: `{movieId}:{section_type}:{part_index}` (e.g., `42:plot:1`)

### Embeddings (to be decided)
- **all-MiniLM-L6-v2** or **EmbeddingGemma** (A/B later)
- L2-normalized vectors for cosine searches

## Environment

```bash
python -m venv .venv
source .venv/bin/activate # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env
```


## Roadmap
1. **Data contract**: finalize columns and null-handling (`docs/data_contract.md`)
2. **Chunk spec**: confirm sizes, overlap, and ID scheme
3. **Embedding choice**: MiniLM vs EmbeddingGemma
4. **Indexing**: FAISS/HNSW params, metadata map
5. **Hybrid retrieval**: (optional) add BM25
6. **Re-ranking** (later): pooling strategy and features
7. **RAG** (later): grounded rationale generation
8. **Evaluation**: nDCG@K, Recall@K, constraint checks

## Contributing
- Open an issue with context, then PR into `main`.
- Keep large data out of git; use `data/` folders and `.gitkeep`.

## License
TBD (MIT/Apache-2.0 recommended)
