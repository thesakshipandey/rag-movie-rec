# Router-Based RAG Movie Recommender UI

A Streamlit web interface for the 4-expert RAG movie recommendation system with trained MLP router.

## Features

### 4-Expert Retrieval System
1. **Dense (FAISS)**: Semantic search using Qwen3-8B embeddings
2. **BM25**: Lexical keyword matching  
3. **LightGCN**: Collaborative filtering based on user preferences
4. **Emotion**: Affective matching using Plutchik's 8 emotions

### Learned Router
- Automatically determines optimal expert weights for each query
- Trained MLP that predicts weights [α, β, γ, δ] based on query characteristics
- Adaptive fusion based on query type (plot-based, mood-based, title-based, etc.)

### UI Capabilities
- ✅ Interactive query input with real-time search
- ✅ Top-K movie recommendations with rich metadata
- ✅ Query emotion analysis (Plutchik's 8 emotions)
- ✅ Router-predicted expert weights visualization
- ✅ Per-movie score breakdown (raw, z-normalized, contributions)
- ✅ Movie emotion profiles with bar charts
- ✅ Chunk evidence from Dense and BM25 retrievers
- ✅ Filters: language, year range, content type, adult content
- ✅ User personalization via LightGCN
- ✅ Configurable aggregation (sum/max)
- ✅ Debug mode with full result tables and router features

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Indices
Ensure all indices are built:
```bash
# Dense (FAISS) index
python -m src.cli.build_index --indices_dir artifacts/indices/qwen_fullmovie

# BM25 index  
python -m src.cli.build_bm25 --out artifacts/indices/bm25

# Emotion index
python -m src.cli.build_emotion_index --out artifacts/indices/emotion

# LightGCN (if not already present)
python -m src.cli.precompute_lgcn --out artifacts/indices/lightgcn/sim_user_item.npy
```

### 3. Train Router
Train the MLP router on preference data:
```bash
# Build features
python -m src.router.build_router_features \
  --prompts_dir data/prompts \
  --indices_dir artifacts/indices \
  --out artifacts/router/features_sum.parquet \
  --agg_kind sum

# Train router
python -m src.cli.train_router \
  --features artifacts/router/features_sum.parquet \
  --out artifacts/router/router_mlp_sum.pt \
  --epochs 10
```

See [ROUTER_QUICKSTART.md](ROUTER_QUICKSTART.md) for detailed training instructions.

## Usage

### Quick Start
```bash
# Launch the app
./run_router_app.sh
```

Or manually:
```bash
streamlit run src/app/router_app.py --server.port 8501
```

Then open your browser to: **http://localhost:8501**

### Configuration

#### Sidebar Settings
- **Paths**: Configure locations of indices, models, and metadata
- **Router Model**: Select from available trained routers
- **Retrieval Settings**: 
  - Top-K chunks (per retriever): 20-2000 (default: 200)
  - Top-M movies: 5-100 (default: 10)
  - Aggregation: sum or max
- **User Context**: LightGCN user index (0-942)
- **Filters**: Language, year range, type, adult content

#### Query Examples
Try these queries to see different expert behaviors:

**Title-based** (BM25 dominant):
```
Toy Story
The Matrix
```

**Plot-based** (Dense dominant):
```
animated toys that come to life
humans fighting machines in a simulated reality
```

**Mood-based** (Emotion dominant):
```
uplifting feel-good movie
dark psychological thriller
```

**Personalized** (LightGCN contributes):
```
movies I might like
recommend something for me
```

### Understanding the Results

#### Query Analysis
- **Emotion Distribution**: 8 Plutchik emotions inferred from your query
- Bar chart visualization of emotional profile

#### Router Weights
- **α (Dense)**: Weight for semantic search
- **β (BM25)**: Weight for lexical matching
- **γ (LightGCN)**: Weight for collaborative filtering
- **δ (Emotion)**: Weight for affective matching

The router automatically determines these based on query characteristics!

#### Per-Movie Results
Each recommendation shows:
1. **Poster**: Movie poster from TMDb
2. **Metadata**: Title, release date, movieId
3. **Final Score**: Router-fused score
4. **Score Breakdown**:
   - Raw scores from each expert
   - Z-normalized scores (comparable across experts)
   - Weighted contributions (weight × z-score)
5. **Emotion Profile**: Movie's emotion distribution
6. **Chunk Evidence**: Retrieved text chunks with scores

## Architecture

### Data Flow
```
User Query
    ↓
[4 Experts Run in Parallel]
    ├─ Dense (FAISS) → chunks → movie scores
    ├─ BM25 → chunks → movie scores  
    ├─ LightGCN → item scores → movie scores
    └─ Emotion → JS-divergence → movie scores
    ↓
[Z-Score Normalization]
    ↓
[Router Computes Δz Features]
    ↓
[Router MLP Predicts Weights]
    ↓
[Weighted Fusion: Σ (weight_i × z_i)]
    ↓
[Final Ranking: Top-K Movies]
```

### Router Inference
For single-query inference, the router uses **Δz summary features**:
```python
dz_i = max(z_i) - median(z_i)  # discriminative power per expert
dz = [dz_dense, dz_bm25, dz_lgcn, dz_emo]  # [4] input to router
weights = RouterMLP(dz)  # [4] predicted weights
```

Unlike training (pairwise comparisons), inference summarizes the distribution of scores to predict which experts are most useful for the query.

## File Structure
```
src/app/router_app.py          # Main Streamlit application
run_router_app.sh              # Launch script
artifacts/
  indices/
    qwen_fullmovie/            # Dense FAISS index
    bm25/                      # BM25 index
    lightgcn/                  # LightGCN similarity matrix
    emotion/                   # Emotion index
  router/
    router_mlp_sum.pt          # Trained router model
    features_sum.parquet       # Router training features
```

## Troubleshooting

### "No module named 'src'"
Make sure you're using the launch script `./run_router_app.sh` which sets up the PYTHONPATH correctly. If running manually:
```bash
export PYTHONPATH=/mnt/nas/sakshipandey/main/projects/rag-movie-rec:$PYTHONPATH
streamlit run src/app/router_app.py
```

### "Router model not found"
Ensure you've trained the router:
```bash
python -m src.cli.train_router \
  --features artifacts/router/features_sum.parquet \
  --out artifacts/router/router_mlp_sum.pt
```

### "HFValidationError: Repo id must be in the form..."
This occurs when loading the emotion model from a local path with newer HuggingFace transformers. 

**Solution**: By default, the emotion model is disabled (checkbox unchecked). The app will use keyword-based emotion inference instead, which doesn't require loading the RoBERTa model. If you want to use the model-based inference:
1. Check the "Use emotion model (RoBERTa)" checkbox in the sidebar
2. Ensure the model path is correct
3. If loading fails, the app will automatically fall back to keyword-based inference

### "Emotion index not available"
Build the emotion index:
```bash
python -m src.cli.build_emotion_index \
  --movie_emotions data/raw/movie_emotions.json \
  --out artifacts/indices/emotion
```

### "User index out of range"
LightGCN supports users 0-942. Adjust in sidebar or use a different user_idx.

### Slow performance
- Reduce Top-K chunks (try 100 instead of 200)
- Use max aggregation instead of sum
- Ensure indices are properly cached (warm-up query)

## Advanced Usage

### Custom Router Models
The UI automatically detects all `.pt` files in `artifacts/router/`. To use a custom router:
1. Train your router: `python -m src.cli.train_router --out artifacts/router/my_router.pt`
2. Select it from the dropdown in the sidebar

### Comparing Aggregation Methods
Toggle between "sum" and "max" aggregation to see how chunk-to-movie aggregation affects results:
- **sum**: Broader coverage (rewards multiple relevant chunks)
- **max**: Precision-focused (rewards single best chunk)

### Filter Combinations
Combine multiple filters for targeted search:
```
Query: "animated adventure"
Filters:
  - Language: en
  - Year >= 1995
  - Year <= 2010  
  - Type: movie
```

## References
- [Architecture Documentation](architecture.md)
- [Router Training Guide](ROUTER_QUICKSTART.md)
- [Main README](README.md)

## Support
For issues or questions, refer to the full architecture documentation or examine debug output in the UI's expandable debug sections.


