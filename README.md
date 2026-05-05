# RAG Movie Recommendation System

A comprehensive movie recommendation system using RAG (Retrieval-Augmented Generation) with multiple retrieval experts, routing mechanisms, and evaluation frameworks.

## 📁 Project Structure

```
rag-movie-rec/
├── src/                          # Source code
│   ├── router/                   # Router implementations (MLP, RankNet, Listwise)
│   ├── evaluations/              # Evaluation framework
│   ├── retrieval/                # Retrieval experts (BM25, LightGCN, FAISS)
│   ├── aggregation/              # Result aggregation
│   ├── embeddings/               # Embedding backends
│   ├── emotions/                 # Emotion-based retrieval
│   └── ...
├── scripts/                      # All executable scripts
│   ├── run_cascade_pipeline.sh   # Run cascade system
│   ├── run_listwise_pipeline.sh  # Run listwise training
│   ├── run_router_app.sh         # Run router application
│   ├── train_all_cascade_models.sh
│   └── ...
├── docs/                         # Documentation organized by topic
│   ├── general/                  # Getting started & overview
│   ├── cascade/                  # Cascade system docs
│   ├── router/                   # Router training & usage
│   ├── evaluation/               # Evaluation guides
│   └── fixes_and_summaries/      # Implementation notes
├── configs/                      # Configuration files
├── data/                         # Dataset files
├── artifacts/                    # Generated outputs
│   ├── evaluation_results/       # All evaluation outputs
│   ├── indices/                  # Built indices
│   └── router/                   # Trained models
├── notebooks/                    # Jupyter notebooks
└── tests/                        # Test files

```

## 🚀 Quick Start

1. **First Time Setup**
   - Read: [`docs/general/START_HERE.md`](docs/general/START_HERE.md)
   - Then: [`docs/general/QUICK_START.md`](docs/general/QUICK_START.md)

2. **Run the System**
   ```bash
   # Run cascade pipeline
   bash scripts/run_cascade_pipeline.sh
   
   # Run router application
   bash scripts/run_router_app.sh
   ```

## 📚 Documentation Index

### Getting Started
- [`docs/general/START_HERE.md`](docs/general/START_HERE.md) - **Start here!**
- [`docs/general/QUICK_START.md`](docs/general/QUICK_START.md) - Quick setup guide
- [`docs/general/QUICK_REFERENCE.md`](docs/general/QUICK_REFERENCE.md) - Command reference
- [`docs/general/architecture.md`](docs/general/architecture.md) - System architecture
- [`docs/general/RUN_INSTRUCTIONS.md`](docs/general/RUN_INSTRUCTIONS.md) - Detailed run instructions

### Cascade System
- [`docs/cascade/CASCADE_QUICK_START.md`](docs/cascade/CASCADE_QUICK_START.md) - Get started with cascade
- [`docs/cascade/CASCADE_TRAINING_GUIDE.md`](docs/cascade/CASCADE_TRAINING_GUIDE.md) - Training guide
- [`docs/cascade/CASCADE_IMPLEMENTATION_SUMMARY.md`](docs/cascade/CASCADE_IMPLEMENTATION_SUMMARY.md) - Implementation details
- [`docs/cascade/CASCADE_FILES_INDEX.md`](docs/cascade/CASCADE_FILES_INDEX.md) - File reference

### Router System (MLP, RankNet, Listwise)
- [`docs/router/ROUTER_QUICKSTART.md`](docs/router/ROUTER_QUICKSTART.md) - Router quick start
- [`docs/router/ROUTER_APP_README.md`](docs/router/ROUTER_APP_README.md) - Router application
- [`docs/router/LISTWISE_ROUTER_README.md`](docs/router/LISTWISE_ROUTER_README.md) - Listwise training
- [`docs/router/RUN_ROUTER_TRAINING.sh`](docs/router/RUN_ROUTER_TRAINING.sh) - Training script

### Evaluation
- [`docs/evaluation/QUICKSTART_EVALUATION.md`](docs/evaluation/QUICKSTART_EVALUATION.md) - Evaluation quick start
- [`docs/evaluation/EVALUATION_SYSTEM_GUIDE.md`](docs/evaluation/EVALUATION_SYSTEM_GUIDE.md) - Complete guide
- [`docs/evaluation/RUN_EVALUATION_INSTRUCTIONS.md`](docs/evaluation/RUN_EVALUATION_INSTRUCTIONS.md) - Run instructions
- [`docs/evaluation/EXAMPLE_RESULTS_PREVIEW.md`](docs/evaluation/EXAMPLE_RESULTS_PREVIEW.md) - Example results

### Implementation Notes & Fixes
- [`docs/fixes_and_summaries/`](docs/fixes_and_summaries/) - All implementation summaries and fixes

## 🎯 Key Features

1. **Multiple Retrieval Experts**
   - BM25 (keyword-based)
   - LightGCN (collaborative filtering)
   - FAISS (semantic search with Gemma/Qwen embeddings)
   - Emotion-based retrieval

2. **Router Systems**
   - **MLP Router** - Multi-layer perceptron
   - **RankNet** - Pairwise ranking
   - **Listwise Router** - Listwise learning-to-rank

3. **Cascade System**
   - Sequential expert invocation
   - Confidence-based routing
   - Efficient inference

4. **Comprehensive Evaluation**
   - Multiple metrics (NDCG, Precision, Recall, MRR)
   - Expert analysis
   - Comparative benchmarking

## 🛠️ Available Scripts

All scripts are in the `scripts/` folder:

```bash
# Training
scripts/train_all_cascade_models.sh      # Train all cascade models
scripts/train_roberta_plutchik.py        # Train emotion classifier

# Running Systems
scripts/run_cascade_pipeline.sh          # Run cascade system
scripts/run_listwise_pipeline.sh         # Run listwise training
scripts/run_router_app.sh                # Run router app
scripts/rag_recsys.sh                    # Main RAG system

# Evaluation
scripts/run_comprehensive_eval.sh        # Comprehensive evaluation

# Utilities
scripts/validate_paths.py                # Validate file paths
scripts/predict_roberta_plutchik.py      # Predict emotions
scripts/Makefile                         # Make commands
```

## 📦 Dependencies

Install requirements:
```bash
pip install -r requirements.txt
```

## 🔍 Need Help?

- General questions: See [`docs/general/help.md`](docs/general/help.md)
- Issues with paths: See [`docs/fixes_and_summaries/PATH_FIXES_COMPLETE.md`](docs/fixes_and_summaries/PATH_FIXES_COMPLETE.md)
- Implementation details: See [`docs/fixes_and_summaries/IMPLEMENTATION_SUMMARY.md`](docs/fixes_and_summaries/IMPLEMENTATION_SUMMARY.md)

## 📊 Results

Evaluation results are stored in `artifacts/evaluation_results/`:
- Cascade results: `artifacts/evaluation_results/cascade_analysis/`
- Listwise results: `artifacts/evaluation_results/listwise/`
- RankNet results: `artifacts/evaluation_results/ranknet/`
- Expert comparisons: `artifacts/evaluation_results/experts/`

---

**Note**: This is an active research project. Documentation is continuously updated.



