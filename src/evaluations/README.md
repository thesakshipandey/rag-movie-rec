# Comprehensive Evaluation System

Complete evaluation framework for the RAG Movie Recommender system.

## Quick Start

Run the comprehensive evaluation with a single command:

```bash
./run_comprehensive_eval.sh
```

Or with custom paths:

```bash
./run_comprehensive_eval.sh \
  artifacts/router/features_sum.with_splits.bal.parquet \
  artifacts/evaluation_results \
  test
```

## Manual Execution

```bash
python -m src.evaluations.comprehensive_eval \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --output_dir artifacts/evaluation_results \
  --router_dir artifacts/router \
  --split test \
  --models all \
  --prompts_json data/prompts/prompts.json \
  --movie_text artifacts/movies/movie_text.parquet \
  --emotion_index artifacts/indices/emotion/emotion.parquet
```

## What Gets Evaluated

### 1. Individual Expert Systems
- **Alpha (Dense/FAISS)**: Semantic search with Qwen embeddings
- **Beta (BM25)**: Lexical keyword matching
- **Gamma (LightGCN)**: Collaborative filtering
- **Delta (Emotion)**: Affective matching with Plutchik emotions

### 2. MoE Router Models
- `router_mlp_sum.pt`: Sum aggregation
- `router_mlp_attn.pt`: Attention aggregation
- `router_mlp_combo.pt`: Combination features

### 3. RankNet Baselines
- `ranknet_mlp.pt`: MLP-based ranking
- `ranknet_global.pt`: Global MLP
- `ranknet_global_linear.pt`: Linear baseline

### 4. Fine-tuned RoBERTa Emotion Classifier
- Test set accuracy, F1-scores
- Per-class metrics
- Confusion matrix

## Output Structure

```
artifacts/evaluation_results/
├── dataset_analysis/
│   ├── prompts_statistics.json
│   ├── pairs_statistics.json
│   ├── features_statistics.json
│   ├── movies_statistics.json
│   └── dataset_summary.json
├── experts/
│   ├── alpha_metrics.json
│   ├── beta_metrics.json
│   ├── gamma_metrics.json
│   ├── delta_metrics.json
│   └── experts_comparison.csv
├── moe/
│   ├── router_mlp_sum_metrics.json
│   ├── router_mlp_attn_metrics.json
│   ├── router_mlp_combo_metrics.json
│   └── moe_comparison.csv
├── ranknet/
│   ├── ranknet_mlp_metrics.json
│   ├── ranknet_global_metrics.json
│   ├── ranknet_global_linear_metrics.json
│   └── ranknet_comparison.csv
├── emotion_classifier/
│   ├── test_metrics.json
│   ├── confusion_matrix.csv
│   ├── per_class_metrics.csv
│   └── classification_report.txt
├── comparisons/
│   ├── all_models_comparison.csv
│   ├── category_performance.csv
│   ├── difficulty_analysis.csv
│   ├── best_models.json
│   └── comparison_summary.json
├── errors/
│   ├── expert_disagreements.csv
│   ├── failure_patterns_by_category.csv
│   ├── failure_patterns_by_difficulty.csv
│   ├── case_studies.json
│   └── error_analysis_summary.json
├── plots/
│   ├── performance_comparison.png
│   ├── category_heatmap.png
│   ├── difficulty_heatmap.png
│   ├── expert_correlations.png
│   ├── confusion_matrix.png
│   └── (more visualizations)
├── summary_report.md
├── best_results_for_presentation.md
└── all_metrics.json
```

## Key Metrics

### Agreement Metrics
- **agree_no_ties**: Accuracy excluding tie predictions
- **agree_ties_0p5**: Accuracy counting ties as 0.5 credit

### Breakdown Dimensions
- **By Category**: plot_based, mood_based, title_based, multi_genre, etc.
- **By Difficulty**: easy, medium, hard

### Model Comparison
- Overall ranking
- Category-wise best models
- Difficulty-wise best models
- Most robust model (lowest variance)

## Modules

### 1. `comprehensive_eval.py`
Main orchestrator that runs all evaluations in sequence.

### 2. `dataset_analysis.py`
Analyzes prompt, pair, feature, and movie distributions.

### 3. `comparison_analysis.py`
Compares models across all metrics and identifies best performers.

### 4. `error_analysis.py`
Identifies failure patterns and generates case studies.

### 5. `visualizations.py`
Generates publication-ready plots and charts.

### 6. `generate_report.py`
Creates markdown reports with executive summaries.

### 7. `emotion_classifier_eval.py`
Evaluates fine-tuned RoBERTa emotion classifier.

## Individual Module Usage

### Dataset Analysis Only
```bash
python -m src.evaluations.dataset_analysis \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --output_dir artifacts/evaluation_results/dataset_analysis \
  --prompts_json data/prompts/prompts.json \
  --movie_text artifacts/movies/movie_text.parquet \
  --emotion_index artifacts/indices/emotion/emotion.parquet
```

### Emotion Classifier Only
```bash
python -m src.evaluations.emotion_classifier_eval \
  --model_path /path/to/roberta-emotion-model \
  --test_data data/prompts/prompts.json \
  --output_dir artifacts/evaluation_results/emotion_classifier \
  --split test
```

### Error Analysis Only
```bash
python -m src.evaluations.error_analysis \
  --features artifacts/router/features_sum.with_splits.bal.parquet \
  --output_dir artifacts/evaluation_results/errors \
  --split test \
  --prompts_json data/prompts/prompts.json
```

### Comparison Analysis Only
```bash
python -m src.evaluations.comparison_analysis \
  --results_dir artifacts/evaluation_results \
  --output_dir artifacts/evaluation_results/comparisons \
  --features artifacts/router/features_sum.with_splits.bal.parquet
```

### Report Generation Only
```bash
python -m src.evaluations.generate_report \
  --results_json artifacts/evaluation_results/all_metrics.json \
  --output_dir artifacts/evaluation_results
```

## Interpreting Results

### Best Results for Presentation

The file `best_results_for_presentation.md` contains:
- 🏆 Best overall model
- Top 3 models ranked
- Best model per query category
- Best model per difficulty level
- Key highlights for presentations

### Summary Report

The file `summary_report.md` contains:
- Executive summary
- Detailed results tables
- Key findings
- Recommendations for deployment
- Dataset statistics

### All Metrics JSON

The file `all_metrics.json` contains the complete results in JSON format for further analysis or integration into other tools.

## Customization

### Evaluate Specific Models Only

```bash
# Only experts
python -m src.evaluations.comprehensive_eval \
  --features ... --output_dir ... --models experts

# Only MoE routers
python -m src.evaluations.comprehensive_eval \
  --features ... --output_dir ... --models moe

# Only RankNet baselines
python -m src.evaluations.comprehensive_eval \
  --features ... --output_dir ... --models ranknet
```

### Different Splits

```bash
# Validation set
python -m src.evaluations.comprehensive_eval \
  --features ... --output_dir ... --split val

# All data
python -m src.evaluations.comprehensive_eval \
  --features ... --output_dir ... --split all
```

## Requirements

All required packages are in `requirements.txt`:
- pandas
- numpy
- torch
- scikit-learn
- matplotlib
- seaborn
- transformers (for emotion classifier)

## Troubleshooting

### Missing Features
If you see "Missing column: dz_alpha", ensure your features file has all expert Δz columns.

### Missing Models
If a model file doesn't exist, it will be skipped with a warning.

### Emotion Classifier Errors
The emotion classifier evaluation is optional. If the model path is not provided or invalid, it will be skipped.

## Citation

If you use this evaluation framework, please cite the RAG Movie Recommender project.

## License

See project LICENSE file.

