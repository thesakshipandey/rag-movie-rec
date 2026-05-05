# BERT-based Expert Router

Production-ready training and evaluation pipeline for a text-conditioned expert router that outputs 4 weights [α, β, γ, δ] from a prompt using a BERT encoder with per-expert attention heads.

## Architecture

```
Text → BERT/DistilBERT → H [T, d]
     → 4 expert queries Q [4, d] (learnable)
     → Per-expert attention: scores_k = H @ q_k / √d → softmax → context_k
     → Stack contexts C [4, d]
     → Linear scorer → 4 logits → softmax → weights w [4]
```

## Installation

### Prerequisites

Ensure you have PyTorch installed. If not, install it:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Required Packages

All dependencies should already be in `requirements.txt`:
- torch
- transformers
- pandas
- numpy
- matplotlib
- tqdm
- scikit-learn

## Quick Start

### Training

Train a router from scratch:

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

python -m src.router_bert.train_router \
    --parquet artifacts/router/features_sum.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --encoder distilbert-base-uncased \
    --epochs 5 \
    --lr 2e-5 \
    --batch_prompts 16 \
    --freeze_encoder
```

**Key arguments:**
- `--freeze_encoder`: Freeze encoder weights (default, faster training)
- `--unfreeze`: Fine-tune encoder (slower but potentially better)
- `--encoder`: HuggingFace model name (default: distilbert-base-uncased)
- `--epochs`: Number of training epochs
- `--batch_prompts`: Batch size in number of prompts (not pairs)
- `--temperature`: Temperature for Bradley-Terry loss (default: 1.0)
- `--entropy_min`: Minimum entropy for regularization (default: 0.6)
- `--entropy_lambda`: Weight for entropy penalty (default: 1e-3)

### Evaluation

Evaluate a trained model:

```bash
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/run_<timestamp>/best_model \
    --parquet artifacts/router/features_sum.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --split test \
    --tol 0.05
```

**Outputs:**
- `metrics_overall.json`: Overall agreement metrics
- `metrics_by_difficulty.csv`: Metrics grouped by difficulty
- `metrics_by_category.csv`: Metrics grouped by category
- `weights_histogram.png`: Distribution of predicted weights
- `attn_examples/`: Attention visualization for sample prompts
- `predictions.csv`: All predictions with scores and weights

## Data Format

### Input Parquet (`features_sum.with_splits.bal.parquet`)

Required columns:
- `prompt_id`: Unique identifier for each prompt
- `pair_id`: Unique identifier for each pair
- `y`: Binary label (0 or 1, where 1 means A > B)
- `dz_alpha`, `dz_beta`, `dz_gamma`, `dz_delta`: Delta features per expert
- `split`: Train/val/test split (optional, will be created if missing)
- `difficulty`: Difficulty level (optional, for grouped metrics)
- `category`: Category label (optional, for grouped metrics)

### Prompts Parquet (`prompt_text.parquet`)

Required columns:
- `prompt_id`: Matches the prompt_id in features parquet
- `text`: Prompt text string

## Training Objective

The model is trained using Bradley-Terry pairwise loss:

```
L_BTL = mean_i log(1 + exp(-y'_i * s_i / T))
```

where:
- `s_i = w · dz_i` (dot product of weights and delta features)
- `y'_i = 2*y_i - 1` (convert {0,1} to {-1,+1})
- `T` is temperature (default 1.0)

Optional entropy regularization encourages diverse weight distributions:

```
L_entropy = max(0, entropy_min - H(w))
```

## Evaluation Metrics

### Agreement Metrics

1. **agree_no_ties**: Fraction where sign(score) matches true label
   - Ties (score=0) are counted as wrong

2. **agree_ties_0p5**: Partial credit for ties
   - If |score| ≤ tol: 0.5 credit
   - Otherwise: 1.0 if correct, 0.0 if wrong

### Grouped Metrics

Metrics are also computed by:
- **Difficulty**: If `difficulty` column exists
- **Category**: If `category` column exists

## Model Details

### Frozen vs Fine-tuned Encoder

**Frozen (default, `--freeze_encoder`):**
- Only trains expert queries and scorer head
- Faster training (~2-3x speedup)
- Lower memory usage
- Good for most cases

**Fine-tuned (`--unfreeze`):**
- Trains entire BERT encoder
- Slower but potentially better performance
- Higher memory usage
- Recommended for domain-specific text

### Attention Mechanism

Each of the 4 experts has a learned query vector. For each prompt:
1. Encode text to get token embeddings H [T, d]
2. Compute attention scores per expert: `score_k = H @ q_k / √d`
3. Apply softmax over tokens to get attention weights
4. Compute context vector: `c_k = Σ attn_k[j] * H[j]`
5. Score contexts to get logits, then softmax to weights

This allows each expert to focus on different parts of the prompt.

## Output Structure

```
artifacts/router/bert_router/
└── run_<timestamp>/
    ├── config.json              # Training configuration
    ├── training_log.csv         # Per-epoch metrics
    ├── best_model/              # Best checkpoint (by val agreement)
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── tokenizer/
    └── latest_model/            # Latest checkpoint
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer/
```

After evaluation:
```
best_model/
└── eval_test/
    ├── metrics_overall.json
    ├── metrics_by_difficulty.csv
    ├── metrics_by_category.csv
    ├── weights_histogram.png
    ├── predictions.csv
    └── attn_examples/
        ├── example_1.txt
        ├── example_2.txt
        └── ...
```

## HuggingFace Cache

All HuggingFace models and datasets are cached to:
- `/mnt/nas/sakshipandey/main/models/transformers`
- `/mnt/nas/sakshipandey/main/models/datasets`

This is configured via environment variables in both training and evaluation scripts.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
--batch_prompts 8
```

Or use a smaller encoder:
```bash
--encoder distilbert-base-uncased  # Smaller than bert-base-uncased
```

### Poor Performance

Try fine-tuning the encoder:
```bash
--unfreeze
```

Adjust temperature:
```bash
--temperature 0.5  # Sharper decisions
```

Increase entropy regularization:
```bash
--entropy_lambda 1e-2  # Encourage more diverse weights
```

## Citation

If you use this router in your research, please cite:

```bibtex
@software{bert_router_2024,
  title={BERT-based Text-Conditioned Expert Router},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

