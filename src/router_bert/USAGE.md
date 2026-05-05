# BERT Router - Usage Guide

This guide provides step-by-step instructions for training and evaluating the BERT-based expert router.

## Prerequisites

1. **Install PyTorch** (if not already installed):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR for CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2. **Install other dependencies**:

```bash
pip install transformers pandas numpy matplotlib tqdm scikit-learn
```

3. **Verify setup**:

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
bash src/router_bert/setup_env.sh
```

## Quick Start

### 1. Test Data Loading

First, verify that the data files are accessible:

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
python3 src/router_bert/test_data_loading.py
```

Expected output:
```
✓ Data loading test PASSED
```

### 2. Train a Model (Frozen Encoder - Fast)

Train with frozen encoder (recommended for first run):

```bash
python -m src.router_bert.train_router \
    --parquet artifacts/router/features_sum.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --encoder distilbert-base-uncased \
    --epochs 5 \
    --lr 2e-5 \
    --batch_prompts 16 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/frozen_run
```

**Training time estimate:**
- With GPU: ~5-10 minutes for 5 epochs
- With CPU: ~30-60 minutes for 5 epochs

### 3. Evaluate the Model

After training completes, evaluate on the test set:

```bash
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/frozen_run/best_model \
    --parquet artifacts/router/features_sum.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --split test \
    --tol 0.05
```

**Outputs:**
- `metrics_overall.json` - Overall agreement metrics
- `metrics_by_difficulty.csv` - Metrics per difficulty level
- `metrics_by_category.csv` - Metrics per category
- `weights_histogram.png` - Weight distribution plots
- `attn_examples/` - Attention visualization examples
- `predictions.csv` - All predictions with scores

## Advanced Usage

### Fine-tune the Encoder

For potentially better performance, fine-tune the entire encoder:

```bash
python -m src.router_bert.train_router \
    --parquet artifacts/router/features_sum.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --encoder distilbert-base-uncased \
    --epochs 5 \
    --lr 1e-5 \
    --batch_prompts 8 \
    --unfreeze \
    --out_dir artifacts/router/bert_router/finetuned_run
```

**Note:** Use lower learning rate (1e-5) and smaller batch size (8) when fine-tuning.

### Use a Different Encoder

Try different BERT variants:

```bash
# Standard BERT (larger, potentially better)
python -m src.router_bert.train_router \
    --encoder bert-base-uncased \
    --batch_prompts 8 \
    ...

# Smaller/faster models
python -m src.router_bert.train_router \
    --encoder distilbert-base-uncased \
    --batch_prompts 16 \
    ...
```

### Adjust Hyperparameters

#### Temperature

Controls sharpness of Bradley-Terry loss:

```bash
--temperature 0.5   # Sharper decisions (more confident)
--temperature 2.0   # Softer decisions (less confident)
```

#### Entropy Regularization

Encourages diverse weight distributions:

```bash
--entropy_min 0.8    # Encourage more uniform weights
--entropy_lambda 1e-2  # Stronger regularization
```

#### Learning Rate

```bash
--lr 5e-5   # Higher learning rate (faster but less stable)
--lr 1e-6   # Lower learning rate (slower but more stable)
```

## Training Output

During training, you'll see output like:

```
Epoch 1/5
--------------------------------------------------------------------------------
Training: 100%|████████████| 394/394 [00:45<00:00, loss: 0.6234, bt: 0.6198, ent: 0.0036]
Evaluating: 100%|████████████| 85/85 [00:08<00:00]

Train Loss: 0.6234 (BT: 0.6198, Ent: 0.0036)
Val Loss: 0.5987
Val Agree (no ties): 0.7234
Val Agree (ties 0.5): 0.7456
New best validation agreement: 0.7234
```

**Metrics explained:**
- **BT (Bradley-Terry)**: Main pairwise loss
- **Ent (Entropy)**: Entropy penalty (should be small)
- **Agree (no ties)**: Hard agreement (higher is better)
- **Agree (ties 0.5)**: Soft agreement with partial credit for ties

## Evaluation Output

After evaluation, you'll see:

```
================================================================================
OVERALL METRICS
================================================================================
n_pairs: 1350.0000
agree_no_ties: 0.7456
agree_ties_0p5: 0.7678
mean_abs_score: 0.0234
std_score: 0.0456

================================================================================
METRICS BY DIFFICULTY
================================================================================
  difficulty  n_pairs  agree_no_ties  agree_ties_0p5
        easy      450          0.823           0.845
      medium      450          0.734           0.756
        hard      450          0.678           0.701

================================================================================
METRICS BY CATEGORY
================================================================================
       category  n_pairs  agree_no_ties  agree_ties_0p5
         action      168          0.756           0.778
         comedy      172          0.741           0.765
          drama      165          0.732           0.754
        fantasy      171          0.748           0.771
         horror      168          0.739           0.762
        romance      169          0.745           0.768
         sci-fi      170          0.751           0.774
       thriller      167          0.743           0.766
```

## Interpreting Results

### Agreement Metrics

- **agree_no_ties**: Strict agreement (no partial credit)
  - > 0.75: Excellent
  - 0.65-0.75: Good
  - 0.55-0.65: Fair
  - < 0.55: Poor

- **agree_ties_0p5**: Soft agreement (partial credit for close calls)
  - Usually 2-5% higher than agree_no_ties
  - Better reflects model uncertainty

### Weight Distributions

Check `weights_histogram.png`:
- **Uniform weights** (~0.25 each): Model treats all experts equally
- **Skewed weights**: Model has learned to prefer certain experts
- **Diverse weights**: Model adapts weights to different prompts (good!)

### Attention Examples

Check `attn_examples/`:
- Shows which tokens each expert focuses on
- Helps understand what each expert has learned
- Look for patterns (e.g., genre words, sentiment, etc.)

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_prompts 8

# Or use CPU
--device cpu
```

### Poor Performance

1. **Try fine-tuning**:
   ```bash
   --unfreeze --lr 1e-5
   ```

2. **Adjust temperature**:
   ```bash
   --temperature 0.5
   ```

3. **Train longer**:
   ```bash
   --epochs 10
   ```

4. **Increase entropy regularization**:
   ```bash
   --entropy_lambda 1e-2
   ```

### Slow Training

1. **Use frozen encoder**:
   ```bash
   --freeze_encoder
   ```

2. **Use smaller model**:
   ```bash
   --encoder distilbert-base-uncased
   ```

3. **Increase batch size** (if memory allows):
   ```bash
   --batch_prompts 32
   ```

## Example Workflow

Complete workflow from scratch:

```bash
# 1. Navigate to project
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# 2. Test data loading
python3 src/router_bert/test_data_loading.py

# 3. Train with frozen encoder (fast baseline)
python -m src.router_bert.train_router \
    --encoder distilbert-base-uncased \
    --epochs 5 \
    --batch_prompts 16 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/baseline

# 4. Evaluate on test set
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/baseline/best_model \
    --split test

# 5. Review results
cat artifacts/router/bert_router/baseline/best_model/eval_test/metrics_overall.json
open artifacts/router/bert_router/baseline/best_model/eval_test/weights_histogram.png
cat artifacts/router/bert_router/baseline/best_model/eval_test/attn_examples/example_1.txt

# 6. If baseline is good, try fine-tuning for better performance
python -m src.router_bert.train_router \
    --encoder distilbert-base-uncased \
    --epochs 5 \
    --batch_prompts 8 \
    --lr 1e-5 \
    --unfreeze \
    --out_dir artifacts/router/bert_router/finetuned

# 7. Compare results
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/finetuned/best_model \
    --split test
```

## Next Steps

After successful training and evaluation:

1. **Compare with baselines**: Check how BERT router compares to XGBoost or MLP routers
2. **Error analysis**: Look at predictions.csv to find failure cases
3. **Hyperparameter tuning**: Try different learning rates, temperatures, etc.
4. **Ensemble**: Combine multiple models for better performance
5. **Deploy**: Use the trained model in production (see deployment guide)

## Support

For issues or questions:
1. Check the main README.md
2. Review the code documentation
3. Check training logs in `artifacts/router/bert_router/*/training_log.csv`

