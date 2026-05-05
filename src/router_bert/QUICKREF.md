# BERT Router - Quick Reference

## Installation

```bash
# Install PyTorch (choose one)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu    # CPU

# Verify setup
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
bash src/router_bert/setup_env.sh
```

## Common Commands

### Test Data Loading
```bash
python3 src/router_bert/test_data_loading.py
```

### Training

**Basic (Frozen Encoder - Recommended)**
```bash
python -m src.router_bert.train_router
```

**With Custom Settings**
```bash
python -m src.router_bert.train_router \
    --encoder distilbert-base-uncased \
    --epochs 5 \
    --batch_prompts 16 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/my_run
```

**Fine-tune Encoder**
```bash
python -m src.router_bert.train_router \
    --unfreeze \
    --lr 1e-5 \
    --batch_prompts 8 \
    --epochs 5
```

**CPU Only**
```bash
python -m src.router_bert.train_router \
    --device cpu \
    --batch_prompts 4
```

### Evaluation

**Basic**
```bash
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/run_<timestamp>/best_model
```

**Custom Split**
```bash
python -m src.router_bert.eval_router \
    --ckpt_dir <path_to_model> \
    --split val \
    --tol 0.05
```

## Key Arguments

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder` | `distilbert-base-uncased` | HuggingFace model |
| `--freeze_encoder` | `True` | Freeze encoder (fast) |
| `--unfreeze` | `False` | Fine-tune encoder (slow) |
| `--epochs` | `5` | Training epochs |
| `--lr` | `2e-5` | Learning rate |
| `--batch_prompts` | `16` | Batch size |
| `--temperature` | `1.0` | BT loss temperature |
| `--entropy_min` | `0.6` | Min entropy |
| `--entropy_lambda` | `1e-3` | Entropy weight |
| `--out_dir` | Auto | Output directory |
| `--device` | Auto | `cuda` or `cpu` |

### Evaluation

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt_dir` | Required | Model directory |
| `--split` | `test` | Split to evaluate |
| `--tol` | `0.05` | Tie tolerance |
| `--n_attn_examples` | `10` | Attention examples |

## File Locations

**Data:**
- Features: `artifacts/router/features_sum.with_splits.bal.parquet`
- Prompts: `artifacts/prompts/prompt_text.parquet`

**Outputs:**
- Models: `artifacts/router/bert_router/run_<timestamp>/`
- Logs: `artifacts/router/bert_router/run_<timestamp>/training_log.csv`
- Eval: `artifacts/router/bert_router/run_<timestamp>/best_model/eval_<split>/`

**Cache:**
- HF Models: `/mnt/nas/sakshipandey/main/models/transformers/`

## Typical Workflow

```bash
# 1. Test data
python3 src/router_bert/test_data_loading.py

# 2. Quick training test (1 epoch)
python -m src.router_bert.train_router --epochs 1 --batch_prompts 8

# 3. Full training
python -m src.router_bert.train_router --epochs 5

# 4. Evaluate
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/run_<timestamp>/best_model

# 5. Check results
cat artifacts/router/bert_router/run_<timestamp>/best_model/eval_test/metrics_overall.json
```

## Troubleshooting

**CUDA OOM:** `--batch_prompts 8` or `--device cpu`

**Slow training:** `--freeze_encoder` (default)

**Poor performance:** `--unfreeze --lr 1e-5`

**Check logs:** `cat artifacts/router/bert_router/run_<timestamp>/training_log.csv`

## Expected Performance

**Training time (5 epochs, frozen encoder):**
- GPU: ~5-10 minutes
- CPU: ~30-60 minutes

**Agreement metrics:**
- Good: > 0.70
- Excellent: > 0.75

## Help

```bash
python -m src.router_bert.train_router --help
python -m src.router_bert.eval_router --help
```

## Documentation

- Full docs: `src/router_bert/README.md`
- Usage guide: `src/router_bert/USAGE.md`
- Implementation: `src/router_bert/IMPLEMENTATION_SUMMARY.md`

