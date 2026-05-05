# Setup Instructions for BERT Router

## Environment Setup

You have a virtual environment with PyTorch already installed at: `./venvs/rag_recsys/`

### Activate the Virtual Environment

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate
```

### Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch 2.8.0 - CUDA available: True
```

## Running the Pipeline

### 1. Test Data Loading (No PyTorch Required)

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
python src/router_bert/test_data_loading.py
```

### 2. Train Model

```bash
# Activate venv first!
source venvs/rag_recsys/bin/activate

# Quick test (1 epoch)
python -m src.router_bert.train_router \
    --epochs 1 \
    --batch_prompts 8 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/test_run

# Full training (5 epochs)
python -m src.router_bert.train_router \
    --epochs 5 \
    --batch_prompts 16 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/run1
```

### 3. Evaluate Model

```bash
# Activate venv first!
source venvs/rag_recsys/bin/activate

python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/run1/best_model \
    --split test
```

## Common Issues

### Issue: ModuleNotFoundError: No module named 'torch'

**Solution:** Activate the virtual environment first:
```bash
source venvs/rag_recsys/bin/activate
```

### Issue: ModuleNotFoundError: No module named 'router_bert'

**Solution:** This has been fixed. The imports now use `src.router_bert` which is correct.

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size:
```bash
python -m src.router_bert.train_router --batch_prompts 4
```

## User Information

This implementation is for user: **sakshipandey**

All paths are configured for:
- Project root: `/mnt/nas/sakshipandey/main/projects/rag-movie-rec`
- HuggingFace cache: `/mnt/nas/sakshipandey/main/models`
- Virtual environment: `./venvs/rag_recsys/`

## Quick Start Script

Create a convenience script to activate venv and run commands:

```bash
#!/bin/bash
# File: run_bert_router.sh

cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate

# Run the command passed as argument
"$@"
```

Then use it like:
```bash
bash run_bert_router.sh python -m src.router_bert.train_router --epochs 5
```

