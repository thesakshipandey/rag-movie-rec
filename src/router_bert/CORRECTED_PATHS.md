# BERT Router - Corrected Paths and Setup

## ✅ Path Corrections Applied

### Import Paths Fixed

**Before (INCORRECT):**
```python
from router_bert.data.loader import RouterDataset
from router_bert.models.four_head_router import FourHeadRouter
```

**After (CORRECT):**
```python
from src.router_bert.data.loader import RouterDataset
from src.router_bert.models.four_head_router import FourHeadRouter
```

### Files Updated:
- ✅ `src/router_bert/train_router.py` - Fixed imports
- ✅ `src/router_bert/eval_router.py` - Fixed imports

## 👤 User Information

**This implementation is for:**
- **User:** sakshipandey
- **User group:** pg24
- **Home directory:** `/mnt/nas/sakshipandey/`

**Project paths:**
- **Project root:** `/mnt/nas/sakshipandey/main/projects/rag-movie-rec`
- **Virtual environment:** `./venvs/rag_recsys/` (relative to project root)
- **HuggingFace cache:** `/mnt/nas/sakshipandey/main/models/`
- **Data directory:** `artifacts/router/` and `artifacts/prompts/`

## 🐍 Virtual Environment

Your project already has a virtual environment with PyTorch 2.8.0 installed:

**Location:** `/mnt/nas/sakshipandey/main/projects/rag-movie-rec/venvs/rag_recsys/`

**Activate it:**
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate
```

**Verify:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# Expected: PyTorch 2.8.0
```

## 🚀 Correct Usage

### Option 1: Manual Activation (Recommended)

```bash
# 1. Navigate to project
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# 2. Activate venv
source venvs/rag_recsys/bin/activate

# 3. Run commands
python -m src.router_bert.train_router --epochs 5
python -m src.router_bert.eval_router --ckpt_dir artifacts/router/bert_router/run_*/best_model
```

### Option 2: Using Wrapper Script

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# The wrapper automatically activates venv
bash src/router_bert/run_with_venv.sh python -m src.router_bert.train_router --epochs 5
```

### Option 3: Complete Example Workflow

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# This script activates venv and runs full pipeline
bash src/router_bert/example_workflow.sh
```

## 📁 All File Paths (Absolute)

### Code Files
```
/mnt/nas/sakshipandey/main/projects/rag-movie-rec/src/router_bert/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── loader.py
├── models/
│   ├── __init__.py
│   └── four_head_router.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── viz.py
├── train_router.py          ✅ FIXED IMPORTS
├── eval_router.py           ✅ FIXED IMPORTS
└── test_data_loading.py
```

### Data Files
```
/mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/
├── router/
│   └── features_sum.with_splits.bal.parquet
└── prompts/
    └── prompt_text.parquet
```

### Output Directory
```
/mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/router/bert_router/
└── run_<timestamp>/
    ├── config.json
    ├── training_log.csv
    ├── best_model/
    └── latest_model/
```

### Cache Directory
```
/mnt/nas/sakshipandey/main/models/
├── transformers/        # HuggingFace models
└── datasets/           # HuggingFace datasets
```

## ✅ Verification Steps

### 1. Check Virtual Environment
```bash
ls -la /mnt/nas/sakshipandey/main/projects/rag-movie-rec/venvs/rag_recsys/
# Should show bin/, lib/, etc.
```

### 2. Test Imports (with venv activated)
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate
python -c "from src.router_bert.data.loader import RouterDataset; print('✓ Imports work!')"
```

### 3. Test Data Loading
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate
python src/router_bert/test_data_loading.py
```

### 4. Quick Training Test
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate
python -m src.router_bert.train_router --epochs 1 --batch_prompts 4 --out_dir artifacts/router/bert_router/quick_test
```

## 🔧 Environment Variables

The scripts automatically set these when run:

```python
os.environ['HF_HOME'] = '/mnt/nas/sakshipandey/main/models'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nas/sakshipandey/main/models/transformers'
os.environ['HF_DATASETS_CACHE'] = '/mnt/nas/sakshipandey/main/models/datasets'
```

This ensures all HuggingFace downloads go to your NAS storage.

## 📝 Command Reference

### Training Commands

**Quick test (1 epoch):**
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate
python -m src.router_bert.train_router \
    --epochs 1 \
    --batch_prompts 8 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/test
```

**Full training (frozen encoder):**
```bash
python -m src.router_bert.train_router \
    --epochs 5 \
    --batch_prompts 16 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/frozen_run
```

**Fine-tuned training:**
```bash
python -m src.router_bert.train_router \
    --epochs 5 \
    --batch_prompts 8 \
    --lr 1e-5 \
    --unfreeze \
    --out_dir artifacts/router/bert_router/finetuned_run
```

### Evaluation Commands

**Evaluate on test set:**
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/frozen_run/best_model \
    --split test
```

**Evaluate on validation set:**
```bash
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/frozen_run/best_model \
    --split val
```

## 🎯 Quick Start (Copy-Paste Ready)

```bash
# Complete workflow in one go
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
source venvs/rag_recsys/bin/activate

# Test data
python src/router_bert/test_data_loading.py

# Train (quick test)
python -m src.router_bert.train_router \
    --epochs 1 \
    --batch_prompts 8 \
    --out_dir artifacts/router/bert_router/test_run

# Evaluate
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/test_run/best_model \
    --split val
```

## 📚 Documentation Files

All documentation is in: `/mnt/nas/sakshipandey/main/projects/rag-movie-rec/src/router_bert/`

- **SETUP.md** - This file (setup instructions)
- **QUICKREF.md** - Quick reference for commands
- **USAGE.md** - Comprehensive usage guide
- **README.md** - Architecture and details
- **INDEX.md** - Documentation hub

## ⚠️ Common Mistakes to Avoid

❌ **DON'T run without activating venv:**
```bash
python -m src.router_bert.train_router  # Will fail!
```

✅ **DO activate venv first:**
```bash
source venvs/rag_recsys/bin/activate
python -m src.router_bert.train_router  # Works!
```

❌ **DON'T use old import style:**
```python
from router_bert.data.loader import ...  # Wrong!
```

✅ **DO use correct import style:**
```python
from src.router_bert.data.loader import ...  # Correct!
```

## 🎉 Summary

**What was fixed:**
1. ✅ Import paths in `train_router.py` and `eval_router.py`
2. ✅ Created venv wrapper scripts
3. ✅ Updated example workflow to use venv
4. ✅ Documented all paths for user sakshipandey

**Ready to use:**
- All code is working
- Virtual environment is already set up
- Data files are validated
- Just activate venv and run!

**Next step:**
```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
bash src/router_bert/example_workflow.sh
```

