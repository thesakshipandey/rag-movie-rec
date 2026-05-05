# BERT Router - Documentation Index

Welcome to the BERT-based text-conditioned expert router implementation!

## 📚 Documentation

### Getting Started
1. **[QUICKREF.md](QUICKREF.md)** - Quick reference for common commands
   - Installation instructions
   - Common commands
   - Key arguments
   - Troubleshooting

2. **[USAGE.md](USAGE.md)** - Comprehensive usage guide
   - Step-by-step tutorials
   - Advanced usage examples
   - Hyperparameter tuning
   - Interpreting results

3. **[README.md](README.md)** - Main documentation
   - Architecture overview
   - Data format
   - Training objective
   - Model details

### Implementation Details
4. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation status
   - Complete feature list
   - Design decisions
   - Code structure
   - Testing status

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install PyTorch (choose based on your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify setup
bash src/router_bert/setup_env.sh
```

### 2. Test Data Loading
```bash
python3 src/router_bert/test_data_loading.py
```

### 3. Run Example Workflow
```bash
bash src/router_bert/example_workflow.sh
```

This will:
- ✅ Validate data loading
- ✅ Run 1 epoch of training
- ✅ Evaluate on validation set
- ✅ Show results

### 4. Train Full Model
```bash
python -m src.router_bert.train_router \
    --epochs 5 \
    --batch_prompts 16 \
    --freeze_encoder
```

### 5. Evaluate
```bash
python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/run_<timestamp>/best_model \
    --split test
```

## 📁 File Structure

```
src/router_bert/
├── 📖 Documentation
│   ├── INDEX.md                     ← You are here
│   ├── QUICKREF.md                  Quick reference
│   ├── USAGE.md                     Usage guide
│   ├── README.md                    Main docs
│   └── IMPLEMENTATION_SUMMARY.md    Implementation details
│
├── 🔧 Scripts
│   ├── setup_env.sh                 Environment setup
│   ├── example_workflow.sh          Example workflow
│   └── test_data_loading.py         Data loading test
│
├── 🎯 Main Entrypoints
│   ├── train_router.py              Training script
│   └── eval_router.py               Evaluation script
│
├── 📦 Core Modules
│   ├── data/
│   │   └── loader.py                Data loading & dataset
│   ├── models/
│   │   └── four_head_router.py      Model architecture
│   └── utils/
│       ├── metrics.py               Evaluation metrics
│       └── viz.py                   Visualization
│
└── __init__.py                      Module initialization
```

## 🎯 Use Cases

### Scenario 1: Quick Test
**Goal:** Verify everything works

```bash
bash src/router_bert/example_workflow.sh
```

**Time:** ~5-10 minutes (GPU) or ~20-30 minutes (CPU)

---

### Scenario 2: Baseline Model
**Goal:** Train a fast baseline with frozen encoder

```bash
python -m src.router_bert.train_router \
    --encoder distilbert-base-uncased \
    --epochs 5 \
    --batch_prompts 16 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/baseline
```

**Time:** ~10-15 minutes (GPU)

---

### Scenario 3: Best Performance
**Goal:** Fine-tune for maximum accuracy

```bash
python -m src.router_bert.train_router \
    --encoder bert-base-uncased \
    --epochs 10 \
    --batch_prompts 8 \
    --lr 1e-5 \
    --unfreeze \
    --out_dir artifacts/router/bert_router/finetuned
```

**Time:** ~30-60 minutes (GPU)

---

### Scenario 4: CPU Training
**Goal:** Train without GPU

```bash
python -m src.router_bert.train_router \
    --encoder distilbert-base-uncased \
    --epochs 3 \
    --batch_prompts 4 \
    --freeze_encoder \
    --device cpu
```

**Time:** ~60-90 minutes

## 📊 Expected Results

### Good Performance
- **agree_no_ties**: > 0.70
- **agree_ties_0p5**: > 0.72

### Excellent Performance
- **agree_no_ties**: > 0.75
- **agree_ties_0p5**: > 0.77

### Weight Distribution
- **Diverse**: Different prompts get different weights (good!)
- **Not collapsed**: All 4 experts are used (not all weight on one)

## 🔍 Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```bash
--batch_prompts 8  # or even 4
```

### Issue: Slow Training
**Solution:**
```bash
--freeze_encoder  # Use frozen encoder (default)
--encoder distilbert-base-uncased  # Use smaller model
```

### Issue: Poor Performance
**Solutions:**
1. Try fine-tuning: `--unfreeze --lr 1e-5`
2. Adjust temperature: `--temperature 0.5`
3. Train longer: `--epochs 10`
4. Increase entropy regularization: `--entropy_lambda 1e-2`

### Issue: PyTorch Not Found
**Solution:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Evaluation Outputs

After evaluation, you'll find:

```
artifacts/router/bert_router/run_<timestamp>/best_model/eval_test/
├── metrics_overall.json          Overall performance
├── metrics_by_difficulty.csv     Performance by difficulty
├── metrics_by_category.csv       Performance by category
├── weights_histogram.png         Weight distributions
├── predictions.csv               All predictions
└── attn_examples/                Attention visualizations
    ├── example_1.txt
    ├── example_2.txt
    └── ...
```

## 🎓 Learning Path

### Beginner
1. Read [QUICKREF.md](QUICKREF.md)
2. Run `bash src/router_bert/example_workflow.sh`
3. Review outputs

### Intermediate
1. Read [USAGE.md](USAGE.md)
2. Train full model (5 epochs)
3. Experiment with hyperparameters
4. Compare frozen vs fine-tuned

### Advanced
1. Read [README.md](README.md) and [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
2. Modify model architecture
3. Implement custom losses
4. Integrate with other systems

## 🔗 Related Components

This BERT router is part of a larger movie recommendation system:

- **XGBoost Router**: `src/router/xgb_router.py`
- **MLP Router**: `src/router/mlp_router.py`
- **Expert Retrieval**: `src/retrieval/`
- **Evaluation Framework**: `src/evaluations/`

## 💡 Tips

1. **Start small**: Use frozen encoder and 1-2 epochs to verify everything works
2. **Monitor training**: Watch validation agreement to detect overfitting
3. **Check attention**: Look at attention examples to understand what model learned
4. **Compare methods**: Try both frozen and fine-tuned to see which works better
5. **Save results**: Keep training logs and evaluation outputs for comparison

## 📞 Support

### Documentation
- Quick questions → [QUICKREF.md](QUICKREF.md)
- How-to guides → [USAGE.md](USAGE.md)
- Architecture details → [README.md](README.md)
- Implementation → [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

### Debugging
1. Check training logs: `artifacts/router/bert_router/run_*/training_log.csv`
2. Run data test: `python3 src/router_bert/test_data_loading.py`
3. Verify environment: `bash src/router_bert/setup_env.sh`

### Help Commands
```bash
python -m src.router_bert.train_router --help
python -m src.router_bert.eval_router --help
```

## ✅ Checklist

Before training:
- [ ] PyTorch installed
- [ ] Data files accessible
- [ ] Environment verified (`setup_env.sh`)
- [ ] Data loading test passed

After training:
- [ ] Training log saved
- [ ] Best model checkpoint saved
- [ ] Validation agreement > 0.65

After evaluation:
- [ ] Metrics computed (overall + grouped)
- [ ] Visualizations generated
- [ ] Attention examples saved
- [ ] Results documented

## 🎉 Success Criteria

Your implementation is working well if:
1. ✅ Training completes without errors
2. ✅ Validation agreement improves over epochs
3. ✅ Test agreement > 0.70 (good) or > 0.75 (excellent)
4. ✅ Weight distributions are diverse (not collapsed)
5. ✅ Attention patterns make sense (focus on relevant tokens)

---

**Ready to get started?** Run:
```bash
bash src/router_bert/example_workflow.sh
```

Good luck! 🚀

