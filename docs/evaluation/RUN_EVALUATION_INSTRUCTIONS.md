# How to Run the Comprehensive Evaluation

## Issue Detected

The system Python (`python3`) doesn't have PyTorch installed. You need to use the same Python environment that you use for training the models.

## Solution: Find Your Python Environment

### Option 1: Using the Same Environment as Training

Since your training scripts (`RUN_ROUTER_TRAINING.sh`) use `python`, you likely have a conda environment or virtual environment. Activate it first:

```bash
# If using conda
conda activate your-env-name

# If using venv
source /path/to/venv/bin/activate

# Then check torch is available
python -c "import torch; print('✅ Torch available:', torch.__version__)"
```

### Option 2: Find the Python with Torch

```bash
# Search for python installations
which -a python python3

# Test each one for torch
python -c "import torch" 2>/dev/null && echo "✅ This python has torch" || echo "❌ No torch"
python3 -c "import torch" 2>/dev/null && echo "✅ This python has torch" || echo "❌ No torch"
```

## Once You Have the Right Python

### Update the run script

Edit `run_comprehensive_eval.sh` and change line 14 to use your python:

```bash
# Replace this line:
if command -v python &> /dev/null; then

# With your specific python path, for example:
if true; then
    PYTHON_CMD="/path/to/your/python"  # or "conda run -n yourenv python"
```

### Or Run Directly

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Replace 'python' with your actual python command
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

## What You'll Get

After running (takes 2-3 minutes), you'll have:

```
artifacts/evaluation_results/
├── 🏆 best_results_for_presentation.md    ← YOUR BEST RESULTS
├── 📊 summary_report.md                    ← FULL REPORT  
├── all_metrics.json                        ← ALL DATA
│
├── experts/                                ← 4 expert evaluations
│   ├── alpha_metrics.json (Dense/FAISS)
│   ├── beta_metrics.json (BM25)
│   ├── gamma_metrics.json (LightGCN)
│   ├── delta_metrics.json (Emotion)
│   └── experts_comparison.csv
│
├── moe/                                    ← MoE router evaluations
│   ├── router_mlp_sum_metrics.json
│   ├── router_mlp_attn_metrics.json
│   ├── router_mlp_combo_metrics.json
│   └── moe_comparison.csv
│
├── ranknet/                                ← RankNet baselines
│   ├── ranknet_mlp_metrics.json
│   ├── ranknet_global_metrics.json
│   ├── ranknet_global_linear_metrics.json
│   └── ranknet_comparison.csv
│
├── comparisons/                            ← Model comparisons
│   ├── all_models_comparison.csv          ← ALL MODELS RANKED
│   ├── category_performance.csv
│   ├── difficulty_analysis.csv
│   └── best_models.json                   ← BEST MODEL IDENTIFIED
│
├── dataset_analysis/                       ← Dataset statistics
│   ├── dataset_summary.json
│   ├── prompts_statistics.json
│   ├── pairs_statistics.json
│   └── features_statistics.json
│
├── errors/                                 ← Failure analysis
│   ├── expert_disagreements.csv
│   ├── case_studies.json
│   └── error_analysis_summary.json
│
└── plots/                                  ← Visualizations (if matplotlib installed)
    ├── performance_comparison.png
    ├── category_heatmap.png
    ├── difficulty_heatmap.png
    └── expert_correlations.png
```

## Expected Output Example

When you run it successfully, you'll see:

```
================================================================================
COMPREHENSIVE EVALUATION PIPELINE
================================================================================

[1/9] Loading features from artifacts/router/features_sum.with_splits.bal.parquet...
Loaded 450 pairs

[2/9] Running dataset analysis...
Saved prompts statistics to artifacts/evaluation_results/dataset_analysis/prompts_statistics.json
✅ Dataset analysis complete

[3/9] Evaluating individual experts...
  Evaluating expert: alpha...
  Evaluating expert: beta...
  Evaluating expert: gamma...
  Evaluating expert: delta...
✅ Expert evaluations saved

[4/9] Evaluating MoE routers...
  Evaluating: router_mlp_sum...
  Evaluating: router_mlp_attn...
  Evaluating: router_mlp_combo...
✅ MoE evaluations saved

[5/9] Evaluating RankNet baselines...
  Evaluating: ranknet_mlp...
✅ RankNet evaluations saved

[6/9] Skipping emotion classifier (model path not provided)

[7/9] Running comparison analysis...
Creating overall comparison table...
✅ Comparison analysis complete

[8/9] Running error analysis...
Analyzing expert disagreements...
✅ Error analysis complete

[9/9] Generating visualizations...
Saved plot to artifacts/evaluation_results/plots/performance_comparison.png
✅ Visualizations complete

[10/10] Generating reports...
Generated report: artifacts/evaluation_results/summary_report.md
Generated presentation summary: artifacts/evaluation_results/best_results_for_presentation.md

================================================================================
EVALUATION COMPLETE!
================================================================================

🏆 BEST MODEL: moe_router_mlp_sum
   Agreement Score: 78.23%

Results saved to: artifacts/evaluation_results
```

## View Your Results

```bash
# Best results for your presentation
cat artifacts/evaluation_results/best_results_for_presentation.md

# Full detailed report
cat artifacts/evaluation_results/summary_report.md

# All models comparison table
cat artifacts/evaluation_results/comparisons/all_models_comparison.csv

# Check plots
ls artifacts/evaluation_results/plots/
```

## Quick Test (Without Running Full Evaluation)

To test if your Python environment is ready:

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Test imports
python -c "
import pandas as pd
import numpy as np
import torch
from src.router.mlp_router import RouterMLP
print('✅ All imports successful!')
print('PyTorch version:', torch.__version__)
"
```

If this works, you're ready to run the full evaluation!

## Troubleshooting

### "No module named 'torch'"
You're not in the right Python environment. Activate your conda/venv first.

### "No module named 'src'"
You need to set PYTHONPATH:
```bash
export PYTHONPATH=/mnt/nas/sakshipandey/main/projects/rag-movie-rec:$PYTHONPATH
```

### "File not found: features_sum.with_splits.bal.parquet"
Your features file might be named differently. Check:
```bash
ls artifacts/router/features*.parquet
```

## Need Help?

Check these files in the repo:
- `QUICKSTART_EVALUATION.md` - Quick start guide
- `EVALUATION_SYSTEM_GUIDE.md` - Complete guide  
- `EVALUATION_IMPLEMENTATION_SUMMARY.md` - What was implemented

---

**Once you have the right Python environment, running the evaluation takes just one command and 2-3 minutes!**

