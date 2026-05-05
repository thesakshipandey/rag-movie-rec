#!/bin/bash
# Example workflow for BERT router training and evaluation
# This script demonstrates the complete pipeline from data validation to evaluation

set -e  # Exit on error

echo "================================================================================"
echo "BERT Router - Example Workflow"
echo "================================================================================"
echo ""

# Navigate to project root
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Activate virtual environment
if [ -d "venvs/rag_recsys" ]; then
    source venvs/rag_recsys/bin/activate
    echo "✓ Activated virtual environment: venvs/rag_recsys"
    echo ""
else
    echo "❌ Virtual environment not found at: venvs/rag_recsys"
    echo "Please activate your Python environment with PyTorch installed."
    exit 1
fi

# Step 1: Test data loading
echo "Step 1: Testing data loading..."
echo "--------------------------------------------------------------------------------"
python3 src/router_bert/test_data_loading.py
if [ $? -ne 0 ]; then
    echo "❌ Data loading test failed!"
    exit 1
fi
echo ""

# Step 2: Quick training test (1 epoch)
echo "Step 2: Running quick training test (1 epoch)..."
echo "--------------------------------------------------------------------------------"
echo "This will verify that the training pipeline works correctly."
echo ""

python -m src.router_bert.train_router \
    --parquet artifacts/router/features_sum.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --encoder distilbert-base-uncased \
    --epochs 1 \
    --lr 2e-5 \
    --batch_prompts 8 \
    --freeze_encoder \
    --out_dir artifacts/router/bert_router/test_run

if [ $? -ne 0 ]; then
    echo "❌ Training test failed!"
    exit 1
fi
echo ""

# Step 3: Evaluate the test model
echo "Step 3: Evaluating test model on validation set..."
echo "--------------------------------------------------------------------------------"

python -m src.router_bert.eval_router \
    --ckpt_dir artifacts/router/bert_router/test_run/best_model \
    --parquet artifacts/router/features_sum.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --split val \
    --tol 0.05

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi
echo ""

# Step 4: Show results
echo "Step 4: Results Summary"
echo "--------------------------------------------------------------------------------"
echo ""
echo "Training log:"
cat artifacts/router/bert_router/test_run/training_log.csv
echo ""
echo "Overall metrics:"
cat artifacts/router/bert_router/test_run/best_model/eval_val/metrics_overall.json
echo ""

echo "================================================================================"
echo "✅ Example workflow completed successfully!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review results in: artifacts/router/bert_router/test_run/"
echo "  2. Run full training (5+ epochs) for better performance:"
echo "     python -m src.router_bert.train_router --epochs 5 --batch_prompts 16"
echo "  3. Evaluate on test set:"
echo "     python -m src.router_bert.eval_router --ckpt_dir <path> --split test"
echo "  4. Compare with other router methods (XGBoost, MLP)"
echo ""
echo "Documentation:"
echo "  - Quick reference: src/router_bert/QUICKREF.md"
echo "  - Full guide: src/router_bert/USAGE.md"
echo "  - README: src/router_bert/README.md"
echo ""

