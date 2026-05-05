#!/bin/bash
# Complete cascade training and evaluation pipeline
# This script trains all cascade models and evaluates them

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "CASCADE TRAINING & EVALUATION PIPELINE"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date)"
echo "============================================================"
echo

# Find python
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo

# Configuration
FEATURES="artifacts/router/features_sum.with_splits.bal.parquet"
OUTPUT_DIR="artifacts/evaluation_results/cascade_training"

# Check if features exist
if [ ! -f "$FEATURES" ]; then
    echo "ERROR: Features file not found: $FEATURES"
    echo "Please ensure the features file exists before running this pipeline."
    exit 1
fi

echo "Features file: $FEATURES"
echo "Output directory: $OUTPUT_DIR"
echo

# Step 1: Train all cascade models
echo "============================================================"
echo "STEP 1: Training All Cascade Models"
echo "============================================================"
echo "This will train 6 models:"
echo "  1. Baseline (no gating)"
echo "  2. Cascade threshold 0.70"
echo "  3. Cascade threshold 0.75"
echo "  4. Cascade threshold 0.80"
echo "  5. Cascade threshold 0.85"
echo "  6. Cascade threshold 0.90"
echo
echo "Estimated time: 15-30 minutes (depending on hardware)"
echo

read -p "Press Enter to start training (or Ctrl+C to cancel)..."
echo

START_TRAIN=$(date +%s)

./train_all_cascade_models.sh

END_TRAIN=$(date +%s)
TRAIN_TIME=$((END_TRAIN - START_TRAIN))
TRAIN_MIN=$((TRAIN_TIME / 60))
TRAIN_SEC=$((TRAIN_TIME % 60))

echo
echo "✓ Training complete! Time: ${TRAIN_MIN}m ${TRAIN_SEC}s"
echo

# Step 2: Evaluate all models
echo "============================================================"
echo "STEP 2: Evaluating All Cascade Models"
echo "============================================================"
echo "This will evaluate all trained models on the test set."
echo "Estimated time: 1-2 minutes"
echo

read -p "Press Enter to start evaluation (or Ctrl+C to cancel)..."
echo

START_EVAL=$(date +%s)

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

$PYTHON_CMD -m src.evaluations.models.MoE_cascade_eval \
    --features "$FEATURES" \
    --output_dir "$OUTPUT_DIR" \
    --split test

END_EVAL=$(date +%s)
EVAL_TIME=$((END_EVAL - START_EVAL))

echo
echo "✓ Evaluation complete! Time: ${EVAL_TIME}s"
echo

# Step 3: Display results
echo "============================================================"
echo "STEP 3: Results Summary"
echo "============================================================"
echo

if [ -f "$OUTPUT_DIR/cascade_training_comparison.csv" ]; then
    echo "Comparison Table:"
    echo "----------------"
    if command -v column &> /dev/null; then
        cat "$OUTPUT_DIR/cascade_training_comparison.csv" | column -t -s,
    else
        cat "$OUTPUT_DIR/cascade_training_comparison.csv"
    fi
    echo
fi

if [ -f "$OUTPUT_DIR/CASCADE_TRAINING_SUMMARY.md" ]; then
    echo "Summary Report: $OUTPUT_DIR/CASCADE_TRAINING_SUMMARY.md"
    echo
    echo "Preview (first 50 lines):"
    echo "------------------------"
    head -n 50 "$OUTPUT_DIR/CASCADE_TRAINING_SUMMARY.md"
    echo
fi

# Final summary
TOTAL_TIME=$((END_EVAL - START_TRAIN))
TOTAL_MIN=$((TOTAL_TIME / 60))
TOTAL_SEC=$((TOTAL_TIME % 60))

echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo "Total time: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  - Training: ${TRAIN_MIN}m ${TRAIN_SEC}s"
echo "  - Evaluation: ${EVAL_TIME}s"
echo
echo "Results Location:"
echo "  - Models: artifacts/router/router_*.pt"
echo "  - Evaluations: $OUTPUT_DIR"
echo "  - Summary: $OUTPUT_DIR/CASCADE_TRAINING_SUMMARY.md"
echo "  - Comparison: $OUTPUT_DIR/cascade_training_comparison.csv"
echo
echo "Next Steps:"
echo "  1. Review the summary report: $OUTPUT_DIR/CASCADE_TRAINING_SUMMARY.md"
echo "  2. Compare model performance in the CSV table"
echo "  3. Select the best model for your use case"
echo "  4. Check individual model results: $OUTPUT_DIR/*_results.json"
echo "============================================================"

