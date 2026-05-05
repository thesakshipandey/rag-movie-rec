#!/bin/bash
# Batch script to train all cascade router configurations
# This trains 6 models: 1 without gating + 5 with different thresholds

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Configuration
FEATURES="artifacts/router/features_sum.with_splits.bal.parquet"
OUTPUT_DIR="artifacts/router"
EPOCHS=20
LR=5e-4
SEED=42

# Common training args
COMMON_ARGS="--features $FEATURES --epochs $EPOCHS --lr $LR --seed $SEED"
COMMON_ARGS="$COMMON_ARGS --ab_shuffle_easy train --swap_prob 0.5"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "CASCADE ROUTER TRAINING - Batch Training All Configurations"
echo "============================================================"
echo "Features: $FEATURES"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Seed: $SEED"
echo "============================================================"
echo

# Find python command
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

# Export PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Function to train a model
train_model() {
    local name=$1
    local threshold=$2
    local gating=$3
    local output="$OUTPUT_DIR/router_$name.pt"
    
    echo "----------------------------------------"
    echo "Training: $name"
    echo "  Threshold: $threshold"
    echo "  Gating: $gating"
    echo "  Output: $output"
    echo "----------------------------------------"
    
    if [ "$gating" = "true" ]; then
        $PYTHON_CMD -m src.cli.train_router_cascade \
            $COMMON_ARGS \
            --out "$output" \
            --cascade_threshold "$threshold" \
            --gating \
            --gating_strength 0.1
    else
        $PYTHON_CMD -m src.cli.train_router_cascade \
            $COMMON_ARGS \
            --out "$output" \
            --cascade_threshold "$threshold"
    fi
    
    echo
    echo "✓ Completed: $name"
    echo
}

# Track start time
START_TIME=$(date +%s)

# Train baseline (no gating)
echo "========================================"
echo "MODEL 1/6: Baseline (No Gating)"
echo "========================================"
echo
train_model "no_gating" "0.75" "false"

# Train with cascade gating at different thresholds
THRESHOLDS=("0.70" "0.75" "0.80" "0.85" "0.90")
MODEL_NUM=2

for threshold in "${THRESHOLDS[@]}"; do
    echo "========================================"
    echo "MODEL $MODEL_NUM/6: Cascade threshold $threshold"
    echo "========================================"
    echo
    train_model "cascade_${threshold}" "$threshold" "true"
    MODEL_NUM=$((MODEL_NUM + 1))
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "============================================================"
echo "ALL CASCADE MODELS TRAINED SUCCESSFULLY!"
echo "============================================================"
echo "Time elapsed: ${MINUTES}m ${SECONDS}s"
echo
echo "Trained models:"
ls -lh "$OUTPUT_DIR"/router_cascade_*.pt "$OUTPUT_DIR"/router_no_gating.pt 2>/dev/null || true
echo
echo "Next steps:"
echo "  1. Run evaluation:"
echo "     python -m src.evaluations.models.MoE_cascade_eval \\"
echo "       --features $FEATURES \\"
echo "       --output_dir artifacts/evaluation_results/cascade_training \\"
echo "       --split test"
echo
echo "  2. View results in:"
echo "     artifacts/evaluation_results/cascade_training/"
echo "============================================================"

