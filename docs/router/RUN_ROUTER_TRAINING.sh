#!/bin/bash
# RUN_ROUTER_TRAINING.sh
# Complete pipeline to train the MoE router for RAG Movie Recommender
# Author: RAG Movie Rec Team
# Date: October 24, 2025

set -e  # Exit on error

echo "=============================================="
echo "  MoE Router Training Pipeline"
echo "=============================================="
echo ""

# Configuration
PROMPTS_DIR="/mnt/nas/sakshipandey/main/projects/rag-movie-rec/data/prompts"
INDICES_DIR="artifacts/indices"
OUTPUT_DIR="artifacts/router"
LOGS_DIR="logs"
ENCODER="qwen"
MODEL_PATH="/mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR

echo "Configuration:"
echo "  Prompts: $PROMPTS_DIR"
echo "  Indices: $INDICES_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Logs:    $LOGS_DIR"
echo ""

# ============================================
# STEP 1: Build Features (Sum Aggregation)
# ============================================
echo "=============================================="
echo "STEP 1: Building Features (Sum Aggregation)"
echo "=============================================="
echo ""

python -m src.router.build_router_features \
  --prompts_dir "$PROMPTS_DIR" \
  --indices_dir "$INDICES_DIR" \
  --out "$OUTPUT_DIR/features_sum.parquet" \
  --agg_kind sum \
  --topk 200 \
  --encoder "$ENCODER" \
  --model "$MODEL_PATH" \
  --logs_dir "$LOGS_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Features built successfully: $OUTPUT_DIR/features_sum.parquet"
    echo ""
else
    echo "✗ Feature building failed!"
    exit 1
fi

# ============================================
# STEP 2: Train Router
# ============================================
echo "=============================================="
echo "STEP 2: Training Router"
echo "=============================================="
echo ""

python -m src.cli.train_router \
  --features "$OUTPUT_DIR/features_sum.parquet" \
  --out "$OUTPUT_DIR/router_mlp_sum.pt" \
  --epochs 10 \
  --lr 5e-4 \
  --ent_lambda 1e-3 \
  --ent_target 1.2 \
  --tie_tol 0.05 \
  --logs_dir "$LOGS_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Router trained successfully: $OUTPUT_DIR/router_mlp_sum.pt"
    echo ""
else
    echo "✗ Training failed!"
    exit 1
fi

# ============================================
# STEP 3: Evaluate Router
# ============================================
echo "=============================================="
echo "STEP 3: Evaluating Router"
echo "=============================================="
echo ""

python -m src.evaluations.router.eval_router \
  --features "$OUTPUT_DIR/features_sum.parquet" \
  --weights "$OUTPUT_DIR/router_mlp_sum.pt" \
  --tol 0.05 \
  --logs_dir "$LOGS_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation complete!"
    echo ""
else
    echo "✗ Evaluation failed!"
    exit 1
fi

# ============================================
# Summary
# ============================================
echo "=============================================="
echo "  Training Pipeline Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Features:  $OUTPUT_DIR/features_sum.parquet"
echo "  Model:     $OUTPUT_DIR/router_mlp_sum.pt"
echo "  Logs:      $LOGS_DIR/"
echo ""
echo "Latest log files:"
ls -t "$LOGS_DIR"/*router*.log | head -3
echo ""
echo "View training summary:"
echo "  tail -n 50 \$(ls -t $LOGS_DIR/train_router_*.log | head -1)"
echo ""
echo "View evaluation results:"
echo "  tail -n 30 \$(ls -t $LOGS_DIR/eval_router_*.log | head -1)"
echo ""
echo "=============================================="

