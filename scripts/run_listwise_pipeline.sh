#!/bin/bash
# Complete pipeline for listwise router training and evaluation
# Usage: bash run_listwise_pipeline.sh [--skip-generation] [--skip-training] [--skip-eval]

set -e  # Exit on error

# Detect Python command (prefer conda's python if available, else python3)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Neither 'python' nor 'python3' found in PATH"
    exit 1
fi

# Allow override via environment variable
PYTHON_CMD="${PYTHON:-$PYTHON_CMD}"

# Default paths (absolute paths from project root)
PROJECT_ROOT="/mnt/nas/sakshipandey/main"
DATA_DIR="${PROJECT_ROOT}/projects/Data"
INDICES_DIR="${PROJECT_ROOT}/projects/rag-movie-rec/artifacts/indices"
ROUTER_DIR="${PROJECT_ROOT}/projects/rag-movie-rec/artifacts/router"
EVAL_DIR="${PROJECT_ROOT}/projects/rag-movie-rec/artifacts/evaluation_results/listwise"
ENCODER="qwen"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen3-Embedding-8B"

# Parse arguments
SKIP_GENERATION=false
SKIP_TRAINING=false
SKIP_EVAL=false

for arg in "$@"; do
    case $arg in
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        *)
            ;;
    esac
done

echo "=============================================="
echo "Listwise Router Training Pipeline"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data Dir: $DATA_DIR"
echo "  Indices Dir: $INDICES_DIR"
echo "  Router Dir: $ROUTER_DIR"
echo "  Eval Dir: $EVAL_DIR"
echo "  Encoder: $ENCODER"
echo "  Model: $MODEL_PATH"
echo ""
echo "Flags:"
echo "  Skip Generation: $SKIP_GENERATION"
echo "  Skip Training: $SKIP_TRAINING"
echo "  Skip Evaluation: $SKIP_EVAL"
echo ""
echo "=============================================="
echo ""

# Create directories
mkdir -p $ROUTER_DIR
mkdir -p $EVAL_DIR
mkdir -p $EVAL_DIR/comparison

# Step 1: Generate Expert Scores
if [ "$SKIP_GENERATION" = false ]; then
    echo "[Step 1/5] Generating expert scores..."
    echo "=============================================="
    $PYTHON_CMD -m src.router.generate_expert_scores \
        --data_dir $DATA_DIR \
        --indices_dir $INDICES_DIR \
        --out $ROUTER_DIR/listwise_expert_scores.parquet \
        --encoder $ENCODER \
        --model $MODEL_PATH \
        --emotion_model_path "${PROJECT_ROOT}/models/roberta-plutchik-query_noKD/final" \
        --topk_retrieval 1000 \
        --agg_kind sum
    echo ""
    echo "✓ Expert scores generated!"
    echo ""
else
    echo "[Step 1/5] Skipping expert score generation"
    echo ""
fi

# Step 2: Train Router
if [ "$SKIP_TRAINING" = false ]; then
    echo "[Step 2/5] Training contextual hedge router..."
    echo "=============================================="
    $PYTHON_CMD -m src.cli.train_router_listwise \
        --expert_scores $ROUTER_DIR/listwise_expert_scores.parquet \
        --prompts_path $DATA_DIR/prompts.json \
        --out $ROUTER_DIR/router_listwise.pt \
        --loss listmle \
        --epochs 50 \
        --lr 1e-4 \
        --batch_size 32 \
        --d_context 128 \
        --d_hidden 256 \
        --dropout 0.2 \
        --temperature 1.0 \
        --entropy_weight 0.001 \
        --entropy_target 1.2 \
        --seed 42 \
        --device cuda
    echo ""
    echo "✓ Router trained!"
    echo ""
else
    echo "[Step 2/5] Skipping router training"
    echo ""
fi

# Step 3: Evaluate Single Experts
if [ "$SKIP_EVAL" = false ]; then
    echo "[Step 3/5] Evaluating single experts..."
    echo "=============================================="
    $PYTHON_CMD -m src.evaluations.eval_single_experts \
        --expert_scores $ROUTER_DIR/listwise_expert_scores.parquet \
        --prompts_path $DATA_DIR/prompts.json \
        --out $EVAL_DIR/single_expert_metrics.json \
        --k 10 \
        --relevance_threshold 0.9 \
        --seed 42
    echo ""
    echo "✓ Single expert evaluation complete!"
    echo ""
else
    echo "[Step 3/5] Skipping single expert evaluation"
    echo ""
fi

# Step 4: Evaluate Router
if [ "$SKIP_EVAL" = false ]; then
    echo "[Step 4/5] Evaluating router..."
    echo "=============================================="
    $PYTHON_CMD -m src.evaluations.eval_router_listwise \
        --expert_scores $ROUTER_DIR/listwise_expert_scores.parquet \
        --prompts_path $DATA_DIR/prompts.json \
        --router_checkpoint $ROUTER_DIR/router_listwise.pt \
        --out $EVAL_DIR/router_metrics.json \
        --k 10 \
        --relevance_threshold 0.9 \
        --device cuda
    echo ""
    echo "✓ Router evaluation complete!"
    echo ""
else
    echo "[Step 4/5] Skipping router evaluation"
    echo ""
fi

# Step 5: Compare Methods
if [ "$SKIP_EVAL" = false ]; then
    echo "[Step 5/5] Comparing all methods..."
    echo "=============================================="
    $PYTHON_CMD -m src.evaluations.compare_methods \
        --single_expert_metrics $EVAL_DIR/single_expert_metrics.json \
        --router_metrics $EVAL_DIR/router_metrics.json \
        --out_dir $EVAL_DIR/comparison \
        --k 10
    echo ""
    echo "✓ Comparison complete!"
    echo ""
else
    echo "[Step 5/5] Skipping comparison"
    echo ""
fi

# Done
echo "=============================================="
echo "✓ Pipeline Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Expert Scores:     $ROUTER_DIR/listwise_expert_scores.parquet"
echo "  Router Checkpoint: $ROUTER_DIR/router_listwise.pt"
echo "  Evaluation:        $EVAL_DIR/"
echo "  Comparison:        $EVAL_DIR/comparison/"
echo ""
echo "Next steps:"
echo "  1. Review results in $EVAL_DIR/comparison/comparison_table.csv"
echo "  2. Check plots in $EVAL_DIR/comparison/*.png"
echo "  3. Analyze per-prompt results in $EVAL_DIR/router_metrics_per_prompt.json"
echo ""

