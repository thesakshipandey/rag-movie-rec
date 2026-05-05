#!/bin/bash
# Convenience script to run comprehensive evaluation

set -e

# Project root
PROJECT_ROOT="/mnt/nas/sakshipandey/main/projects/rag-movie-rec"
cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Find python command (try python, python3, or from existing scripts)
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found. Please ensure Python is in your PATH."
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Default paths
FEATURES="${1:-artifacts/router/features_sum.with_splits.bal.parquet}"
OUTPUT_DIR="${2:-artifacts/evaluation_results}"
SPLIT="${3:-test}"

echo "=================================================="
echo "RAG Movie Recommender - Comprehensive Evaluation"
echo "=================================================="
echo ""
echo "Features:    $FEATURES"
echo "Output dir:  $OUTPUT_DIR"
echo "Split:       $SPLIT"
echo ""
echo "=================================================="
echo ""

# Run evaluation
$PYTHON_CMD -m src.evaluations.comprehensive_eval \
    --features "$FEATURES" \
    --output_dir "$OUTPUT_DIR" \
    --router_dir artifacts/router \
    --split "$SPLIT" \
    --models all \
    --prompts_json data/prompts/prompts.json \
    --movie_text artifacts/movies/movie_text.parquet \
    --emotion_index artifacts/indices/emotion/emotion.parquet

echo ""
echo "=================================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="
echo ""
echo "View results:"
echo "  - Summary:       cat $OUTPUT_DIR/summary_report.md"
echo "  - Best results:  cat $OUTPUT_DIR/best_results_for_presentation.md"
echo "  - All metrics:   cat $OUTPUT_DIR/all_metrics.json | jq"
echo ""

