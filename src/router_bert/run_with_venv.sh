#!/bin/bash
# Convenience script to run BERT router commands with the correct virtual environment
# Usage: bash run_with_venv.sh <command>

set -e

# Navigate to project root
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Activate virtual environment
if [ -d "venvs/rag_recsys" ]; then
    source venvs/rag_recsys/bin/activate
    echo "✓ Activated virtual environment: venvs/rag_recsys"
else
    echo "❌ Virtual environment not found at: venvs/rag_recsys"
    echo "Please create it or update the path in this script."
    exit 1
fi

# Verify PyTorch is available
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not found in virtual environment"
    exit 1
fi

# Run the command
echo "Running: $@"
echo ""
"$@"

