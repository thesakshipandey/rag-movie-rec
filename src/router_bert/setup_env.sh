#!/bin/bash
# Setup script for BERT router environment

echo "Setting up BERT Router environment..."

# Check if PyTorch is installed
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch is already installed"
    python3 -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
else
    echo "✗ PyTorch not found"
    echo ""
    echo "Please install PyTorch manually:"
    echo ""
    echo "For CUDA 11.8:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    echo "For CPU only:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    echo ""
    exit 1
fi

# Check other dependencies
echo ""
echo "Checking other dependencies..."

MISSING_DEPS=()

if ! python3 -c "import transformers" 2>/dev/null; then
    MISSING_DEPS+=("transformers")
fi

if ! python3 -c "import pandas" 2>/dev/null; then
    MISSING_DEPS+=("pandas")
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    MISSING_DEPS+=("numpy")
fi

if ! python3 -c "import matplotlib" 2>/dev/null; then
    MISSING_DEPS+=("matplotlib")
fi

if ! python3 -c "import tqdm" 2>/dev/null; then
    MISSING_DEPS+=("tqdm")
fi

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo "✓ All dependencies are installed"
else
    echo "✗ Missing dependencies: ${MISSING_DEPS[*]}"
    echo ""
    echo "Install from requirements.txt:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "You can now run:"
echo "  python -m src.router_bert.train_router --help"
echo "  python -m src.router_bert.eval_router --help"

