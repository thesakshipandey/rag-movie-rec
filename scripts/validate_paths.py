#!/usr/bin/env python
"""
Validate that all required paths and files exist before running the pipeline.
"""
import os
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path("/mnt/nas/sakshipandey/main")

def check_file(path: Path, name: str) -> bool:
    """Check if a file exists."""
    if path.exists():
        print(f"✓ {name}: {path}")
        return True
    else:
        print(f"✗ {name}: {path} (NOT FOUND)")
        return False

def check_dir(path: Path, name: str) -> bool:
    """Check if a directory exists."""
    if path.is_dir():
        print(f"✓ {name}: {path}")
        return True
    else:
        print(f"✗ {name}: {path} (NOT FOUND)")
        return False

def main():
    print("=" * 80)
    print("PATH VALIDATION FOR LISTWISE ROUTER PIPELINE")
    print("=" * 80)
    print()
    
    all_good = True
    
    # Data files
    print("[1/5] Checking data files...")
    data_dir = PROJECT_ROOT / "projects/Data"
    all_good &= check_file(data_dir / "merged_all.json", "Listwise rankings")
    all_good &= check_file(data_dir / "prompts.json", "Prompt metadata")
    print()
    
    # Indices
    print("[2/5] Checking indices...")
    indices_dir = PROJECT_ROOT / "projects/rag-movie-rec/artifacts/indices"
    all_good &= check_dir(indices_dir, "Indices directory")
    
    # Dense index (FAISS)
    qwen_index = indices_dir / "qwen_fullmovie"
    gemma_index = indices_dir / "gemma"
    if qwen_index.is_dir():
        all_good &= check_file(qwen_index / "faiss.index", "FAISS index (qwen)")
        all_good &= check_file(qwen_index / "meta.parquet", "FAISS metadata (qwen)")
    elif gemma_index.is_dir():
        all_good &= check_file(gemma_index / "faiss.index", "FAISS index (gemma)")
        all_good &= check_file(gemma_index / "meta.parquet", "FAISS metadata (gemma)")
    else:
        print(f"✗ No FAISS index found (checked qwen_fullmovie and gemma)")
        all_good = False
    
    # BM25 index
    bm25_dir = indices_dir / "bm25"
    if bm25_dir.is_dir():
        all_good &= check_file(bm25_dir / "bm25.pkl", "BM25 index")
        all_good &= check_file(bm25_dir / "meta.parquet", "BM25 metadata")
    else:
        print(f"✗ BM25 directory not found: {bm25_dir}")
        all_good = False
    
    # LightGCN (optional)
    lgcn_dir = indices_dir / "lightgcn"
    if lgcn_dir.is_dir():
        check_file(lgcn_dir / "lgcn_scores.npy", "LightGCN scores (optional)")
        check_file(lgcn_dir / "lgcn_meta.json", "LightGCN metadata (optional)")
    else:
        print(f"⚠ LightGCN not found: {lgcn_dir} (optional, will use zeros)")
    
    # Emotion index (optional)
    emotion_dir = indices_dir / "emotion"
    if emotion_dir.is_dir():
        print(f"✓ Emotion directory: {emotion_dir} (optional)")
    else:
        print(f"⚠ Emotion directory not found: {emotion_dir} (optional, will use zeros)")
    print()
    
    # Model
    print("[3/5] Checking model...")
    model_path = PROJECT_ROOT / "models/Qwen3-Embedding-8B"
    all_good &= check_dir(model_path, "Qwen3 model")
    print()
    
    # Output directories
    print("[4/5] Checking output directories...")
    router_dir = PROJECT_ROOT / "projects/rag-movie-rec/artifacts/router"
    eval_dir = PROJECT_ROOT / "projects/rag-movie-rec/artifacts/evaluation_results/listwise"
    
    # Create if they don't exist
    router_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "comparison").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Router output directory: {router_dir} (created if needed)")
    print(f"✓ Evaluation output directory: {eval_dir} (created if needed)")
    print()
    
    # Python environment
    print("[5/5] Checking Python environment...")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        all_good = False
    
    try:
        import pandas
        print(f"✓ Pandas: {pandas.__version__}")
    except ImportError:
        print("✗ Pandas not installed")
        all_good = False
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        all_good = False
    
    try:
        import faiss
        print(f"✓ FAISS installed")
    except ImportError:
        print("✗ FAISS not installed")
        all_good = False
    
    print()
    print("=" * 80)
    
    if all_good:
        print("✓ ALL CHECKS PASSED - Ready to run pipeline!")
        print()
        print("Run the pipeline with:")
        print("  cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec")
        print("  conda activate rag_recsys")
        print("  ./run_listwise_pipeline.sh")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        print()
        print("Common issues:")
        print("  1. Not in conda environment: run 'conda activate rag_recsys'")
        print("  2. Missing indices: Build indices first")
        print("  3. Wrong paths: Check PROJECT_ROOT in this script")
        return 1

if __name__ == "__main__":
    sys.exit(main())

