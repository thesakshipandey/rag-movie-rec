#!/bin/bash
# Complete pipeline: Regenerate features → Balance → Train

set -e  # Exit on error

# cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
# source /mnt/nas/sakshipandey/venvs/rag_recsys/bin/activate

echo "================================================================================"
echo "BERT Router - Complete Pipeline"
echo "================================================================================"
echo ""

# Step 1: Regenerate features with all fixes
echo "Step 1: Regenerating features with all 1682 movies..."
echo "--------------------------------------------------------------------------------"
python -m src.router.build_router_features \
    --prompts_dir data/prompts \
    --indices_dir artifacts/indices \
    --out artifacts/router/features_sum_fixed.parquet \
    --topk 1682 \
    --user_idx_default 1 \
    --agg_kind sum \
    --encoder qwen \
    --model /mnt/nas/sakshipandey/main/models/Qwen3-Embedding-8B \
    --logs_dir logs

if [ $? -ne 0 ]; then
    echo "❌ Feature generation failed!"
    exit 1
fi
echo ""

# Step 2: Add splits
echo "Step 2: Adding train/val/test splits..."
echo "--------------------------------------------------------------------------------"
python -m src.cli.make_split \
    --input artifacts/router/features_sum_fixed.parquet \
    --output artifacts/router/features_sum_fixed.with_splits.parquet \
    --split_by prompt_id \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Split creation failed!"
    exit 1
fi
echo ""

# Step 3: Balance the dataset
echo "Step 3: Balancing dataset (equal y=0 and y=1)..."
echo "--------------------------------------------------------------------------------"
python -m src.cli.ab_balance \
    --in artifacts/router/features_sum_fixed.with_splits.parquet \
    --out artifacts/router/features_sum_fixed.with_splits.bal.parquet \
    --group_by split difficulty \
    --target 0.5 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Balancing failed!"
    exit 1
fi
echo ""

# Step 4: Verify features
echo "Step 4: Verifying fixed features..."
echo "--------------------------------------------------------------------------------"
python3 << 'EOF'
import pandas as pd
import numpy as np

df = pd.read_parquet('artifacts/router/features_sum_fixed.with_splits.bal.parquet')

print("="*80)
print("FEATURE VERIFICATION")
print("="*80)

# Basic stats
print(f"\nTotal pairs: {len(df)}")
print(f"Unique prompts: {df['prompt_id'].nunique()}")
print(f"\nSplits:")
for split in ['train', 'val', 'test']:
    count = (df['split'] == split).sum()
    print(f"  {split}: {count}")

# Delta feature statistics
dz_cols = ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta']
print("\n" + "="*80)
print("DELTA FEATURE STATISTICS")
print("="*80)
for col in dz_cols:
    nonzero = (df[col].abs() > 1e-8).sum()
    pct = 100 * nonzero / len(df)
    mean = df[col].mean()
    std = df[col].std()
    print(f"{col:12s}: {nonzero:5d}/{len(df)} non-zero ({pct:5.1f}%) | mean={mean:7.4f} std={std:6.4f}")

all_zero = (df[dz_cols].abs() <= 1e-8).all(axis=1).sum()
print(f"\nPairs with ALL zeros: {all_zero}/{len(df)} ({100*all_zero/len(df):.1f}%)")

# Label balance
print("\n" + "="*80)
print("LABEL BALANCE")
print("="*80)
for split in ['train', 'val', 'test']:
    split_df = df[df['split'] == split]
    y_counts = split_df['y'].value_counts().to_dict()
    print(f"{split:6s}: y=0: {y_counts.get(0, 0):4d}, y=1: {y_counts.get(1, 0):4d}")

print("\n" + "="*80)
if all_zero == 0 and (df['dz_gamma'].abs() > 1e-8).sum() > 0:
    print("✓ VERIFICATION PASSED - All features look good!")
else:
    print("⚠ WARNING - Some issues detected")
print("="*80)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Verification failed!"
    exit 1
fi
echo ""

# Step 5: Train the router
echo "Step 5: Training BERT router..."
echo "--------------------------------------------------------------------------------"
python -m src.router_bert.train_router \
    --parquet artifacts/router/features_sum_fixed.with_splits.bal.parquet \
    --prompts artifacts/prompts/prompt_text.parquet \
    --encoder distilbert-base-uncased \
    --epochs 10 \
    --batch_prompts 16 \
    --freeze_encoder \
    --lr 2e-5 \
    --temperature 1.0 \
    --out_dir artifacts/router/bert_router/fixed_run

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi
echo ""

echo "================================================================================"
echo "✅ Complete pipeline finished successfully!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Check training results in: artifacts/router/bert_router/fixed_run/"
echo "  2. Run evaluation:"
echo "     python -m src.router_bert.eval_router \\"
echo "       --ckpt_dir artifacts/router/bert_router/fixed_run/best_model \\"
echo "       --split test"
echo ""

