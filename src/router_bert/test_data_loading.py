"""
Simple test script to verify data loading works.

This can be run without PyTorch to test the data pipeline.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np


def test_data_loading():
    """Test that data files can be loaded and joined."""
    
    print("="*80)
    print("Testing Data Loading")
    print("="*80)
    
    # Paths
    parquet_path = 'artifacts/router/features_sum.with_splits.bal.parquet'
    prompts_path = 'artifacts/prompts/prompt_text.parquet'
    
    print(f"\n1. Loading features from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    print(f"\n2. Loading prompts from: {prompts_path}")
    prompts_df = pd.read_parquet(prompts_path)
    print(f"   ✓ Loaded {len(prompts_df)} prompts")
    print(f"   Columns: {prompts_df.columns.tolist()}")
    
    print(f"\n3. Joining data...")
    df = df.merge(prompts_df, on='prompt_id', how='left')
    print(f"   ✓ Joined successfully")
    
    print(f"\n4. Checking required columns...")
    required_cols = ['prompt_id', 'pair_id', 'y', 'dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta', 'text']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"   ✗ Missing columns: {missing}")
        return False
    print(f"   ✓ All required columns present")
    
    print(f"\n5. Checking for NaN values in dz columns...")
    dz_cols = ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta']
    nan_counts = df[dz_cols].isna().sum()
    print(f"   NaN counts: {nan_counts.to_dict()}")
    df_clean = df.dropna(subset=dz_cols)
    print(f"   ✓ After dropping NaN: {len(df_clean)} rows")
    
    print(f"\n6. Checking splits...")
    if 'split' in df.columns:
        split_counts = df['split'].value_counts()
        print(f"   Split distribution:")
        for split, count in split_counts.items():
            print(f"     {split}: {count}")
    else:
        print(f"   ✗ No split column found")
    
    print(f"\n7. Checking metadata columns...")
    if 'difficulty' in df.columns:
        print(f"   ✓ difficulty column present: {df['difficulty'].nunique()} unique values")
    else:
        print(f"   ✗ difficulty column not found")
    
    if 'category' in df.columns:
        print(f"   ✓ category column present: {df['category'].nunique()} unique values")
    else:
        print(f"   ✗ category column not found")
    
    print(f"\n8. Analyzing prompt structure...")
    n_prompts = df['prompt_id'].nunique()
    pairs_per_prompt = df.groupby('prompt_id').size()
    print(f"   Total prompts: {n_prompts}")
    print(f"   Pairs per prompt - Mean: {pairs_per_prompt.mean():.1f}, "
          f"Min: {pairs_per_prompt.min()}, Max: {pairs_per_prompt.max()}")
    
    print(f"\n9. Sample data:")
    sample = df.iloc[0]
    print(f"   Prompt ID: {sample['prompt_id']}")
    print(f"   Prompt text: {sample['text'][:100]}...")
    print(f"   Pair ID: {sample['pair_id']}")
    print(f"   Label (y): {sample['y']}")
    print(f"   Delta features: alpha={sample['dz_alpha']:.4f}, "
          f"beta={sample['dz_beta']:.4f}, "
          f"gamma={sample['dz_gamma']:.4f}, "
          f"delta={sample['dz_delta']:.4f}")
    
    print("\n" + "="*80)
    print("✓ Data loading test PASSED")
    print("="*80)
    
    return True


if __name__ == '__main__':
    try:
        success = test_data_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

