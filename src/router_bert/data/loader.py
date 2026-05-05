"""
Data loader for router training.

Loads pairwise comparison data and joins with prompt text,
organizing data by prompt for efficient batching.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


def load_router_data(
    parquet_path: str,
    prompts_path: str,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and join router features with prompt text.
    
    Args:
        parquet_path: Path to features parquet with pairs
        prompts_path: Path to prompts parquet with text
        split: Optional split to filter ('train', 'val', 'test')
    
    Returns:
        DataFrame with joined data
    """
    # Load features
    df = pd.read_parquet(parquet_path)
    
    # Load prompts
    prompts_df = pd.read_parquet(prompts_path)
    
    # Join on prompt_id
    df = df.merge(prompts_df, on='prompt_id', how='left')
    
    # Validate required columns
    required_cols = ['prompt_id', 'pair_id', 'y', 'dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta', 'text']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with NaN in dz columns
    dz_cols = ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta']
    df = df.dropna(subset=dz_cols)
    
    # Drop rows where ALL delta features are zero (no learning signal)
    has_signal = (df[dz_cols].abs() > 1e-8).any(axis=1)
    n_before = len(df)
    df = df[has_signal].copy()
    n_after = len(df)
    if n_before > n_after:
        print(f"Filtered out {n_before - n_after} pairs with all-zero deltas (no learning signal)")
    
    # Cast y to int
    df['y'] = df['y'].astype(int)
    
    # Filter by split if provided
    if split is not None:
        if 'split' not in df.columns:
            raise ValueError(f"Split column not found in data, but split={split} requested")
        df = df[df['split'] == split].copy()
        print(f"Filtered to split={split}: {len(df)} pairs")
    
    return df


class RouterDataset(Dataset):
    """
    Dataset that groups pairs by prompt for efficient batching.
    
    Each item is a prompt with all its associated pairs.
    """
    
    def __init__(
        self,
        parquet_path: str,
        prompts_path: str,
        split: Optional[str] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            parquet_path: Path to features parquet
            prompts_path: Path to prompts parquet
            split: Optional split to load ('train', 'val', 'test')
        """
        self.df = load_router_data(parquet_path, prompts_path, split)
        
        # Group by prompt
        self.prompts = []
        for prompt_id, group in self.df.groupby('prompt_id'):
            prompt_data = {
                'prompt_id': prompt_id,
                'prompt_text': group['text'].iloc[0],
                'pairs': []
            }
            
            for _, row in group.iterrows():
                pair_data = {
                    'dz': np.array([
                        row['dz_alpha'],
                        row['dz_beta'],
                        row['dz_gamma'],
                        row['dz_delta']
                    ], dtype=np.float32),
                    'y': int(row['y']),
                    'pair_id': row['pair_id'],
                }
                
                # Add optional metadata
                if 'difficulty' in row:
                    pair_data['difficulty'] = row['difficulty']
                if 'category' in row:
                    pair_data['category'] = row['category']
                
                prompt_data['pairs'].append(pair_data)
            
            self.prompts.append(prompt_data)
        
        print(f"Loaded {len(self.prompts)} prompts with {len(self.df)} total pairs")
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.prompts[idx]


def collate_prompts_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for batching prompts.
    
    Args:
        batch: List of prompt dicts from RouterDataset
    
    Returns:
        Collated batch with:
        - prompt_texts: List[str]
        - prompt_ids: List[str]
        - all_pairs: List of all pairs across batch
        - pair_to_prompt_idx: mapping from pair index to prompt index in batch
    """
    prompt_texts = []
    prompt_ids = []
    all_pairs = []
    pair_to_prompt_idx = []
    
    for prompt_idx, prompt_data in enumerate(batch):
        prompt_texts.append(prompt_data['prompt_text'])
        prompt_ids.append(prompt_data['prompt_id'])
        
        for pair in prompt_data['pairs']:
            all_pairs.append(pair)
            pair_to_prompt_idx.append(prompt_idx)
    
    return {
        'prompt_texts': prompt_texts,
        'prompt_ids': prompt_ids,
        'all_pairs': all_pairs,
        'pair_to_prompt_idx': torch.tensor(pair_to_prompt_idx, dtype=torch.long),
    }

