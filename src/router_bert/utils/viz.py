"""
Visualization utilities for router analysis.

Includes attention visualization and weight distribution plots.
"""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_weights_histogram(
    weights_array: np.ndarray,
    save_path: str,
    expert_names: Optional[List[str]] = None,
):
    """
    Plot histogram of expert weights.
    
    Args:
        weights_array: [N, 4] array of weights
        save_path: Path to save figure
        expert_names: Names for the 4 experts
    """
    if expert_names is None:
        expert_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, expert_names)):
        weights = weights_array[:, i]
        ax.hist(weights, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{name} Weight')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Weight Distribution')
        ax.axvline(weights.mean(), color='red', linestyle='--', 
                   label=f'Mean: {weights.mean():.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved weight histogram to {save_path}")


def top_k_tokens(
    tokenizer,
    input_ids: torch.Tensor,
    attn_vec: torch.Tensor,
    k: int = 8,
) -> List[Tuple[str, float]]:
    """
    Get top-k tokens by attention weight.
    
    Args:
        tokenizer: HuggingFace tokenizer
        input_ids: [T] token ids for one sequence
        attn_vec: [T] attention weights for one sequence
        k: Number of top tokens to return
    
    Returns:
        List of (token_string, weight) tuples
    """
    # Convert to numpy
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    if isinstance(attn_vec, torch.Tensor):
        attn_vec = attn_vec.cpu().numpy()
    
    # Get top-k indices
    top_indices = np.argsort(attn_vec)[-k:][::-1]
    
    # Decode tokens
    results = []
    for idx in top_indices:
        token_id = input_ids[idx]
        token_str = tokenizer.decode([token_id])
        weight = float(attn_vec[idx])
        results.append((token_str, weight))
    
    return results


def format_attention_html(
    prompt_text: str,
    tokens: List[str],
    attn_weights: np.ndarray,
    expert_names: Optional[List[str]] = None,
) -> str:
    """
    Format attention weights as HTML for visualization.
    
    Args:
        prompt_text: Original prompt text
        tokens: List of token strings [T]
        attn_weights: [4, T] attention weights per expert
        expert_names: Names for the 4 experts
    
    Returns:
        HTML string
    """
    if expert_names is None:
        expert_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
    
    html = f"<h3>Prompt: {prompt_text}</h3>\n"
    
    for expert_idx, expert_name in enumerate(expert_names):
        html += f"<h4>{expert_name} Expert Attention</h4>\n"
        html += "<p>"
        
        weights = attn_weights[expert_idx]
        max_weight = weights.max()
        
        for token, weight in zip(tokens, weights):
            # Normalize for coloring
            intensity = int(255 * (1 - weight / max_weight)) if max_weight > 0 else 255
            color = f"rgb(255, {intensity}, {intensity})"
            html += f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px;">{token}</span> '
        
        html += "</p>\n"
    
    return html


def format_attention_text(
    prompt_text: str,
    tokenizer,
    input_ids: torch.Tensor,
    attn_weights: torch.Tensor,
    expert_names: Optional[List[str]] = None,
    top_k: int = 8,
) -> str:
    """
    Format attention as simple text output.
    
    Args:
        prompt_text: Original prompt text
        tokenizer: HuggingFace tokenizer
        input_ids: [T] token ids
        attn_weights: [4, T] attention weights per expert
        expert_names: Names for the 4 experts
        top_k: Number of top tokens to show per expert
    
    Returns:
        Formatted text string
    """
    if expert_names is None:
        expert_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
    
    lines = []
    lines.append(f"Prompt: {prompt_text}")
    lines.append("=" * 80)
    
    for expert_idx, expert_name in enumerate(expert_names):
        lines.append(f"\n{expert_name} Expert - Top {top_k} Attended Tokens:")
        
        top_tokens = top_k_tokens(
            tokenizer,
            input_ids,
            attn_weights[expert_idx],
            k=top_k,
        )
        
        for rank, (token, weight) in enumerate(top_tokens, 1):
            lines.append(f"  {rank}. '{token}' (weight: {weight:.4f})")
    
    lines.append("")
    return "\n".join(lines)


def save_attention_examples(
    examples: List[dict],
    save_dir: str,
    expert_names: Optional[List[str]] = None,
):
    """
    Save attention examples to text files.
    
    Args:
        examples: List of dicts with 'prompt_text', 'tokenizer', 'input_ids', 'attn_weights'
        save_dir: Directory to save examples
        expert_names: Names for the 4 experts
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for i, example in enumerate(examples):
        text = format_attention_text(
            prompt_text=example['prompt_text'],
            tokenizer=example['tokenizer'],
            input_ids=example['input_ids'],
            attn_weights=example['attn_weights'],
            expert_names=expert_names,
        )
        
        filepath = os.path.join(save_dir, f'example_{i+1}.txt')
        with open(filepath, 'w') as f:
            f.write(text)
    
    print(f"Saved {len(examples)} attention examples to {save_dir}")

