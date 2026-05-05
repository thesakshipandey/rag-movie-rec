"""
Listwise ranking loss functions for router training.

Implements:
- ListMLE: Plackett-Luce likelihood
- ListNet: Cross-entropy on top-1 distributions  
- ApproxNDCG: Differentiable nDCG approximation
"""
import torch
import torch.nn.functional as F
from typing import Optional


def listmle_loss(
    predicted_scores: torch.Tensor,
    ground_truth_scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    ListMLE loss: negative log-likelihood of Plackett-Luce model.
    
    The Plackett-Luce model assumes items are selected sequentially according to:
    P(π) = ∏ᵢ exp(s_πᵢ) / Σⱼ≥ᵢ exp(s_πⱼ)
    
    Loss = -log P(π) = -Σᵢ [s_πᵢ - log(Σⱼ≥ᵢ exp(s_πⱼ))]
    
    Args:
        predicted_scores: [B, N] predicted scores for N items in each batch
        ground_truth_scores: [B, N] ground truth scores (higher = better rank)
        mask: [B, N] optional binary mask (1=valid, 0=ignore)
        eps: small constant for numerical stability
        
    Returns:
        Scalar loss (mean over batch)
    """
    B, N = predicted_scores.shape
    
    # Sort by ground truth scores (descending) to get optimal permutation
    # ground_truth_scores should already be in ranking order (1.0, 0.95, 0.93, ...)
    gt_ranks = torch.argsort(ground_truth_scores, dim=1, descending=True)  # [B, N]
    
    # Gather predicted scores in ground truth order
    predicted_sorted = torch.gather(predicted_scores, 1, gt_ranks)  # [B, N]
    
    if mask is not None:
        mask_sorted = torch.gather(mask, 1, gt_ranks)
    else:
        mask_sorted = torch.ones_like(predicted_sorted)
    
    # Compute log-likelihood for each position
    # For position i: log(exp(s_i) / sum_{j>=i} exp(s_j))
    #               = s_i - log(sum_{j>=i} exp(s_j))
    
    max_scores = predicted_sorted.max(dim=1, keepdim=True)[0]  # [B, 1] for stability
    exp_scores = torch.exp(predicted_sorted - max_scores) * mask_sorted  # [B, N]
    
    # Cumulative sum from right to left: sum_{j>=i} exp(s_j)
    cumsum_exp = torch.flip(
        torch.cumsum(torch.flip(exp_scores, dims=[1]), dim=1),
        dims=[1]
    )  # [B, N]
    
    # Log-likelihood at each position
    log_pl = predicted_sorted - max_scores.squeeze(1) - torch.log(cumsum_exp + eps)  # [B, N]
    log_pl = log_pl * mask_sorted
    
    # Sum over positions, mean over batch
    loss = -log_pl.sum(dim=1).mean()
    
    return loss


def listnet_loss(
    predicted_scores: torch.Tensor,
    ground_truth_scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    ListNet loss: cross-entropy between top-1 probability distributions.
    
    P(i is top-1) = exp(s_i / T) / Σⱼ exp(s_j / T)
    Loss = -Σᵢ P_gt(i) * log P_pred(i)
    
    Args:
        predicted_scores: [B, N] predicted scores
        ground_truth_scores: [B, N] ground truth scores
        mask: [B, N] optional mask
        temperature: softmax temperature
        
    Returns:
        Scalar loss (mean over batch)
    """
    if mask is not None:
        # Apply large negative value to masked positions
        predicted_scores = predicted_scores + (1.0 - mask) * (-1e9)
        ground_truth_scores = ground_truth_scores + (1.0 - mask) * (-1e9)
    
    # Compute top-1 probability distributions
    p_pred = F.softmax(predicted_scores / temperature, dim=1)  # [B, N]
    p_gt = F.softmax(ground_truth_scores / temperature, dim=1)  # [B, N]
    
    # Cross-entropy loss
    loss = -(p_gt * torch.log(p_pred + 1e-10)).sum(dim=1).mean()
    
    return loss


def approx_ndcg_loss(
    predicted_scores: torch.Tensor,
    ground_truth_scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    k: Optional[int] = None
) -> torch.Tensor:
    """
    ApproxNDCG loss: differentiable approximation of nDCG.
    
    Uses smooth approximation of rank via softmax-based sorting.
    
    Args:
        predicted_scores: [B, N] predicted scores
        ground_truth_scores: [B, N] ground truth scores (relevance labels)
        mask: [B, N] optional mask
        temperature: temperature for soft ranking (lower = sharper)
        k: compute nDCG@k (None = use all items)
        
    Returns:
        Loss = 1 - nDCG (mean over batch)
    """
    B, N = predicted_scores.shape
    
    if mask is None:
        mask = torch.ones_like(predicted_scores)
    
    # Apply mask
    pred_masked = predicted_scores * mask + (1.0 - mask) * (-1e9)
    
    # Compute ideal DCG (sort ground truth descending)
    gt_sorted, _ = torch.sort(ground_truth_scores, dim=1, descending=True)
    if k is not None:
        gt_sorted = gt_sorted[:, :k]
    
    positions = torch.arange(1, gt_sorted.shape[1] + 1, device=gt_sorted.device).float()
    discounts = 1.0 / torch.log2(positions + 1.0)  # [k]
    ideal_dcg = (gt_sorted * discounts).sum(dim=1)  # [B]
    
    # Compute predicted DCG using soft ranking
    # For each item, compute its "soft rank" via comparisons with all other items
    # soft_rank_i ≈ Σⱼ sigmoid((s_j - s_i) / T)
    
    diff = pred_masked.unsqueeze(2) - pred_masked.unsqueeze(1)  # [B, N, N]
    soft_comparisons = torch.sigmoid(diff / temperature)  # [B, N, N]
    soft_ranks = soft_comparisons.sum(dim=2)  # [B, N] (higher score = lower rank)
    
    # Convert soft ranks to soft positions (1-indexed)
    soft_positions = N + 1 - soft_ranks  # [B, N]
    
    # Compute soft discounts
    soft_discounts = 1.0 / torch.log2(soft_positions + 1.0)  # [B, N]
    
    # Predicted DCG
    pred_dcg = (ground_truth_scores * soft_discounts * mask).sum(dim=1)  # [B]
    
    if k is not None:
        # Approximate top-k by using soft attention weights
        top_k_weights = F.softmax(-soft_ranks / temperature, dim=1)  # [B, N]
        # Keep top k by re-normalizing
        _, topk_idx = torch.topk(top_k_weights, k=min(k, N), dim=1)
        topk_mask = torch.zeros_like(top_k_weights)
        topk_mask.scatter_(1, topk_idx, 1.0)
        pred_dcg = (ground_truth_scores * soft_discounts * topk_mask).sum(dim=1)
    
    # nDCG
    ndcg = pred_dcg / (ideal_dcg + 1e-10)
    
    # Loss = 1 - nDCG (minimize)
    loss = (1.0 - ndcg).mean()
    
    return loss


def ranknet_loss(
    predicted_scores: torch.Tensor,
    ground_truth_scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    RankNet pairwise loss (for reference/comparison).
    
    For each pair (i, j) where gt_i > gt_j:
    Loss += -log σ(s_i - s_j)
    
    Args:
        predicted_scores: [B, N] predicted scores
        ground_truth_scores: [B, N] ground truth scores
        mask: [B, N] optional mask
        
    Returns:
        Scalar loss (mean over valid pairs)
    """
    B, N = predicted_scores.shape
    
    if mask is None:
        mask = torch.ones_like(predicted_scores)
    
    # Pairwise differences
    pred_diff = predicted_scores.unsqueeze(2) - predicted_scores.unsqueeze(1)  # [B, N, N]
    gt_diff = ground_truth_scores.unsqueeze(2) - ground_truth_scores.unsqueeze(1)  # [B, N, N]
    
    # Pair mask: both items valid
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, N, N]
    
    # Only consider pairs where gt_i > gt_j (sign = +1) or gt_i < gt_j (sign = -1)
    # Use sign of gt_diff as target
    sign_gt = torch.sign(gt_diff)  # [B, N, N]
    valid_pairs = (sign_gt != 0).float() * pair_mask
    
    # RankNet loss: -log σ(sign * pred_diff)
    loss = F.softplus(-sign_gt * pred_diff) * valid_pairs
    
    # Mean over valid pairs
    num_pairs = valid_pairs.sum() + 1e-10
    loss = loss.sum() / num_pairs
    
    return loss


def compute_expert_weighted_scores(
    expert_scores: torch.Tensor,
    expert_weights: torch.Tensor
) -> torch.Tensor:
    """
    Combine expert scores with router weights.
    
    Args:
        expert_scores: [B, N, K] scores from K experts for N items
        expert_weights: [B, K] router weights over K experts
        
    Returns:
        final_scores: [B, N] weighted combination
    """
    # Broadcast and sum: [B, N, K] * [B, 1, K] -> [B, N]
    weights_expanded = expert_weights.unsqueeze(1)  # [B, 1, K]
    final_scores = (expert_scores * weights_expanded).sum(dim=2)  # [B, N]
    
    return final_scores


