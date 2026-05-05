import torch

_LOG2 = torch.log(torch.tensor(2.0))

def _entropy_bits(w: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy in *bits* for a probability vector.
    """
    w = w / (w.sum() + 1e-9)
    w = w.clamp_min(1e-9)
    return -(w * w.log()).sum() / _LOG2

def should_gate(
    w: torch.Tensor,
    tau: float = 0.75,
    entropy_bits: float = 1.3,
    margin: float = 0.20,
    return_stats: bool = False,
):
    """
    Decide whether to trigger dominance gating.

    Gate if ALL hold:
      1) max(w) >= tau                         (peakiness)
      2) entropy(w) <= entropy_bits (in bits)  (concentration)
      3) (top1 - top2) >= margin               (clear winner)

    Args:
      w: 1D tensor of length 4 (weights over [α,β,γ,δ]); need not be normalized.
      tau: dominance threshold for top1
      entropy_bits: entropy ceiling in *bits* (2.0 = uniform over 4)
      margin: gap between top1 and top2
      return_stats: if True, also return a dict of diagnostics

    Returns:
      (gate_bool, dominant_idx)   or
      (gate_bool, dominant_idx, stats_dict) if return_stats=True
    """
    assert w.ndim == 1 and w.numel() == 4, "w must be 1D length-4"
    # normalize; protect against NaNs/infs
    if not torch.isfinite(w).all():
        if return_stats:
            return False, 0, {"reason": "nonfinite", "w": w.detach().cpu().tolist()}
        return False, 0

    w = w / (w.sum() + 1e-9)
    w = w.clamp_min(1e-9)

    wmax, idx = torch.max(w, dim=-1)
    ent_b = _entropy_bits(w)
    top2v, _ = torch.topk(w, k=2)
    gap = top2v[0] - top2v[1]

    do_gate = (wmax >= tau) and (ent_b <= entropy_bits) and (gap >= margin)
    if return_stats:
        return bool(do_gate), int(idx.item()), {
            "wmax": float(wmax.item()),
            "entropy_bits": float(ent_b.item()),
            "gap": float(gap.item()),
            "top_idx": int(idx.item()),
            "w": [float(x) for x in w.tolist()],
        }
    return bool(do_gate), int(idx.item())
