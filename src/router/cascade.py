import torch
from typing import Dict, Callable, Iterable, List, Tuple, Optional
from .gating import should_gate

# canonical expert order
EXPERTS = ["alpha","beta","gamma","delta"]

ScoresDict = Dict[int, Dict[str, float]]   # movieId -> {"z_alpha":..., "z_beta":..., ...}

def _topk(scores: ScoresDict, key: str, k: int) -> List[int]:
    return [mid for mid, _ in sorted(scores.items(), key=lambda kv: kv[1][key], reverse=True)[:k]]

def _dz_summary(scores: ScoresDict, actives: List[str]) -> torch.Tensor:
    """
    Stable Δz summary per expert for routing: (top1 - median) across movies.
    Returns Tensor[4] aligned to EXPERTS (fill missing with 0).
    """
    out = torch.zeros(4, dtype=torch.float32)
    for e in actives:
        idx = EXPERTS.index(e)
        zs = torch.tensor([v[f"z_{e}"] for v in scores.values()], dtype=torch.float32)
        if zs.numel() == 0:
            out[idx] = 0.0
        else:
            out[idx] = (zs.max() - zs.median())
    return out

def _restrict_weights(w_full: torch.Tensor, actives: List[str]) -> torch.Tensor:
    """Return weights aligned to actives order, renormalized."""
    idxs = [EXPERTS.index(e) for e in actives]
    w = w_full[idxs].clone()
    w = w / (w.sum() + 1e-9)
    return w

def _fuse(scores: ScoresDict, w: torch.Tensor, actives: List[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for mid, feat in scores.items():
        s = 0.0
        for i, e in enumerate(actives):
            s += float(w[i]) * float(feat[f"z_{e}"])
        out[mid] = s
    return out

def cascade_route(
    get_scores_fn: Callable[[List[str], Optional[Iterable[int]]], ScoresDict],
    router_fn: Callable[[torch.Tensor], torch.Tensor],
    prompt_text: str,
    user_idx: Optional[int],
    *,
    use_cascade: bool = True,            # 🔁 ablation switch: False = no cascade (flat fuse)
    max_layers: int = 3,
    tau: float = 0.75,
    entropy_bits: float = 1.3,
    margin: float = 0.20,
    K_delta: int = 200,
    K_gamma: int = 200,
    K_alphabeta: int = 250,
) -> Tuple[Dict[int, float], List[dict]]:
    """
    Cascade routing with an ablation switch.

    - If use_cascade=False: single-shot fuse over all experts (no gating, no filtering).
    - If use_cascade=True:  up to 3 layers, optional dominance gating, Top-K filtering.

    get_scores_fn(active_experts, candidate_pool) -> { movieId: { "z_alpha":..., ... } }
    router_fn(dz_summary[4]) -> weights over [α,β,γ,δ] (Tensor[4])

    Returns:
      final_scores: {movieId: fused_score}
      route_log:    list of layer decisions/weights
    """
    # L1: score with all experts
    act1 = ["alpha","beta","gamma","delta"]
    scores1 = get_scores_fn(act1, None)
    dz1 = _dz_summary(scores1, act1)          # Tensor[4]
    w1 = router_fn(dz1)                        # Tensor[4], may be unnormalized → gating normalizes

    log: List[dict] = [{
        "layer": 1,
        "use_cascade": bool(use_cascade),
        "weights_raw": [float(x) for x in w1.detach().cpu().tolist()],
    }]

    # Flat (no cascade) ablation
    if not use_cascade or max_layers <= 1:
        w1r = _restrict_weights(w1, act1)
        log[-1]["weights"] = [float(x) for x in w1r.tolist()]
        log[-1]["gated"] = False
        return _fuse(scores1, w1r, act1), log

    # Decide gating
    gated, idx = should_gate(w1, tau=tau, entropy_bits=entropy_bits, margin=margin)
    log[-1]["gated"] = bool(gated)
    log[-1]["expert"] = (EXPERTS[idx] if gated else None)

    if not gated:
        w1r = _restrict_weights(w1, act1)
        log[-1]["weights"] = [float(x) for x in w1r.tolist()]
        return _fuse(scores1, w1r, act1), log

    # Dominance filtering
    dom = EXPERTS[idx]
    if dom == "delta":
        pool = _topk(scores1, "z_delta", K_delta)
        act2 = ["alpha","beta","gamma"]  # drop δ downstream
    elif dom == "gamma":
        pool = _topk(scores1, "z_gamma", K_gamma)
        act2 = ["alpha","beta"]          # γ as prior; refine with αβ
    else:
        other = "beta" if dom == "alpha" else "alpha"
        pool = list(sorted(set(_topk(scores1, f"z_{dom}", K_alphabeta)) |
                           set(_topk(scores1, f"z_{other}", K_alphabeta))))
        act2 = ["alpha","beta","gamma"]  # drop δ in α/β dominance

    # L2
    scores2 = get_scores_fn(act2, pool)
    dz2 = _dz_summary(scores2, act2)
    w2_full = router_fn(dz2)                    # Tensor[4]
    w2 = _restrict_weights(w2_full, act2)       # align & renorm
    log.append({
        "layer": 2,
        "actives": act2,
        "weights": {e: float(w2[i]) for i, e in enumerate(act2)}
    })

    if max_layers == 2:
        return _fuse(scores2, w2, act2), log

    # L3: final fuse (no extra filtering)
    final_scores = _fuse(scores2, w2, act2)
    log.append({"layer": 3, "note": "final rerank"})
    return final_scores, log
