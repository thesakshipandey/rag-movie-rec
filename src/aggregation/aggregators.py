# src/aggregation/aggregators.py
from __future__ import annotations
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Iterable, Dict

class SumAggregator:
    def __init__(self, mode: str = "sum"):
        assert mode in ("sum","max")
        self.mode = mode
    def __call__(self, chunk_hits: Iterable[dict]) -> Dict[int, float]:
        if self.mode == "sum":
            agg = defaultdict(float)
            for h in chunk_hits:
                agg[int(h["movieId"])] += float(h["score"])
            return dict(agg)
        else:
            maxv = {}
            for h in chunk_hits:
                m = int(h["movieId"]); s = float(h["score"])
                maxv[m] = s if m not in maxv else max(maxv[m], s)
            return maxv

class AttentionAggregator(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(temperature))))
    def forward(self, chunk_hits: Iterable[dict]) -> Dict[int, float]:
        buckets = {}
        for h in chunk_hits:
            m = int(h["movieId"])
            buckets.setdefault(m, []).append(float(h["score"]))
        out = {}
        for m, scores in buckets.items():
            t = torch.exp(self.log_tau).clamp_min(0.05)
            s = torch.tensor(scores, dtype=torch.float32)
            w = torch.softmax(s / t, dim=0)
            out[m] = float((w * s).sum().item())
        return out
