# src/router/mlp_router.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class RouterMLP(nn.Module):
#     """
#     Router that outputs weights over experts [α, β, γ, δ] and a fused pairwise margin.
#     Inputs:  dz ∈ R^{B×4}  (Δz per expert: [dz_alpha, dz_beta, dz_gamma, dz_delta])
#     Returns: margin_logits ∈ R^{B}, weights ∈ R^{B×4}
#     """
#     def __init__(self, d_in: int = 4, d_hidden: int = 64, temperature: float = 1.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_in, d_hidden), nn.ReLU(),
#             nn.Linear(d_hidden, d_hidden), nn.ReLU(),
#             nn.Linear(d_hidden, 4)  # logits for α, β, γ, δ
#         )
#         # Learnable temperature for weight calibration
#         self.temp = nn.Parameter(torch.tensor(float(temperature)))
#         # Optional per-expert affine calibration on Δz
#         self.a = nn.Parameter(torch.ones(4))
#         self.b = nn.Parameter(torch.zeros(4))

#     def forward(self, dz: torch.Tensor):
#         """
#         dz: torch.FloatTensor [B,4]
#         """
#         logits = self.net(dz) / self.temp.clamp_min(0.2)
#         w = torch.softmax(logits, dim=-1)  # [B,4]
#         dz_tilde = self.a * dz + self.b    # per-expert calibration
#         margin_logits = (w * dz_tilde).sum(dim=-1)  # fused margin favoring A
#         return margin_logits, w


# def btl_loss(margin_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     Bradley–Terry (logistic) loss for pairwise preference.
#     y ∈ {0,1}: 1 means A preferred over B; margin_logits corresponds to A over B.
#     """
#     return F.binary_cross_entropy_with_logits(margin_logits, y.float())

import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterMLP(nn.Module):
    """
    MLP router over per-expert Δz features plus optional context features.
    Input: x ∈ R^{B×d_in}. First dz_dim entries are Δz in expert order [α,β,γ,δ].
    Output:
      - margin logits s ∈ R^{B} (favoring A over B)
      - weights w ∈ R^{B×4} (softmax over [α,β,γ,δ])
    """
    def __init__(self, d_in: int = 4, dz_dim: int = 4, d_hidden: int = 128,
                 temperature: float = 1.0, mix_indices: list[int] | None = None,
                 dropout: float = 0.1):
        super().__init__()
        if dz_dim > d_in:
            raise ValueError(f"dz_dim ({dz_dim}) cannot exceed d_in ({d_in})")
        self.dz_dim = dz_dim
        self.mix_indices = tuple(mix_indices) if mix_indices else None
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 4)  # logits for α,β,γ,δ
        )
        self.bias_head = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1)
        )
        self.temp = nn.Parameter(torch.tensor(float(temperature)))
        # Optional per-expert calibration (helps cross-expert scale drift)
        self.a = nn.Parameter(torch.ones(dz_dim))
        self.b = nn.Parameter(torch.zeros(dz_dim))
        if self.mix_indices:
            self.mix_scale = nn.Parameter(torch.tensor(1.0))
            self.mix_margin_weight = nn.Parameter(torch.zeros(len(self.mix_indices)))
        else:
            self.register_parameter("mix_scale", None)
            self.register_parameter("mix_margin_weight", None)

    def forward(self, x: torch.Tensor):
        """x: [B, d_in] with first dz_dim columns = Δz per expert"""
        logits = self.net(x) / self.temp.clamp_min(0.2)
        mix_margin = 0.0
        if self.mix_indices:
            mix = x[:, self.mix_indices]  # [B,|mix|]
            mix_prior = torch.log(mix.clamp(min=1e-6))
            scale = self.mix_scale.clamp(0.1, 5.0)
            logits = logits + scale * mix_prior
            mix_margin = (self.mix_margin_weight * mix).sum(dim=-1)
        w = F.softmax(logits, dim=-1)             # [B,4]
        dz = x[:, :self.dz_dim]
        dz_tilde = self.a * dz + self.b           # [B, dz_dim]
        bias = self.bias_head(x).squeeze(-1)
        s = (w * dz_tilde).sum(dim=-1) + bias + mix_margin  # [B]
        return s, w

import torch.nn.functional as F


def btl_loss(margin_logits: torch.Tensor, y: torch.Tensor, focal_gamma: float = 0.0) -> torch.Tensor:
    """
    Bradley–Terry–Luce pairwise loss (NLL) on the A-vs-B margin.
    s = margin_logits (A better if s>0)
    y ∈ {0,1}  (1 means A preferred, 0 means B preferred)

    Loss = -log P(y | s) with BTL link:
      if y=1: -log σ(s)     → softplus(-s)
      if y=0: -log σ(-s)    → softplus(s)
    """
    y = y.float()
    loss = y * F.softplus(-margin_logits) + (1.0 - y) * F.softplus(margin_logits)
    if focal_gamma > 0.0:
        p = torch.sigmoid(margin_logits)
        pt = torch.where(y > 0.5, p, 1.0 - p).clamp(1e-6, 1 - 1e-6)
        focal_weight = torch.pow(1.0 - pt, focal_gamma)
        loss = loss * focal_weight
    return loss.mean()

def logistic_loss(margin_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Logistic Regression loss.
    y ∈ {0,1} with 1 meaning "A preferred", 0 meaning "B preferred".
    We map y -> j ∈ {+1,-1} and minimize -log σ(j * s) = softplus(-j*s).
    """
    j = torch.where(
        y > 0.5,
        torch.tensor(1.0, device=margin_logits.device),
        torch.tensor(-1.0, device=margin_logits.device),
    )
    return F.softplus(-j * margin_logits).mean()
