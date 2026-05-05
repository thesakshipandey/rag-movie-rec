# src/router/__init__.py
try:
    from .cascade import cascade_route
except ModuleNotFoundError:  # pragma: no cover - optional torch dependency
    cascade_route = None

try:
    from .gating import should_gate
    from .mlp_router import RouterMLP, btl_loss
except ModuleNotFoundError:  # pragma: no cover
    RouterMLP = None
    btl_loss = None
    should_gate = None

from .xgb_router import RouterHeadXGB

__all__ = ["RouterMLP", "btl_loss", "should_gate", "cascade_route", "RouterHeadXGB"]
