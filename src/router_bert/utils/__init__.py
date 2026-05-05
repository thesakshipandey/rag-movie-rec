"""Utility functions for metrics and visualization."""

from .metrics import agreement_no_ties, agreement_ties_half, compute_metrics_by_group
from .viz import plot_weights_histogram, format_attention_html, top_k_tokens

__all__ = [
    "agreement_no_ties",
    "agreement_ties_half", 
    "compute_metrics_by_group",
    "plot_weights_histogram",
    "format_attention_html",
    "top_k_tokens",
]

