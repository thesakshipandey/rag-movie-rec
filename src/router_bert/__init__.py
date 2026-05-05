"""
BERT-based text-conditioned expert router.

This module implements a production-ready training and evaluation pipeline
for a text-conditioned expert router that outputs 4 weights [alpha, beta, gamma, delta]
from a prompt using a BERT-style transformer encoder with per-expert attention heads.
"""

__version__ = "0.1.0"

