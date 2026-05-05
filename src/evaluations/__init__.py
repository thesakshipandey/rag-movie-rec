"""Comprehensive evaluation framework for RAG movie recommender system."""

from . import dataset_analysis
from . import comparison_analysis
from . import error_analysis
from . import generate_report

# Optional: visualizations require matplotlib/seaborn
try:
    from . import visualizations
    __all__ = [
        'dataset_analysis',
        'comparison_analysis',
        'error_analysis',
        'visualizations',
        'generate_report',
    ]
except ImportError:
    __all__ = [
        'dataset_analysis',
        'comparison_analysis',
        'error_analysis',
        'generate_report',
    ]

