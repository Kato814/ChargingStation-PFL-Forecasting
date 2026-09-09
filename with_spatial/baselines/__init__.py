"""Paper comparison baselines and locked comparison-result adapters.

The main paper method is fully rerunnable from ``src/``.  The neural comparison
methods were produced by an earlier deterministic three-seed training workflow;
this package keeps their locked out-of-fold predictions as the comparison source
of truth while the training provenance is documented separately.
"""

from .protocols import NEURAL_BASELINE_SEEDS, PFEDME_SHARED_PROTOCOL, RESIDUAL_MLP_PROTOCOL
from .reference import (
    BASELINE_COLUMN_MAP,
    TABLE2_DISPLAY_ORDER,
    load_locked_comparison_predictions,
    reproduce_table2_from_locked_predictions,
)

__all__ = [
    "NEURAL_BASELINE_SEEDS",
    "PFEDME_SHARED_PROTOCOL",
    "RESIDUAL_MLP_PROTOCOL",
    "BASELINE_COLUMN_MAP",
    "TABLE2_DISPLAY_ORDER",
    "load_locked_comparison_predictions",
    "reproduce_table2_from_locked_predictions",
]
