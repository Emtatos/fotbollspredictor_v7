"""
Uncertainty module for entropy-based half-guard selection.

Provides normalized entropy calculation for 3-class probability distributions.
"""
import math
from typing import Tuple


def entropy_norm(p1: float, px: float, p2: float) -> float:
    """
    Normalized entropy in [0, 1] for a 3-class distribution.
    0 = fully certain, 1 = maximally uncertain (uniform).

    Args:
        p1: Probability of home win
        px: Probability of draw
        p2: Probability of away win

    Returns:
        Normalized entropy value between 0 and 1
    """
    eps = 1e-15
    ps = [max(eps, min(1.0, float(p))) for p in (p1, px, p2)]
    s = sum(ps)
    ps = [p / s for p in ps]  # re-normalize
    h = -sum(p * math.log(p) for p in ps)
    return float(h / math.log(3.0))
