"""Small helpers used inside agents' adapt_parameters()."""

from typing import Dict


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def scale_param(value: float, factor: float, lo: float, hi: float) -> float:
    return clamp(value * factor, lo, hi)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(abs(w) for w in weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}