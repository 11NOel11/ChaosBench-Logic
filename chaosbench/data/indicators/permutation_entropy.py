import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def permutation_entropy(
    series: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute permutation entropy of a time series.

    Args:
        series: 1D time series array.
        order: Embedding dimension (pattern length). Default 3.
        delay: Time delay between elements in a pattern. Default 1.
        normalize: If True, normalize by log2(order!) to get value in [0, 1].

    Returns:
        Permutation entropy value. If normalized, in [0, 1].
        0 = perfectly regular, 1 = maximally complex.
    """
    patterns = _extract_ordinal_patterns(series, order, delay)
    if len(patterns) == 0:
        return 0.0

    dist = _pattern_distribution(patterns, order)
    if len(dist) <= 1:
        return 0.0

    h = -sum(p * math.log2(p) for p in dist.values() if p > 0)

    if normalize:
        max_entropy = math.log2(math.factorial(order))
        if max_entropy == 0:
            return 0.0
        h = h / max_entropy

    return h


def _extract_ordinal_patterns(
    series: np.ndarray, order: int, delay: int
) -> List[Tuple[int, ...]]:
    """Extract ordinal patterns from a time series.

    For each valid starting index i, extract the subsequence
    [series[i], series[i+delay], ..., series[i+(order-1)*delay]]
    and compute its ordinal pattern (rank permutation).

    Args:
        series: 1D time series.
        order: Pattern length.
        delay: Time delay.

    Returns:
        List of ordinal patterns as tuples of ranks.
    """
    n = len(series)
    span = (order - 1) * delay + 1
    if n < span:
        return []

    patterns: List[Tuple[int, ...]] = []
    for i in range(n - span + 1):
        window = series[i : i + span : delay]
        pattern = tuple(int(x) for x in np.argsort(window))
        patterns.append(pattern)

    return patterns


def _pattern_distribution(
    patterns: List[Tuple[int, ...]], order: int
) -> Dict[Tuple[int, ...], float]:
    """Compute probability distribution over ordinal patterns.

    Args:
        patterns: List of observed ordinal patterns.
        order: Pattern length (for computing total possible patterns).

    Returns:
        Dict mapping each observed pattern to its relative frequency.
    """
    total = len(patterns)
    if total == 0:
        return {}

    counts = Counter(patterns)
    return {pat: count / total for pat, count in counts.items()}
