"""Gottwald-Melbourne 0-1 test for chaos.

Distinguishes regular from chaotic dynamics in a scalar time series.
K near 0 indicates regular dynamics, K near 1 indicates chaos.

Reference: Gottwald & Melbourne, Proc. R. Soc. A 460 (2004).
"""

import numpy as np
from typing import Optional


def _translation_variables(series: np.ndarray, c: float) -> tuple:
    """Compute translation variables p(n) and q(n).

    p(n) = sum_{j=0}^{n-1} x(j) * cos(j*c)
    q(n) = sum_{j=0}^{n-1} x(j) * sin(j*c)

    Args:
        series: 1D time series.
        c: Frequency parameter in (0, pi).

    Returns:
        Tuple of (p, q) arrays.
    """
    N = len(series)
    js = np.arange(N)
    cos_jc = np.cos(js * c)
    sin_jc = np.sin(js * c)
    p = np.cumsum(series * cos_jc)
    q = np.cumsum(series * sin_jc)
    return p, q


def _mean_square_displacement(
    p: np.ndarray,
    q: np.ndarray,
    n_max: int,
    series: np.ndarray,
    c: float,
) -> np.ndarray:
    """Compute modified mean square displacement M_c(n).

    M(n) = (1/N) sum_{j=0}^{N-1} [(p(j+n)-p(j))^2 + (q(j+n)-q(j))^2]

    The modified version subtracts the oscillatory term:
    M_c(n) = M(n) - E[x]^2 * (1 - cos(n*c)) / (1 - cos(c))

    Args:
        p: Translation variable p.
        q: Translation variable q.
        n_max: Maximum lag for MSD computation.
        series: Original time series (for oscillatory correction).
        c: Frequency parameter (for oscillatory correction).

    Returns:
        Array of MSD values for lags 1..n_max.
    """
    N = len(p)
    mean_x = np.mean(series)
    denom = 1.0 - np.cos(c)
    msd = np.empty(n_max)

    for n in range(1, n_max + 1):
        dp = p[n:N] - p[:N - n]
        dq = q[n:N] - q[:N - n]
        M_n = np.mean(dp ** 2 + dq ** 2)
        correction = mean_x ** 2 * (1.0 - np.cos(n * c)) / denom
        msd[n - 1] = M_n - correction

    return msd


def _correlation_K(msd: np.ndarray, n_max: int) -> float:
    """Compute correlation coefficient K from MSD.

    K = corr(n, M(n)) for n = 1..n_max.
    Uses the correlation method (not regression).
    Clamp result to [0, 1].

    Args:
        msd: Mean square displacement array.
        n_max: Number of points to use.

    Returns:
        K value clamped to [0, 1].
    """
    ns = np.arange(1, n_max + 1, dtype=float)
    vals = msd[:n_max]
    cov_matrix = np.corrcoef(ns, vals)
    K = cov_matrix[0, 1]
    return float(np.clip(K, 0.0, 1.0))


def zero_one_test(
    series: np.ndarray,
    c: Optional[float] = None,
    n_c: int = 10,
    seed: int = 42,
) -> float:
    """Compute the 0-1 test for chaos (Gottwald-Melbourne).

    Returns K in [0, 1]. Averages over n_c random c values for robustness.
    If a single c is provided, uses just that value.

    Args:
        series: 1D time series array.
        c: Optional fixed c value in (0, pi). If None, samples n_c random values.
        n_c: Number of random c values to average over.
        seed: Random seed for reproducibility.

    Returns:
        K value in [0, 1]. Near 0 = regular, near 1 = chaotic.
    """
    series = np.asarray(series, dtype=float).ravel()
    N = len(series)
    n_max = min(N // 10, 1000)

    if n_max < 2:
        return 0.0

    if c is not None:
        c_values = [c]
    else:
        rng = np.random.default_rng(seed)
        candidates = _sample_c_values(rng, n_c)
        c_values = candidates

    K_values = []
    for c_val in c_values:
        p, q = _translation_variables(series, c_val)
        msd = _mean_square_displacement(p, q, n_max, series, c_val)
        K_c = _correlation_K(msd, n_max)
        K_values.append(K_c)

    K = float(np.median(K_values))
    return float(np.clip(K, 0.0, 1.0))


def _sample_c_values(rng: np.random.Generator, n_c: int) -> list:
    """Sample c values in (0, pi) avoiding resonances near multiples of pi/5.

    Args:
        rng: NumPy random generator.
        n_c: Number of values to sample.

    Returns:
        List of valid c values.
    """
    resonances = [k * np.pi / 5.0 for k in range(1, 5)]
    exclusion_radius = 0.05
    values = []
    while len(values) < n_c:
        batch = rng.uniform(0.01, np.pi - 0.01, size=n_c * 3)
        for val in batch:
            if all(abs(val - r) > exclusion_radius for r in resonances):
                values.append(float(val))
                if len(values) >= n_c:
                    break
    return values
