"""MEGNO (Mean Exponential Growth of Nearby Orbits) indicator.

Computes the MEGNO chaos indicator for dynamical systems in the
ChaosBench-Logic v2 benchmark. Uses variational equations for ODE
systems and finite-difference Lyapunov approximation for maps.

MEGNO ~ 2 indicates regular (quasi-periodic) motion.
MEGNO > 2 indicates chaotic motion.
MEGNO < 2 indicates stable periodic motion.
"""

from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from chaosbench.data.indicators.time_series import (
    _DEFAULT_PARAMS,
    _make_map_callable,
    _make_ode_callable,
    get_default_ic,
    get_system_type,
)

_UNSUPPORTED_ODE_SYSTEMS = frozenset({
    "mackey_glass",
    "stochastic_ou",
    "kuramoto_sivashinsky",
    "sine_gordon",
})


def _numerical_jacobian(
    f_ode: callable,
    t: float,
    y: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Approximate the Jacobian of f_ode at (t, y) via central differences.

    Args:
        f_ode: Right-hand side callable f(t, y) -> dy/dt.
        t: Current time.
        y: Current state vector.
        eps: Finite difference step size.

    Returns:
        Jacobian matrix of shape (n, n).
    """
    n = len(y)
    jac = np.empty((n, n))
    f0 = f_ode(t, y)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = eps
        fp = f_ode(t, y + e_j)
        fm = f_ode(t, y - e_j)
        jac[:, j] = (fp - fm) / (2.0 * eps)
    return jac


def _make_augmented_rhs(
    f_ode: callable,
    dim: int,
    eps: float = 1e-7,
) -> callable:
    """Build the augmented ODE [dy/dt, dw/dt] for MEGNO variational equations.

    Args:
        f_ode: Right-hand side callable f(t, y) -> dy/dt.
        dim: State space dimension.
        eps: Finite difference step for Jacobian approximation.

    Returns:
        Augmented RHS callable for the combined (y, w) system.
    """
    def augmented_rhs(t: float, state: np.ndarray) -> np.ndarray:
        y = state[:dim]
        w = state[dim:]
        dydt = f_ode(t, y)
        jac = _numerical_jacobian(f_ode, t, y, eps)
        dwdt = jac @ w
        return np.concatenate([dydt, dwdt])
    return augmented_rhs


def _compute_megno_ode(
    system_id: str,
    params: Dict[str, float],
    ic: np.ndarray,
    t_max: float,
    n_points: int,
    seed: int,
) -> Optional[float]:
    """Compute MEGNO for an ODE system via variational equations.

    Args:
        system_id: System identifier from the benchmark.
        params: Merged system parameters.
        ic: Initial conditions.
        t_max: Maximum integration time.
        n_points: Number of output time points.
        seed: Random seed for tangent vector initialization.

    Returns:
        Time-averaged MEGNO value, or None if integration fails.
    """
    if system_id in _UNSUPPORTED_ODE_SYSTEMS:
        return None

    f_ode = _make_ode_callable(system_id, params)
    dim = len(ic)

    rng = np.random.default_rng(seed)
    w0 = rng.standard_normal(dim)
    w0 = w0 / np.linalg.norm(w0)

    augmented_rhs = _make_augmented_rhs(f_ode, dim)
    state0 = np.concatenate([ic, w0])

    t_start = 1e-6
    t_eval = np.linspace(t_start, t_max, n_points)

    try:
        sol = solve_ivp(
            augmented_rhs,
            (t_start, t_max),
            state0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8,
            max_step=t_max / n_points * 2.0,
        )
    except Exception:
        return None

    if not sol.success:
        return None

    # Check if solution has valid finite values
    if not np.all(np.isfinite(sol.y)):
        return None

    times = sol.t
    states = sol.y.T

    megno_sum = 0.0
    for i in range(1, len(times)):
        t_i = times[i]
        w_i = states[i, dim:]
        y_i = states[i, :dim]

        w_norm_sq = np.dot(w_i, w_i)
        if w_norm_sq < 1e-300:
            return None

        f_val = _make_ode_callable(system_id, params)
        jac = _numerical_jacobian(f_val, t_i, y_i)
        w_dot = jac @ w_i

        y_inst = t_i * np.dot(w_dot, w_i) / w_norm_sq
        dt = times[i] - times[i - 1]
        megno_sum += y_inst * dt

    total_time = times[-1] - times[0]
    if total_time < 1e-12:
        return None

    megno_value = (2.0 / total_time) * megno_sum
    return float(megno_value)


def _compute_megno_map(
    system_id: str,
    params: Dict[str, float],
    ic: np.ndarray,
    n_points: int,
    seed: int,
) -> Optional[float]:
    """Compute MEGNO-like indicator for a discrete map via finite-difference Lyapunov.

    Args:
        system_id: System identifier from the benchmark.
        params: Merged system parameters.
        ic: Initial conditions.
        n_points: Number of map iterations.
        seed: Random seed for tangent vector initialization.

    Returns:
        Averaged MEGNO-like value, or None if computation fails.
    """
    step_fn = _make_map_callable(system_id, params)
    dim = len(ic)
    eps = 1e-7

    rng = np.random.default_rng(seed)
    w = rng.standard_normal(dim)
    w = w / np.linalg.norm(w)

    state = ic.copy()
    megno_sum = 0.0
    renorm_interval = 10

    for k in range(1, n_points):
        try:
            next_state = step_fn(state)
        except Exception:
            return None

        if not np.all(np.isfinite(next_state)):
            return None

        jac = np.empty((dim, dim))
        for j in range(dim):
            e_j = np.zeros(dim)
            e_j[j] = eps
            fp = step_fn(state + e_j)
            fm = step_fn(state - e_j)
            jac[:, j] = (fp - fm) / (2.0 * eps)

        w_new = jac @ w
        w_norm = np.linalg.norm(w_new)

        if w_norm < 1e-300 or not np.isfinite(w_norm):
            return None

        log_growth = np.log(w_norm)
        megno_sum += k * log_growth

        if k % renorm_interval == 0:
            w = w_new / w_norm
        else:
            w = w_new / w_norm

        state = next_state

    n = n_points - 1
    if n < 1:
        return None

    time_weight = n * (n + 1) / 2.0
    megno_value = (2.0 / n) * (megno_sum / time_weight) * n
    return float(megno_value)


def compute_megno(
    system_id: str,
    params: Optional[Dict[str, float]] = None,
    ic: Optional[np.ndarray] = None,
    t_max: float = 100.0,
    n_points: int = 2000,
    seed: int = 42,
    validate: bool = True,
    max_abs_megno: float = 50.0,
) -> Optional[float]:
    """Compute MEGNO for a dynamical system.

    Uses the variational equations approach for ODEs:
    - Augments the ODE with tangent vector equations
    - Integrates the augmented system
    - Computes time-averaged MEGNO: Y(t) ~ 2 for regular, Y(t) > 2 for chaotic

    For maps: uses finite-difference Lyapunov approximation.
    Returns None if the system type is not supported or integration fails.

    Args:
        system_id: System identifier from the benchmark.
        params: System parameters. Uses defaults if None.
        ic: Initial conditions. Uses defaults if None.
        t_max: Maximum integration time.
        n_points: Number of output time points.
        seed: Random seed for reproducibility.
        validate: If True, reject values with |MEGNO| > max_abs_megno.
        max_abs_megno: Maximum absolute MEGNO value considered valid.

    Returns:
        Time-averaged MEGNO value, or None if not computable or invalid.
    """
    try:
        sys_type = get_system_type(system_id)
    except ValueError:
        return None

    merged_params = dict(_DEFAULT_PARAMS.get(system_id, {}))
    if params is not None:
        merged_params.update(params)

    if ic is None:
        try:
            ic = get_default_ic(system_id)
        except ValueError:
            return None

    rng = np.random.default_rng(seed)
    ic = ic + 1e-10 * rng.standard_normal(ic.shape)

    megno_value = None
    if sys_type == "ode":
        megno_value = _compute_megno_ode(
            system_id, merged_params, ic, t_max, n_points, seed,
        )

    elif sys_type == "map":
        megno_value = _compute_megno_map(
            system_id, merged_params, ic, n_points, seed,
        )

    # Validate MEGNO value if requested
    if validate and megno_value is not None:
        if not np.isfinite(megno_value) or abs(megno_value) > max_abs_megno:
            # Reject unrealistic values as computation errors
            return None

    return megno_value
