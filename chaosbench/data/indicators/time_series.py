"""Time series engine for ChaosBench-Logic v2.

Provides ODE and map callables for all 30 benchmark systems,
trajectory generation via scipy integration and direct iteration,
and default initial conditions for each system.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# ODE right-hand sides: f(t, y) -> dy/dt
# ---------------------------------------------------------------------------

def _lorenz63(t: float, y: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    """Lorenz 1963 attractor."""
    x, yy, z = y
    return np.array([sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z])


def _rossler(t: float, y: np.ndarray, a: float = 0.2, b: float = 0.2, c: float = 5.7) -> np.ndarray:
    """Rossler attractor."""
    x, yy, z = y
    return np.array([-yy - z, x + a * yy, b + z * (x - c)])


def _chen_system(t: float, y: np.ndarray, a: float = 35.0, b: float = 3.0, c: float = 28.0) -> np.ndarray:
    """Chen system."""
    x, yy, z = y
    return np.array([a * (yy - x), (c - a) * x - x * z + c * yy, x * yy - b * z])


def _chua_circuit(t: float, y: np.ndarray, alpha: float = 15.6, beta: float = 28.0, m0: float = -1.143, m1: float = -0.714) -> np.ndarray:
    """Chua circuit with piecewise-linear nonlinearity."""
    x, yy, z = y
    hx = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
    return np.array([alpha * (yy - x - hx), x - yy + z, -beta * yy])


def _double_pendulum(t: float, y: np.ndarray, m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0, g: float = 9.81) -> np.ndarray:
    """Double pendulum via full Lagrangian equations of motion.

    State: [theta1, theta2, omega1, omega2].
    """
    th1, th2, w1, w2 = y
    delta = th1 - th2
    sd, cd = np.sin(delta), np.cos(delta)
    s1, s2 = np.sin(th1), np.sin(th2)

    M = m1 + m2
    den1 = l1 * (M - m2 * cd * cd)
    den2 = l2 * (M - m2 * cd * cd)

    dw1 = (-m2 * l1 * w1 * w1 * sd * cd
            + m2 * g * s2 * cd
            - m2 * l2 * w2 * w2 * sd
            - M * g * s1) / den1
    dw2 = (m2 * l2 * w2 * w2 * sd * cd
            + M * l1 * w1 * w1 * sd
            + M * g * s1 * cd
            - M * g * s2) / den2

    return np.array([w1, w2, dw1, dw2])


def _duffing_chaotic(t: float, y: np.ndarray, delta: float = 0.3, gamma: float = 0.5, omega: float = 1.2) -> np.ndarray:
    """Duffing oscillator in the chaotic regime."""
    x, v = y
    return np.array([v, -delta * v - x ** 3 + gamma * np.cos(omega * t)])


def _fitzhugh_nagumo(t: float, y: np.ndarray, epsilon: float = 0.08, a: float = 0.7, b: float = 0.8, I: float = 0.5) -> np.ndarray:
    """FitzHugh-Nagumo neuron model."""
    v, w = y
    return np.array([v - v ** 3 / 3.0 - w + I, epsilon * (v + a - b * w)])


def _hindmarsh_rose(t: float, y: np.ndarray, I: float = 3.5) -> np.ndarray:
    """Hindmarsh-Rose neuron model with standard parameters."""
    a = 1.0
    b = 3.0
    c = 1.0
    d = 5.0
    s = 4.0
    r = 0.006
    x_r = -1.6
    x, yy, z = y
    return np.array([
        yy - a * x ** 3 + b * x ** 2 - z + I,
        c - d * x ** 2 - yy,
        r * (s * (x - x_r) - z),
    ])


def _lorenz84(t: float, y: np.ndarray, a: float = 0.25, b: float = 4.0, F: float = 8.0, G: float = 1.0) -> np.ndarray:
    """Lorenz 1984 low-order atmospheric model."""
    x, yy, z = y
    return np.array([
        -a * x - yy ** 2 - z ** 2 + a * F,
        -yy + x * yy - b * x * z + G,
        -z + b * x * yy + x * z,
    ])


def _lorenz96(t: float, y: np.ndarray, N: int = 5, F: float = 8.0) -> np.ndarray:
    """Lorenz 1996 model with N coupled variables."""
    dy = np.zeros(N)
    for i in range(N):
        dy[i] = (y[(i + 1) % N] - y[(i - 2) % N]) * y[(i - 1) % N] - y[i] + F
    return dy


def _lotka_volterra(t: float, y: np.ndarray, alpha: float = 1.5, beta: float = 1.0, gamma: float = 3.0, delta: float = 1.0) -> np.ndarray:
    """Lotka-Volterra predator-prey model."""
    x, yy = y
    return np.array([alpha * x - beta * x * yy, delta * x * yy - gamma * yy])


def _mackey_glass_ode(t: float, y: np.ndarray, beta_mg: float = 0.2, gamma_mg: float = 0.1, tau: float = 17.0, n: float = 10.0) -> np.ndarray:
    """Placeholder; actual Mackey-Glass uses delay (handled in generate_ode_trajectory)."""
    raise NotImplementedError("Use generate_ode_trajectory for mackey_glass")


def _brusselator(t: float, y: np.ndarray, A: float = 1.0, B: float = 3.0) -> np.ndarray:
    """Brusselator chemical oscillator."""
    x, yy = y
    return np.array([A + x ** 2 * yy - (B + 1) * x, B * x - x ** 2 * yy])


def _damped_oscillator(t: float, y: np.ndarray, omega: float = 1.0, zeta: float = 0.1) -> np.ndarray:
    """Damped harmonic oscillator."""
    x, v = y
    return np.array([v, -2.0 * zeta * omega * v - omega ** 2 * x])


def _damped_driven_pendulum(t: float, y: np.ndarray, beta_ddp: float = 0.5, g_over_L: float = 1.0, A: float = 1.2, omega: float = 2.0 / 3.0) -> np.ndarray:
    """Damped driven pendulum (non-chaotic regime).

    theta'' + beta*theta' + (g/L)*sin(theta) = A*cos(omega*t)
    State: [theta, omega_angular].
    """
    th, w = y
    return np.array([w, -beta_ddp * w - g_over_L * np.sin(th) + A * np.cos(omega * t)])


def _oregonator(t: float, y: np.ndarray, epsilon: float = 0.1, q: float = 8e-4, f: float = 1.0) -> np.ndarray:
    """Oregonator model for Belousov-Zhabotinsky reaction (2-variable reduction)."""
    x, z = y
    return np.array([
        (x * (1.0 - x) - f * z * (x - q) / (x + q)) / epsilon,
        x - z,
    ])


def _rikitake_dynamo(t: float, y: np.ndarray, mu: float = 1.0, a: float = 5.0) -> np.ndarray:
    """Rikitake two-disk dynamo model."""
    x1, x2, x3 = y
    return np.array([
        -mu * x1 + x3 * x2,
        -mu * x2 + (x3 - a) * x1,
        1.0 - x1 * x2,
    ])


def _shm(t: float, y: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """Simple harmonic motion."""
    x, v = y
    return np.array([v, -omega ** 2 * x])


def _vdp(t: float, y: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """Van der Pol oscillator."""
    x, v = y
    return np.array([v, mu * (1.0 - x ** 2) * v - x])


def _kuramoto_sivashinsky(t: float, y: np.ndarray, L: float = 22.0) -> np.ndarray:
    """Kuramoto-Sivashinsky PDE via spectral (Fourier) discretization.

    u_t = -u*u_x - u_xx - u_xxxx on [0, L] with periodic BCs.
    State y holds the real-space values on a uniform grid of size N = len(y).
    """
    N = len(y)
    k = np.fft.rfftfreq(N, d=L / (2.0 * np.pi * N))
    yhat = np.fft.rfft(y)
    ux = np.fft.irfft(1j * k * yhat, n=N)
    lin = np.fft.irfft((-k ** 2 + k ** 4) * yhat, n=N)
    return -y * ux - lin


def _sine_gordon(t: float, y: np.ndarray, L: float = 20.0) -> np.ndarray:
    """Sine-Gordon PDE via finite differences.

    u_tt = u_xx - sin(u) on [0, L] with periodic BCs.
    State: y = [u_0..u_{N-1}, v_0..v_{N-1}] where v = u_t.
    """
    N = len(y) // 2
    u = y[:N]
    v = y[N:]
    dx = L / N
    uxx = (np.roll(u, -1) + np.roll(u, 1) - 2.0 * u) / (dx ** 2)
    du = v
    dv = uxx - np.sin(u)
    return np.concatenate([du, dv])


def _stochastic_ou_drift(t: float, y: np.ndarray, theta: float = 1.0, mu: float = 0.0, sigma: float = 0.5) -> np.ndarray:
    """Drift term for Ornstein-Uhlenbeck process (SDE handled specially)."""
    return theta * (mu - y)


# ---------------------------------------------------------------------------
# Map callables: f(state, params) -> next_state
# ---------------------------------------------------------------------------

def _logistic_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Logistic map x_{n+1} = r*x_n*(1-x_n)."""
    r = params["r"]
    x = state[0]
    return np.array([r * x * (1.0 - x)])


def _henon_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Henon map."""
    a = params["a"]
    b = params["b"]
    x, y = state
    return np.array([1.0 - a * x ** 2 + y, b * x])


def _ikeda_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Ikeda map."""
    u = params["u"]
    x, y = state
    tn = 0.4 - 6.0 / (1.0 + x ** 2 + y ** 2)
    ct, st = np.cos(tn), np.sin(tn)
    return np.array([1.0 + u * (x * ct - y * st), u * (x * st + y * ct)])


def _arnold_cat_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Arnold cat map on the unit torus."""
    x, y = state
    return np.array([(2.0 * x + y) % 1.0, (x + y) % 1.0])


def _bakers_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Baker's map on the unit square."""
    x, y = state
    if x < 0.5:
        return np.array([2.0 * x, y / 2.0])
    return np.array([2.0 * x - 1.0, (y + 1.0) / 2.0])


def _circle_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Circle map (quasi-periodic regime)."""
    Omega = params["Omega"]
    K = params["K"]
    theta = state[0]
    return np.array([(theta + Omega - K / (2.0 * np.pi) * np.sin(2.0 * np.pi * theta)) % 1.0])


def _standard_map(state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Chirikov standard map."""
    K = params["K"]
    theta, p = state
    p_new = p + K * np.sin(theta)
    theta_new = theta + p_new
    return np.array([theta_new % (2.0 * np.pi), p_new])


# ---------------------------------------------------------------------------
# System registry and metadata
# ---------------------------------------------------------------------------

_ODE_PARAM_MAP: Dict[str, Tuple[Callable, List[str]]] = {
    "lorenz63": (_lorenz63, ["sigma", "rho", "beta"]),
    "rossler": (_rossler, ["a", "b", "c"]),
    "chen_system": (_chen_system, ["a", "b", "c"]),
    "chua_circuit": (_chua_circuit, ["alpha", "beta", "m0", "m1"]),
    "double_pendulum": (_double_pendulum, ["m1", "m2", "l1", "l2", "g"]),
    "duffing_chaotic": (_duffing_chaotic, ["delta", "gamma", "omega"]),
    "fitzhugh_nagumo": (_fitzhugh_nagumo, ["epsilon", "a", "b", "I"]),
    "hindmarsh_rose": (_hindmarsh_rose, ["I"]),
    "lorenz84": (_lorenz84, ["a", "b", "F", "G"]),
    "lorenz96": (_lorenz96, ["N", "F"]),
    "lotka_volterra": (_lotka_volterra, ["alpha", "beta", "gamma", "delta"]),
    "mackey_glass": (_mackey_glass_ode, ["beta_mg", "gamma_mg", "tau", "n"]),
    "brusselator": (_brusselator, ["A", "B"]),
    "damped_oscillator": (_damped_oscillator, ["omega", "zeta"]),
    "damped_driven_pendulum_nonchaotic": (_damped_driven_pendulum, ["beta_ddp", "g_over_L", "A", "omega"]),
    "oregonator": (_oregonator, ["epsilon", "q", "f"]),
    "rikitake_dynamo": (_rikitake_dynamo, ["mu", "a"]),
    "shm": (_shm, ["omega"]),
    "vdp": (_vdp, ["mu"]),
    "kuramoto_sivashinsky": (_kuramoto_sivashinsky, ["L"]),
    "sine_gordon": (_sine_gordon, ["L"]),
    "stochastic_ou": (_stochastic_ou_drift, ["theta", "mu", "sigma"]),
}

_MAP_PARAM_MAP: Dict[str, Tuple[Callable, List[str]]] = {
    "logistic_r4": (_logistic_map, ["r"]),
    "logistic_r2_8": (_logistic_map, ["r"]),
    "henon": (_henon_map, ["a", "b"]),
    "ikeda_map": (_ikeda_map, ["u"]),
    "arnold_cat_map": (_arnold_cat_map, []),
    "bakers_map": (_bakers_map, []),
    "circle_map_quasiperiodic": (_circle_map, ["Omega", "K"]),
    "standard_map": (_standard_map, ["K"]),
}

_SYSTEM_TYPES: Dict[str, str] = {}
for _sid in _ODE_PARAM_MAP:
    _SYSTEM_TYPES[_sid] = "ode"
for _sid in _MAP_PARAM_MAP:
    _SYSTEM_TYPES[_sid] = "map"


def _make_ode_callable(system_id: str, params: Dict[str, float]) -> Callable:
    """Build a closure f(t, y) with parameters baked in."""
    base_fn, param_names = _ODE_PARAM_MAP[system_id]
    bound = {k: params[k] for k in param_names if k in params}
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return base_fn(t, y, **bound)
    return rhs


def _make_map_callable(system_id: str, params: Dict[str, float]) -> Callable:
    """Build a closure f(state) with parameters baked in."""
    base_fn, param_names = _MAP_PARAM_MAP[system_id]
    bound = {k: params[k] for k in param_names if k in params}
    def step(state: np.ndarray) -> np.ndarray:
        return base_fn(state, bound)
    return step


SYSTEM_REGISTRY: Dict[str, Callable] = {}

for _sid, (_fn, _pnames) in _ODE_PARAM_MAP.items():
    SYSTEM_REGISTRY[_sid] = _fn

for _sid, (_fn, _pnames) in _MAP_PARAM_MAP.items():
    SYSTEM_REGISTRY[_sid] = _fn


# ---------------------------------------------------------------------------
# Default initial conditions
# ---------------------------------------------------------------------------

_DEFAULT_ICS: Dict[str, np.ndarray] = {
    "lorenz63": np.array([1.0, 1.0, 1.0]),
    "rossler": np.array([1.0, 1.0, 0.0]),
    "chen_system": np.array([-10.0, 0.0, 37.0]),
    "chua_circuit": np.array([0.7, 0.0, 0.0]),
    "double_pendulum": np.array([np.pi / 2.0, np.pi / 2.0, 0.0, 0.0]),
    "duffing_chaotic": np.array([0.1, 0.0]),
    "fitzhugh_nagumo": np.array([0.0, 0.0]),
    "hindmarsh_rose": np.array([-1.5, -10.0, 2.0]),
    "lorenz84": np.array([1.0, 1.0, 1.0]),
    "lorenz96": np.array([0.01, 0.0, 0.0, 0.0, 0.0]),
    "lotka_volterra": np.array([10.0, 5.0]),
    "mackey_glass": np.array([1.2]),
    "brusselator": np.array([1.0, 1.0]),
    "damped_oscillator": np.array([1.0, 0.0]),
    "damped_driven_pendulum_nonchaotic": np.array([0.1, 0.0]),
    "oregonator": np.array([0.5, 0.5]),
    "rikitake_dynamo": np.array([-1.4, -1.0, 2.0]),
    "shm": np.array([1.0, 0.0]),
    "vdp": np.array([2.0, 0.0]),
    "kuramoto_sivashinsky": None,
    "sine_gordon": None,
    "stochastic_ou": np.array([0.0]),
    "logistic_r4": np.array([0.1]),
    "logistic_r2_8": np.array([0.1]),
    "henon": np.array([0.0, 0.0]),
    "ikeda_map": np.array([0.1, 0.1]),
    "arnold_cat_map": np.array([0.1, 0.3]),
    "bakers_map": np.array([0.1, 0.3]),
    "circle_map_quasiperiodic": np.array([0.0]),
    "standard_map": np.array([0.1, 0.0]),
}

_DEFAULT_PARAMS: Dict[str, Dict[str, float]] = {
    "lorenz63": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    "rossler": {"a": 0.2, "b": 0.2, "c": 5.7},
    "chen_system": {"a": 35.0, "b": 3.0, "c": 28.0},
    "chua_circuit": {"alpha": 15.6, "beta": 28.0, "m0": -1.143, "m1": -0.714},
    "double_pendulum": {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81},
    "duffing_chaotic": {"delta": 0.3, "gamma": 0.5, "omega": 1.2},
    "fitzhugh_nagumo": {"epsilon": 0.08, "a": 0.7, "b": 0.8, "I": 0.5},
    "hindmarsh_rose": {"I": 3.5},
    "lorenz84": {"a": 0.25, "b": 4.0, "F": 8.0, "G": 1.0},
    "lorenz96": {"N": 5, "F": 8.0},
    "lotka_volterra": {"alpha": 1.5, "beta": 1.0, "gamma": 3.0, "delta": 1.0},
    "mackey_glass": {"beta_mg": 0.2, "gamma_mg": 0.1, "tau": 17.0, "n": 10.0},
    "brusselator": {"A": 1.0, "B": 3.0},
    "damped_oscillator": {"omega": 1.0, "zeta": 0.1},
    "damped_driven_pendulum_nonchaotic": {"beta_ddp": 0.5, "g_over_L": 1.0, "A": 1.2, "omega": 2.0 / 3.0},
    "oregonator": {"epsilon": 0.1, "q": 8e-4, "f": 1.0},
    "rikitake_dynamo": {"mu": 1.0, "a": 5.0},
    "shm": {"omega": 1.0},
    "vdp": {"mu": 1.0},
    "kuramoto_sivashinsky": {"L": 22.0},
    "sine_gordon": {"L": 20.0},
    "stochastic_ou": {"theta": 1.0, "mu": 0.0, "sigma": 0.5},
    "logistic_r4": {"r": 4.0},
    "logistic_r2_8": {"r": 2.8},
    "henon": {"a": 1.4, "b": 0.3},
    "ikeda_map": {"u": 0.9},
    "arnold_cat_map": {},
    "bakers_map": {},
    "circle_map_quasiperiodic": {"Omega": 0.6066, "K": 0.5},
    "standard_map": {"K": 0.97},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_system_type(system_id: str) -> str:
    """Return 'ode' or 'map' for a system."""
    if system_id not in _SYSTEM_TYPES:
        raise ValueError(f"Unknown system: {system_id}")
    return _SYSTEM_TYPES[system_id]


def get_default_ic(system_id: str) -> np.ndarray:
    """Return sensible default initial conditions for a system.

    For PDE-based systems (kuramoto_sivashinsky, sine_gordon) a small
    random perturbation on a 32-point grid is returned using seed 42.
    """
    if system_id not in _DEFAULT_ICS:
        raise ValueError(f"Unknown system: {system_id}")

    ic = _DEFAULT_ICS[system_id]
    if ic is not None:
        return ic.copy()

    rng = np.random.default_rng(42)
    N_grid = 32
    if system_id == "kuramoto_sivashinsky":
        return 0.01 * rng.standard_normal(N_grid)
    if system_id == "sine_gordon":
        u0 = 0.01 * rng.standard_normal(N_grid)
        v0 = np.zeros(N_grid)
        return np.concatenate([u0, v0])
    return np.array([0.0])


def _generate_mackey_glass(
    params: Dict[str, float],
    ic: np.ndarray,
    t_span: Tuple[float, float],
    n_points: int,
) -> np.ndarray:
    """Generate Mackey-Glass trajectory via fixed-timestep delay discretization.

    Args:
        params: Must contain beta_mg, gamma_mg, tau, n.
        ic: Scalar initial value (used to fill history).
        t_span: Integration interval.
        n_points: Number of output points.
    """
    beta_mg = params.get("beta_mg", 0.2)
    gamma_mg = params.get("gamma_mg", 0.1)
    tau = params.get("tau", 17.0)
    n_exp = params.get("n", 10.0)

    t0, t1 = t_span
    dt = (t1 - t0) / n_points
    delay_steps = max(1, int(tau / dt))
    total_steps = n_points + delay_steps

    x = np.empty(total_steps)
    x[:delay_steps] = ic[0] if len(ic) >= 1 else 1.2

    for i in range(delay_steps, total_steps):
        x_tau = x[i - delay_steps]
        dx = beta_mg * x_tau / (1.0 + x_tau ** n_exp) - gamma_mg * x[i - 1]
        x[i] = x[i - 1] + dt * dx

    return x[delay_steps:].reshape(-1, 1)


def _generate_stochastic_ou(
    params: Dict[str, float],
    ic: np.ndarray,
    t_span: Tuple[float, float],
    n_points: int,
    seed: int,
) -> np.ndarray:
    """Generate Ornstein-Uhlenbeck trajectory via Euler-Maruyama.

    Args:
        params: Must contain theta, mu, sigma.
        ic: Initial state.
        t_span: Integration interval.
        n_points: Number of output points.
        seed: Random seed for reproducibility.
    """
    theta = params.get("theta", 1.0)
    mu_ou = params.get("mu", 0.0)
    sigma = params.get("sigma", 0.5)

    rng = np.random.default_rng(seed)
    t0, t1 = t_span
    dt = (t1 - t0) / n_points
    sqrt_dt = np.sqrt(dt)

    dim = len(ic)
    traj = np.empty((n_points, dim))
    traj[0] = ic.copy()

    for i in range(1, n_points):
        dW = rng.standard_normal(dim) * sqrt_dt
        traj[i] = traj[i - 1] + theta * (mu_ou - traj[i - 1]) * dt + sigma * dW

    return traj


def generate_ode_trajectory(
    system_id: str,
    params: Dict[str, float],
    ic: Optional[np.ndarray] = None,
    t_span: Tuple[float, float] = (0, 100),
    n_points: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Generate trajectory for an ODE system using RK45.

    Args:
        system_id: Identifier from the benchmark (e.g. 'lorenz63').
        params: System parameters as a dict.
        ic: Initial conditions. If None, defaults are used.
        t_span: Integration interval (t0, t1).
        n_points: Number of output time points.
        seed: Random seed for IC perturbation if needed.

    Returns:
        Array of shape (n_points, state_dim).
    """
    if get_system_type(system_id) != "ode":
        raise ValueError(f"{system_id} is not an ODE system")

    merged = {**_DEFAULT_PARAMS.get(system_id, {}), **params}

    if ic is None:
        rng = np.random.default_rng(seed)
        ic = get_default_ic(system_id)
        ic = ic + 1e-10 * rng.standard_normal(ic.shape)

    if system_id == "mackey_glass":
        return _generate_mackey_glass(merged, ic, t_span, n_points)

    if system_id == "stochastic_ou":
        return _generate_stochastic_ou(merged, ic, t_span, n_points, seed)

    if system_id == "lorenz96":
        N_l96 = int(merged.pop("N", 5))
        if len(ic) != N_l96:
            rng = np.random.default_rng(seed)
            ic = rng.standard_normal(N_l96) * 0.01
        merged["N"] = N_l96

    rhs = _make_ode_callable(system_id, merged)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        rhs,
        t_span,
        ic,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9,
        max_step=0.01 if system_id in ("double_pendulum", "kuramoto_sivashinsky", "sine_gordon") else np.inf,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed for {system_id}: {sol.message}")

    return sol.y.T


def generate_map_trajectory(
    system_id: str,
    params: Dict[str, float],
    ic: Optional[np.ndarray] = None,
    n_iter: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Generate trajectory for a discrete map via direct iteration.

    Args:
        system_id: Identifier from the benchmark (e.g. 'henon').
        params: Map parameters as a dict.
        ic: Initial state. If None, defaults are used.
        n_iter: Number of iterations.
        seed: Random seed for IC perturbation if needed.

    Returns:
        Array of shape (n_iter, state_dim).
    """
    if get_system_type(system_id) != "map":
        raise ValueError(f"{system_id} is not a map system")

    merged = {**_DEFAULT_PARAMS.get(system_id, {}), **params}

    if ic is None:
        rng = np.random.default_rng(seed)
        ic = get_default_ic(system_id)
        ic = ic + 1e-10 * rng.standard_normal(ic.shape)

    step_fn = _make_map_callable(system_id, merged)
    dim = len(ic)
    traj = np.empty((n_iter, dim))
    traj[0] = ic.copy()

    for i in range(1, n_iter):
        traj[i] = step_fn(traj[i - 1])

    return traj
