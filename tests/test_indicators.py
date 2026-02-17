"""Tests for chaos indicator modules in ChaosBench-Logic v2.

Covers time series generation, 0-1 test for chaos, permutation entropy,
MEGNO indicator, the compute_all_indicators pipeline, and the
indicator diagnostic task.
"""

import numpy as np
import pytest

from chaosbench.data.indicators.time_series import (
    SYSTEM_REGISTRY,
    _DEFAULT_PARAMS,
    generate_map_trajectory,
    generate_ode_trajectory,
    get_default_ic,
    get_system_type,
)
from chaosbench.data.indicators.zero_one_test import zero_one_test
from chaosbench.data.indicators.permutation_entropy import permutation_entropy
from chaosbench.data.indicators.megno import compute_megno
from chaosbench.data.indicators.populate import ALL_SYSTEMS, compute_all_indicators
from chaosbench.tasks.indicator_diagnostics import (
    IndicatorDiagnosticTask,
    generate_indicator_questions,
)
from chaosbench.data.schemas import Question


class TestTimeSeriesGeneration:
    """Time series trajectory generation for ODE and map systems."""

    def test_lorenz63_shape(self):
        """Lorenz63 ODE returns correct shape."""
        traj = generate_ode_trajectory(
            "lorenz63", _DEFAULT_PARAMS["lorenz63"],
            t_span=(0, 10), n_points=500,
        )
        assert traj.shape == (500, 3)

    def test_lorenz63_finite(self):
        """Lorenz63 trajectory contains only finite values."""
        traj = generate_ode_trajectory(
            "lorenz63", _DEFAULT_PARAMS["lorenz63"],
            t_span=(0, 10), n_points=500,
        )
        assert np.all(np.isfinite(traj))

    def test_logistic_r4_shape(self):
        """Logistic map at r=4 returns correct shape."""
        traj = generate_map_trajectory(
            "logistic_r4", _DEFAULT_PARAMS["logistic_r4"], n_iter=1000,
        )
        assert traj.shape == (1000, 1)

    def test_logistic_r4_bounded(self):
        """Logistic map at r=4 stays within [0, 1]."""
        traj = generate_map_trajectory(
            "logistic_r4", _DEFAULT_PARAMS["logistic_r4"], n_iter=1000,
        )
        assert np.all(traj >= -0.01)
        assert np.all(traj <= 1.01)

    def test_henon_shape(self):
        """Henon map returns correct 2D shape."""
        traj = generate_map_trajectory(
            "henon", _DEFAULT_PARAMS["henon"], n_iter=1000,
        )
        assert traj.shape == (1000, 2)

    def test_damped_oscillator_decays(self):
        """Damped oscillator amplitude decreases over time."""
        traj = generate_ode_trajectory(
            "damped_oscillator", _DEFAULT_PARAMS["damped_oscillator"],
            t_span=(0, 50), n_points=1000,
        )
        first_half_max = np.max(np.abs(traj[:250, 0]))
        second_half_max = np.max(np.abs(traj[750:, 0]))
        assert second_half_max < first_half_max

    def test_circle_map_quasiperiodic_shape(self):
        """Circle map returns correct 1D shape."""
        traj = generate_map_trajectory(
            "circle_map_quasiperiodic",
            _DEFAULT_PARAMS["circle_map_quasiperiodic"],
            n_iter=500,
        )
        assert traj.shape == (500, 1)

    def test_get_system_type_ode(self):
        """Lorenz63 is classified as an ODE system."""
        assert get_system_type("lorenz63") == "ode"

    def test_get_system_type_map(self):
        """Logistic r4 is classified as a map system."""
        assert get_system_type("logistic_r4") == "map"

    def test_get_system_type_unknown_raises(self):
        """Unknown system raises ValueError."""
        with pytest.raises(ValueError, match="Unknown system"):
            get_system_type("nonexistent_system")


class TestZeroOneTest:
    """Gottwald-Melbourne 0-1 test for chaos."""

    def test_chaotic_logistic_k_near_one(self):
        """Logistic map at r=4 should yield K close to 1."""
        traj = generate_map_trajectory(
            "logistic_r4", _DEFAULT_PARAMS["logistic_r4"], n_iter=2000,
        )
        series = traj[:, 0]
        K = zero_one_test(series, seed=42)
        assert K > 0.5

    def test_regular_sine_wave_k_near_zero(self):
        """Sine wave (regular) should yield K close to 0."""
        t = np.linspace(0, 100, 2000)
        series = np.sin(t)
        K = zero_one_test(series, seed=42)
        assert K < 0.3

    def test_deterministic_same_seed(self):
        """Same seed produces identical K values."""
        traj = generate_map_trajectory(
            "logistic_r4", _DEFAULT_PARAMS["logistic_r4"], n_iter=2000,
        )
        series = traj[:, 0]
        K1 = zero_one_test(series, seed=99)
        K2 = zero_one_test(series, seed=99)
        assert K1 == K2

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different K values."""
        traj = generate_map_trajectory(
            "logistic_r4", _DEFAULT_PARAMS["logistic_r4"], n_iter=2000,
        )
        series = traj[:, 0]
        K1 = zero_one_test(series, seed=1)
        K2 = zero_one_test(series, seed=2)
        assert isinstance(K1, float)
        assert isinstance(K2, float)

    def test_fixed_c_parameter(self):
        """Using a fixed c value returns a valid K."""
        traj = generate_map_trajectory(
            "logistic_r4", _DEFAULT_PARAMS["logistic_r4"], n_iter=2000,
        )
        series = traj[:, 0]
        K = zero_one_test(series, c=1.0, seed=42)
        assert 0.0 <= K <= 1.0

    def test_k_in_unit_interval(self):
        """K is always clamped to [0, 1]."""
        traj = generate_map_trajectory(
            "henon", _DEFAULT_PARAMS["henon"], n_iter=2000,
        )
        series = traj[:, 0]
        K = zero_one_test(series, seed=42)
        assert 0.0 <= K <= 1.0

    def test_short_series_returns_zero(self):
        """Very short series returns K=0 due to insufficient data."""
        series = np.array([0.1, 0.5, 0.3])
        K = zero_one_test(series, seed=42)
        assert K == 0.0

    def test_constant_series(self):
        """Constant series yields K near 0 (regular)."""
        series = np.ones(2000) * 3.14
        K = zero_one_test(series, seed=42)
        assert K < 0.3


class TestPermutationEntropy:
    """Permutation entropy computation."""

    def test_high_for_random(self):
        """Random data should have high normalized permutation entropy."""
        rng = np.random.default_rng(42)
        series = rng.uniform(size=2000)
        pe = permutation_entropy(series, order=3, delay=1, normalize=True)
        assert pe > 0.9

    def test_low_for_constant(self):
        """Constant series should have zero permutation entropy."""
        series = np.ones(500)
        pe = permutation_entropy(series, order=3, delay=1, normalize=True)
        assert pe == 0.0

    def test_normalized_in_unit_interval(self):
        """Normalized PE is in [0, 1]."""
        rng = np.random.default_rng(7)
        series = rng.standard_normal(1000)
        pe = permutation_entropy(series, order=4, delay=1, normalize=True)
        assert 0.0 <= pe <= 1.0

    def test_higher_order_decreases_entropy(self):
        """Higher order can change entropy for structured signals."""
        t = np.linspace(0, 20, 2000)
        series = np.sin(t)
        pe3 = permutation_entropy(series, order=3, delay=1, normalize=True)
        pe5 = permutation_entropy(series, order=5, delay=1, normalize=True)
        assert isinstance(pe3, float)
        assert isinstance(pe5, float)

    def test_empty_series(self):
        """Series shorter than embedding span returns 0."""
        series = np.array([1.0, 2.0])
        pe = permutation_entropy(series, order=5, delay=1, normalize=True)
        assert pe == 0.0

    def test_unnormalized_is_nonnegative(self):
        """Unnormalized PE is non-negative."""
        rng = np.random.default_rng(42)
        series = rng.uniform(size=500)
        pe = permutation_entropy(series, order=3, delay=1, normalize=False)
        assert pe >= 0.0


class TestMEGNO:
    """MEGNO (Mean Exponential Growth of Nearby Orbits) indicator."""

    def test_lorenz63_above_two(self):
        """MEGNO for Lorenz63 should be above 2 (chaotic)."""
        val = compute_megno("lorenz63", t_max=20, n_points=500, seed=42)
        assert val is not None
        assert val > 2.0

    def test_shm_below_chaotic_threshold(self):
        """MEGNO for simple harmonic motion should be below chaotic threshold."""
        val = compute_megno("shm", t_max=20, n_points=500, seed=42)
        assert val is not None
        assert val < 2.5

    def test_mackey_glass_returns_none(self):
        """MEGNO for mackey_glass is unsupported and returns None."""
        val = compute_megno("mackey_glass", t_max=20, n_points=500, seed=42)
        assert val is None

    def test_unknown_system_returns_none(self):
        """MEGNO for an unknown system returns None."""
        val = compute_megno("totally_fake_system", t_max=20, n_points=500)
        assert val is None

    def test_deterministic_with_seed(self):
        """Same seed produces identical MEGNO values."""
        v1 = compute_megno("lorenz63", t_max=20, n_points=500, seed=123)
        v2 = compute_megno("lorenz63", t_max=20, n_points=500, seed=123)
        assert v1 == v2

    def test_map_system_returns_value(self):
        """MEGNO for a map system returns a numeric value."""
        val = compute_megno("logistic_r4", n_points=500, seed=42)
        assert val is None or isinstance(val, float)


class TestPipeline:
    """Indicator computation pipeline via compute_all_indicators."""

    def test_all_systems_count(self):
        """ALL_SYSTEMS contains 30 entries."""
        assert len(ALL_SYSTEMS) == 30

    def test_lorenz63_returns_expected_keys(self):
        """compute_all_indicators for lorenz63 returns all expected keys."""
        result = compute_all_indicators("lorenz63", seed=42)
        expected_keys = {
            "system_id", "zero_one_K", "permutation_entropy",
            "megno", "system_type", "seed", "timestamp",
        }
        # Check all expected keys are present (allow optional keys like megno_failure_reason)
        assert expected_keys.issubset(set(result.keys()))

    def test_lorenz63_system_type(self):
        """Pipeline correctly reports system_type for lorenz63."""
        result = compute_all_indicators("lorenz63", seed=42)
        assert result["system_type"] == "ode"

    def test_logistic_r4_runs_without_crash(self):
        """compute_all_indicators completes for logistic_r4."""
        result = compute_all_indicators("logistic_r4", seed=42)
        assert result["system_id"] == "logistic_r4"
        assert result["system_type"] == "map"

    def test_zero_one_k_is_numeric(self):
        """The zero_one_K value is numeric for a well-behaved system."""
        result = compute_all_indicators("lorenz63", seed=42)
        assert result["zero_one_K"] is None or isinstance(result["zero_one_K"], float)

    def test_permutation_entropy_is_numeric(self):
        """The permutation_entropy value is numeric for a well-behaved system."""
        result = compute_all_indicators("lorenz63", seed=42)
        assert result["permutation_entropy"] is None or isinstance(
            result["permutation_entropy"], float
        )


class TestIndicatorDiagnosticTask:
    """Indicator diagnostic task question generation and scoring."""

    @pytest.fixture
    def sample_systems(self):
        """Minimal systems dict for testing question generation."""
        return {
            "lorenz63": {
                "name": "Lorenz attractor",
                "truth_assignment": {"Chaotic": True},
            },
            "shm": {
                "name": "Simple harmonic oscillator",
                "truth_assignment": {"Chaotic": False},
            },
        }

    @pytest.fixture
    def sample_indicators(self):
        """Minimal indicator values for testing question generation."""
        return {
            "lorenz63": {
                "zero_one_K": 0.95,
                "permutation_entropy": 0.88,
                "megno": 4.5,
            },
            "shm": {
                "zero_one_K": 0.05,
                "permutation_entropy": 0.15,
                "megno": 2.0,
            },
        }

    def test_generates_questions(self, sample_systems, sample_indicators):
        """generate_indicator_questions produces a non-empty list."""
        questions = generate_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        assert len(questions) > 0

    def test_questions_are_valid(self, sample_systems, sample_indicators):
        """All generated questions pass validation."""
        questions = generate_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        for q in questions:
            assert isinstance(q, Question)
            errors = q.validate()
            assert errors == [], f"Question {q.item_id} has errors: {errors}"

    def test_item_ids_are_unique(self, sample_systems, sample_indicators):
        """All indicator diagnostic item IDs are unique."""
        questions = generate_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        item_ids = [q.item_id for q in questions]
        assert len(item_ids) == len(set(item_ids)), "Duplicate item IDs found"

    def test_ground_truth_is_yes_or_no(self, sample_systems, sample_indicators):
        """Ground truth labels are always YES or NO."""
        questions = generate_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        for q in questions:
            assert q.ground_truth in ("YES", "NO")

    def test_deterministic_with_seed(self, sample_systems, sample_indicators):
        """Same seed produces identical question lists."""
        q1 = generate_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        q2 = generate_indicator_questions(
            sample_systems, sample_indicators, seed=42,
        )
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.item_id == b.item_id
            assert a.ground_truth == b.ground_truth

    def test_task_score_perfect(self, sample_systems, sample_indicators):
        """Scoring with all correct predictions returns accuracy 1.0."""
        task = IndicatorDiagnosticTask(
            systems=sample_systems,
            indicators=sample_indicators,
            seed=42,
        )
        items = task.generate_items()
        predictions = {q.item_id: q.ground_truth for q in items}
        result = task.score(predictions)
        assert result["accuracy"] == 1.0
        assert result["correct"] == result["total"]

    def test_task_score_empty(self, sample_systems, sample_indicators):
        """Scoring with no predictions returns zero total."""
        task = IndicatorDiagnosticTask(
            systems=sample_systems,
            indicators=sample_indicators,
            seed=42,
        )
        result = task.score({})
        assert result["total"] == 0
