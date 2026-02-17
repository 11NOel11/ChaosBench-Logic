"""ChaosBench-Logic v2: Benchmarking LLMs on dynamical systems reasoning."""

__version__ = "2.1.0"

from chaosbench.data.schemas import (
    SystemInstance,
    Question,
    Dialogue,
    AnswerKey,
    DatasetConfig,
)
from chaosbench.logic.ontology import PREDICATES, KEYWORD_MAP
from chaosbench.logic.axioms import get_fol_rules, check_fol_violations
from chaosbench.logic.extract import extract_predicate_from_question
from chaosbench.logic.solver_repair import repair_assignment, validate_repair
from chaosbench.eval.metrics import (
    EvalResult,
    compute_summary,
    normalize_label,
)
from chaosbench.eval.runner import load_jsonl, load_batches
from chaosbench.eval.belief_dynamics import (
    hamming_distance,
    belief_divergence_curve,
    instability_score,
)
from chaosbench.models.prompt import ModelConfig, ModelClient, DummyEchoModel, make_model_client
from chaosbench.data.indicators.zero_one_test import zero_one_test
from chaosbench.data.indicators.permutation_entropy import permutation_entropy
from chaosbench.data.indicators.megno import compute_megno
from chaosbench.data.indicators.populate import compute_all_indicators
from chaosbench.data.bifurcations import BIFURCATION_DATA, get_regime_at_param
from chaosbench.tasks.regime_transition import RegimeTransitionTask
from chaosbench.tasks.indicator_diagnostics import IndicatorDiagnosticTask
from chaosbench.data.adversarial import generate_adversarial_set, CONFUSABLE_PAIRS
from chaosbench.tasks.hard_split import identify_hard_items, create_hard_split
from chaosbench.tasks.fol_inference import FOLInferenceTask, generate_fol_questions
from chaosbench.tasks.extended_systems import ExtendedSystemsTask, generate_extended_system_questions
from chaosbench.tasks.cross_indicator import CrossIndicatorTask, generate_cross_indicator_questions
from chaosbench.tasks.atomic import AtomicTask, generate_atomic_questions
from chaosbench.tasks.multi_hop import MultiHopTask, generate_multi_hop_questions
from chaosbench.tasks.perturbation_robustness import PerturbationRobustnessTask, generate_perturbation_questions
from chaosbench.data.splits import assign_splits, validate_splits, get_split_items
from chaosbench.eval.cache import ResponseCache
from chaosbench.eval.metrics import compute_axis_metrics, AxisMetricResult
