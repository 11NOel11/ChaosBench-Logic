"""Atomic predicate questions scaled to all systems.

This module generates direct factual questions about dynamical systems predicates
scaled across all 30 benchmark systems. It tests whether models can correctly
identify single predicate properties for each system.

The task probes all 11 predicates from the ontology for all systems in the benchmark,
providing comprehensive coverage of atomic predicate knowledge.

Question Format
---------------
Eight semantically distinct template variations with natural language phrasings:
1. "Is the {system_name} {predicate_display}?"
2. "Does the {system_name} exhibit {predicate_display}?"
3. "Would you classify the {system_name} as {predicate_display}?"
4. "Can we characterize the {system_name} as {predicate_display}?"
5. "Is it accurate to say that the {system_name} is {predicate_display}?"
6. "Should the {system_name} be considered {predicate_display}?"
7. "Does the behavior of the {system_name} qualify as {predicate_display}?"
8. "Is it correct to describe the {system_name} as {predicate_display}?"

Each system-predicate pair can generate multiple questions using different templates.

Predicates Probed
-----------------
All 11 predicates from the ontology:
1. Chaotic - Exhibits chaotic dynamics
2. Deterministic - Fully determined by initial conditions
3. PosLyap - Has positive Lyapunov exponent
4. Sensitive - Sensitive to initial conditions
5. StrangeAttr - Has strange attractor
6. PointUnpredictable - Pointwise unpredictable
7. StatPredictable - Statistically predictable
8. QuasiPeriodic - Has quasi-periodic motion
9. Random - Exhibits random/stochastic behavior
10. FixedPointAttr - Has fixed point attractor
11. Periodic - Has periodic orbits

Expected Distribution
---------------------
- Base: 330-2640 questions (30 systems × 11 predicates × 1-8 templates)
- Scalable: Up to 1815-14520 with all 165 systems (165 × 11 × 1-8)
- Coverage: Configurable across core and dysts systems
- Template variety: 8 semantically distinct phrasings
- Ground truth: Based on truth_assignment in system metadata
- Balance: Approximately 50% YES, 50% NO (via interleaving)

Implementation Notes
--------------------
- Questions are shuffled using provided seed for reproducibility
- YES/NO balance is enforced by sorting and interleaving ground truths
- Optional target_count parameter allows truncating to specific size
- All questions grounded in system's truth_assignment from JSON metadata

Design Rationale
----------------
This task ensures that:
1. All systems and all predicates are comprehensively tested
2. Models are evaluated on fundamental atomic predicate knowledge
3. Results provide detailed per-predicate accuracy breakdowns
4. Questions are naturally phrased with variety in templates

See Also
--------
- `chaosbench.logic.ontology.PREDICATES` : List of 11 predicates
- System metadata files in `systems/` directory
- `tests/test_atomic.py` : Test suite (if created)
"""

import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chaosbench.data.schemas import Question
from chaosbench.logic.ontology import PREDICATES


PREDICATE_DISPLAY = {
    "Chaotic": "chaotic",
    "Deterministic": "deterministic",
    "PosLyap": "having a positive Lyapunov exponent",
    "Sensitive": "sensitive to initial conditions",
    "StrangeAttr": "having a strange attractor",
    "PointUnpredictable": "pointwise unpredictable",
    "StatPredictable": "statistically predictable",
    "QuasiPeriodic": "quasi-periodic",
    "Random": "random",
    "FixedPointAttr": "having a fixed point attractor",
    "Periodic": "periodic",
    # v2.2 Extension: New predicates for 4-5 hop chains
    "Dissipative": "dissipative (volume-contracting)",
    "Bounded": "bounded",
    "Mixing": "mixing",
    "Ergodic": "ergodic",
    # v2.3 Extension: 12 new predicates from metadata dimensions
    "HyperChaotic": "hyperchaotic",
    "Conservative": "conservative (Hamiltonian)",
    "HighDimensional": "high-dimensional (high Kaplan-Yorke dimension)",
    "Multifractal": "multifractal",
    "HighDimSystem": "a high-dimensional system (state space dimension ≥ 4)",
    "ContinuousTime": "a continuous-time system",
    "DiscreteTime": "a discrete-time map",
    "DelaySystem": "a delay differential equation system",
    "Forced": "externally forced (non-autonomous)",
    "Autonomous": "autonomous",
    "StrongMixing": "strongly mixing",
    "WeakMixing": "weakly mixing",
}

# Short formal definitions used in Frame B (definition-based) templates.
# Each value completes the sentence "a system is {predicate_display} if {definition}".
PREDICATE_DEFINITIONS = {
    "Chaotic": "it exhibits sensitive dependence on initial conditions, has a positive largest Lyapunov exponent, and is deterministic",
    "Deterministic": "its future state is uniquely determined by its current state with no randomness",
    "PosLyap": "nearby trajectories diverge exponentially, as measured by a positive largest Lyapunov exponent",
    "Sensitive": "arbitrarily small perturbations to initial conditions lead to exponentially diverging trajectories",
    "StrangeAttr": "its long-term dynamics are confined to a fractal set with non-integer dimension in phase space",
    "PointUnpredictable": "individual long-term trajectories cannot be predicted reliably due to sensitivity to initial conditions",
    "StatPredictable": "statistical averages over trajectories (ensemble or time averages) can be computed reliably",
    "QuasiPeriodic": "its motion is composed of two or more incommensurate frequencies forming an invariant torus",
    "Random": "it incorporates genuine stochasticity or noise that cannot be explained by deterministic rules",
    "FixedPointAttr": "its trajectories converge asymptotically to a single stationary point in phase space",
    "Periodic": "its trajectories exactly repeat after a fixed finite time (period)",
    "Dissipative": "it contracts phase-space volume over time (sum of Lyapunov exponents < 0)",
    "Bounded": "all trajectories remain within a finite region of phase space for all time",
    "Mixing": "initially separated regions of phase space become uniformly spread over time under time evolution",
    "Ergodic": "time averages along trajectories equal ensemble (phase-space) averages almost everywhere",
    "HyperChaotic": "it has two or more positive Lyapunov exponents, producing chaos in multiple independent directions",
    "Conservative": "it preserves phase-space volume under time evolution (sum of Lyapunov exponents ≈ 0)",
    "HighDimensional": "its chaotic attractor has a Kaplan-Yorke dimension of at least 3.0",
    "Multifractal": "its attractor has a correlation dimension that differs significantly from its Kaplan-Yorke dimension",
    "HighDimSystem": "its state space has four or more dimensions",
    "ContinuousTime": "its equations of motion are ordinary differential equations in continuous time",
    "DiscreteTime": "its dynamics are defined by an iterated map applied at discrete time steps",
    "DelaySystem": "its current state depends on past states through an explicit time delay (DDE)",
    "Forced": "it is subject to an explicit external time-dependent forcing or driving term",
    "Autonomous": "its equations of motion have no explicit time dependence",
    "StrongMixing": "correlations between observables decay to zero as the time separation goes to infinity",
    "WeakMixing": "its time evolution is ergodic and has no eigenvalues of the Koopman operator on the unit circle except 1",
}

# Frame A: Direct predicate queries (original 8 templates)
TEMPLATES_A = [
    "Is the {name} {predicate_display}?",
    "Does the {name} exhibit {predicate_display}?",
    "Would you classify the {name} as {predicate_display}?",
    "Can we characterize the {name} as {predicate_display}?",
    "Is it accurate to say that the {name} is {predicate_display}?",
    "Should the {name} be considered {predicate_display}?",
    "Does the behavior of the {name} qualify as {predicate_display}?",
    "Is it correct to describe the {name} as {predicate_display}?",
]

# Frame B: Definition-based templates (8 new)
# Uses {definition} from PREDICATE_DEFINITIONS; ground truth same as base question.
TEMPLATES_B = [
    "A system is {predicate_display} if {definition}. Is the {name} such a system?",
    "By definition, {predicate_display} means {definition}. Does the {name} fit this description?",
    "Formally, {predicate_display} refers to a property where {definition}. Does the {name} satisfy this?",
    "In dynamical systems theory, a {predicate_display} system is one where {definition}. Is the {name} one?",
    "The property {predicate_display} is defined as: {definition}. Does the {name} possess this property?",
    "We say a system is {predicate_display} when {definition}. Would the {name} qualify?",
    "Mathematically, {predicate_display} means {definition}. Can this apply to the {name}?",
    "A system satisfies {predicate_display} if {definition}. Does the {name} satisfy this condition?",
]

# Frame C: Negation-frame templates (8 new)
# Ground truth is INVERTED relative to the base question: TRUE→NO, FALSE→YES.
TEMPLATES_C = [
    "Is it false that the {name} is {predicate_display}?",
    "Would it be incorrect to say the {name} is {predicate_display}?",
    "Can we rule out that the {name} is {predicate_display}?",
    "Is the claim '{name} is {predicate_display}' wrong?",
    "Should we reject the assertion that the {name} is {predicate_display}?",
    "Is the {name} NOT {predicate_display}?",
    "Would you say the {name} lacks the property of being {predicate_display}?",
    "Is it untrue that the {name} exhibits {predicate_display}?",
]

# Frame D: Comparative/contextual templates (8 new)
# Ground truth same as base question.
TEMPLATES_D = [
    "Compared to a purely periodic oscillator, is the {name} {predicate_display}?",
    "Among dynamical systems, would the {name} be classified as {predicate_display}?",
    "A physicist studying the {name} would say it is {predicate_display}: true or false?",
    "If you were to categorize the {name}, would {predicate_display} be an accurate label?",
    "In the context of nonlinear dynamics, is the {name} considered {predicate_display}?",
    "From a dynamical systems perspective, does the {name} qualify as {predicate_display}?",
    "An expert in chaos theory would describe the {name} as {predicate_display}: agree or disagree?",
    "In standard mathematical parlance, is the {name} {predicate_display}?",
]

# Combined template list (Frame A) for backward compatibility
TEMPLATES = TEMPLATES_A

# All templates with their frame labels and inversion flags
ALL_TEMPLATE_FRAMES = [
    (TEMPLATES_A, "A", False),  # (templates, frame_id, invert_ground_truth)
    (TEMPLATES_B, "B", False),
    (TEMPLATES_C, "C", True),   # Frame C inverts ground truth
    (TEMPLATES_D, "D", False),
]


def _load_systems(systems_dir: str = "systems") -> Dict[str, Dict]:
    """Load system JSONs from directory.

    Args:
        systems_dir: Path to systems directory.

    Returns:
        Dict mapping system_id to system data.
    """
    systems = {}
    if not os.path.isdir(systems_dir):
        return systems
    for fname in sorted(os.listdir(systems_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(systems_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            systems[sid] = data
    return systems


def generate_atomic_questions(
    systems: Dict[str, Dict],
    seed: int = 42,
    target_count: Optional[int] = None,
    templates_per_predicate: int = 1,
    enable_multiframe: bool = False,
    frames: Optional[List[str]] = None,
) -> List[Question]:
    """Generate atomic predicate questions for all systems.

    Args:
        systems: Dict mapping system_id to system data with truth_assignment.
        seed: Random seed for reproducibility.
        target_count: Optional target number of questions (truncates if provided).
        templates_per_predicate: Number of template variants to use per
            system-predicate pair (1-8). Default 1 for backward compatibility.
        enable_multiframe: If True, generate questions from all 4 template frames
            (A, B, C, D) instead of only Frame A. Default False for backward compat.
        frames: Optional list of frame IDs to use (e.g. ["A", "B", "C", "D"]).
            Only used when enable_multiframe=True.

    Returns:
        List of Question objects with roughly balanced YES/NO answers.
    """
    rng = random.Random(seed)
    questions: List[Question] = []
    counter = [0]  # Mutable list for shared state

    # Determine which template frames to use
    if enable_multiframe:
        active_frames = [
            (tmpl_list, frame_id, invert)
            for tmpl_list, frame_id, invert in ALL_TEMPLATE_FRAMES
            if frames is None or frame_id in frames
        ]
    else:
        # Backward-compatible: only Frame A
        active_frames = [(TEMPLATES_A, "A", False)]

    # Total templates across active frames (for clamping templates_per_predicate)
    n_frame_a_templates = len(TEMPLATES_A)
    templates_per_predicate = max(1, min(templates_per_predicate, n_frame_a_templates))

    # Generate all system-predicate combinations
    for sid in sorted(systems.keys()):
        sys_data = systems[sid]
        truth = sys_data.get("truth_assignment", {})
        name = sys_data.get("name", sid)

        for pred in PREDICATES:
            if pred not in PREDICATE_DISPLAY:
                continue
            pred_disp = PREDICATE_DISPLAY[pred]
            pred_defn = PREDICATE_DEFINITIONS.get(pred, pred_disp)
            gt_val = truth.get(pred, False)

            for tmpl_list, frame_id, invert_gt in active_frames:
                # Determine ground truth for this frame
                # Frame C inverts the answer; all other frames keep it the same
                if invert_gt:
                    ground_truth = "NO" if gt_val else "YES"
                else:
                    ground_truth = "YES" if gt_val else "NO"

                # For Frame C (negation), limit to 1 template per predicate to
                # avoid trivially easy negations flooding the dataset.
                if frame_id == "C":
                    # Pick one deterministic template per (sid, pred)
                    _key_bytes = f"{sid}:{pred}:C".encode()
                    _stable_int = int(hashlib.sha256(_key_bytes).hexdigest(), 16)
                    c_rng = random.Random(seed ^ (_stable_int & 0xFFFFFFFF))
                    frame_templates = [c_rng.choice(tmpl_list)]
                else:
                    # FALSE predicates use all available frame templates to enrich
                    # the minority pool before balancing.
                    if not gt_val:
                        frame_templates = tmpl_list
                    elif templates_per_predicate >= len(tmpl_list):
                        frame_templates = tmpl_list
                    else:
                        _key_bytes = f"{sid}:{pred}:{frame_id}".encode()
                        _stable_int = int(hashlib.sha256(_key_bytes).hexdigest(), 16)
                        local_rng = random.Random(seed ^ (_stable_int & 0xFFFFFFFF))
                        frame_templates = local_rng.sample(tmpl_list, templates_per_predicate)

                for template in frame_templates:
                    counter[0] += 1
                    # Format template (Frame B uses {definition})
                    try:
                        q_text = template.format(
                            name=name,
                            predicate_display=pred_disp,
                            definition=pred_defn,
                        )
                    except KeyError:
                        q_text = template.format(
                            name=name,
                            predicate_display=pred_disp,
                        )

                    # Compute template index within the frame's list
                    try:
                        tmpl_idx = tmpl_list.index(template)
                    except ValueError:
                        tmpl_idx = 0

                    questions.append(Question(
                        item_id=f"atomic_{counter[0]:04d}",
                        question_text=q_text,
                        system_id=sid,
                        task_family="atomic",
                        ground_truth=ground_truth,
                        predicates=[pred],
                        metadata={
                            "question_type": "atomic_predicate",
                            "predicate": pred,
                            "template_frame": frame_id,
                            "template_index": tmpl_idx,
                            "template_id": f"{frame_id}{tmpl_idx:02d}",
                            "negated": invert_gt,
                        },
                    ))

    # Enforce strict 50/50 YES/NO balance by downsampling the majority class.
    yes_questions = [q for q in questions if q.ground_truth == "YES"]
    no_questions = [q for q in questions if q.ground_truth == "NO"]

    rng.shuffle(yes_questions)
    rng.shuffle(no_questions)

    min_pool = min(len(yes_questions), len(no_questions))
    if target_count is not None:
        per_class = min(target_count // 2, min_pool)
    else:
        per_class = min_pool

    balanced_questions = yes_questions[:per_class] + no_questions[:per_class]

    # Shuffle the balanced list
    rng.shuffle(balanced_questions)

    return balanced_questions


@dataclass
class AtomicTask:
    """Task for testing atomic predicate knowledge across all systems.

    Attributes:
        task_family: Always "atomic".
        systems: Dict mapping system_id to system data.
        seed: Random seed for reproducibility.
        target_count: Optional target number of questions.
        templates_per_predicate: Number of template variants per predicate (1-8).
        enable_multiframe: If True, use all 4 linguistic frames (A/B/C/D).
        frames: Optional list of specific frame IDs to use.
    """

    task_family: str = "atomic"
    systems: Dict[str, Dict] = field(default_factory=dict)
    seed: int = 42
    target_count: Optional[int] = None
    templates_per_predicate: int = 1
    enable_multiframe: bool = False
    frames: Optional[List[str]] = None

    def generate_items(self) -> List[Question]:
        """Generate atomic predicate questions.

        Returns:
            List of Question objects.
        """
        if not self.systems:
            self.systems = _load_systems()
        return generate_atomic_questions(
            self.systems,
            self.seed,
            self.target_count,
            self.templates_per_predicate,
            self.enable_multiframe,
            self.frames,
        )

    def score(self, predictions: Dict[str, str]) -> Dict[str, Any]:
        """Score model predictions against ground truth.

        Args:
            predictions: Dict mapping item_id to predicted label.

        Returns:
            Dict with accuracy and per-predicate accuracy breakdowns.
        """
        items = self.generate_items()
        correct = 0
        total = 0
        by_predicate: Dict[str, List[bool]] = defaultdict(list)

        for q in items:
            pred = predictions.get(q.item_id)
            if pred is None:
                continue
            total += 1
            is_correct = pred.upper() == q.ground_truth
            if is_correct:
                correct += 1

            # Track per-predicate accuracy
            predicate = q.metadata.get("predicate")
            if predicate:
                by_predicate[predicate].append(is_correct)

        predicate_accuracy = {
            k: sum(v) / len(v) for k, v in sorted(by_predicate.items())
        }

        return {
            "accuracy": correct / total if total > 0 else None,
            "correct": correct,
            "total": total,
            "predicate_accuracy": predicate_accuracy,
        }
