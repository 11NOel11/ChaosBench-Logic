"""Integration tests for v2 dataset build process."""

import json
import os

import pytest

from chaosbench.data.schemas import Question
from chaosbench.data.bifurcations import BIFURCATION_DATA
from chaosbench.tasks.regime_transition import RegimeTransitionTask, SYSTEM_DISPLAY_NAMES
from chaosbench.tasks.fol_inference import generate_fol_questions
from chaosbench.tasks.extended_systems import generate_extended_system_questions
from chaosbench.tasks.cross_indicator import generate_cross_indicator_questions


class TestBifurcationDataExpansion:
    """Bifurcation data has been expanded with new systems."""

    def test_seven_systems_in_bifurcation_data(self):
        """BIFURCATION_DATA now contains 7 systems."""
        assert len(BIFURCATION_DATA) == 7

    def test_new_systems_present(self):
        """All 4 new systems are present."""
        for sid in ["rossler", "duffing_chaotic", "chua_circuit", "vdp"]:
            assert sid in BIFURCATION_DATA

    def test_display_names_match(self):
        """Every system in BIFURCATION_DATA has a display name."""
        for sid in BIFURCATION_DATA:
            assert sid in SYSTEM_DISPLAY_NAMES

    def test_vdp_is_nonchaotic(self):
        """VDP transitions do not contain any chaotic regimes."""
        from chaosbench.data.bifurcations import is_chaotic_regime
        vdp = BIFURCATION_DATA["vdp"]
        for t in vdp.transitions:
            assert not is_chaotic_regime(t.regime), (
                f"VDP regime '{t.regime}' should not be chaotic"
            )


class TestRegimeTransitionExpanded:
    """Regime transition task with expanded systems."""

    def test_generates_more_questions(self):
        """Expanded systems produce more questions than original 3."""
        task = RegimeTransitionTask(seed=42)
        questions = task.generate_items()
        assert len(questions) > 30

    def test_all_questions_valid(self):
        """All generated questions pass validation."""
        task = RegimeTransitionTask(seed=42)
        questions = task.generate_items()
        for q in questions:
            errors = q.validate()
            assert errors == [], f"{q.item_id}: {errors}"


class TestQuestionToJSONLConversion:
    """Question to JSONL dict conversion."""

    def test_basic_conversion(self):
        """Converts YES/NO to TRUE/FALSE correctly."""
        from scripts.build_v2_dataset import question_to_jsonl
        q = Question(
            item_id="test_001",
            question_text="Is the system chaotic?",
            system_id="lorenz63",
            task_family="atomic",
            ground_truth="YES",
        )
        d = question_to_jsonl(q)
        assert d["id"] == "test_001"
        assert d["question"] == "Is the system chaotic?"
        assert d["ground_truth"] == "TRUE"
        assert d["type"] == "atomic"
        assert d["system_id"] == "lorenz63"
        assert d["template"] == "V2"

    def test_no_conversion(self):
        """NO maps to FALSE."""
        from scripts.build_v2_dataset import question_to_jsonl
        q = Question(
            item_id="test_002",
            question_text="Is it periodic?",
            system_id="lorenz63",
            task_family="atomic",
            ground_truth="NO",
        )
        d = question_to_jsonl(q)
        assert d["ground_truth"] == "FALSE"


class TestJSONLValidation:
    """JSONL file format validation (runs after build)."""

    @pytest.fixture
    def batch_files(self):
        """Return paths to new batch files if they exist."""
        data_dir = "data"
        batches = [
            "batch8_indicator_diagnostics.jsonl",
            "batch9_regime_transitions.jsonl",
            "batch10_adversarial.jsonl",
            "batch11_consistency_paraphrase.jsonl",
            "batch12_fol_inference.jsonl",
            "batch13_extended_systems.jsonl",
            "batch14_cross_indicator.jsonl",
        ]
        existing = []
        for b in batches:
            path = os.path.join(data_dir, b)
            if os.path.isfile(path):
                existing.append(path)
        return existing

    def test_batch_files_exist(self, batch_files):
        """At least some batch files have been generated."""
        if not batch_files:
            pytest.skip("Batch files not yet generated (run build_v2_dataset.py first)")

    def test_valid_json_lines(self, batch_files):
        """Every line in each batch file is valid JSON."""
        if not batch_files:
            pytest.skip("Batch files not yet generated")
        for path in batch_files:
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        pytest.fail(f"{path}:{i} is not valid JSON")

    def test_ground_truth_values(self, batch_files):
        """Every ground_truth is TRUE or FALSE."""
        if not batch_files:
            pytest.skip("Batch files not yet generated")
        for path in batch_files:
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    gt = record.get("ground_truth")
                    assert gt in ("TRUE", "FALSE"), (
                        f"{path}:{i} has ground_truth={gt!r}"
                    )

    def test_required_fields(self, batch_files):
        """Every record has required fields."""
        if not batch_files:
            pytest.skip("Batch files not yet generated")
        required = {"id", "question", "ground_truth", "type", "system_id", "template"}
        for path in batch_files:
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    missing = required - set(record.keys())
                    assert not missing, (
                        f"{path}:{i} missing fields: {missing}"
                    )
