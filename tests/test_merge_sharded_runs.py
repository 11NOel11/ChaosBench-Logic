"""Tests for sharded run merge utilities."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest

from chaosbench.eval.metrics import EvalResult

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "merge_sharded_runs.py"
SPEC = importlib.util.spec_from_file_location("merge_sharded_runs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

discover_shard_dirs = MODULE.discover_shard_dirs
merge_results_by_item = MODULE.merge_results_by_item
validate_shard_set = MODULE.validate_shard_set


def _make_result(item_id: str, pred: str = "YES") -> EvalResult:
    return EvalResult(
        item_id=item_id,
        batch_file="batch1.jsonl",
        task_family="atomic",
        bias_family=None,
        dialogue_id=None,
        turn_index=None,
        system_id="lorenz63",
        gold="YES",
        pred_raw=pred,
        pred_norm=pred,
        correct=True,
        error_type=None,
        question="Is Lorenz chaotic?",
    )


def test_discover_shard_dirs_and_validate(tmp_path: Path):
    (tmp_path / "gpt4_zeroshot_shard1of3").mkdir()
    (tmp_path / "gpt4_zeroshot_shard2of3").mkdir()
    (tmp_path / "gpt4_zeroshot_shard3of3").mkdir()
    (tmp_path / "gpt4_cot_shard1of2").mkdir()

    shards = discover_shard_dirs(tmp_path, "gpt4", "zeroshot")

    assert [s[0] for s in shards] == [1, 2, 3]
    assert validate_shard_set(shards) == 3


def test_validate_shard_set_detects_missing_shard(tmp_path: Path):
    (tmp_path / "gpt4_zeroshot_shard1of3").mkdir()
    (tmp_path / "gpt4_zeroshot_shard3of3").mkdir()

    shards = discover_shard_dirs(tmp_path, "gpt4", "zeroshot")

    with pytest.raises(ValueError, match="Missing shard directories"):
        validate_shard_set(shards)


def test_merge_results_accepts_identical_duplicates():
    a = _make_result("q0001")
    b = _make_result("q0002")
    dup = EvalResult(**asdict(a))

    merged, report = merge_results_by_item([[a, b], [dup]])

    assert [r.item_id for r in merged] == ["q0001", "q0002"]
    assert report["duplicate_rows"] == 1
    assert report["unique_items"] == 2


def test_merge_results_rejects_conflicting_duplicates():
    a = _make_result("q0001", pred="YES")
    conflict = _make_result("q0001", pred="NO")
    conflict.correct = False

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        merge_results_by_item([[a], [conflict]])
