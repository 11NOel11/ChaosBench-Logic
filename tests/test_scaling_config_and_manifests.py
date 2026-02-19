"""Tests for generation config loading and run manifest registry."""

import json
from pathlib import Path

from scripts.build_v2_dataset import load_generation_config
from run_benchmark import write_run_manifest


def test_load_generation_config_defaults():
    cfg = load_generation_config(None)
    assert cfg["seed"] == 42
    assert cfg["template"] == "V2"
    assert cfg["adversarial"]["n_per_type"] == 50


def test_load_generation_config_nested_override(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "seed: 7\nconsistency:\n  paraphrase_variants: 5\nadversarial:\n  drop_unknown: false\n",
        encoding="utf-8",
    )

    cfg = load_generation_config(str(cfg_path))

    assert cfg["seed"] == 7
    assert cfg["consistency"]["paraphrase_variants"] == 5
    assert cfg["consistency"]["batch2_take"] == 50
    assert cfg["adversarial"]["drop_unknown"] is False


def test_write_run_manifest_creates_registry_and_run_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    batch_file = data_dir / "batch1_atomic_implication.jsonl"
    batch_file.write_text('{"id":"q1"}\n', encoding="utf-8")

    run_out_dir = tmp_path / "results" / "gpt4_zeroshot"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = write_run_manifest(
        model_name="gpt4",
        mode="zeroshot",
        run_out_dir=str(run_out_dir),
        data_dir="data",
        batch_files=["batch1_atomic_implication.jsonl"],
        total_items=1,
        shard_items=1,
        workers=2,
        checkpoint_interval=25,
        shard_index=0,
        num_shards=1,
        max_items=None,
    )

    assert Path(manifest_path).is_file()
    run_copy = run_out_dir / "run_manifest.json"
    assert run_copy.is_file()

    manifest = json.loads(run_copy.read_text(encoding="utf-8"))
    assert manifest["model"] == "gpt4"
    assert manifest["mode"] == "zeroshot"
    assert manifest["items_in_this_run"] == 1
    assert len(manifest["batches"]) == 1
