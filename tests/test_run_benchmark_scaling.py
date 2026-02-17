"""Tests for run_benchmark scaling helpers."""

from pathlib import Path

from run_benchmark import discover_batch_files, slice_items_for_shard


def test_discover_batch_files_sorts_by_numeric_index(tmp_path: Path):
    (tmp_path / "batch10_z.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "batch2_b.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "batch1_a.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "notes.jsonl").write_text("{}\n", encoding="utf-8")

    discovered = discover_batch_files(str(tmp_path))

    assert discovered == ["batch1_a.jsonl", "batch2_b.jsonl", "batch10_z.jsonl"]


def test_discover_batch_files_respects_explicit_order(tmp_path: Path):
    explicit = ["batch9_x.jsonl", "batch1_a.jsonl"]
    discovered = discover_batch_files(str(tmp_path), requested_batches=explicit)
    assert discovered == explicit


def test_slice_items_for_shard_partitions_deterministically():
    items = [{"id": f"q{i:04d}"} for i in range(12)]

    shard_0 = slice_items_for_shard(items, shard_index=0, num_shards=3)
    shard_1 = slice_items_for_shard(items, shard_index=1, num_shards=3)
    shard_2 = slice_items_for_shard(items, shard_index=2, num_shards=3)

    assert len(shard_0) == len(shard_1) == len(shard_2) == 4
    assert {item["id"] for item in shard_0}.isdisjoint({item["id"] for item in shard_1})
    assert {item["id"] for item in shard_0}.isdisjoint({item["id"] for item in shard_2})
    assert {item["id"] for item in shard_1}.isdisjoint({item["id"] for item in shard_2})
