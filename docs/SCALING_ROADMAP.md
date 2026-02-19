# ChaosBench-Logic Scaling Roadmap (v2.x -> v3)

This roadmap defines major rescale initiatives to evolve ChaosBench-Logic from a
single benchmark release into a repeatable benchmark platform.

## North Star

- Grow dataset scale from 1,879 to 15,000+ high-quality items.
- Expand system coverage from 30 to 80+ systems.
- Run distributed evaluations with shard-native orchestration and canonical merge.
- Gate releases with strict scientific QA and reproducibility checks.

## Workstream A: Dataset Factory at 10x Scale

### Scope

- Introduce config-driven generation for all batch families.
- Add hard-negative and adversarial template mining.
- Add automatic balancing by task family, label, and system class.
- Add deduplication checks (ID uniqueness + text-level near duplicates).

### Deliverables

- `configs/generation/*.yaml` for batch recipes.
- `scripts/build_v2_dataset.py` extension for recipe-driven generation.
- `scripts/validate_v2.py` extension for balance and ambiguity checks.

### Success Criteria

- 15,000+ valid JSONL items.
- <1% ambiguous/flagged items in validation pass.
- Stable label balance within target tolerance bands.

## Workstream B: System Expansion Program

### Scope

- Add 50+ systems in waves across ODE/map/PDE/stochastic families.
- Compute indicators for all newly onboarded systems.
- Track indicator reliability per system and gate low-confidence ones.

### Deliverables

- Extended `systems/*.json` inventory.
- Extended `systems/indicators/*_indicators.json` inventory.
- Reliability report for indicator completeness and confidence.

### Success Criteria

- 80+ systems registered.
- >=90% systems with complete indicator records.
- Reliability metadata attached to each system card.

## Workstream C: Distributed Evaluation Backbone

### Scope

- Make sharding first-class in local and SLURM workflows.
- Merge shard outputs into canonical per-model run artifacts.
- Add reproducible run manifests for all benchmark executions.

### Deliverables

- Shard-aware runner CLI (`run_benchmark.py`).
- Shard-aware cluster script (`scripts/run_cluster_eval.py`).
- Canonical merge utility (`scripts/merge_sharded_runs.py`).

### Success Criteria

- 5-10x faster full matrix throughput on cluster.
- Re-runnable run IDs with deterministic shard merges.
- No manual JSON stitching for published results.

## Workstream D: QA and Scientific Integrity Gates

### Scope

- Add release-blocking checks for:
  - schema correctness
  - deterministic generation
  - logic consistency / FOL checks
  - leakage and template overfitting checks
  - split contamination checks

### Deliverables

- CI quality gate workflow for release tags.
- Validation report artifacts for every release candidate.

### Success Criteria

- Zero critical QA failures for release candidates.
- Metrics reproducible within expected variance bands.

## Workstream E: Public Benchmark Packaging

### Scope

- Introduce fixed public splits (`core`, `robustness`, `heldout_systems`, `hard`).
- Publish strict evaluation protocol and submission format.
- Automate release notes from manifests and validation reports.

### Deliverables

- Split manifests in `data/`.
- Public protocol doc in `docs/`.
- Auto-generated changelog section for each release.

### Success Criteria

- One-command reproducible benchmark report generation.
- Stable leaderboard ingestion format.

## Execution Plan

### Phase 1 (Weeks 1-2): Infrastructure

- Finalize shard run + merge workflow.
- Add release-gate validation jobs.
- Define run manifest schema.

### Phase 2 (Weeks 3-6): Expansion Wave 1

- Add ~8k items and ~25 systems.
- Run baseline model matrix and QA audit.
- Publish interim v2.x data refresh.

### Phase 3 (Weeks 7-10): Expansion Wave 2

- Reach 15k+ items and 80+ systems.
- Finalize public split protocol and packaging.
- Publish v3-ready benchmark bundle.

## Risks and Mitigations

- Label drift at scale -> deterministic templates + validator audits.
- Indicator instability -> reliability flags and fallback computation paths.
- Evaluation cost growth -> staged run tiers (smoke/core/full) and budget caps.
- Overfitting to templates -> held-out templates and paraphrase diversification.

## Immediate Next Actions

1. Adopt shard merge workflow for all cluster runs.
2. Add generation config spec and migrate one batch family.
3. Wire release QA gates into CI before next dataset refresh.
