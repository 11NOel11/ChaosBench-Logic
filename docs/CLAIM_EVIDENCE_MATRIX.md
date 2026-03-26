# Claim-Evidence Matrix

This matrix maps paper-facing claims to concrete repo artifacts for camera-ready verification.

| Claim | Evidence Path |
|------|---------------|
| v2 contains 40,886 questions across 10 canonical files | `data/v2_manifest.json` |
| Canonical v2 file list is stable and machine-readable | `data/canonical_v2_files.json` |
| Dataset release notes and hash are documented | `docs/RELEASE_NOTES_V2.md` |
| Full dataset card metadata is published in-repo | `../DATASET_CARD.md` |
| Strict online-vs-static safety evaluation exists | `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/M6_STRICT_EVAL.md` |
| Strict sweep includes LOPO and temporal replay metrics | `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/m6_online_strict_eval.csv` |
| LOPO replay outputs are materialized per run | `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/replay_fix7_mix0p5_a0p05_lag0_openai0_guard/cycle/m6_lopo_replay.csv` |
| Temporal replay outputs are materialized per run | `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/replay_fix7_mix0p5_a0p05_lag0_openai0_guard/cycle/m6_temporal_backtest_replay.csv` |
| EM-style decision brief for M6 candidate selection exists | `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/m6_em_pack/M6_EM_BRIEF.md` |
| Final results tables for paper pack are generated | `artifacts/results_pack_v2/20260221_114551/FINAL_RESULTS_REPORT.md` |

Notes:
- Paths under `workspace/` and `artifacts/` are reproducibility outputs and may be regenerated.
- Core release identity should be cited from `data/v2_manifest.json` and `docs/RELEASE_NOTES_V2.md`.
