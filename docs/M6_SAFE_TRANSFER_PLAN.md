# M6 Safe Transfer Plan

## Goal

Move from static transfer guardrails to online safe adaptation under context shift.

Primary target claim:

- An online controller can reduce harmful transfer events under shift while
  preserving or improving utility versus static threshold baselines.

## Implemented

- `chaosbench/repair/online_controller.py`
  - Shift score via JSD on `(family, margin_bucket, support_bucket)` histograms
  - Provider-specific reference distributions with provider/prefix fallback
  - Harm and utility signals
  - Adaptive primal-dual updates for `lambda` and `rho`
  - Local risk-aware threshold sweep around current threshold
  - Provider-specific step dampening (`provider_step_multipliers`)
  - Non-degradation rollback guard using online-vs-static signal
  - CUSUM-like alarm with emergency threshold tightening
  - Serializable controller state
- `scripts/run_m5_instance_guardrail.py`
  - `--controller-mode {static,online}`
  - Online controller config flags
  - `--online-update-splits` to avoid split leakage
  - Delayed-label queue simulation via `--online-label-lag-runs`
  - `m6_online_trace.csv` + `m6_online_update_events.csv` + `m6_controller_state.json`
  - Provider reference map support via `--provider-reference-json`
  - Provider dampening flags and non-degrade guard flags
  - Per-run `online_minus_static_mcc` signal for strict slice checks
  - No implicit provider-threshold map loading by default
- `scripts/calibrate_m5_provider_thresholds.py`
  - Added prior artifact: `provider_threshold_priors_crossfit_v1.json`
  - Added provider reference artifact: `provider_reference_dists_crossfit_v1.json`
  - Added per-provider threshold std in summary
- `scripts/run_m5_crossfit_cycle.py`
  - Runs static and online transfer (online enabled by default)
  - Sync to stable provider map is now opt-in (`--sync-stable-config`)
  - Forwards online sweep/risk/lag args
  - Supports stress offsets on calibrated map (`--provider-threshold-offsets`)
  - Comparison table now includes harm and alarm rates
  - Adds true LOPO replay orchestration (`--run-lopo-replay`) with
    holdout-provider transfer rows in `cycle/m6_lopo_replay.csv`
  - Adds temporal replay orchestration (`--run-temporal-backtest`) with
    prefix-train/suffix-test execution using generated split maps and
    per-cut rows in `cycle/m6_temporal_backtest_replay.csv`
- `scripts/analyze_m6_strict_eval.py`
  - Computes paired run bootstrap CI + provider-cluster bootstrap CI
  - Computes sign test and worst-provider non-degradation checks
  - Writes strict eval summary tables (`m6_online_strict_eval.csv`, `M6_STRICT_EVAL.md`)
- `scripts/run_prompt_variant_suite.py`
  - Cell cache skip now validates expected deterministic `run_id`
  - Optional post-suite M5/M6 transfer call

## Theory Statement Candidate

Under bounded convex surrogates for utility/risk and projected primal-dual
updates with step sizes `eta_t = O(1/sqrt(t))`, M6 achieves sublinear
`O(sqrt(T))` dynamic regret and sublinear cumulative constraint violation for
harm and shift constraints. This implies vanishing average violation.

## Devil's Advocate Guardrails (must pass before claim)

- No leakage:
  - leave-one-provider-out transfer evaluation
  - blocked split by item/system where needed
- Delayed-label realism:
  - update controller only on label-available windows
  - report lag sensitivity
- Statistical rigor:
  - paired cluster bootstrap CIs
  - pre-registered primary endpoint and comparator
  - secondary metrics corrected for multiplicity
- Safety-first reporting:
  - downside tails (CVaR-like)
  - harmful-transfer count and alarm rate
  - worst-slice non-degradation checks

## Highest-Potential Leads (ordered)

1. **Exploration bucket for unbiased learning**
   - Reserve small randomized fraction to estimate counterfactual harm.
2. **Worst-slice safety constraints**
   - Enforce no degradation on critical families or providers before accepting
     global improvements.
3. **Temporal robustness checks**
   - Rolling-window backtests to verify stability under model/provider drift.
4. **Risk objective refinement**
   - Couple sweep objective to worst-slice penalties instead of global degrade-only
     utility.

## Current Grid Snapshot

- Sweep root: `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep`
- Ranked summary: `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/m6_online_grid_rank.csv`
- Strict eval summary: `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/M6_STRICT_EVAL.md`
- Strict eval CSV: `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/m6_online_strict_eval.csv`
- Provider-slice table: `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/m6_online_strict_provider_slices.csv`
- Top mean-diff config observed: `mix0p0_a0p05_lag0` (online mean diff `+0.002182`)
- Top safety-weighted configs observed: `mix0p75_a0p02_lag1`, `mix0p75_a0p05_lag1`
  with zero mean harm and zero alarm rate on this sweep slice.
- Strict paired online-vs-static result: best observed online effect is tie (`0.000000`);
  many configs show small negative online-minus-static mean (`-0.000179`) driven by
  openai slice.
- Mitigation run with openai frozen updates (`provider_step_multipliers: openai=0.0`)
  restores strict tie/non-degradation for the high-sweep online config
  (`fix3_mix1p0_a0p05_lag0_openai0_guard`).
- Non-zero mitigation also works: `openai=0.2` with `sweep_mix=0.5`
  (`fix4_mix0p5_a0p05_lag0_openai02_guard`) preserves strict tie while keeping
  online adaptation active.

## Stress-Test Snapshot (stale threshold robustness)

- Stress root: `workspace/deep_survey_2026-03-01/repair_v3/m6_stress_suite`
- Strict summary: `workspace/deep_survey_2026-03-01/repair_v3/m6_stress_suite/m6_online_strict_eval.csv`
- Provider slices: `workspace/deep_survey_2026-03-01/repair_v3/m6_stress_suite/m6_online_strict_provider_slices.csv`
- Under strong stale-threshold shift (`openai=+0.09`, static threshold = `0.20`):
  - online-minus-static mean: `+0.001277` (strict pass)
  - openai slice mean online-minus-static: `+0.005108`
- Under overly permissive stale-threshold shift (`openai=-0.09`, static threshold = `0.02`):
  - online-minus-static mean: `+0.000327` (strict pass)
  - openai slice mean online-minus-static: `+0.001308`

## Execution-Level Replay Snapshot

- Replay config roots:
  - `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/replay_fix6_mix0p5_a0p05_lag0_openai02_guard`
  - `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/replay_fix7_mix0p5_a0p05_lag0_openai0_guard`
- Each run used true execution-time replay checks:
  - LOPO replay orchestration (`cycle/m6_lopo_replay.csv`)
  - Temporal prefix->suffix replay orchestration (`cycle/m6_temporal_backtest_replay.csv`)
- Observed for both replay configs:
  - strict online-minus-static mean: `+0.000000` (`strict_pass=1`)
  - LOPO worst holdout online-minus-static: `+0.000000`
  - Temporal worst slice online-minus-static: `+0.000000`
  - One positive LOPO cell remains (`openrouter_live`, `+0.001559`), with no negative
    replay cells.
- Current safety-first replay candidate: `replay_fix7_mix0p5_a0p05_lag0_openai0_guard`
  (openai updates frozen, no downside observed vs static in replay checks).

## EM Briefing Pack

- Pack root:
  - `workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep/m6_em_pack`
- Decision tables:
  - `m6_em_master_summary.csv`
  - `m6_em_stress_summary.csv`
  - `m6_em_replay_summary.csv`
  - `m6_em_candidate_comparison.csv`
  - `m6_em_decision_snapshot.json`
- Figure-ready assets:
  - `em_fig_strict_top_configs.{png,pdf}`
  - `em_fig_stress_suite.{png,pdf}`
  - `em_fig_replay_guardrails.{png,pdf}`
  - `m6_em_figure_manifest.csv`
- Brief markdown:
  - `M6_EM_BRIEF.md`

## Diagnostic Finding

- The observed online regression is a threshold cliff on one openai run
  (`pv_openai_gpt-4o_v1_logiccheck_25e2110e97`):
  - one degraded candidate has score `0.102509`;
  - static threshold `0.11` vetoes it;
  - aggressive online updates lower threshold below `0.11`, admitting this flip and
    reducing MCC by about `0.00214` on that run.
- This behavior is path/order-sensitive for the 3 openai heldout runs (some
  permutations tie, some regress), which is why dampening the effective step/sweep
  for openai removes downside.

## Next Experiment Matrix

- Static baseline: `--controller-mode static`
- Online baseline: `--controller-mode online`
- Update split sensitivity:
  - `--online-update-splits heldout`
  - `--online-update-splits dev,heldout`
- Controller sweeps:
  - `--online-risk-budget-b0 {0.0005,0.001,0.002}`
  - `--online-shift-kappa {1.0,2.0,4.0}`
  - `--online-alarm-threshold {0.01,0.02,0.05}`
  - `--online-sweep-mix {0.0,0.75,1.0}`
- Delay simulation:
  - `--online-label-lag-runs {0,1}`

## Repro Commands

```bash
uv run python scripts/run_m5_crossfit_cycle.py \
  --provider-dirs openai,deepseek,gemini_v2,openrouter_live \
  --calibration-out-dir workspace/deep_survey_2026-03-01/repair_v3/m6_cycle_main \
  --transfer-out-tag m5_static_m6_main
```

```bash
uv run python scripts/run_m5_instance_guardrail.py \
  --repair-dir workspace/deep_survey_2026-03-01/prompt_variants_parallel/openrouter_live \
  --policy-json workspace/deep_survey_2026-03-01/repair_v3/m5_instance/m5_policy.json \
  --controller-mode online \
  --provider-thresholds-json workspace/deep_survey_2026-03-01/repair_v3/m6_cycle_main/provider_thresholds_crossfit_v1.json \
  --provider-reference-json workspace/deep_survey_2026-03-01/repair_v3/m6_cycle_main/provider_reference_dists_crossfit_v1.json \
  --online-label-lag-runs 1 \
  --default-split heldout \
  --out-dir workspace/deep_survey_2026-03-01/prompt_variants_parallel/openrouter_live/m6_online_main
```

```bash
uv run python scripts/analyze_m6_strict_eval.py \
  --grid-root workspace/deep_survey_2026-03-01/repair_v3/m6_grid_sweep \
  --bootstrap 5000 \
  --seed 42
```

```bash
uv run python scripts/run_m5_crossfit_cycle.py \
  --provider-dirs openai,deepseek,gemini_v2,openrouter_live \
  --calibration-out-dir workspace/deep_survey_2026-03-01/repair_v3/m6_stress_suite/stress_openai_high_mix1 \
  --transfer-out-tag m5_static_stress_openai_high_mix1 \
  --online-transfer-out-tag m6_online_stress_openai_high_mix1 \
  --provider-threshold-offsets openai=0.09 \
  --skip-full-suite

uv run python scripts/run_m5_crossfit_cycle.py \
  --provider-dirs openai,deepseek,gemini_v2,openrouter_live \
  --calibration-out-dir workspace/deep_survey_2026-03-01/repair_v3/m6_stress_suite/stress_openai_verylow_mix1 \
  --transfer-out-tag m5_static_stress_openai_verylow_mix1 \
  --online-transfer-out-tag m6_online_stress_openai_verylow_mix1 \
  --provider-threshold-offsets openai=-0.09 \
  --skip-full-suite

uv run python scripts/analyze_m6_strict_eval.py \
  --grid-root workspace/deep_survey_2026-03-01/repair_v3/m6_stress_suite \
  --bootstrap 5000 \
  --seed 42
```
