PZT Evaluation Plan

Objective
- Validate that PZT (MISS + FA + zone band) promotes “few‑switches” sources and reduces duplicate/superset dominance compared to the legacy MISS‑only score.

Datasets
- `SLM_PZT_Receipes/all_recipes.ndjson` with subsets N ∈ {200, 1k, 5k} and a 10k exact baseline (no ANN).

Defaults
- Gates: min_ingr_recall=0.55, min_ingr_precision=0.55, min_ops_recall=0.45, min_ops_precision=0.30, min_instr_sim=0.50
- Zone (few_switches): min_switch_cost=0.02, max_switch_cost=0.25 (set `--mode closest` to disable)
- Edit weights: a_m=0.55, a_f=0.15, b_m=0.15, b_f=0.05, c=0.10

Analyses
- Coverage: fraction of targets with ≥1 result under gates/band; sensitivity to band/gate changes.
- Duplicate suppression: share of old top‑1 with near‑zero switch_cost removed when switching from `closest` to `few_switches` (use `scripts/compare_old_new_scores.py`).
- Distributions: quantiles/histograms for pzt_edit_score and switch_cost.
- Head diagnostics: top missing/extra heads across top‑1; ensure staples/extras dominate extras as expected.
- Ranking stability: check tie‑break by sim_full on narrow bands.

Protocol
1) Build artifacts for each subset (ANN for ≤5k, exact for 10k).
2) Run `mode=closest` and `mode=few_switches` with same gates; keep `cand_k` identical.
3) Compare summaries via `analysis_summary_edit.json` and spot‑check JSONL explanations.

Ablations
- Precision gates OFF vs ON; coverage and quality shift.
- Band width: widen to (0.00, 0.30); narrow to (0.03, 0.20).
- Edit weights: increase `a_f` to penalize extras more if supersets persist.

Artifacts
- `targets_to_sources.(jsonl|csv)`, `sources_to_targets.(jsonl|csv)` with edit metrics and explanations.
- `analysis_summary_edit.json` per run comparison.
