# Human validation package for PZT V2 model v0.2.0

Send only the files under `blind/` to the evaluator:

- `human_scores_blind.csv`
- `human_scoring_rubric.json`

Do not send or open `model_key/human_scores_key.csv` during scoring. It contains
the hidden model outputs, deciles, and repeat mapping.

The blind CSV contains 50 distinct directional pairs and five randomly placed
repeats. For each presentation, the evaluator must:

1. set `source_known_yes_no`;
2. enter an integer `human_score_1_to_6` only when the source is known;
3. optionally add notes.

After scoring, analyze the completed blind file without changing the frozen
model:

```sh
python -m pzt_v2.run_human_analysis \
  --completed-blind human_validation/pzt-v2-model-v0.2.0/blind/human_scores_blind.csv \
  --key human_validation/pzt-v2-model-v0.2.0/model_key/human_scores_key.csv \
  --output human_validation/pzt-v2-model-v0.2.0/human_analysis.json \
  --manifest human_validation/pzt-v2-model-v0.2.0/human_analysis_manifest.json \
  --upstream-manifest evidence/runs/pzt-v2-20260715/human_validation_manifest.json \
  --run-id pzt-v2-human-analysis-v0.2.0
```

The evaluator-dependent outcome will be `SUPPORTED`, `REFUTED`, or
`INCONCLUSIVE`. These 50 pairs must never be used to tune model v0.2.0.
