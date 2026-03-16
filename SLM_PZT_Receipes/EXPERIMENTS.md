# Experiments that match the January 2026 slides

The slide deck’s charts (slides 9–12) were produced from the exact—non-ANN—10,000-recipe run plus the downstream measurement script that contrasts similarity-based and edit-aware scores. This document explains how to recreate those artifacts using the code in this directory.

## 1. Prerequisites
- Python 3.10+ (3.13 used historically), activated virtual environment, and `pip install -r requirements.txt`.
- `all_recipes_sample_200.ndjson` is already provided for fast onboarding. The full corpus (`corbt/all-recipes`, 2,147,248 recipes) is used for the 10K experiment; download it with:
  ```sh
  python download_dataset_json.py --output all_recipes.ndjson
  ```
  (The script hits Hugging Face’s `corbt/all-recipes` card and writes NDJSON with the same `input` structure shown in slide 5.)
- Ensure the heavy files are stored locally; they are ignored by git but assumed by the next steps.

## 2. Build the 10K exact run

This replicates the “exact brute-force” experiment on slide 9. The `build` command parses the corpus, extracts ingredient/operation vocabularies, saves IDF weights, computes SBERT embeddings (`all-MiniLM-L6-v2`), and, because `--no-ann` is set, skips ANN indexes to keep the run identical to the brute force baseline.

```sh
python pztpp_transfer.py build \
  --ndjson all_recipes.ndjson \
  --limit 10000 \
  --out-dir pztpp_out_10k_noann \
  --strict \
  --preset quality \
  --no-ann \
  --no-head-neighbors
```

The run directory now contains the parsed recipes (`parse_audit.json`), embeddings (float16 memmaps), vocabularies, IDF weights, head postings, and `build_config.json`.

## 3. Run the rankings/measurements

Use the same NDJSON & directory so the `targets_to_sources.*` outputs match the slide’s reported distributions.

```sh
python pztpp_transfer.py run \
  --ndjson all_recipes.ndjson \
  --limit 10000 \
  --out-dir pztpp_out_10k_noann \
  --preset quality \
  --cand-k 1500 \
  --top-k 20 \
  --mode few_switches \
  --rerank-by-sim-full \
  --gap-threshold 0.50 \
  --min-ingr-precision 0.55 \
  --min-ops-precision 0.30
```

The command writes:

- `targets_to_sources.jsonl/csv`: per-target top-K sources with scores, costs, extra/missing heads/ops, `in_pzt_zone` flag, etc.
- `sources_to_targets.*`: reverse rankings (S→T).  
- `run_config_run.json`: decks the actual flag values used for the run.

These artifacts feed the charts in slides 10–12; each column (`pzt_score`, `cov_ingr_soft`, `miss_ingr`, etc.) corresponds to the CSV columns mentioned in slide 8 (scores, masses, explanations).

## 4. Reproduce the measure comparison visuals

The per-method bar charts use the `scripts/exp_compare_measures_10k.py` helper. It reads `pztpp_out_10k_noann` and compares:

| Method | Description |
| --- | --- |
| full | dense SBERT full-text similarity |
| ingr | SBERT ingredient embedding similarity |
| instr | SBERT instruction embedding similarity |
| head-set/op-set | exact-part overlap |
| legacy | MISS-only cost |
| pzt++ | Edit-aware PZT score (MISS+FA + few-switch zone) |

Run the script as follows:

```sh
python scripts/exp_compare_measures_10k.py \
  --run-dir pztpp_out_10k_noann \
  --out-dir exp_10k_measure_compare \
  --k 20
```

Outputs:

- `per_target_top1_comparison.csv`: used for slide 10 to compute “% targets with a proximal top-1”.  
- `measure_compare_summary.json`, `agreement_at_k.csv`, `disagreement_examples.jsonl`: feed the textual explanation about which measures agree/disagree.  
- `exp_10k_measure_compare.jsonl`: optional data for deeper plotting.

## 5. Explainable few-switch narrative (slide 12)

Use `scripts/generate_llm_adaptations.py` to turn the missing/extra heads/ops into natural-language instructions matching slide 12’s example. It can be run in two modes: `--model gpt-4o-mini` (requires API access) or `--model template` (uses deterministic templates over head/op vocab).

## 6. Optional variations

- `--ann-engine hnswlib` with `--ef-search` and `cand_k≈1000` reproduces the ANN-based runs cited under “Settings that work well”.  
- The `scripts/sweep_pzt.py` command sweeps weights/gates, enabling deeper analysis beyond the slide’s reference runs.
- `scripts/compare_old_new_scores.py` shows the performance change when adding precision penalties; use it to justify the bars labeled “PZT++-Edit” vs “legacy”.

## 7. Verification and lineage

Documented files such as `docs/PZTpp_Algorithm.md`, `README.md`, `CODE_DOC.md`, and `REVIEW.md` explain how each artifact contributes to the slide deck and how to verify each metric.
