# Code overview (PZT++ engine)

This file explains how the directories and modules in `SLM_PZT_Receipes/` realize the “few-switches” evaluation reported in the presentation slideshow.

## Parser & normalization (`pzt/`)
- `pzt/schema.py`: simple `Recipe` dataclass that holds normalized title, ingredients, instructions, and computed IDs.
- `pzt/norm.py`: performs canonicalization of ingredient text, head extraction, TF–IDF weight recipes, and records changes (missing words, substitutions). These normalized heads and ops feed the structured alignment signals on slides (missing heads/operations, operations coverage, etc.).
- `pzt/parser.py`: strict parser for NDJSON lines with an `input` field (title + Ingredients/Instructions). It enforces the format shown on slide 5 (“Ingredients: ... Instructions: ...”), produces warnings for missing sections, and exposes `parse_ndjson()` for the build pipeline.

## Pipeline core (`pztpp/`)
- `build.py`: orchestrates parsing, vocabulary construction, IDF weights, and embedding computation. It persists `head_vocab.json`, `op_vocab.json`, IDF arrays, head postings, and (optionally) ANN indexes (`faiss` or `hnswlib`). This matches the “parts-based coverage + embeddings + index” architecture referenced on slides 6–8.
- `run.py`: loads the persisted assets, performs retrieval (hybrid/ingredient/full) either via ANN indexes or brute-force cosine, and computes per-pair metrics: coverage, precision, switch/miss costs, `pzt_score`, `pzt_edit_score`, `pzt_legacy_score`, and explains missing/extra heads/ops. Zone filtering (`min_switch_cost ≤ switch_cost ≤ max_switch_cost`) is implemented here to reproduce the “few-switch zone” visuals on slides 3–5.
- `score.py`: contains the directional recall/precision helpers, the PZT edit-cost model, and zone membership logic. The `pzt_score` (Edit) and legacy MISS-only variant match the curves on slides 10–12.
- `embeddings.py`, `index.py`, `ops.py`, `vocab.py`, `neighbors.py`: provide the utilities for dense embedding generation (SBERT), building ANN indexes, TF–IDF vocab management, and head substitution neighbors (used to produce modulable explanations like those in slide 12).

## CLI + experiments
- `pztpp_transfer.py`: the single entry point. Subcommands `build` and `run` expose presets (`quality`, `fast`, `debug`), strict parsing, gating toggles, ANN options, and zone weighting. `EXPERIMENTS.md` references the exact commands used for the “10K exact” bar graphs, measure comparisons, and gating experiments shown on slides 10–12.
- `scripts/exp_compare_measures_10k.py`: consumes `pztpp_out_10k_noann` run outputs and computes the metrics that produce the slides’ charts (“proximal top-1 rate”, “median switch cost”). The script evaluates full-text, ingredient, op, head, & PZT++ scores to recreate the comparison bars from the deck.
- `scripts/generate_llm_adaptations.py`: optional natural-language explanation generator that rounds out the adaptation narratives shown in slide 12 (“Banana Bread → Pumpkin Bread”).
- Additional scripts (`generate_edit_sentences.py`, `compare_old_new_scores.py`, `sweep_pzt.py`, `mvp_smoke.py`) support exploratory analysis, scoring sweeps, and smoke testing for the auxiliary web/UI stack; `EXPERIMENTS.md` points to the subset that produced the slide visuals.

## Documentation tie-ins
- `docs/PZTpp_Algorithm.md` spells out the parsing, normalization, retrieval, gating, scoring, and zone-filtering steps that correspond directly to the slide narrative.
- `README.md`, `CODE_DOC.md`, and `EXPERIMENTS.md` lift the same story into reproducible instructions so future collaborators can audit the slide claims.
