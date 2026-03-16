# PZT++ Recipe Transfer Engine (SLM)

This directory contains the live implementation of the Proximal Zone of Transferability (PZT++) experiment stack: parsing, part extraction, embeddings, ANN/indexing, and ranking logic that powers the “few-switches” evaluation reported in the January 2026 slides.

## Layout
- `pztpp_transfer.py`: single CLI that exposes `build`/`run` commands with presets (`quality`, `fast`, `debug`), strict parsing, ANN toggles, and edit-sensitive scoring options (`pzt_score`, `pzt_edit_score`, PZT zone filters).
- `pztpp/`: engine modules (`build`, `run`, `embeddings`, `embedding index wrappers`, `score`, `ops`, `vocab`, `neighbors`).
- `pzt/`: parsing, normalization, schema, and helper utilities for structured ingredient/operation representations.
- `scripts/`: experimental helpers (`exp_compare_measures_10k.py`, `generate_llm_adaptations.py`, etc.) that reproduce the graphs and tables from the slides.
- `docs/`: narrative references (`docs/PZTpp_Algorithm.md`, `docs/EXPERIMENT_PLAN.md`, `docs/README.md`).
- `requirements.txt`: Python packages used by the baked virtual environment.
- `all_recipes_sample_200.ndjson`: lightweight sample for quick prototyping; the full `corbt/all-recipes` corpus is documented in `EXPERIMENTS.md` and is not tracked to keep the repository manageable.

## Environment
- Install Python 3.10+ (3.13 was used for the captured venv). Create a fresh venv and install the pinned dependencies:
  ```sh
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  ```
- Node.js 18+ is needed only for the ancillary tooling (`package.json`/`package-lock.json`); run `npm install` inside this directory if you plan to inspect those scripts.
- Large datasets, ANN indexes, and run outputs are ignored by git. Use `download_dataset_json.py` to fetch the Hugging Face `corbt/all-recipes` split (see `EXPERIMENTS.md` for the exact data location) if you need the full 2.1M recipe corpus.

## Running the pipeline
1. **Build artifacts** (embeddings, vocabularies, ANN indexes):  
   ```sh
   python -m pztpp_transfer build \
     --ndjson all_recipes_sample_200.ndjson \
     --out-dir pztpp_out \
     --strict --preset quality
   ```
2. **Run rankings** for a dataset/run:  
   ```sh
   python -m pztpp_transfer run \
     --ndjson all_recipes_sample_200.ndjson \
     --out-dir pztpp_out \
     --preset quality
   ```
3. **Inspect artifacts**: results are written to `targets_to_sources.(jsonl|csv)`, `sources_to_targets.(jsonl|csv)`, and `parse_audit.json`. Edit explainability information is stored under `explanations/` fields (missing heads/ops, etc.)

For the slide-aligned experiments (10K exact run, measurement comparison), see `EXPERIMENTS.md`. The same `pztpp_transfer` commands populate the run directories that `scripts/exp_compare_measures_10k.py` consumes.

## Additional references
- `CODE_DOC.md`: module-by-module explanation of how parsing, embeddings, and scoring components connect to the PZT++ narrative.
- `EXPERIMENTS.md`: step-by-step reproduction of the “10K exact” comparisons, gate settings, and few-switch band evaluation from the slides.
- `REVIEW.md`: a short audit that ties the code artifacts to the slide deck and flags the experiments we verified.
