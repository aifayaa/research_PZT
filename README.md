# PZT Recipe Transfer Workspace

The active research code lives under `SLM_PZT_Receipes/`; the rest of the repository has been archived to `archive/legacy_root/` for reference. Everything you need to parse, embed, score, and benchmark the PZT++ engine is self-contained within that directory, including:

- the NDJSON corpus (`all_recipes_sample_200.ndjson` plus tooling to download the full `corbt/all-recipes` split),
- the Python CLI (`pztpp_transfer.py`) that builds data artifacts and runs the ranking pipeline,
- experiment scripts (e.g., `scripts/exp_compare_measures_10k.py`) and documentation,
- generated outputs and caches (ignored via `.gitignore` so the git history only tracks source/control files).

To recreate the experiments in this workspace, follow the instructions in `SLM_PZT_Receipes/EXPERIMENTS.md`. The canonical overview of the modules, parser, and parts-based scoring lives in `SLM_PZT_Receipes/CODE_DOC.md`. Slide-specific validation and alignment notes are captured in `SLM_PZT_Receipes/REVIEW.md`.

If you need the older MVP apps, npm/react bundles, or unrelated `pzt`/`pztpp` prototypes, you will find them under `archive/legacy_root/` to preserve them without cluttering the active git tree.
