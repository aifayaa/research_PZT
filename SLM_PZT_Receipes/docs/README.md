Project: PZT Recipe Transferability Engine

Overview
- PZT is a directional (S→T) “few‑switches” transferability engine using Sentence‑Transformers (all‑MiniLM‑L6‑v2) + parts‑based coverage.
- It computes ingredient/operation recall and precision, an edit‑sensitive cost (MISS + FA + instruction gap), and ranks candidates with a few‑switches zone filter.
- ANN retrieval (mac: hnswlib; Linux: FAISS) is supported; exact brute‑force available for baselines.

Docs Map
- Algorithm: `docs/PZTpp_Algorithm.md` (current spec: recall+precision, switch/edit costs, gates, zone, outputs).
- Reproduction: `docs/project_overview.tex` (10K exact commands and summary table).

Key Paths
- CLI: `SLM_PZT_Receipes/pztpp_transfer.py` (build/run)
- Engine modules: `pztpp/` (build, run, embeddings, neighbors, vocab, score)
- Outputs: `targets_to_sources.(jsonl|csv)`, `sources_to_targets.(jsonl|csv)`, `parse_audit.json`, `run_config*.json`

Data Format
- NDJSON with an `input` field (title, Ingredients:, Instructions:). See README.md for examples.

Note
- This docs folder reflects the current PZT++ engine exclusively.
