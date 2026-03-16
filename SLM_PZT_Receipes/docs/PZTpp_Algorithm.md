# PZT Transferability Engine — Algorithm and Implementation

## Overview

PZT++ is a directional (S→T) “few‑switches” transferability engine for recipes. It represents a recipe via its parts (ingredients and operations), retrieves candidate sources for each target, and re‑ranks them with an adaptation‑cost score that favors sources covering most of the target’s parts while requiring as few changes (“switches”) as possible.

The engine is deterministic (no randomness beyond documented ANN configs), auditable (strict parsing and explicit outputs), and scalable (ANN retrieval and/or brute‑force with batching and memmaps). It produces both T→S and S→T ranked lists with structured explanations.

## Data Model and Parsing (Strict)

Input: NDJSON, one JSON object per line with an `input` string containing:

- Title line
- "Ingredients:" block (case‑insensitive)
- "Instructions:" or "Directions:" block (case‑insensitive)

Parsing rules:
- Strict, no silent fallbacks. If either Ingredients or Instructions is missing:
  - In strict mode: raise error with recipe_id and line number
  - With `--skip-bad-records`: skip and record warning
- Clean bullet prefixes (e.g., `-`, `•`) and leading numbering (e.g., `1.`, `Step 1:`). Remove empty lines.
- Deterministic `recipe_id`:
  - Use `id`, `_id`, or `recipe_id` if present and non‑empty
  - Else `sha1(normalized(full_text))`, where normalization enforces `\n` newlines, strips trailing spaces, and collapses repeated blank lines
- The parser records: `recipe_id`, `title`, `ingredients_lines_raw`, `instructions_lines_raw`, `full_text`, `parse_warnings`, `source_line_number` (1‑based)

Canonical record (superset):
- `ingredients_lines_norm` (see normalization below)
- `ingredients_text_norm` (joined normalized ingredient lines)
- `ingredients_heads` (canonical head terms per line)
- `ingredient_weights` (heuristic importance per line; proteins↑, flavors↑, staples↓; normalized to mean 1.0)
- `operations`/`ops_ids` extracted from instructions (see ops extraction)

## Ingredient Normalization and Head Extraction

Normalization is conservative and stable:
- Remove quantities, units, countables, and parentheticals: e.g., `"2 tsp", "1/2 c.", "oz", "lb", "g", "jar", "can"`
- Trim prep adjectives: `chopped`, `minced`, `sliced`, `fresh`, `ground`, `grated`
- Canonicalize frequent variants via explicit mapping (case‑insensitive; preserves word boundaries):
  - Beans: `chickpea/garbanzo` → `chickpea`
  - Bell peppers: `bell peppers/capsicum` → `bell pepper` (never maps `black pepper`)
  - Cheeses: `cheddar cheese` → `cheddar`; `mozzarella cheese` → `mozzarella`
  - Milks: `condensed milk`, `evaporated milk` retained as multiword heads
  - Soups: `cream of X soup` → `X soup`; `X soup mix` → `X soup`
- Heuristic head reductions: `brown sugar` → `sugar`, `all purpose flour` → `flour`, etc.

Head extraction yields a canonical head per normalized line:
- Prefer known multi‑word heads: `bell pepper`, `evaporated milk`, `condensed milk`, `mushroom soup`, etc.
- Else pick a noun‑like head token (safe singularization) after drop‑ping stopwords (`of`, `and`, `with`, `in`, `for`, `the`, `a`).

Global head vocabulary maps `head → head_id (int)`; per‑recipe `ingredients_heads` are stored as sorted, unique `head_id[]` (sparse and fast).

## Operations (Technique) Extraction

Lexicon‑based extractor (no external NLP downloads). It detects multi‑word patterns first (e.g., `bring_to_boil`, `preheat_oven`) and canonical single verbs (`bake`, `saute`, `simmer`, `boil`, `mix`, `grill`, etc.), unifying spellings (`sauté`→`saute`). The result is a sorted, unique list of op strings, then mapped to `op_id` via a global vocabulary.

## TF‑IDF Weights (Heads + Ops)

Corpus‑wide DF/IDF computed over per‑recipe sets:
- `idf_head(head) = log(1 + N / (1 + df_head))`
- `idf_op(op)     = log(1 + N / (1 + df_op))`

Per‑target T weights (normalized):
- `w_head(T,h) = normalize( heuristic_importance(h) × idf_head(h) )`
- `w_op(T,op)  = normalize( idf_op(op) )`
- Normalization rescales the set of weights per recipe to mean 1.0 for numerical stability.

Toggles: `--no-tfidf-heads`, `--no-tfidf-ops` (default ON for both in presets).

## Embeddings and Caching

Model: `SentenceTransformer('all‑MiniLM‑L6‑v2')` (384‑dim). The engine computes and caches:
- `E_full`: embedding of `full_text`
- `E_ingr`: embedding of `ingredients_text_norm`
- `E_instr`: embedding of instructions (light cleaning only)

Storage:
- Float16 NumPy arrays (memmaps) for `E_full.f16.npy`, `E_ingr.f16.npy`, `E_instr.f16.npy` (cast to float32 for dot products)
- Metadata: `embeddings_meta.json` (model name, dim, dtype)

## Indexes and Retrieval

Two retrieval spaces:
- Ingredients space: `E_ingr`
- Full‑text space: `E_full`

ANN (optional):
- macOS dev: `hnswlib` HNSW (cosine space), defaults: `M=16`, `efConstruction=100`, `efSearch=128–256`
- Linux prod (50K+): FAISS IVF‑PQ or HNSW (recommended IVF‑PQ with OPQ)

Brute‑force (exact):
- Fallback to exact dot products (cosine with L2‑normalized vectors) when `--no-ann` or indexes absent. Implemented as streamed/batched BLAS matmul over memmaps.

Heads inverted index (exact recall booster):
- Build `head_postings.json`: `head_id → [recipe_idx]`. At run time, union these postings into the ANN/brute candidates to guarantee recall of exact head overlaps (bounded cap per target).

## Candidate Generation (High Recall)

Per target T:
1) Retrieve top `cand_k` from ingredients space
2) Retrieve top `cand_k` from full‑text space
3) Union and dedupe (exclude self)
4) Optional: merge in `head_postings` for target’s heads (capped) for recall of exact overlaps

Default retrieval: `hybrid` (both spaces) with `hybrid_w=0.6` for scoring tie‑breaks.

## PZT Scoring, Gates, and Zone

This scoring corrects the MISS‑only limitation by adding precision (false‑alarm) penalties and an explicit “few‑switches” zone filter. The main score is `pzt_score`; the legacy MISS‑only score is preserved for comparability as `pzt_legacy_score`.

Notation per recipe X:
- Heads: `H(X)`; Ops: `O(X)`; Head weights: `w_H(X,h)`; Op weights: `w_O(X,o)`
- Instruction similarity: `sim_instr(S,T) ∈ [0,1]`; Full‑text similarity: `sim_full(S,T)`
- Substitution credit for heads uses precomputed neighbors: `credit(h,Y) = 1` if `h∈H(Y)`, else `max_nb sim(h,nb)^2` if any neighbor in `H(Y)`; else `0`.

Directional recall (aliases of existing coverage):
- Ingredients: `ingr_recall = Σ_{h∈H(T)} w_H(T,h)·credit(h,S) / Σ_{h∈H(T)} w_H(T,h)`
- Operations:  `ops_recall  = Σ_{o∈O(T)} w_O(T,o)·1[o∈O(S)] / Σ_{o∈O(T)} w_O(T,o)`

Directional precision (new):
- Ingredients: `ingr_precision = Σ_{h∈H(S)} w_H(S,h)·credit(h,T) / Σ_{h∈H(S)} w_H(S,h)`
- Operations:  `ops_precision  = Σ_{o∈O(S)} w_O(S,o)·1[o∈O(T)] / Σ_{o∈O(S)} w_O(S,o)`

Masses (MISS + FA) and instruction gap:
- `miss_ingr=1−ingr_recall`, `fa_ingr=1−ingr_precision`
- `miss_ops=1−ops_recall`,  `fa_ops=1−ops_precision`
- `instr_gap=1−sim_instr`

Switch/edit costs and score:
- `C_switch = a_m·miss_ingr + a_f·fa_ingr + b_m·miss_ops + b_f·fa_ops`
- `C_edit = C_switch + c·instr_gap`
- `pzt_edit_score = exp(−C_edit)`

Gates (now include precision):
- `min_ingr_recall=0.55`, `min_ingr_precision=0.55`
- `min_ops_recall=0.45`, `min_ops_precision=0.30`
- `min_instr_sim=0.50`

Few‑switches zone filter (UX default):
- Keep if `min_switch_cost ≤ C_switch ≤ max_switch_cost` (defaults `0.02 ≤ C_switch ≤ 0.25`)
- Mode `few_switches` applies the band; mode `closest` disables (keeps duplicates).

Ranking rule:
- Sort by `pzt_edit_score` desc, then `C_switch` asc (fewer edits), then `sim_full` desc.

## Explanations and Diagnostics

For each (S,T):
- Components: `ingr_recall`, `ingr_precision`, `ops_recall`, `ops_precision`, `sim_instr`, `sim_full`
- Masses: `miss_ingr`, `fa_ingr`, `miss_ops`, `fa_ops`
- Costs/scores: `switch_cost (C_switch)`, `edit_cost (C_edit)`, `pzt_edit_score`; legacy `pzt_score` retained
- Missing/extra explanations:
  - `missing_heads_top`: weighted residuals `w·(1−credit)` (partial subs show smaller residual)
  - `extra_heads_top`: top weighted S heads not needed by T
  - `missing_ops_top`, `extra_ops_top`
- Substitution suggestions (if neighbors enabled): `[(target_head, neighbor_head, sim)]`

## Outputs

Two artifacts are written in one run:

A) per‑target sources (T→S): `targets_to_sources.jsonl` and `.csv`

B) per‑source targets (S→T): `sources_to_targets.jsonl` and `.csv`

Each JSONL record has:
- `query_id`, `query_title`
- `results`: list of entries with:
  - Legacy: `pzt_score`
  - `switch_cost`, `edit_cost`, `scores = { pzt_score, pzt_edit_score }`
  - `components = { cov_ingr_soft, cov_ops_soft, ingr_recall, ingr_precision, ops_recall, ops_precision, sim_instr, sim_full }`
  - `masses = { miss_ingr, fa_ingr, miss_ops, fa_ops }`
  - `explanations = { missing_heads_top, extra_heads_top, missing_ops_top, extra_ops_top, substitution_suggestions_top }`
  - `in_pzt_zone` (boolean)

Always saved:
- `parse_audit.json`, `run_config.json`

## Presets and Flags (Key)

Presets:
- `quality`: `retrieval=hybrid`, `hybrid_w=0.6`, `cand_k=500–1000`, `top_k=20`, `a=0.60`, `b=0.30`, `c=0.10`, gates ON, tie‑break ON, explain_top_n=30
- `fast`: smaller `cand_k`, explain_top_n=10
- `debug`: `limit=200`, `top_k=10`, verbose

Build flags:
- `--no-ann` (refuse when limit > 50k; otherwise brute‑force)
- `--ann-engine {faiss,hnswlib}`
- `--head-neighbors-k`, `--tau-sub` (if head neighbors enabled)

Run flags:
- `--retrieval {hybrid,ingr,full}`, `--hybrid-w`
- `--cand-k`, `--top-k`
- Legacy weights: `--a --b --c` (for `pzt_score`)
- Edit weights: `--a-m --a-f --b-m --b-f --c`
- Gates: `--min-ingr-cov`, `--min-ops-cov`, `--min-instr-sim`, `--min-ingr-precision`, `--min-ops-precision`, `--no-gates`
- Zone: `--mode {closest,few_switches}`, `--min-switch-cost`, `--max-switch-cost`
- `--explain-top-n`, `--gap-threshold`, `--rerank-by-sim-full`

## Performance and Scaling

Embeddings (float16 memmaps): minimal RAM; cast blocks to float32 during matmul/dot products. Retrieval scales via:

- ANN (preferred for 50k+):
  - macOS: `hnswlib` with `M=16`, `efConstruction=100`, `efSearch=128–256`; `cand_k≈1000`
  - Linux: FAISS IVF‑PQ (e.g., `nlist=4096`, `nprobe=32–64`, `PQ m=16`, OPQ if available)

- Exact brute‑force (scientific baseline):
  - Stream Q targets and compute E_ingr/E_full dot products in column chunks; keep per‑query top‑L heaps; union and re‑rank.
  - With heads‑overlap pruning, exact under gates but much faster at 50k.

## Determinism and Auditability

- Strict parsing, explicit warnings (no silent replacements)
- Deterministic IDs and run configurations saved
- No randomness in scoring; ANN settings are versioned and logged

## 10K Brute‑Force Summary (Example)

For a 10K exact run (no ANN; gates ON):

- Targets with ≥1 result: 6199/10000
- Top‑1 (mean): `pzt≈0.8512`, `cov_ingr≈0.7825`, `cov_ops≈0.9730`, `sim_instr≈0.7233`, `sim_full≈0.7050`
- Top missing heads (aggregated): staples/prep traces (`butter`, `milk`, `salt`, `vanilla`, `flour`, ...)

With PZT++‑Edit, duplicates are de‑emphasized by the few‑switches band and precision penalties, surface better “switch a few things” candidates.
