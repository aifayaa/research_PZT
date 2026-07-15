# PZT V2 evidence snapshot — 2026-07-15

Frozen model commit: `ea801a88edc16fe8877240ab46f10dbe36dcc38f`

Frozen tag: `pzt-v2-model-v0.2.0`

All committed manifests have verdict `PASS`, `code_dirty=false`, and reference
the frozen commit. Large vocabularies, graph pairs, and registries remain under
`/tmp/pzt_v2_*`; their SHA-256 hashes are recorded in the manifests.

Key results:

- parsing: 20,000/20,000 valid, zero raw-preservation failure;
- extraction: 20,000/20,000 completed, zero operation-order violation;
- vocabulary: 436,929/436,929 occurrences represented;
- validated semantic alias mass: 65.76%; remaining concepts are conservative
  provisional singletons, not unreviewed synonym merges;
- pair ladder: 200, 1,000, and 10,000 directional pairs completed;
- 10K graph/alignment errors: zero;
- 10K raw zero fraction: 11.28%; score resolution: 6,262 values at 8 decimals;
- blinded human package: 50 distinct pairs across ten deciles plus five repeats.

The human empirical verdict is intentionally absent until the blind CSV is
completed by the evaluator and analyzed with `pzt_v2.run_human_analysis`.
