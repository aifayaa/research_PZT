# PZT V2 Refutation Protocol

## Status and claims

PZT V2 has two different validation targets:

1. implementation validity: every artifact is a faithful output of the declared model;
2. empirical validity: the frozen model agrees with later blinded human scores.

Passing implementation gates does not establish human validity. Human labels must
not be used to modify the exact model version evaluated against those labels.

## Hard-gated data flow

```text
NDJSON -> parsing -> extraction -> vocabulary registry -> recipe graphs
       -> directional alignment -> edit script -> T_raw -> PZT_score
```

Every stage writes a `pzt-evidence/v1` manifest with the code commit, command,
configuration hash, input/output hashes, metrics, thresholds, and a `PASS` or
`FAIL` verdict. A downstream stage verifies all upstream manifests and exits
non-zero when one is failed, stale, or modified.

The vocabulary registry is conservative. A validated alias may share a concept
with other validated aliases. An unvalidated surface receives a deterministic
`provisional_singleton` ID. This guarantees complete representation without
inventing a semantic merge. Later clustering may propose merges but cannot
activate them without explicit validation and must-not-link checks.

## Frozen score contract v0.1

Contract identifier: `pzt-v2-equal-unit-edits/v0.1`.

The graph alignment treats typed nodes and typed edges as auditable units.

- identical typed signatures are preserved at cost `0`;
- a same-kind substitution costs `1`;
- an insertion costs `1`;
- a deletion costs `1`;
- every source unit is consumed once;
- every target unit is produced once;
- replaying the edit script must reconstruct the complete target unit set.

For a directional alignment `source -> target`:

```text
C = preserved_target_units / target_units
D_norm = min(1, total_edit_cost / target_units)
T_raw = C * (1 - D_norm)
N = D_norm
PZT_score = T_raw * N
```

`T_raw` and `PZT_score` are always emitted separately. Identity therefore has
maximal raw transferability and zero novelty-adjusted PZT. These equal costs are
a pre-registered, deliberately simple first hypothesis; they must not be tuned
on the first human evaluation set.

## Required implementation refutations

- Parsing: zero silent loss; every record is valid or explicitly rejected.
- Extraction: quantities and units never enter ingredient identity; operations
  appear in textual order and multiple verbs are retained.
- Vocabulary: every token has a validated or provisional singleton concept;
  one alias cannot map to conflicting canonical IDs.
- Graphs: no dangling endpoints; all ingredients, states, operations, results,
  and dependencies retain provenance.
- Alignment: complete source consumption, complete target reconstruction, and
  exact cost recomputation from edit operations.
- Scores: identity, lexical equivalence, state changes, directionality, and
  monotonic edit properties are fixture-tested.

## Scale ladder

Runs progress through fixtures, 200 recipes, 1,000 pairs, and finally 10,000
directional pairs. Each scale has its own manifests. The final 10K run uses a
deterministic pair seed and scores every selected pair exactly; embeddings do
not enter `T_raw` or `PZT_score`.

## Blinded human comparison

From the frozen 10K output, five pairs are selected reproducibly from each
`T_raw` decile. Fifty distinct directional pairs plus five blind repeats are
presented without model scores. The evaluator declares whether the source is
known and, if so, assigns an integer transferability score from 1 to 6.

The pre-registered primary metric is Spearman correlation. Support requires
`rho >= 0.50`, bootstrap lower bound `> 0.20`, fixed-projection MAE `<= 1`, and
stable scores on at least four of five blind repeats. A non-positive rank
correlation refutes the frozen hypothesis; intermediate outcomes are
inconclusive. Any remediated model must be tested on new held-out pairs.
