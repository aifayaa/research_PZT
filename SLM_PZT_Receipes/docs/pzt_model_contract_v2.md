# PZT V2 Model Contract

## 1. Purpose

PZT V2 defines the scientific and technical contract for the next recipe transferability model.

The model estimates whether a human can mentally transform a known source recipe into a target recipe through a limited number of cognitive edit operations.

The implementation must therefore represent recipes as semantic transformation graphs before any transferability score is accepted.

## 2. Scientific Definition of PZT V2

PZT V2 scores a cognitive transformation cost between two conceptual recipe graphs.

The intended pipeline is:

```text
raw recipe text
-> semantic canonical concepts
-> recipe graph
-> cognitive edit distance
-> transferability score
```

A source recipe `A` is transferable to a target recipe `B` when the graph for `A` can be transformed into the graph for `B` with limited, explainable edits.

## 3. What the Model Is Not

PZT V2 does not score raw lexical similarity.

PZT V2 does not score global culinary proximity.

PZT V2 does not use embeddings as the final transferability score.

Embedding similarity, instruction similarity, full-text similarity, and other latent measures may be used for candidate retrieval, semantic canonicalization, auditing, or final tie-breaks. They must not replace the graph-based cognitive edit model.

A pair must never be considered highly transferable only because its full-text embedding is close.

## 4. Semantic Memory and Latent Representation

Semantic memory is used to recognize that different lexical forms can express the same concept.

Examples:

```text
mix
combine
stir together
incorporate
```

may canonicalize to:

```text
OP_MIX
```

Likewise:

```text
zucchini
courgette
```

may canonicalize to:

```text
ING_ZUCCHINI
```

Latent representation is used for semantic canonicalization. It is not the final transferability score.

## 5. Canonical Concepts

Canonicalization merges lexical synonyms while preserving real conceptual differences.

Important example:

```text
tomato
chopped tomato
```

must not become the same single concept.

The correct representation must preserve at least:

```text
tomato -> ING_TOMATO
chopped tomato -> ING_TOMATO + STATE_CHOPPED
```

or, transformationally:

```text
OP_CHOP(ING_TOMATO) -> STATE_CHOPPED_TOMATO
```

Therefore, `chopped tomato != tomato`, while both share `ING_TOMATO`.

## 6. Recipe Graph Definition

A recipe graph represents ingredients, preparation states, operations, intermediate states, and results.

The graph must support input/output dependencies:

```text
ingredient/state -> operation -> state/result
```

Example:

```text
mash(ING_BANANA) -> STATE_MASHED_BANANA
mix(ING_FLOUR, ING_EGG, STATE_MASHED_BANANA) -> STATE_BATTER
bake(STATE_BATTER) -> RESULT_BANANA_BREAD
```

Recipe graphs are the substrate for cognitive edit distance.

## 7. Ingredient Objects, Preparation States, Operations, and Intermediate States

The required conceptual entities are:

- `IngredientObject`: base ingredient concept such as `ING_TOMATO`.
- `PreparationState`: ingredient state such as `STATE_CHOPPED`.
- `Operation`: transformation such as `OP_CHOP`, `OP_MIX`, or `OP_BAKE`.
- `IntermediateState`: produced mixture, batter, base, or prepared object.
- `RecipeStep`: a documented or inferred transformation step.
- `RecipeGraph`: the directed input/output graph for a recipe.
- `EditOperation`: a future alignment edit such as insert, delete, substitute, or preserve.
- `TransferabilityScore`: future raw transferability score before novelty correction.
- `PZTScore`: future useful transferability score after non-identity correction.

Ingredient preparation modifiers must not be dropped.

A head-only ingredient representation is scientifically insufficient.

Example:

```text
"2 cups chopped tomatoes" must not be reduced to only ING_TOMATO.
```

It must preserve at least:

- `ING_TOMATO`
- `STATE_CHOPPED`
- optionally implied operation `OP_CHOP`

## 8. Directionality: Source -> Target

Transferability is directional.

`A -> B` asks how a human can transform the source recipe graph into the target recipe graph. The reverse direction may have a different edit script and a different cognitive cost.

All future alignments, edit operations, and scores must record the direction.

## 9. Cognitive Edit Distance: Future Scope

Cognitive edit distance is not implemented in PR 1.

Future implementations must produce explicit edit operations before calculating a distance:

- matched units
- substitutions
- insertions
- deletions
- unmatched source units
- unmatched target units

The final distance must remain traceable to those edit operations.

## 10. Transferability Score: Future Scope

Transferability scoring is not implemented in PR 1.

The intended future shape is:

```text
T_raw(A,B) = C(A,B) * (1 - D_norm(A,B))
PZT_score(A,B) = T_raw(A,B) * N(A,B)
```

Where:

- `C(A,B)` is structural conservation.
- `D_norm(A,B)` is normalized cognitive edit distance.
- `N(A,B)` is a non-identity or novelty factor.

The principal score must depend on canonical recipe graphs, conservation, and edit distance, not directly on embeddings.

## 11. Required Acceptance Cases

Future work must preserve the following cases:

1. Lexical variation without cognitive difference: `stir` and `combine` canonicalize to `OP_MIX`.
2. Same ingredient head but different preparation state: `tomato` and `chopped tomato` share `ING_TOMATO`, but `chopped tomato` preserves `STATE_CHOPPED` or `STATE_CHOPPED_TOMATO`.
3. Same operational structure, different ingredient: chicken stir-fry to tofu stir-fry should preserve structure while representing ingredient substitution.
4. Same ingredient, different operations: boiled egg to scrambled eggs should not be strong transfer only because both share `ING_EGG`.
5. Identity: identical graphs should have maximal raw transferability, but low useful PZT after non-identity correction.
6. Strong non-trivial transfer: banana bread to zucchini bread should conserve mix/bake structure while preserving ingredient/prep substitutions.

## 12. Implementation Gates for Future PRs

No scoring implementation can be accepted unless it produces these objects or direct equivalents:

- `IngredientObject`
- `PreparationState`
- `Operation`
- `IntermediateState`
- `RecipeStep`
- `RecipeGraph`
- `EditOperation`
- `TransferabilityScore`
- `PZTScore`

Required gates:

- Parsing must preserve raw semantic information.
- Extraction must distinguish base object, preparation state, explicit operation, and intermediate state.
- Canonicalization must merge lexical synonyms while preserving conceptual differences.
- Graph building must represent ingredients, operations, states, intermediate states, and dependencies.
- Alignment must explain what is conserved and what changes.
- Scoring must be traceable to edit operations.
- Candidate retrieval may use embeddings, but graph-based PZT scoring must be the scientific ranking layer.
