from __future__ import annotations

from .canonicalize_states import canonicalize_state_candidate
from .extraction_schema import IngredientCandidate
from .semantic_memory import DEFAULT_SEMANTIC_MEMORY, SemanticMemory
from .semantic_schema import CanonicalIngredient


def canonicalize_ingredient_candidate(
    candidate: IngredientCandidate,
    *,
    memory: SemanticMemory = DEFAULT_SEMANTIC_MEMORY,
) -> CanonicalIngredient:
    warnings = list(candidate.warnings)
    concept = memory.lookup_ingredient(candidate.base_object_candidate)
    if concept is None:
        canonical_id = None
        canonical_label = None
        if candidate.base_object_candidate:
            warnings.append(f"unknown_ingredient:{candidate.base_object_candidate}")
        else:
            warnings.append("missing_base_object_candidate")
    else:
        canonical_id = concept.canonical_id
        canonical_label = concept.label

    states = []
    for state_raw in candidate.preparation_state_candidates:
        state, state_warnings = canonicalize_state_candidate(state_raw, memory=memory)
        warnings.extend(state_warnings)
        if state is not None:
            states.append(state)

    return CanonicalIngredient(
        raw=candidate.raw,
        base_object_raw=candidate.base_object_candidate,
        canonical_ingredient_id=canonical_id,
        canonical_ingredient_label=canonical_label,
        quantity=candidate.quantity,
        unit=candidate.unit,
        descriptors=list(candidate.descriptors),
        preparation_states=states,
        warnings=warnings,
    )
