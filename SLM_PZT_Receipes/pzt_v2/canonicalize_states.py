from __future__ import annotations

from .semantic_memory import DEFAULT_SEMANTIC_MEMORY, SemanticMemory
from .semantic_schema import CanonicalPreparationState


def canonicalize_state_candidate(
    raw: str,
    *,
    memory: SemanticMemory = DEFAULT_SEMANTIC_MEMORY,
) -> tuple[CanonicalPreparationState | None, list[str]]:
    concept = memory.lookup_state(raw)
    if concept is None:
        return None, [f"unknown_state:{raw}"]
    return (
        CanonicalPreparationState(
            raw=raw,
            canonical_state_id=concept.canonical_id,
            canonical_state_label=concept.label,
            implied_operation_id=concept.metadata.get("implied_operation_id"),
            confidence=1.0,
        ),
        [],
    )
