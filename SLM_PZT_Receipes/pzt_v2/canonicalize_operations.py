from __future__ import annotations

from .extraction_schema import OperationCandidate
from .semantic_memory import DEFAULT_SEMANTIC_MEMORY, SemanticMemory
from .semantic_schema import CanonicalOperation


def canonicalize_operation_candidate(
    candidate: OperationCandidate,
    *,
    memory: SemanticMemory = DEFAULT_SEMANTIC_MEMORY,
) -> CanonicalOperation:
    warnings = list(candidate.warnings)
    lookup_key = _operation_lookup_key(candidate)
    concept = memory.lookup_operation(lookup_key) or memory.lookup_operation(candidate.operation_lemma_candidate)
    if concept is None:
        canonical_id = None
        canonical_label = None
        warnings.append(f"unknown_operation:{candidate.operation_lemma_candidate}")
    else:
        canonical_id = concept.canonical_id
        canonical_label = concept.label

    return CanonicalOperation(
        raw_span=candidate.raw_span,
        operation_lemma_candidate=candidate.operation_lemma_candidate,
        canonical_operation_id=canonical_id,
        canonical_operation_label=canonical_label,
        input_candidates=list(candidate.input_candidates),
        output_state_candidate=candidate.output_state_candidate,
        warnings=warnings,
    )


def _operation_lookup_key(candidate: OperationCandidate) -> str:
    raw = candidate.raw_span.lower()
    if candidate.operation_lemma_candidate == "cook" and "pan" in raw:
        return "cook in pan"
    if candidate.operation_lemma_candidate == "mix" and "stir" in raw and "together" in raw:
        return "stir together"
    return candidate.operation_lemma_candidate
