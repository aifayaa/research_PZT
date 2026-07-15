from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CanonicalConcept:
    canonical_id: str
    kind: str
    label: str
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalPreparationState:
    raw: str
    canonical_state_id: str
    canonical_state_label: str
    implied_operation_id: Optional[str]
    confidence: float


@dataclass(frozen=True)
class CanonicalIngredient:
    raw: str
    base_object_raw: Optional[str]
    canonical_ingredient_id: Optional[str]
    canonical_ingredient_label: Optional[str]
    quantity: Optional[str]
    unit: Optional[str]
    descriptors: List[str] = field(default_factory=list)
    preparation_states: List[CanonicalPreparationState] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalOperation:
    raw_span: str
    operation_lemma_candidate: str
    canonical_operation_id: Optional[str]
    canonical_operation_label: Optional[str]
    input_candidates: List[str] = field(default_factory=list)
    output_state_candidate: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalInstructionStep:
    raw: str
    step_index: int
    canonical_operations: List[CanonicalOperation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalizationAudit:
    ingredient_candidates_total: int
    canonical_ingredients_total: int
    operation_candidates_total: int
    canonical_operations_total: int
    unknown_ingredients: List[str] = field(default_factory=list)
    unknown_operations: List[str] = field(default_factory=list)
    unknown_states: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalizedRecipe:
    recipe_id: str
    title: str
    raw_ingredients: List[str]
    raw_instructions: List[str]
    canonical_ingredients: List[CanonicalIngredient]
    canonical_instruction_steps: List[CanonicalInstructionStep]
    canonicalization_audit: CanonicalizationAudit
