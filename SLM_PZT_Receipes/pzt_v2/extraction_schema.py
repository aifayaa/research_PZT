from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class IngredientCandidate:
    raw: str
    quantity: Optional[str]
    unit: Optional[str]
    base_object_candidate: Optional[str]
    preparation_state_candidates: List[str] = field(default_factory=list)
    descriptors: List[str] = field(default_factory=list)
    form_candidates: List[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OperationCandidate:
    raw_span: str
    operation_lemma_candidate: str
    input_candidates: List[str] = field(default_factory=list)
    output_state_candidate: Optional[str] = None
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class InstructionStepCandidate:
    raw: str
    step_index: int
    operation_candidates: List[OperationCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionAudit:
    ingredient_lines_total: int
    ingredient_candidates_total: int
    instruction_steps_total: int
    operation_candidates_total: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecipeExtraction:
    recipe_id: str
    title: str
    raw_ingredients: List[str]
    raw_instructions: List[str]
    ingredient_candidates: List[IngredientCandidate]
    instruction_step_candidates: List[InstructionStepCandidate]
    extraction_audit: ExtractionAudit
