from __future__ import annotations

from typing import List

from .canonicalize_ingredients import canonicalize_ingredient_candidate
from .canonicalize_operations import canonicalize_operation_candidate
from .extraction_schema import RecipeExtraction
from .semantic_memory import DEFAULT_SEMANTIC_MEMORY, SemanticMemory
from .semantic_schema import (
    CanonicalInstructionStep,
    CanonicalizationAudit,
    CanonicalizedRecipe,
)


def canonicalize_recipe_extraction(
    extraction: RecipeExtraction,
    *,
    memory: SemanticMemory = DEFAULT_SEMANTIC_MEMORY,
) -> CanonicalizedRecipe:
    canonical_ingredients = [
        canonicalize_ingredient_candidate(candidate, memory=memory)
        for candidate in extraction.ingredient_candidates
    ]

    canonical_steps: List[CanonicalInstructionStep] = []
    for step in extraction.instruction_step_candidates:
        canonical_ops = [
            canonicalize_operation_candidate(operation, memory=memory)
            for operation in step.operation_candidates
        ]
        canonical_steps.append(
            CanonicalInstructionStep(
                raw=step.raw,
                step_index=step.step_index,
                canonical_operations=canonical_ops,
                warnings=list(step.warnings),
            )
        )

    unknown_ingredients = [
        ingredient.base_object_raw or ingredient.raw
        for ingredient in canonical_ingredients
        if ingredient.canonical_ingredient_id is None
    ]
    unknown_states = [
        warning.split(":", 1)[1]
        for ingredient in canonical_ingredients
        for warning in ingredient.warnings
        if warning.startswith("unknown_state:")
    ]
    all_ops = [operation for step in canonical_steps for operation in step.canonical_operations]
    unknown_operations = [
        operation.operation_lemma_candidate
        for operation in all_ops
        if operation.canonical_operation_id is None
    ]

    warnings: List[str] = []
    for idx, ingredient in enumerate(canonical_ingredients):
        warnings.extend(f"ingredient[{idx}]:{warning}" for warning in ingredient.warnings)
    for step in canonical_steps:
        warnings.extend(f"instruction[{step.step_index}]:{warning}" for warning in step.warnings)
        for op_idx, operation in enumerate(step.canonical_operations):
            warnings.extend(f"instruction[{step.step_index}].operation[{op_idx}]:{warning}" for warning in operation.warnings)

    canonical_operation_count = sum(1 for operation in all_ops if operation.canonical_operation_id is not None)
    audit = CanonicalizationAudit(
        ingredient_candidates_total=len(extraction.ingredient_candidates),
        canonical_ingredients_total=sum(1 for ingredient in canonical_ingredients if ingredient.canonical_ingredient_id is not None),
        operation_candidates_total=sum(len(step.operation_candidates) for step in extraction.instruction_step_candidates),
        canonical_operations_total=canonical_operation_count,
        unknown_ingredients=unknown_ingredients,
        unknown_operations=unknown_operations,
        unknown_states=unknown_states,
        warnings=warnings,
        errors=[],
    )

    return CanonicalizedRecipe(
        recipe_id=extraction.recipe_id,
        title=extraction.title,
        raw_ingredients=list(extraction.raw_ingredients),
        raw_instructions=list(extraction.raw_instructions),
        canonical_ingredients=canonical_ingredients,
        canonical_instruction_steps=canonical_steps,
        canonicalization_audit=audit,
    )
