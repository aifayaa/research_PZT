from __future__ import annotations

from typing import List

from .extraction_schema import ExtractionAudit, RecipeExtraction
from .ingredient_extraction import extract_ingredient_candidates
from .instruction_extraction import extract_instruction_steps
from .parsing import ParsedRecipe


def extract_recipe_candidates(parsed_recipe: ParsedRecipe) -> RecipeExtraction:
    ingredient_candidates = extract_ingredient_candidates(parsed_recipe.raw_ingredients)
    instruction_step_candidates = extract_instruction_steps(parsed_recipe.raw_instructions)

    warnings: List[str] = []
    errors: List[str] = []
    for idx, candidate in enumerate(ingredient_candidates):
        warnings.extend(f"ingredient[{idx}]:{warning}" for warning in candidate.warnings)
    for step in instruction_step_candidates:
        warnings.extend(f"instruction[{step.step_index}]:{warning}" for warning in step.warnings)
        for op_idx, op in enumerate(step.operation_candidates):
            warnings.extend(f"instruction[{step.step_index}].operation[{op_idx}]:{warning}" for warning in op.warnings)

    operation_count = sum(len(step.operation_candidates) for step in instruction_step_candidates)
    audit = ExtractionAudit(
        ingredient_lines_total=len(parsed_recipe.raw_ingredients),
        ingredient_candidates_total=len(ingredient_candidates),
        instruction_steps_total=len(parsed_recipe.raw_instructions),
        operation_candidates_total=operation_count,
        warnings=warnings,
        errors=errors,
    )
    return RecipeExtraction(
        recipe_id=parsed_recipe.recipe_id,
        title=parsed_recipe.title,
        raw_ingredients=list(parsed_recipe.raw_ingredients),
        raw_instructions=list(parsed_recipe.raw_instructions),
        ingredient_candidates=ingredient_candidates,
        instruction_step_candidates=instruction_step_candidates,
        extraction_audit=audit,
    )
