from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from .semantic_schema import CanonicalizedRecipe


@dataclass(frozen=True)
class SemanticCoverageReport:
    recipe_count: int
    ingredient_candidates_total: int
    ingredient_known_total: int
    ingredient_unknown_total: int
    ingredient_coverage_ratio: float
    operation_candidates_total: int
    operation_known_total: int
    operation_unknown_total: int
    operation_coverage_ratio: float
    state_candidates_total: int
    state_known_total: int
    state_unknown_total: int
    state_coverage_ratio: float
    unknown_ingredients: List[str] = field(default_factory=list)
    unknown_operations: List[str] = field(default_factory=list)
    unknown_states: List[str] = field(default_factory=list)
    unknown_ingredient_frequencies: Dict[str, int] = field(default_factory=dict)
    unknown_operation_frequencies: Dict[str, int] = field(default_factory=dict)
    unknown_state_frequencies: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_semantic_coverage_report(canonicalized_recipes: List[CanonicalizedRecipe]) -> SemanticCoverageReport:
    ingredient_total = 0
    ingredient_known = 0
    operation_total = 0
    operation_known = 0
    state_known = 0
    state_unknowns: List[str] = []
    ingredient_unknowns: List[str] = []
    operation_unknowns: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []

    for recipe in canonicalized_recipes:
        audit = recipe.canonicalization_audit
        ingredient_total += audit.ingredient_candidates_total
        ingredient_known += audit.canonical_ingredients_total
        operation_total += audit.operation_candidates_total
        operation_known += audit.canonical_operations_total
        ingredient_unknowns.extend(audit.unknown_ingredients)
        operation_unknowns.extend(audit.unknown_operations)
        state_unknowns.extend(audit.unknown_states)
        warnings.extend(f"{recipe.recipe_id}:{warning}" for warning in audit.warnings)
        errors.extend(f"{recipe.recipe_id}:{error}" for error in audit.errors)
        unrecognized_clauses = _unrecognized_instruction_clauses(audit.warnings)
        operation_total += len(unrecognized_clauses)
        operation_unknowns.extend(unrecognized_clauses)
        for ingredient in recipe.canonical_ingredients:
            state_known += len(ingredient.preparation_states)

    ingredient_unknown = ingredient_total - ingredient_known
    operation_unknown = operation_total - operation_known
    state_unknown = len(state_unknowns)
    state_total = state_known + state_unknown

    ingredient_freq = _sorted_freq(ingredient_unknowns)
    operation_freq = _sorted_freq(operation_unknowns)
    state_freq = _sorted_freq(state_unknowns)

    return SemanticCoverageReport(
        recipe_count=len(canonicalized_recipes),
        ingredient_candidates_total=ingredient_total,
        ingredient_known_total=ingredient_known,
        ingredient_unknown_total=ingredient_unknown,
        ingredient_coverage_ratio=_ratio(ingredient_known, ingredient_total),
        operation_candidates_total=operation_total,
        operation_known_total=operation_known,
        operation_unknown_total=operation_unknown,
        operation_coverage_ratio=_ratio(operation_known, operation_total),
        state_candidates_total=state_total,
        state_known_total=state_known,
        state_unknown_total=state_unknown,
        state_coverage_ratio=_ratio(state_known, state_total),
        unknown_ingredients=list(ingredient_freq.keys()),
        unknown_operations=list(operation_freq.keys()),
        unknown_states=list(state_freq.keys()),
        unknown_ingredient_frequencies=ingredient_freq,
        unknown_operation_frequencies=operation_freq,
        unknown_state_frequencies=state_freq,
        warnings=warnings,
        errors=errors,
    )


def _ratio(known: int, total: int) -> float:
    return 1.0 if total == 0 else known / total


def _sorted_freq(values: List[str]) -> Dict[str, int]:
    counter = Counter(values)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _unrecognized_instruction_clauses(warnings: List[str]) -> List[str]:
    prefix = "instruction["
    marker = "]:unrecognized_instruction_clause:"
    out: List[str] = []
    for warning in warnings:
        if warning.startswith(prefix) and marker in warning:
            out.append(warning.split(marker, 1)[1])
    return out
