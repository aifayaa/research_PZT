from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence

from .graph_schema import RecipeEdge, RecipeGraph, RecipeNode, RecipeStep, validate_recipe_graph
from .semantic_memory import DEFAULT_SEMANTIC_MEMORY, SemanticMemory, normalize_alias
from .semantic_schema import CanonicalizedRecipe
from .vocabulary_registry import singleton_id


_GENERIC_INPUTS = {
    "ingredient",
    "ingredients",
    "dry ingredients",
    "wet ingredients",
    "everything",
    "mixture",
    "batter",
}


def build_recipe_graph(
    recipe: CanonicalizedRecipe,
    *,
    memory: SemanticMemory = DEFAULT_SEMANTIC_MEMORY,
) -> RecipeGraph:
    nodes: List[RecipeNode] = []
    edges: List[RecipeEdge] = []
    steps: List[RecipeStep] = []
    counters: Dict[str, int] = defaultdict(int)
    ingredient_nodes: List[str] = []
    surface_to_nodes: Dict[str, List[str]] = defaultdict(list)
    previous_outputs: List[str] = []

    def add_node(kind: str, label: str, canonical_id: str, source: str, metadata: Optional[dict] = None) -> str:
        counters[kind] += 1
        node_id = f"{kind}_{counters[kind]:04d}"
        nodes.append(
            RecipeNode(
                id=node_id,
                kind=kind,
                label=label,
                canonical_id=canonical_id,
                source=source,
                metadata=dict(metadata or {}),
            )
        )
        return node_id

    def add_edge(source_id: str, target_id: str, relation: str, metadata: Optional[dict] = None) -> None:
        counters["edge"] += 1
        edges.append(
            RecipeEdge(
                id=f"edge_{counters['edge']:05d}",
                source=source_id,
                target=target_id,
                relation=relation,
                metadata=dict(metadata or {}),
            )
        )

    for ingredient_index, ingredient in enumerate(recipe.canonical_ingredients):
        raw_label = ingredient.base_object_raw or ingredient.raw
        canonical_id = ingredient.canonical_ingredient_id or singleton_id("ingredient", normalize_alias(raw_label))
        ingredient_id = add_node(
            "ingredient",
            raw_label,
            canonical_id,
            "ingredient_line",
            {
                "raw": ingredient.raw,
                "quantity": ingredient.quantity,
                "unit": ingredient.unit,
                "descriptors": ingredient.descriptors,
                "provisional": ingredient.canonical_ingredient_id is None,
                "ingredient_index": ingredient_index,
            },
        )
        ingredient_nodes.append(ingredient_id)
        for surface in {raw_label, ingredient.raw, ingredient.canonical_ingredient_label or ""}:
            normalized = normalize_alias(surface)
            if normalized:
                surface_to_nodes[normalized].append(ingredient_id)

        for state_index, state in enumerate(ingredient.preparation_states):
            state_label = state.raw
            state_id = add_node(
                "state",
                state_label,
                state.canonical_state_id or singleton_id("state", normalize_alias(state_label)),
                "ingredient_modifier",
                {
                    "implied_operation_id": state.implied_operation_id,
                    "ingredient_index": ingredient_index,
                    "state_index": state_index,
                },
            )
            add_edge(ingredient_id, state_id, "ingredient_to_state")
            surface_to_nodes[normalize_alias(f"{state_label} {raw_label}")].append(state_id)
            surface_to_nodes[normalize_alias(state_label)].append(state_id)

    for canonical_step in recipe.canonical_instruction_steps:
        step_operation_nodes: List[str] = []
        for operation_index, operation in enumerate(canonical_step.canonical_operations):
            operation_label = operation.operation_lemma_candidate
            canonical_id = operation.canonical_operation_id or singleton_id(
                "operation", normalize_alias(operation_label)
            )
            operation_id = add_node(
                "operation",
                operation_label,
                canonical_id,
                "instruction_step",
                {
                    "raw_span": operation.raw_span,
                    "step_index": canonical_step.step_index,
                    "operation_index": operation_index,
                    "provisional": operation.canonical_operation_id is None,
                },
            )
            step_operation_nodes.append(operation_id)
            input_ids = _resolve_input_nodes(operation.input_candidates, surface_to_nodes)
            if not input_ids and previous_outputs:
                input_ids = list(previous_outputs)
            if not input_ids and any(normalize_alias(item) in _GENERIC_INPUTS for item in operation.input_candidates):
                input_ids = list(ingredient_nodes)
            if not input_ids and not operation.input_candidates:
                input_ids = list(previous_outputs)
            for input_id in _dedupe(input_ids):
                input_kind = next(node.kind for node in nodes if node.id == input_id)
                relation = "state_to_operation" if input_kind == "state" else "ingredient_to_operation"
                add_edge(input_id, operation_id, relation)

            previous_outputs = []
            if operation.output_state_candidate:
                output_label = operation.output_state_candidate
                canonical_state_id, provisional = _canonical_output_state(
                    output_label,
                    operation.canonical_operation_id,
                    input_ids,
                    nodes,
                    recipe.title,
                    memory,
                )
                state_id = add_node(
                    "state",
                    output_label,
                    canonical_state_id,
                    "instruction_step",
                    {
                        "step_index": canonical_step.step_index,
                        "operation_index": operation_index,
                        "provisional": provisional,
                    },
                )
                add_edge(operation_id, state_id, "operation_to_state")
                previous_outputs = [state_id]
                surface_to_nodes[normalize_alias(output_label)].append(state_id)
            else:
                previous_outputs = [operation_id]

        for warning_index, warning in enumerate(canonical_step.warnings):
            marker = "unrecognized_instruction_clause:"
            if marker not in warning:
                continue
            raw_clause = warning.split(marker, 1)[1]
            operation_id = add_node(
                "operation",
                raw_clause,
                singleton_id("operation", normalize_alias(raw_clause)),
                "instruction_step",
                {
                    "raw_span": raw_clause,
                    "step_index": canonical_step.step_index,
                    "warning_index": warning_index,
                    "provisional": True,
                    "unrecognized": True,
                },
            )
            for input_id in previous_outputs:
                input_kind = next(node.kind for node in nodes if node.id == input_id)
                relation = "state_to_operation" if input_kind == "state" else "ingredient_to_operation"
                add_edge(input_id, operation_id, relation)
            previous_outputs = [operation_id]
            step_operation_nodes.append(operation_id)

        steps.append(
            RecipeStep(
                id=f"step_{canonical_step.step_index:04d}",
                raw_text=canonical_step.raw,
                operation_nodes=step_operation_nodes,
            )
        )

    result_id = add_node(
        "result",
        recipe.title,
        _result_canonical_id(recipe.title),
        "inferred",
        {"provisional": True, "parent_canonical_id": _result_parent_id(recipe.title)},
    )
    terminal_ids = previous_outputs or ingredient_nodes[-1:]
    for terminal_id in terminal_ids:
        terminal_kind = next(node.kind for node in nodes if node.id == terminal_id)
        relation = "operation_to_result" if terminal_kind == "operation" else "state_to_result"
        add_edge(terminal_id, result_id, relation)

    graph = RecipeGraph(recipe_id=recipe.recipe_id, title=recipe.title, nodes=nodes, edges=edges, steps=steps)
    validate_recipe_graph(graph)
    return graph


def recipe_graph_to_dict(graph: RecipeGraph) -> dict:
    return asdict(graph)


def _resolve_input_nodes(input_candidates: Sequence[str], surface_to_nodes: Dict[str, List[str]]) -> List[str]:
    resolved: List[str] = []
    for candidate in input_candidates:
        normalized = normalize_alias(candidate)
        if not normalized or normalized in _GENERIC_INPUTS:
            continue
        exact = surface_to_nodes.get(normalized)
        if exact:
            resolved.extend(exact)
            continue
        compatible = [
            (surface, node_ids)
            for surface, node_ids in surface_to_nodes.items()
            if surface and (surface in normalized or normalized in surface)
        ]
        if compatible:
            compatible.sort(key=lambda item: len(item[0]), reverse=True)
            resolved.extend(compatible[0][1])
    return _dedupe(resolved)


def _result_canonical_id(title: str) -> str:
    normalized = normalize_alias(title)
    if "flour" in normalized and ("egg" in normalized or "eggs" in normalized):
        return "RESULT_FLOUR_EGG_MIXTURE"
    if "stir fry" in normalized:
        return "RESULT_STIR_FRY"
    slug = re.sub(r"[^A-Z0-9]+", "_", normalized.upper()).strip("_")[:50] or "UNTITLED"
    return f"RESULT_{slug}"


def _result_parent_id(title: str) -> Optional[str]:
    return "RESULT_QUICK_BREAD" if "bread" in normalize_alias(title) else None


def _canonical_output_state(
    output_label: str,
    operation_id: Optional[str],
    input_ids: Sequence[str],
    nodes: Sequence[RecipeNode],
    title: str,
    memory: SemanticMemory,
) -> tuple[str, bool]:
    normalized = normalize_alias(output_label)
    if "batter" in normalized or (operation_id == "OP_MIX" and "bread" in normalize_alias(title)):
        return "STATE_BATTER", False
    tokens = normalized.split()
    state_prefixes = {
        "mashed": "STATE_MASHED",
        "grated": "STATE_GRATED",
        "chopped": "STATE_CHOPPED",
        "diced": "STATE_DICED",
        "sliced": "STATE_SLICED",
        "melted": "STATE_MELTED",
        "beaten": "STATE_BEATEN",
        "boiled": "STATE_BOILED",
    }
    if tokens and tokens[0] in state_prefixes and len(tokens) > 1:
        suffix = re.sub(r"[^A-Z0-9]+", "_", "_".join(tokens[1:]).upper()).strip("_")
        return f"{state_prefixes[tokens[0]]}_{suffix}", False
    if operation_id == "OP_MIX":
        by_id = {node.id: node for node in nodes}
        ingredients = [
            by_id[node_id].canonical_id.removeprefix("ING_")
            for node_id in input_ids
            if node_id in by_id and by_id[node_id].kind == "ingredient"
        ]
        if ingredients:
            return f"STATE_{'_'.join(ingredients)}_MIXTURE", False
    state_concept = memory.lookup_state(output_label)
    if state_concept:
        return state_concept.canonical_id, False
    return singleton_id("state", normalized), True


def _dedupe(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))
