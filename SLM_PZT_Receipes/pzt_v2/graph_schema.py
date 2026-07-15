from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


NODE_KINDS = {"ingredient", "operation", "state", "result"}
NODE_SOURCES = {"ingredient_line", "instruction_step", "ingredient_modifier", "inferred", "manual_fixture"}
EDGE_RELATIONS = {
    "input_to_operation",
    "operation_to_state",
    "state_to_operation",
    "ingredient_to_state",
    "state_to_result",
    "ingredient_to_operation",
    "operation_to_result",
}


class GraphValidationError(ValueError):
    """Raised when a PZT V2 graph or fixture violates the model contract."""


@dataclass(frozen=True)
class RecipeNode:
    id: str
    kind: str
    label: str
    canonical_id: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecipeNode":
        return cls(
            id=_required_str(data, "id"),
            kind=_required_str(data, "kind"),
            label=_required_str(data, "label"),
            canonical_id=_required_str(data, "canonical_id"),
            source=_required_str(data, "source"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RecipeEdge:
    id: str
    source: str
    target: str
    relation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecipeEdge":
        return cls(
            id=_required_str(data, "id"),
            source=_required_str(data, "source"),
            target=_required_str(data, "target"),
            relation=_required_str(data, "relation"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RecipeStep:
    id: str
    raw_text: str
    operation_nodes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecipeStep":
        return cls(
            id=_required_str(data, "id"),
            raw_text=_required_str(data, "raw_text"),
            operation_nodes=list(data.get("operation_nodes") or []),
        )


@dataclass(frozen=True)
class RecipeGraph:
    recipe_id: str
    title: str
    nodes: List[RecipeNode]
    edges: List[RecipeEdge]
    steps: List[RecipeStep] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecipeGraph":
        return cls(
            recipe_id=_required_str(data, "recipe_id"),
            title=_required_str(data, "title"),
            nodes=[RecipeNode.from_dict(item) for item in data.get("nodes", [])],
            edges=[RecipeEdge.from_dict(item) for item in data.get("edges", [])],
            steps=[RecipeStep.from_dict(item) for item in data.get("steps", [])],
        )


@dataclass(frozen=True)
class RecipeFixture:
    recipe_id: str
    title: str
    raw_ingredients: List[str]
    raw_instructions: List[str]
    graph: RecipeGraph

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecipeFixture":
        graph_data = dict(data.get("graph") or {})
        graph_data.setdefault("recipe_id", data.get("recipe_id"))
        graph_data.setdefault("title", data.get("title"))
        return cls(
            recipe_id=_required_str(data, "recipe_id"),
            title=_required_str(data, "title"),
            raw_ingredients=[str(item) for item in data.get("raw_ingredients", [])],
            raw_instructions=[str(item) for item in data.get("raw_instructions", [])],
            graph=RecipeGraph.from_dict(graph_data),
        )


@dataclass(frozen=True)
class ExpectedAcceptance:
    must_match_units: List[str] = field(default_factory=list)
    must_not_match_units: List[List[str]] = field(default_factory=list)
    expected_difference_type: str = ""
    expected_future_score_behavior: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedAcceptance":
        return cls(
            must_match_units=[str(item) for item in data.get("must_match_units", [])],
            must_not_match_units=[list(item) for item in data.get("must_not_match_units", [])],
            expected_difference_type=str(data.get("expected_difference_type", "")),
            expected_future_score_behavior=str(data.get("expected_future_score_behavior", "")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class GraphFixtureCase:
    case_id: str
    description: str
    source_recipe: RecipeFixture
    target_recipe: RecipeFixture
    expected: ExpectedAcceptance

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphFixtureCase":
        return cls(
            case_id=_required_str(data, "case_id"),
            description=_required_str(data, "description"),
            source_recipe=RecipeFixture.from_dict(data.get("source_recipe") or {}),
            target_recipe=RecipeFixture.from_dict(data.get("target_recipe") or {}),
            expected=ExpectedAcceptance.from_dict(data.get("expected") or {}),
        )


def load_fixture_case(path: Path) -> GraphFixtureCase:
    return GraphFixtureCase.from_dict(json.loads(path.read_text(encoding="utf-8")))


def validate_fixture_case(case: GraphFixtureCase) -> None:
    validate_recipe_graph(case.source_recipe.graph)
    validate_recipe_graph(case.target_recipe.graph)

    source_units = canonical_ids(case.source_recipe.graph)
    target_units = canonical_ids(case.target_recipe.graph)
    for unit in case.expected.must_match_units:
        if unit not in source_units:
            raise GraphValidationError(f"{case.case_id}: expected source unit missing: {unit}")
        if unit not in target_units:
            raise GraphValidationError(f"{case.case_id}: expected target unit missing: {unit}")

    all_units = source_units | target_units
    for pair in case.expected.must_not_match_units:
        if len(pair) != 2:
            raise GraphValidationError(f"{case.case_id}: must_not_match_units entries must be pairs")
        left, right = pair
        if left == right:
            raise GraphValidationError(f"{case.case_id}: must-not-match pair collapsed to one canonical id: {left}")
        if left not in all_units:
            raise GraphValidationError(f"{case.case_id}: must-not-match unit missing: {left}")
        if right not in all_units:
            raise GraphValidationError(f"{case.case_id}: must-not-match unit missing: {right}")


def validate_recipe_graph(graph: RecipeGraph) -> None:
    if not graph.nodes:
        raise GraphValidationError(f"{graph.recipe_id}: graph has no nodes")

    node_ids = [node.id for node in graph.nodes]
    duplicate_nodes = _duplicates(node_ids)
    if duplicate_nodes:
        raise GraphValidationError(f"{graph.recipe_id}: duplicate node ids: {duplicate_nodes}")

    edge_ids = [edge.id for edge in graph.edges]
    duplicate_edges = _duplicates(edge_ids)
    if duplicate_edges:
        raise GraphValidationError(f"{graph.recipe_id}: duplicate edge ids: {duplicate_edges}")

    node_id_set = set(node_ids)
    for node in graph.nodes:
        if node.kind not in NODE_KINDS:
            raise GraphValidationError(f"{graph.recipe_id}: invalid node kind {node.kind!r} for {node.id}")
        if node.source not in NODE_SOURCES:
            raise GraphValidationError(f"{graph.recipe_id}: invalid node source {node.source!r} for {node.id}")
        if not node.canonical_id:
            raise GraphValidationError(f"{graph.recipe_id}: node {node.id} has no canonical_id")

    for edge in graph.edges:
        if edge.relation not in EDGE_RELATIONS:
            raise GraphValidationError(f"{graph.recipe_id}: invalid edge relation {edge.relation!r} for {edge.id}")
        if edge.source not in node_id_set:
            raise GraphValidationError(f"{graph.recipe_id}: edge {edge.id} source does not exist: {edge.source}")
        if edge.target not in node_id_set:
            raise GraphValidationError(f"{graph.recipe_id}: edge {edge.id} target does not exist: {edge.target}")


def canonical_ids(graph: RecipeGraph) -> set[str]:
    return {node.canonical_id for node in graph.nodes}


def canonical_signature(graph: RecipeGraph) -> Dict[str, Any]:
    node_sig = sorted((node.kind, node.canonical_id, node.label) for node in graph.nodes)
    by_id = {node.id: node.canonical_id for node in graph.nodes}
    edge_sig = sorted((edge.relation, by_id[edge.source], by_id[edge.target]) for edge in graph.edges)
    return {"nodes": node_sig, "edges": edge_sig}


def _required_str(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise GraphValidationError(f"missing required string field: {key}")
    return value


def _duplicates(values: List[str]) -> List[str]:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
