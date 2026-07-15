from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .graph_schema import RecipeEdge, RecipeGraph, RecipeNode


class AlignmentValidationError(ValueError):
    pass


@dataclass(frozen=True)
class GraphUnit:
    unit_id: str
    unit_kind: str
    canonical_id: str
    signature: str


@dataclass(frozen=True)
class EditOperation:
    action: str
    unit_kind: str
    source_unit_id: Optional[str]
    target_unit_id: Optional[str]
    source_canonical_id: Optional[str]
    target_canonical_id: Optional[str]
    cost: float
    reason: str


@dataclass(frozen=True)
class GraphAlignment:
    source_recipe_id: str
    target_recipe_id: str
    edits: List[EditOperation]
    source_unit_count: int
    target_unit_count: int
    preserved_unit_count: int
    total_edit_cost: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def align_recipe_graphs(source: RecipeGraph, target: RecipeGraph) -> GraphAlignment:
    source_units = graph_units(source)
    target_units = graph_units(target)
    edits: List[EditOperation] = []
    remaining_source = list(source_units)
    remaining_target = list(target_units)

    target_by_signature: Dict[Tuple[str, str], List[GraphUnit]] = {}
    for unit in remaining_target:
        target_by_signature.setdefault((unit.unit_kind, unit.signature), []).append(unit)

    matched_source_ids: set[str] = set()
    matched_target_ids: set[str] = set()
    for source_unit in source_units:
        candidates = target_by_signature.get((source_unit.unit_kind, source_unit.signature), [])
        target_unit = next((candidate for candidate in candidates if candidate.unit_id not in matched_target_ids), None)
        if target_unit is None:
            continue
        matched_source_ids.add(source_unit.unit_id)
        matched_target_ids.add(target_unit.unit_id)
        edits.append(
            EditOperation(
                action="preserve",
                unit_kind=source_unit.unit_kind,
                source_unit_id=source_unit.unit_id,
                target_unit_id=target_unit.unit_id,
                source_canonical_id=source_unit.canonical_id,
                target_canonical_id=target_unit.canonical_id,
                cost=0.0,
                reason="identical_typed_signature",
            )
        )

    remaining_source = [unit for unit in source_units if unit.unit_id not in matched_source_ids]
    remaining_target = [unit for unit in target_units if unit.unit_id not in matched_target_ids]
    for unit_kind in ("ingredient", "state", "operation", "result", "edge"):
        source_kind = [unit for unit in remaining_source if unit.unit_kind == unit_kind]
        target_kind = [unit for unit in remaining_target if unit.unit_kind == unit_kind]
        substitution_count = min(len(source_kind), len(target_kind))
        for source_unit, target_unit in zip(source_kind[:substitution_count], target_kind[:substitution_count]):
            edits.append(
                EditOperation(
                    action="substitute",
                    unit_kind=unit_kind,
                    source_unit_id=source_unit.unit_id,
                    target_unit_id=target_unit.unit_id,
                    source_canonical_id=source_unit.canonical_id,
                    target_canonical_id=target_unit.canonical_id,
                    cost=1.0,
                    reason="same_kind_different_signature",
                )
            )
            matched_source_ids.add(source_unit.unit_id)
            matched_target_ids.add(target_unit.unit_id)

    for source_unit in source_units:
        if source_unit.unit_id in matched_source_ids:
            continue
        edits.append(
            EditOperation(
                action="delete",
                unit_kind=source_unit.unit_kind,
                source_unit_id=source_unit.unit_id,
                target_unit_id=None,
                source_canonical_id=source_unit.canonical_id,
                target_canonical_id=None,
                cost=1.0,
                reason="source_unit_not_required_by_target",
            )
        )
    for target_unit in target_units:
        if target_unit.unit_id in matched_target_ids:
            continue
        edits.append(
            EditOperation(
                action="insert",
                unit_kind=target_unit.unit_kind,
                source_unit_id=None,
                target_unit_id=target_unit.unit_id,
                source_canonical_id=None,
                target_canonical_id=target_unit.canonical_id,
                cost=1.0,
                reason="target_unit_missing_from_source",
            )
        )

    alignment = GraphAlignment(
        source_recipe_id=source.recipe_id,
        target_recipe_id=target.recipe_id,
        edits=edits,
        source_unit_count=len(source_units),
        target_unit_count=len(target_units),
        preserved_unit_count=sum(edit.action == "preserve" for edit in edits),
        total_edit_cost=sum(edit.cost for edit in edits),
    )
    validate_alignment_replay(alignment, source, target)
    return alignment


def validate_alignment_replay(alignment: GraphAlignment, source: RecipeGraph, target: RecipeGraph) -> None:
    source_ids = {unit.unit_id for unit in graph_units(source)}
    target_ids = {unit.unit_id for unit in graph_units(target)}
    consumed_source = [edit.source_unit_id for edit in alignment.edits if edit.source_unit_id is not None]
    produced_target = [edit.target_unit_id for edit in alignment.edits if edit.target_unit_id is not None]
    if len(consumed_source) != len(set(consumed_source)):
        raise AlignmentValidationError("source unit consumed more than once")
    if len(produced_target) != len(set(produced_target)):
        raise AlignmentValidationError("target unit produced more than once")
    if set(consumed_source) != source_ids:
        raise AlignmentValidationError("edit script does not consume the complete source graph")
    if set(produced_target) != target_ids:
        raise AlignmentValidationError("edit script does not reconstruct the complete target graph")
    expected_cost = sum(edit.cost for edit in alignment.edits)
    if abs(expected_cost - alignment.total_edit_cost) > 1e-12:
        raise AlignmentValidationError("alignment cost is not traceable to edit operations")


def graph_units(graph: RecipeGraph) -> List[GraphUnit]:
    units = [
        GraphUnit(
            unit_id=f"node:{node.id}",
            unit_kind=node.kind,
            canonical_id=node.canonical_id,
            signature=f"{node.kind}:{node.canonical_id}",
        )
        for node in graph.nodes
    ]
    by_id = {node.id: node for node in graph.nodes}
    units.extend(
        GraphUnit(
            unit_id=f"edge:{edge.id}",
            unit_kind="edge",
            canonical_id=edge.relation,
            signature=(
                f"{edge.relation}:{by_id[edge.source].kind}:{by_id[edge.source].canonical_id}"
                f"->{by_id[edge.target].kind}:{by_id[edge.target].canonical_id}"
            ),
        )
        for edge in graph.edges
    )
    return units
