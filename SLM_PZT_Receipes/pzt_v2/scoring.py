from __future__ import annotations

from dataclasses import asdict, dataclass

from .alignment import GraphAlignment


@dataclass(frozen=True)
class TransferabilityScore:
    source_recipe_id: str
    target_recipe_id: str
    structural_conservation: float
    normalized_edit_distance: float
    raw_transferability: float
    preserved_unit_count: int
    target_unit_count: int
    total_edit_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PZTScore:
    source_recipe_id: str
    target_recipe_id: str
    raw_transferability: float
    novelty: float
    pzt_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def score_alignment(alignment: GraphAlignment) -> tuple[TransferabilityScore, PZTScore]:
    target_count = alignment.target_unit_count
    conservation = 1.0 if target_count == 0 else alignment.preserved_unit_count / target_count
    total_units = alignment.source_unit_count + alignment.target_unit_count
    normalized_distance = 0.0 if total_units == 0 else min(1.0, alignment.total_edit_cost / total_units)
    raw = _clamp01(conservation * (1.0 - normalized_distance))
    novelty = normalized_distance
    useful = _clamp01(raw * novelty)
    transferability = TransferabilityScore(
        source_recipe_id=alignment.source_recipe_id,
        target_recipe_id=alignment.target_recipe_id,
        structural_conservation=conservation,
        normalized_edit_distance=normalized_distance,
        raw_transferability=raw,
        preserved_unit_count=alignment.preserved_unit_count,
        target_unit_count=target_count,
        total_edit_cost=alignment.total_edit_cost,
    )
    pzt = PZTScore(
        source_recipe_id=alignment.source_recipe_id,
        target_recipe_id=alignment.target_recipe_id,
        raw_transferability=raw,
        novelty=novelty,
        pzt_score=useful,
    )
    return transferability, pzt


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
