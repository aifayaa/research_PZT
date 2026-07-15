from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VocabularySurface:
    surface: str
    kind: str
    count: int
    canonical_id: Optional[str]
    known: bool
    examples: List[str] = field(default_factory=list)
    source_fields: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VocabularyMiningReport:
    input_path: str
    limit: Optional[int]
    records_seen: int
    parsed_valid: int
    parsed_invalid: int
    extraction_success: int
    canonicalization_success: int
    ingredient_surface_count: int
    operation_surface_count: int
    state_surface_count: int
    descriptor_surface_count: int
    unit_surface_count: int
    ingredient_token_total: int
    operation_token_total: int
    state_token_total: int
    known_ingredient_surface_count: int
    unknown_ingredient_surface_count: int
    known_operation_surface_count: int
    unknown_operation_surface_count: int
    known_state_surface_count: int
    unknown_state_surface_count: int
    top_ingredient_surfaces: List[VocabularySurface] = field(default_factory=list)
    top_operation_surfaces: List[VocabularySurface] = field(default_factory=list)
    top_state_surfaces: List[VocabularySurface] = field(default_factory=list)
    top_unknown_ingredient_surfaces: List[VocabularySurface] = field(default_factory=list)
    top_unknown_operation_surfaces: List[VocabularySurface] = field(default_factory=list)
    top_unknown_state_surfaces: List[VocabularySurface] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data
