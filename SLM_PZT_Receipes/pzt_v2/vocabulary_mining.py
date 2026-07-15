from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .canonicalization_pipeline import canonicalize_recipe_extraction
from .extraction_pipeline import extract_recipe_candidates
from .extraction_schema import OperationCandidate
from .parsing import RecipeParseError, parse_recipe_object
from .semantic_schema import CanonicalizedRecipe
from .vocabulary_schema import VocabularyMiningReport, VocabularySurface


MAX_EXAMPLES_PER_SURFACE = 5


@dataclass
class _SurfaceBucket:
    surface: str
    kind: str
    count: int = 0
    canonical_id: Optional[str] = None
    known: bool = False
    examples: List[str] = field(default_factory=list)
    source_fields: set[str] = field(default_factory=set)
    metadata: Dict[str, object] = field(default_factory=dict)
    canonical_ids_seen: set[str] = field(default_factory=set)

    def add(
        self,
        *,
        canonical_id: Optional[str],
        known: bool,
        example: Optional[str],
        source_field: str,
        metadata: Optional[dict] = None,
    ) -> None:
        self.count += 1
        self.known = self.known or known
        if canonical_id is not None:
            self.canonical_ids_seen.add(canonical_id)
        if self.canonical_id is None and canonical_id is not None:
            self.canonical_id = canonical_id
        if example and len(self.examples) < MAX_EXAMPLES_PER_SURFACE and example not in self.examples:
            self.examples.append(example)
        self.source_fields.add(source_field)
        if metadata:
            self.metadata.update(metadata)
        if self.canonical_ids_seen:
            self.metadata["canonical_ids_seen"] = sorted(self.canonical_ids_seen)
        if len(self.canonical_ids_seen) > 1:
            self.metadata["canonical_conflict"] = True

    def to_surface(self) -> VocabularySurface:
        return VocabularySurface(
            surface=self.surface,
            kind=self.kind,
            count=self.count,
            canonical_id=self.canonical_id,
            known=self.known,
            examples=list(self.examples),
            source_fields=sorted(self.source_fields),
            metadata=dict(self.metadata),
        )


@dataclass
class VocabularyMiningResult:
    report: VocabularyMiningReport
    ingredient_vocab: List[VocabularySurface]
    operation_vocab: List[VocabularySurface]
    state_vocab: List[VocabularySurface]
    descriptor_vocab: List[VocabularySurface]
    unit_vocab: List[VocabularySurface]
    unknown_ingredient_vocab: List[VocabularySurface]
    unknown_operation_vocab: List[VocabularySurface]
    unknown_state_vocab: List[VocabularySurface]


class _VocabularyCollector:
    def __init__(self) -> None:
        self.buckets: Dict[Tuple[str, str], _SurfaceBucket] = {}

    def add(
        self,
        *,
        kind: str,
        surface: Optional[str],
        canonical_id: Optional[str],
        known: bool,
        example: Optional[str],
        source_field: str,
        metadata: Optional[dict] = None,
    ) -> None:
        normalized = _normalize_surface(surface)
        if not normalized:
            return
        key = (kind, normalized)
        if key not in self.buckets:
            self.buckets[key] = _SurfaceBucket(surface=normalized, kind=kind)
        self.buckets[key].add(
            canonical_id=canonical_id,
            known=known,
            example=example,
            source_field=source_field,
            metadata=metadata,
        )

    def surfaces(self, kind: str, *, known: Optional[bool] = None) -> List[VocabularySurface]:
        rows = [bucket.to_surface() for (bucket_kind, _), bucket in self.buckets.items() if bucket_kind == kind]
        if known is not None:
            rows = [row for row in rows if row.known is known]
        return sorted(rows, key=lambda row: (-row.count, row.surface))


def mine_vocabulary(
    input_path: Path,
    *,
    limit: Optional[int] = None,
    skip_invalid: bool = False,
    top_k: int = 200,
) -> VocabularyMiningResult:
    collector = _VocabularyCollector()
    records_seen = 0
    parsed_valid = 0
    parsed_invalid = 0
    extraction_success = 0
    canonicalization_success = 0
    warnings: List[str] = []
    errors: List[str] = []

    with input_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if limit is not None and records_seen >= limit:
                break
            if not line.strip():
                continue
            records_seen += 1
            try:
                parsed = parse_recipe_object(json.loads(line), source_line_number=line_number)
                parsed_valid += 1
            except Exception as exc:
                parsed_invalid += 1
                message = f"line[{line_number}]:parse_error:{exc}"
                errors.append(message)
                if skip_invalid:
                    continue
                if isinstance(exc, RecipeParseError):
                    continue
                continue

            try:
                extraction = extract_recipe_candidates(parsed)
                extraction_success += 1
            except Exception as exc:
                errors.append(f"line[{line_number}]:extraction_error:{exc}")
                if skip_invalid:
                    continue
                continue

            try:
                canonicalized = canonicalize_recipe_extraction(extraction)
                canonicalization_success += 1
            except Exception as exc:
                errors.append(f"line[{line_number}]:canonicalization_error:{exc}")
                if skip_invalid:
                    continue
                continue

            _collect_recipe(collector, canonicalized)
            warnings.extend(f"line[{line_number}]:{warning}" for warning in canonicalized.canonicalization_audit.warnings)

    ingredient_vocab = collector.surfaces("ingredient")
    operation_vocab = collector.surfaces("operation")
    state_vocab = collector.surfaces("state")
    descriptor_vocab = collector.surfaces("descriptor")
    unit_vocab = collector.surfaces("unit")
    unknown_ingredient_vocab = collector.surfaces("ingredient", known=False)
    unknown_operation_vocab = collector.surfaces("operation", known=False)
    unknown_state_vocab = collector.surfaces("state", known=False)

    report = VocabularyMiningReport(
        input_path=str(input_path),
        limit=limit,
        records_seen=records_seen,
        parsed_valid=parsed_valid,
        parsed_invalid=parsed_invalid,
        extraction_success=extraction_success,
        canonicalization_success=canonicalization_success,
        ingredient_surface_count=len(ingredient_vocab),
        operation_surface_count=len(operation_vocab),
        state_surface_count=len(state_vocab),
        descriptor_surface_count=len(descriptor_vocab),
        unit_surface_count=len(unit_vocab),
        ingredient_token_total=sum(row.count for row in ingredient_vocab),
        operation_token_total=sum(row.count for row in operation_vocab),
        state_token_total=sum(row.count for row in state_vocab),
        known_ingredient_surface_count=len([row for row in ingredient_vocab if row.known]),
        unknown_ingredient_surface_count=len(unknown_ingredient_vocab),
        known_operation_surface_count=len([row for row in operation_vocab if row.known]),
        unknown_operation_surface_count=len(unknown_operation_vocab),
        known_state_surface_count=len([row for row in state_vocab if row.known]),
        unknown_state_surface_count=len(unknown_state_vocab),
        top_ingredient_surfaces=ingredient_vocab[:top_k],
        top_operation_surfaces=operation_vocab[:top_k],
        top_state_surfaces=state_vocab[:top_k],
        top_unknown_ingredient_surfaces=unknown_ingredient_vocab[:top_k],
        top_unknown_operation_surfaces=unknown_operation_vocab[:top_k],
        top_unknown_state_surfaces=unknown_state_vocab[:top_k],
        warnings=warnings[:top_k],
        errors=errors[:top_k],
    )
    return VocabularyMiningResult(
        report=report,
        ingredient_vocab=ingredient_vocab,
        operation_vocab=operation_vocab,
        state_vocab=state_vocab,
        descriptor_vocab=descriptor_vocab,
        unit_vocab=unit_vocab,
        unknown_ingredient_vocab=unknown_ingredient_vocab,
        unknown_operation_vocab=unknown_operation_vocab,
        unknown_state_vocab=unknown_state_vocab,
    )


def write_vocabulary_artifacts(result: VocabularyMiningResult, output_dir: Path, *, output_report: Optional[Path] = None) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _write_json(output_dir / "vocab_report.json", result.report.to_dict()),
        _write_jsonl(output_dir / "ingredient_vocab.jsonl", result.ingredient_vocab),
        _write_jsonl(output_dir / "operation_vocab.jsonl", result.operation_vocab),
        _write_jsonl(output_dir / "state_vocab.jsonl", result.state_vocab),
        _write_jsonl(output_dir / "descriptor_vocab.jsonl", result.descriptor_vocab),
        _write_jsonl(output_dir / "unit_vocab.jsonl", result.unit_vocab),
        _write_jsonl(output_dir / "unknown_ingredient_vocab.jsonl", result.unknown_ingredient_vocab),
        _write_jsonl(output_dir / "unknown_operation_vocab.jsonl", result.unknown_operation_vocab),
        _write_jsonl(output_dir / "unknown_state_vocab.jsonl", result.unknown_state_vocab),
    ]
    if output_report is not None:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        paths.append(_write_json(output_report, result.report.to_dict()))
    return paths


def _collect_recipe(collector: _VocabularyCollector, recipe: CanonicalizedRecipe) -> None:
    for ingredient in recipe.canonical_ingredients:
        collector.add(
            kind="ingredient",
            surface=ingredient.base_object_raw,
            canonical_id=ingredient.canonical_ingredient_id,
            known=ingredient.canonical_ingredient_id is not None,
            example=ingredient.raw,
            source_field="base_object_candidate",
        )
        for descriptor in ingredient.descriptors:
            collector.add(
                kind="descriptor",
                surface=descriptor,
                canonical_id=None,
                known=True,
                example=ingredient.raw,
                source_field="descriptor",
            )
        if ingredient.unit:
            collector.add(
                kind="unit",
                surface=ingredient.unit,
                canonical_id=None,
                known=True,
                example=ingredient.raw,
                source_field="unit",
            )
        for state in ingredient.preparation_states:
            collector.add(
                kind="state",
                surface=state.raw,
                canonical_id=state.canonical_state_id,
                known=True,
                example=ingredient.raw,
                source_field="preparation_state_candidate",
                metadata={"implied_operation_id": state.implied_operation_id},
            )
        for warning in ingredient.warnings:
            if warning.startswith("unknown_state:"):
                collector.add(
                    kind="state",
                    surface=warning.split(":", 1)[1],
                    canonical_id=None,
                    known=False,
                    example=ingredient.raw,
                    source_field="preparation_state_candidate",
                )

    for step in recipe.canonical_instruction_steps:
        for operation in step.canonical_operations:
            raw_surface = _operation_surface(operation.raw_span, operation.operation_lemma_candidate)
            for surface, source_field in [
                (operation.operation_lemma_candidate, "operation_lemma_candidate"),
                (raw_surface, "raw_span_operation_surface"),
            ]:
                collector.add(
                    kind="operation",
                    surface=surface,
                    canonical_id=operation.canonical_operation_id,
                    known=operation.canonical_operation_id is not None,
                    example=operation.raw_span,
                    source_field=source_field,
                    metadata={"output_state_candidate": operation.output_state_candidate},
                )
        for warning in step.warnings:
            if "unrecognized_instruction_clause:" in warning:
                collector.add(
                    kind="operation",
                    surface=warning.split("unrecognized_instruction_clause:", 1)[1],
                    canonical_id=None,
                    known=False,
                    example=step.raw,
                    source_field="unrecognized_instruction_clause",
                )


def _operation_surface(raw_span: str, lemma: str) -> str:
    raw = raw_span.lower()
    if "stir" in raw and "together" in raw:
        return "stir together"
    if "cook" in raw and "pan" in raw:
        return "cook in pan"
    if "stir-fry" in raw or "stir fry" in raw:
        return "stir-fry"
    if "sauté" in raw:
        return "sauté"
    match = re.match(r"\s*([a-zA-Z-]+)", raw_span)
    return match.group(1).lower() if match else lemma


def _normalize_surface(surface: Optional[str]) -> str:
    if not surface:
        return ""
    normalized = surface.strip().lower().replace("-", " ")
    normalized = re.sub(r"[^a-z0-9/\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: Iterable[VocabularySurface]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    return path
