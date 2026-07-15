from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


REGISTRY_SCHEMA_VERSION = "pzt-vocabulary/v1"
KINDS = ("ingredient", "operation", "state")


class VocabularyRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class RegistryConcept:
    canonical_id: str
    kind: str
    label: str
    aliases: List[str]
    status: str
    token_count: int
    examples: List[str] = field(default_factory=list)
    source_fields: List[str] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RegistryReport:
    schema_version: str
    concept_count: int
    alias_count: int
    token_total: int
    represented_token_total: int
    represented_token_coverage: float
    validated_alias_token_total: int
    validated_alias_token_coverage: float
    provisional_singleton_token_total: int
    by_kind: Dict[str, dict]
    conflicts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_registry(vocabulary_dir: Path) -> tuple[List[RegistryConcept], RegistryReport]:
    rows_by_kind = {kind: _load_jsonl(vocabulary_dir / f"{kind}_vocab.jsonl") for kind in KINDS}
    concepts: Dict[Tuple[str, str], dict] = {}
    alias_owner: Dict[Tuple[str, str], str] = {}
    conflicts: List[str] = []
    by_kind: Dict[str, dict] = {}
    token_total = 0
    validated_total = 0
    provisional_total = 0

    for kind, rows in rows_by_kind.items():
        kind_total = sum(int(row["count"]) for row in rows)
        kind_validated = 0
        kind_provisional = 0
        for row in rows:
            surface = normalize_surface(str(row["surface"]))
            count = int(row["count"])
            known = bool(row.get("known")) and bool(row.get("canonical_id"))
            canonical_id = str(row["canonical_id"]) if known else singleton_id(kind, surface)
            status = "validated" if known else "provisional_singleton"
            owner_key = (kind, surface)
            previous_owner = alias_owner.get(owner_key)
            if previous_owner is not None and previous_owner != canonical_id:
                conflicts.append(f"alias_conflict:{kind}:{surface}:{previous_owner}:{canonical_id}")
            alias_owner[owner_key] = canonical_id
            key = (kind, canonical_id)
            bucket = concepts.setdefault(
                key,
                {
                    "canonical_id": canonical_id,
                    "kind": kind,
                    "label": surface,
                    "aliases": set(),
                    "status": status,
                    "token_count": 0,
                    "examples": [],
                    "source_fields": set(),
                    "provenance": {"source": "20k_vocabulary_mining"},
                },
            )
            if bucket["status"] != status:
                conflicts.append(f"status_conflict:{kind}:{canonical_id}")
            bucket["aliases"].add(surface)
            bucket["token_count"] += count
            for example in row.get("examples", []):
                if example not in bucket["examples"] and len(bucket["examples"]) < 5:
                    bucket["examples"].append(str(example))
            bucket["source_fields"].update(str(item) for item in row.get("source_fields", []))
            if known:
                kind_validated += count
            else:
                kind_provisional += count

        token_total += kind_total
        validated_total += kind_validated
        provisional_total += kind_provisional
        by_kind[kind] = {
            "surface_count": len(rows),
            "token_total": kind_total,
            "represented_token_total": kind_total,
            "represented_token_coverage": 1.0 if kind_total else 1.0,
            "validated_alias_token_total": kind_validated,
            "validated_alias_token_coverage": _ratio(kind_validated, kind_total),
            "provisional_singleton_token_total": kind_provisional,
        }

    registry = [
        RegistryConcept(
            canonical_id=bucket["canonical_id"],
            kind=bucket["kind"],
            label=bucket["label"],
            aliases=sorted(bucket["aliases"]),
            status=bucket["status"],
            token_count=bucket["token_count"],
            examples=list(bucket["examples"]),
            source_fields=sorted(bucket["source_fields"]),
            provenance=dict(bucket["provenance"]),
        )
        for bucket in concepts.values()
    ]
    registry.sort(key=lambda item: (item.kind, item.canonical_id))
    report = RegistryReport(
        schema_version=REGISTRY_SCHEMA_VERSION,
        concept_count=len(registry),
        alias_count=len(alias_owner),
        token_total=token_total,
        represented_token_total=token_total,
        represented_token_coverage=1.0 if token_total else 1.0,
        validated_alias_token_total=validated_total,
        validated_alias_token_coverage=_ratio(validated_total, token_total),
        provisional_singleton_token_total=provisional_total,
        by_kind=by_kind,
        conflicts=sorted(set(conflicts)),
    )
    validate_registry(registry)
    return registry, report


def validate_registry(concepts: Sequence[RegistryConcept]) -> None:
    owners: Dict[Tuple[str, str], str] = {}
    ids: set[Tuple[str, str]] = set()
    for concept in concepts:
        key = (concept.kind, concept.canonical_id)
        if key in ids:
            raise VocabularyRegistryError(f"duplicate concept id: {key}")
        ids.add(key)
        if concept.kind not in KINDS:
            raise VocabularyRegistryError(f"invalid concept kind: {concept.kind}")
        if concept.status not in {"validated", "provisional_singleton"}:
            raise VocabularyRegistryError(f"invalid concept status: {concept.status}")
        for alias in concept.aliases:
            alias_key = (concept.kind, normalize_surface(alias))
            previous = owners.get(alias_key)
            if previous is not None and previous != concept.canonical_id:
                raise VocabularyRegistryError(f"alias maps to multiple concepts: {alias_key}")
            owners[alias_key] = concept.canonical_id


def write_registry(concepts: Sequence[RegistryConcept], report: RegistryReport, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "concept_registry.jsonl"
    with registry_path.open("w", encoding="utf-8") as handle:
        for concept in concepts:
            handle.write(json.dumps(concept.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    report_path = output_dir / "registry_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [registry_path, report_path]


def load_registry(path: Path) -> List[RegistryConcept]:
    concepts = [RegistryConcept(**row) for row in _load_jsonl(path)]
    validate_registry(concepts)
    return concepts


def singleton_id(kind: str, surface: str) -> str:
    prefix = {"ingredient": "ING", "operation": "OP", "state": "STATE"}[kind]
    slug = re.sub(r"[^A-Z0-9]+", "_", surface.upper()).strip("_")[:40] or "EMPTY"
    digest = hashlib.sha1(f"{kind}:{surface}".encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}_RAW_{slug}_{digest}"


def normalize_surface(value: str) -> str:
    normalized = value.strip().lower().replace("-", " ")
    normalized = re.sub(r"[^a-z0-9/\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise VocabularyRegistryError(f"missing vocabulary artifact: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator
