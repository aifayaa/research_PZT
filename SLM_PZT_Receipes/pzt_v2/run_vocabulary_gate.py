from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import create_manifest, write_manifest
from .vocabulary_registry import build_registry, write_registry


_MEASUREMENT_PREFIXES = ("tsp ", "tbsp ", "c ", "cup ", "cups ", "oz ", "lb ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and hard-gate the conservative PZT V2 vocabulary registry.")
    parser.add_argument("--vocabulary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input", type=Path, required=True, help="Original NDJSON used for vocabulary mining.")
    parser.add_argument("--upstream-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()

    concepts, report = build_registry(args.vocabulary_dir)
    outputs = write_registry(concepts, report, args.output_dir)
    vocabulary_report = json.loads((args.vocabulary_dir / "vocab_report.json").read_text(encoding="utf-8"))
    top_ingredients = vocabulary_report.get("top_ingredient_surfaces", [])[:100]
    polluted = [
        row["surface"]
        for row in top_ingredients
        if str(row.get("surface", "")).startswith(_MEASUREMENT_PREFIXES)
    ]
    conflicts = [
        f"canonical_conflict:{kind}:{row['surface']}"
        for kind in ("ingredient", "operation", "state")
        for row in _read_jsonl(args.vocabulary_dir / f"{kind}_vocab.jsonl")
        if row.get("metadata", {}).get("canonical_conflict")
    ]
    failures = []
    if vocabulary_report.get("parsed_invalid", 0):
        failures.append(f"parsed_invalid:{vocabulary_report['parsed_invalid']}")
    if vocabulary_report.get("extraction_success") != vocabulary_report.get("records_seen"):
        failures.append("not_all_records_extracted")
    if vocabulary_report.get("canonicalization_success") != vocabulary_report.get("records_seen"):
        failures.append("not_all_records_canonicalized")
    if report.represented_token_coverage < 1.0:
        failures.append(f"represented_token_coverage:{report.represented_token_coverage:.6f}")
    failures.extend(f"measurement_pollution:{surface}" for surface in polluted)
    failures.extend(conflicts)
    failures.extend(report.conflicts)

    inputs = [
        args.input,
        args.vocabulary_dir / "vocab_report.json",
        *(args.vocabulary_dir / f"{kind}_vocab.jsonl" for kind in ("ingredient", "operation", "state")),
    ]
    manifest = create_manifest(
        run_id=args.run_id,
        stage="vocabulary_registry",
        inputs=inputs,
        outputs=outputs,
        upstream_manifests=args.upstream_manifest,
        configuration={"singleton_policy": "conservative_no_merge", "top_pollution_check": 100},
        metrics={
            **report.to_dict(),
            "parsed_valid": vocabulary_report.get("parsed_valid"),
            "parsed_invalid": vocabulary_report.get("parsed_invalid"),
            "measurement_pollution": polluted,
        },
        thresholds={
            "represented_token_coverage": 1.0,
            "parsed_invalid": 0,
            "canonical_conflicts": 0,
            "measurement_pollution_top_100": 0,
        },
        failure_reasons=failures,
        root=Path.cwd(),
    )
    write_manifest(manifest, args.manifest)
    print(json.dumps({"verdict": manifest.verdict, "failure_reasons": manifest.failure_reasons, **report.to_dict()}, indent=2))
    if manifest.verdict != "PASS":
        raise SystemExit(2)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
