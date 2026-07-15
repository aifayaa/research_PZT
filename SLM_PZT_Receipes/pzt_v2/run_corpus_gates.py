from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .canonicalization_pipeline import canonicalize_recipe_extraction
from .evidence import create_manifest, write_manifest
from .extraction_pipeline import extract_recipe_candidates
from .parsing import RecipeParseError, parse_recipe_object


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hard parsing and extraction gates over a PZT V2 corpus.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parse_metrics: Dict[str, Any] = {
        "records_seen": 0,
        "parsed_valid": 0,
        "parsed_invalid": 0,
        "raw_preservation_failures": 0,
        "parse_errors": [],
    }
    extraction_metrics: Dict[str, Any] = {
        "recipes_extracted": 0,
        "ingredient_lines": 0,
        "ingredient_candidates": 0,
        "instruction_steps": 0,
        "operation_candidates": 0,
        "unrecognized_clauses": 0,
        "operation_order_violations": 0,
        "canonicalization_failures": 0,
        "warning_counts": {},
        "errors": [],
    }
    warning_counts: Counter[str] = Counter()

    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if args.limit is not None and parse_metrics["records_seen"] >= args.limit:
                break
            parse_metrics["records_seen"] += 1
            try:
                obj = json.loads(line)
                parsed = parse_recipe_object(obj, source_line_number=line_number)
            except Exception as exc:
                parse_metrics["parsed_invalid"] += 1
                if len(parse_metrics["parse_errors"]) < 50:
                    parse_metrics["parse_errors"].append(f"line[{line_number}]:{exc}")
                continue
            parse_metrics["parsed_valid"] += 1
            expected_raw = str(obj["input"]).replace("\r\n", "\n").replace("\r", "\n")
            if parsed.raw_input != expected_raw:
                parse_metrics["raw_preservation_failures"] += 1

            try:
                extraction = extract_recipe_candidates(parsed)
                canonicalized = canonicalize_recipe_extraction(extraction)
            except Exception as exc:
                extraction_metrics["canonicalization_failures"] += 1
                if len(extraction_metrics["errors"]) < 50:
                    extraction_metrics["errors"].append(f"line[{line_number}]:{exc}")
                continue
            extraction_metrics["recipes_extracted"] += 1
            extraction_metrics["ingredient_lines"] += extraction.extraction_audit.ingredient_lines_total
            extraction_metrics["ingredient_candidates"] += extraction.extraction_audit.ingredient_candidates_total
            extraction_metrics["instruction_steps"] += extraction.extraction_audit.instruction_steps_total
            extraction_metrics["operation_candidates"] += extraction.extraction_audit.operation_candidates_total
            extraction_metrics["unrecognized_clauses"] += sum(
                warning.startswith("unrecognized_instruction_clause:")
                for step in extraction.instruction_step_candidates
                for warning in step.warnings
            )
            extraction_metrics["operation_order_violations"] += _operation_order_violations(extraction)
            for warning in canonicalized.canonicalization_audit.warnings:
                warning_counts[_warning_kind(warning)] += 1

    extraction_metrics["warning_counts"] = dict(sorted(warning_counts.items()))
    parse_report = args.output_dir / "parsing_report.json"
    extraction_report = args.output_dir / "extraction_report.json"
    _write_json(parse_report, parse_metrics)
    _write_json(extraction_report, extraction_metrics)

    parse_failures = []
    if parse_metrics["parsed_invalid"]:
        parse_failures.append(f"parsed_invalid:{parse_metrics['parsed_invalid']}")
    if parse_metrics["parsed_valid"] != parse_metrics["records_seen"]:
        parse_failures.append("not_all_records_parsed")
    if parse_metrics["raw_preservation_failures"]:
        parse_failures.append(f"raw_preservation_failures:{parse_metrics['raw_preservation_failures']}")
    parse_manifest_path = args.output_dir / "parsing_manifest.json"
    parse_manifest = create_manifest(
        run_id=args.run_id,
        stage="parsing",
        inputs=[args.input],
        outputs=[parse_report],
        configuration={"limit": args.limit, "strict": True},
        metrics=parse_metrics,
        thresholds={"parsed_invalid": 0, "raw_preservation_failures": 0},
        failure_reasons=parse_failures,
        root=Path.cwd(),
    )
    write_manifest(parse_manifest, parse_manifest_path)

    extraction_failures = []
    if parse_manifest.verdict != "PASS":
        extraction_failures.append("parsing_gate_failed")
    if extraction_metrics["recipes_extracted"] != parse_metrics["parsed_valid"]:
        extraction_failures.append("not_all_valid_recipes_extracted")
    if extraction_metrics["ingredient_candidates"] != extraction_metrics["ingredient_lines"]:
        extraction_failures.append("ingredient_line_candidate_mismatch")
    if extraction_metrics["operation_order_violations"]:
        extraction_failures.append(
            f"operation_order_violations:{extraction_metrics['operation_order_violations']}"
        )
    if extraction_metrics["canonicalization_failures"]:
        extraction_failures.append(
            f"canonicalization_failures:{extraction_metrics['canonicalization_failures']}"
        )
    extraction_manifest = create_manifest(
        run_id=args.run_id,
        stage="extraction",
        inputs=[args.input],
        outputs=[extraction_report],
        upstream_manifests=[parse_manifest_path],
        configuration={"limit": args.limit, "ordered_multi_operation": True},
        metrics=extraction_metrics,
        thresholds={
            "recipes_extracted_ratio": 1.0,
            "ingredient_line_candidate_ratio": 1.0,
            "operation_order_violations": 0,
            "canonicalization_failures": 0,
        },
        failure_reasons=extraction_failures,
        root=Path.cwd(),
    )
    write_manifest(extraction_manifest, args.output_dir / "extraction_manifest.json")
    print(
        json.dumps(
            {
                "parsing": parse_manifest.verdict,
                "extraction": extraction_manifest.verdict,
                "parse_metrics": parse_metrics,
                "extraction_metrics": extraction_metrics,
            },
            indent=2,
        )
    )
    if parse_manifest.verdict != "PASS" or extraction_manifest.verdict != "PASS":
        raise SystemExit(2)


def _operation_order_violations(extraction) -> int:
    violations = 0
    for step in extraction.instruction_step_candidates:
        text = step.raw.lower()
        previous = -1
        search_from = 0
        for operation in step.operation_candidates:
            surface = operation.raw_span.lower()
            position = text.find(surface, search_from)
            if position < previous:
                violations += 1
            if position >= 0:
                previous = position
                search_from = position + max(1, len(surface))
    return violations


def _warning_kind(warning: str) -> str:
    tail = warning.split(":", 1)[-1]
    return tail.split(":", 1)[0]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
