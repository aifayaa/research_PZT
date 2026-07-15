from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .canonicalization_pipeline import canonicalize_recipe_extraction
from .extraction_pipeline import extract_recipe_candidates
from .parsing import parse_ndjson
from .semantic_coverage import SemanticCoverageReport, build_semantic_coverage_report
from .semantic_schema import CanonicalizedRecipe


DEFAULT_INPUT = Path("tests/fixtures/pzt_v2_coverage_recipes/coverage_recipes.ndjson")


def canonicalize_ndjson(path: Path) -> List[CanonicalizedRecipe]:
    parsed = parse_ndjson(path)
    return [
        canonicalize_recipe_extraction(extract_recipe_candidates(recipe))
        for recipe in parsed
    ]


def run_semantic_coverage(input_path: Path, output_path: Path | None = None) -> SemanticCoverageReport:
    recipes = canonicalize_ndjson(input_path)
    report = build_semantic_coverage_report(recipes)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PZT V2 semantic coverage audit.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_semantic_coverage(args.input, args.output)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
