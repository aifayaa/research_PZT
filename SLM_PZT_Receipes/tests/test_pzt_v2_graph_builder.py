from __future__ import annotations

import json
import unittest
from pathlib import Path

from pzt_v2.canonicalization_pipeline import canonicalize_recipe_extraction
from pzt_v2.extraction_pipeline import extract_recipe_candidates
from pzt_v2.graph_builder import build_recipe_graph
from pzt_v2.graph_schema import canonical_ids, canonical_signature, validate_recipe_graph
from pzt_v2.parsing import parse_recipe_object


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pzt_graph_cases"


def _build(recipe: dict):
    raw = (
        f"{recipe['title']}\n\nIngredients:\n"
        + "\n".join(recipe["raw_ingredients"])
        + "\n\nInstructions:\n"
        + "\n".join(recipe["raw_instructions"])
    )
    parsed = parse_recipe_object({"recipe_id": recipe["recipe_id"], "input": raw})
    canonicalized = canonicalize_recipe_extraction(extract_recipe_candidates(parsed))
    return build_recipe_graph(canonicalized)


class PZTV2GraphBuilderTests(unittest.TestCase):
    def test_builder_generates_valid_graphs_for_all_acceptance_fixtures(self):
        for path in sorted(FIXTURE_DIR.glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            for side in ("source_recipe", "target_recipe"):
                with self.subTest(case=case["case_id"], side=side):
                    graph = _build(case[side])
                    validate_recipe_graph(graph)
                    self.assertTrue(graph.nodes)
                    self.assertTrue(any(node.kind == "result" for node in graph.nodes))
                    self.assertEqual(len(graph.steps), len(case[side]["raw_instructions"]))

    def test_builder_preserves_required_fixture_concepts(self):
        for path in sorted(FIXTURE_DIR.glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            source = _build(case["source_recipe"])
            target = _build(case["target_recipe"])
            for expected in case["expected"]["must_match_units"]:
                with self.subTest(case=case["case_id"], expected=expected):
                    self.assertIn(expected, canonical_ids(source))
                    self.assertIn(expected, canonical_ids(target))

    def test_chopped_tomato_is_not_collapsed(self):
        case = json.loads((FIXTURE_DIR / "02_tomato_vs_chopped_tomato.json").read_text(encoding="utf-8"))
        source = _build(case["source_recipe"])
        target = _build(case["target_recipe"])
        self.assertIn("ING_TOMATO", canonical_ids(source))
        self.assertIn("ING_TOMATO", canonical_ids(target))
        self.assertNotIn("STATE_CHOPPED", canonical_ids(source))
        self.assertIn("STATE_CHOPPED", canonical_ids(target))

    def test_identity_recipe_builds_identical_signature(self):
        case = json.loads((FIXTURE_DIR / "05_identity_banana_bread.json").read_text(encoding="utf-8"))
        self.assertEqual(
            canonical_signature(_build(case["source_recipe"])),
            canonical_signature(_build(case["target_recipe"])),
        )


if __name__ == "__main__":
    unittest.main()
