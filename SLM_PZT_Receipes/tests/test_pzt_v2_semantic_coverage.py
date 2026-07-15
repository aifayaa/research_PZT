from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pzt_v2.canonicalization_pipeline import canonicalize_recipe_extraction
from pzt_v2.extraction_pipeline import extract_recipe_candidates
from pzt_v2.parsing import parse_ndjson
from pzt_v2.run_semantic_coverage import canonicalize_ndjson, run_semantic_coverage
from pzt_v2.semantic_coverage import build_semantic_coverage_report
from pzt_v2.semantic_memory import SemanticMemory


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pzt_v2_coverage_recipes"
GOLDEN = FIXTURE_DIR / "golden_coverage_recipes.ndjson"
COVERAGE = FIXTURE_DIR / "coverage_recipes.ndjson"
EXPECTED = FIXTURE_DIR / "expected_coverage_concepts.json"


class PZTV2SemanticCoverageTests(unittest.TestCase):
    def test_coverage_corpus_loads_and_parses(self):
        recipes = parse_ndjson(COVERAGE)

        self.assertGreaterEqual(len(recipes), 40)
        self.assertTrue(all(recipe.parse_audit.is_valid for recipe in recipes))

    def test_all_coverage_recipes_extract_and_canonicalize(self):
        canonicalized = canonicalize_ndjson(COVERAGE)

        self.assertGreaterEqual(len(canonicalized), 40)
        self.assertTrue(all(recipe.canonical_ingredients for recipe in canonicalized))
        self.assertTrue(all(recipe.canonical_instruction_steps for recipe in canonicalized))

    def test_semantic_coverage_report_has_consistent_counters(self):
        report = build_semantic_coverage_report(canonicalize_ndjson(COVERAGE))

        self.assertGreaterEqual(report.recipe_count, 40)
        self.assertEqual(report.ingredient_candidates_total, report.ingredient_known_total + report.ingredient_unknown_total)
        self.assertEqual(report.operation_candidates_total, report.operation_known_total + report.operation_unknown_total)
        self.assertEqual(report.state_candidates_total, report.state_known_total + report.state_unknown_total)
        self.assertGreaterEqual(report.ingredient_coverage_ratio, 0.0)
        self.assertLessEqual(report.ingredient_coverage_ratio, 1.0)
        self.assertIsInstance(report.unknown_ingredient_frequencies, dict)
        self.assertIn("Layer vegetables", report.unknown_operations)
        self.assertIsInstance(report.warnings, list)

    def test_golden_corpus_meets_pragmatic_coverage_thresholds(self):
        report = build_semantic_coverage_report(canonicalize_ndjson(GOLDEN))

        self.assertGreaterEqual(report.ingredient_coverage_ratio, 0.95)
        self.assertGreaterEqual(report.operation_coverage_ratio, 0.90)
        self.assertGreaterEqual(report.state_coverage_ratio, 0.85)

    def test_unknowns_are_explicitly_audited(self):
        from pzt_v2.extraction_schema import (
            ExtractionAudit,
            IngredientCandidate,
            InstructionStepCandidate,
            OperationCandidate,
            RecipeExtraction,
        )

        extraction = RecipeExtraction(
            recipe_id="unknown_coverage",
            title="Unknown Coverage",
            raw_ingredients=["nebula root"],
            raw_instructions=["Teleport batter."],
            ingredient_candidates=[
                IngredientCandidate(
                    raw="nebula root",
                    quantity=None,
                    unit=None,
                    base_object_candidate="nebula root",
                )
            ],
            instruction_step_candidates=[
                InstructionStepCandidate(
                    raw="Teleport batter.",
                    step_index=0,
                    operation_candidates=[
                        OperationCandidate(
                            raw_span="Teleport batter",
                            operation_lemma_candidate="teleport",
                            input_candidates=["batter"],
                        )
                    ],
                )
            ],
            extraction_audit=ExtractionAudit(1, 1, 1, 1),
        )
        report = build_semantic_coverage_report([canonicalize_recipe_extraction(extraction)])

        self.assertEqual(report.unknown_ingredients, ["nebula root"])
        self.assertEqual(report.unknown_operations, ["teleport"])
        self.assertEqual(report.ingredient_unknown_total, 1)
        self.assertEqual(report.operation_unknown_total, 1)

    def test_expected_concepts_file_is_well_formed(self):
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

        self.assertIn("ING_TOMATO", expected["ingredients"])
        self.assertIn("OP_MIX", expected["operations"])
        self.assertIn("STATE_CHOPPED", expected["states"])

    def test_chopped_tomato_keeps_base_and_state(self):
        recipe = canonicalize_ndjson(GOLDEN)[2]
        tomato = next(item for item in recipe.canonical_ingredients if item.raw == "chopped tomatoes")

        self.assertEqual(tomato.canonical_ingredient_id, "ING_TOMATO")
        self.assertIn("STATE_CHOPPED", [state.canonical_state_id for state in tomato.preparation_states])

    def test_zucchini_and_courgette_share_canonical_ingredient(self):
        memory = SemanticMemory()

        self.assertEqual(memory.lookup_ingredient("zucchini").canonical_id, "ING_ZUCCHINI")
        self.assertEqual(memory.lookup_ingredient("courgette").canonical_id, "ING_ZUCCHINI")

    def test_mix_combine_and_stir_together_share_op_mix(self):
        memory = SemanticMemory()

        self.assertEqual(memory.lookup_operation("mix").canonical_id, "OP_MIX")
        self.assertEqual(memory.lookup_operation("combine").canonical_id, "OP_MIX")
        self.assertEqual(memory.lookup_operation("stir together").canonical_id, "OP_MIX")

    def test_near_operations_remain_distinct_when_scientifically_ambiguous(self):
        memory = SemanticMemory()

        self.assertNotEqual(memory.lookup_operation("boil").canonical_id, memory.lookup_operation("simmer").canonical_id)
        self.assertNotEqual(memory.lookup_operation("bake").canonical_id, memory.lookup_operation("roast").canonical_id)
        self.assertNotEqual(memory.lookup_operation("fold").canonical_id, memory.lookup_operation("mix").canonical_id)

    def test_runner_writes_report_json_when_output_is_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "semantic_coverage.json"
            report = run_semantic_coverage(GOLDEN, output)

            self.assertTrue(output.exists())
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["recipe_count"], report.recipe_count)
            self.assertIn("ingredient_coverage_ratio", loaded)


if __name__ == "__main__":
    unittest.main()
