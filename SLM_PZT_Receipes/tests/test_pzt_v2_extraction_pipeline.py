from __future__ import annotations

import unittest
from pathlib import Path

from pzt_v2.extraction_pipeline import extract_recipe_candidates
from pzt_v2.parsing import parse_ndjson


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pzt_v2_raw_recipes"


class PZTV2ExtractionPipelineTests(unittest.TestCase):
    def test_complete_recipe_produces_extraction_with_audit(self):
        parsed = parse_ndjson(FIXTURE_DIR / "ingredient_modifiers.ndjson")[0]
        extraction = extract_recipe_candidates(parsed)

        self.assertEqual(extraction.recipe_id, parsed.recipe_id)
        self.assertEqual(extraction.title, "Modifier Test")
        self.assertEqual(extraction.raw_ingredients, parsed.raw_ingredients)
        self.assertEqual(extraction.raw_instructions, parsed.raw_instructions)
        self.assertEqual(extraction.extraction_audit.ingredient_lines_total, 4)
        self.assertEqual(extraction.extraction_audit.ingredient_candidates_total, 4)
        self.assertEqual(extraction.extraction_audit.instruction_steps_total, 1)
        self.assertEqual(extraction.extraction_audit.operation_candidates_total, 1)
        self.assertEqual(extraction.extraction_audit.errors, [])

    def test_pipeline_preserves_modifier_candidates(self):
        parsed = parse_ndjson(FIXTURE_DIR / "ingredient_modifiers.ndjson")[0]
        extraction = extract_recipe_candidates(parsed)

        by_raw = {candidate.raw: candidate for candidate in extraction.ingredient_candidates}
        self.assertIn("chopped", by_raw["2 cups chopped tomatoes"].preparation_state_candidates)
        self.assertIn("diced", by_raw["1 onion, diced"].preparation_state_candidates)
        self.assertIn("beaten", by_raw["3 large eggs, beaten"].preparation_state_candidates)
        self.assertIn("melted", by_raw["1/2 cup melted butter"].preparation_state_candidates)

    def test_pipeline_collects_instruction_warnings(self):
        parsed = parse_ndjson(FIXTURE_DIR / "instruction_operations.ndjson")[0]
        extraction = extract_recipe_candidates(parsed)

        self.assertEqual(extraction.extraction_audit.instruction_steps_total, 5)
        self.assertEqual(extraction.extraction_audit.operation_candidates_total, 5)
        self.assertTrue(any("unrecognized_instruction_clause" in warning for warning in extraction.extraction_audit.warnings))

    def test_no_raw_information_lost_between_parsing_and_extraction(self):
        parsed = parse_ndjson(FIXTURE_DIR / "instruction_operations.ndjson")[0]
        extraction = extract_recipe_candidates(parsed)

        self.assertEqual(parsed.raw_ingredients, extraction.raw_ingredients)
        self.assertEqual(parsed.raw_instructions, extraction.raw_instructions)
        self.assertIn("Beat the eggs, then cook them in a pan.", extraction.raw_instructions)


if __name__ == "__main__":
    unittest.main()
