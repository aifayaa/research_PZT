from __future__ import annotations

import unittest
from pathlib import Path

from pzt_v2.parsing import RecipeParseError, parse_ndjson


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pzt_v2_raw_recipes"


class PZTV2ParsingTests(unittest.TestCase):
    def test_valid_instructions_recipe_is_accepted(self):
        recipe = parse_ndjson(FIXTURE_DIR / "valid_minimal.ndjson")[0]

        self.assertEqual(recipe.title, "Minimal Soup")
        self.assertEqual(recipe.raw_ingredients, ["2 cups chopped tomatoes", "1 onion, diced"])
        self.assertEqual(recipe.raw_instructions, ["Chop the tomatoes.", "Mix tomatoes and onion."])
        self.assertTrue(recipe.parse_audit.has_input)
        self.assertTrue(recipe.parse_audit.has_ingredients_block)
        self.assertTrue(recipe.parse_audit.has_instructions_block)
        self.assertEqual(recipe.parse_audit.ingredient_count, 2)
        self.assertEqual(recipe.parse_audit.instruction_count, 2)
        self.assertEqual(recipe.parse_audit.errors, [])

    def test_valid_directions_recipe_is_accepted(self):
        recipe = parse_ndjson(FIXTURE_DIR / "valid_directions.ndjson")[0]

        self.assertEqual(recipe.title, "Simple Batter")
        self.assertEqual(recipe.raw_instructions, ["Mix flour, eggs and milk.", "Bake the batter for 40 minutes."])
        self.assertTrue(recipe.parse_audit.has_instructions_block)

    def test_missing_ingredients_is_rejected_with_explicit_error(self):
        with self.assertRaises(RecipeParseError) as ctx:
            parse_ndjson(FIXTURE_DIR / "invalid_missing_ingredients.ndjson")

        self.assertIn("missing_ingredients_block", ctx.exception.audit.errors)
        self.assertFalse(ctx.exception.audit.has_ingredients_block)

    def test_missing_instructions_is_rejected_with_explicit_error(self):
        with self.assertRaises(RecipeParseError) as ctx:
            parse_ndjson(FIXTURE_DIR / "invalid_missing_instructions.ndjson")

        self.assertIn("missing_instructions_block", ctx.exception.audit.errors)
        self.assertFalse(ctx.exception.audit.has_instructions_block)

    def test_parser_preserves_raw_semantic_lines(self):
        recipe = parse_ndjson(FIXTURE_DIR / "ingredient_modifiers.ndjson")[0]

        self.assertIn("2 cups chopped tomatoes", recipe.raw_ingredients)
        self.assertIn("3 large eggs, beaten", recipe.raw_ingredients)
        self.assertNotIn("tomato", recipe.raw_ingredients)


if __name__ == "__main__":
    unittest.main()
