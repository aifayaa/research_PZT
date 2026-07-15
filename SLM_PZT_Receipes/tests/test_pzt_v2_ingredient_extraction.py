from __future__ import annotations

import unittest

from pzt_v2.ingredient_extraction import extract_ingredient_candidate


class PZTV2IngredientExtractionTests(unittest.TestCase):
    def test_chopped_tomatoes_preserves_quantity_unit_base_and_state(self):
        candidate = extract_ingredient_candidate("2 cups chopped tomatoes")

        self.assertEqual(candidate.quantity, "2")
        self.assertEqual(candidate.unit, "cups")
        self.assertEqual(candidate.base_object_candidate, "tomatoes")
        self.assertEqual(candidate.preparation_state_candidates, ["chopped"])
        self.assertEqual(candidate.descriptors, [])
        self.assertEqual(candidate.form_candidates, [])
        self.assertEqual(candidate.warnings, [])

    def test_onion_diced_preserves_comma_modifier(self):
        candidate = extract_ingredient_candidate("1 onion, diced")

        self.assertEqual(candidate.quantity, "1")
        self.assertIsNone(candidate.unit)
        self.assertEqual(candidate.base_object_candidate, "onion")
        self.assertEqual(candidate.preparation_state_candidates, ["diced"])
        self.assertEqual(candidate.warnings, [])

    def test_large_eggs_beaten_preserves_descriptor_and_state(self):
        candidate = extract_ingredient_candidate("3 large eggs, beaten")

        self.assertEqual(candidate.quantity, "3")
        self.assertIsNone(candidate.unit)
        self.assertEqual(candidate.base_object_candidate, "eggs")
        self.assertEqual(candidate.descriptors, ["large"])
        self.assertEqual(candidate.preparation_state_candidates, ["beaten"])
        self.assertEqual(candidate.warnings, [])

    def test_melted_butter_preserves_fraction_unit_and_state(self):
        candidate = extract_ingredient_candidate("1/2 cup melted butter")

        self.assertEqual(candidate.quantity, "1/2")
        self.assertEqual(candidate.unit, "cup")
        self.assertEqual(candidate.base_object_candidate, "butter")
        self.assertEqual(candidate.preparation_state_candidates, ["melted"])
        self.assertEqual(candidate.warnings, [])

    def test_chopped_tomato_is_not_reduced_to_plain_tomato_only(self):
        candidate = extract_ingredient_candidate("chopped tomatoes")

        self.assertEqual(candidate.base_object_candidate, "tomatoes")
        self.assertIn("chopped", candidate.preparation_state_candidates)

    def test_abbreviated_units_are_removed_from_base_object(self):
        cases = [
            ("1 tsp salt", "1", "tsp", "salt", []),
            ("2 tsp. salt", "2", "tsp.", "salt", []),
            ("1 t salt", "1", "t", "salt", []),
            ("1 c sugar", "1", "c", "sugar", []),
            ("2 c. flour", "2", "c.", "flour", []),
            ("1 cup milk", "1", "cup", "milk", []),
            ("2 cups flour", "2", "cups", "flour", []),
            ("8 oz cream cheese", "8", "oz", "cream cheese", []),
            ("1 lb ground beef", "1", "lb", "beef", ["ground"]),
        ]

        for raw, quantity, unit, base_object, descriptors in cases:
            with self.subTest(raw=raw):
                candidate = extract_ingredient_candidate(raw)

                self.assertEqual(candidate.quantity, quantity)
                self.assertEqual(candidate.unit, unit)
                self.assertEqual(candidate.base_object_candidate, base_object)
                self.assertEqual(candidate.descriptors, descriptors)
                self.assertNotIn(unit.lower().rstrip("."), candidate.base_object_candidate.split())

    def test_mixed_fraction_and_decimal_quantities_are_preserved(self):
        cases = [
            ("1/2 cup butter", "1/2", "cup", "butter"),
            ("1 1/2 cups flour", "1 1/2", "cups", "flour"),
            ("¼ tsp salt", "¼", "tsp", "salt"),
            (".5 cup milk", ".5", "cup", "milk"),
            ("0.5 cup milk", "0.5", "cup", "milk"),
        ]

        for raw, quantity, unit, base_object in cases:
            with self.subTest(raw=raw):
                candidate = extract_ingredient_candidate(raw)

                self.assertEqual(candidate.quantity, quantity)
                self.assertEqual(candidate.unit, unit)
                self.assertEqual(candidate.base_object_candidate, base_object)

    def test_unit_fix_preserves_preparation_states(self):
        cases = [
            ("1 cup chopped onion", "cup", "onion", "chopped"),
            ("2 cups shredded cheese", "cups", "cheese", "shredded"),
            ("1 can drained tomatoes", "can", "tomatoes", "drained"),
            ("1/2 cup melted butter", "cup", "butter", "melted"),
        ]

        for raw, unit, base_object, state in cases:
            with self.subTest(raw=raw):
                candidate = extract_ingredient_candidate(raw)

                self.assertEqual(candidate.unit, unit)
                self.assertEqual(candidate.base_object_candidate, base_object)
                self.assertIn(state, candidate.preparation_state_candidates)

    def test_leading_list_markers_do_not_block_quantity_and_unit_extraction(self):
        cases = [
            ("- 1/2 tsp. salt", "1/2", "tsp.", "salt"),
            ("- 1 c. sugar", "1", "c.", "sugar"),
            ("- 2 c. flour", "2", "c.", "flour"),
            ("1. 1 cup milk", "1", "cup", "milk"),
        ]

        for raw, quantity, unit, base_object in cases:
            with self.subTest(raw=raw):
                candidate = extract_ingredient_candidate(raw)

                self.assertEqual(candidate.quantity, quantity)
                self.assertEqual(candidate.unit, unit)
                self.assertEqual(candidate.base_object_candidate, base_object)

    def test_allrecipes_compact_measurement_variants_are_split(self):
        cases = [
            ("- 1-1/2 tsp. salt", "1-1/2", "tsp.", "salt"),
            ("- 3/4 tsp.salt", "3/4", "tsp.", "salt"),
            ("- 1/2 c.milk", "1/2", "c.", "milk"),
            ("- 1\\2 C. Sugar", "1\\2", "C.", "sugar"),
            ("- 2-1/2 c. flour", "2-1/2", "c.", "flour"),
        ]

        for raw, quantity, unit, base_object in cases:
            with self.subTest(raw=raw):
                candidate = extract_ingredient_candidate(raw)

                self.assertEqual(candidate.quantity, quantity)
                self.assertEqual(candidate.unit, unit)
                self.assertEqual(candidate.base_object_candidate, base_object)


if __name__ == "__main__":
    unittest.main()
