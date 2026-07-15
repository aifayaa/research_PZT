from __future__ import annotations

import unittest

from pzt_v2.canonicalize_ingredients import canonicalize_ingredient_candidate
from pzt_v2.ingredient_extraction import extract_ingredient_candidate


class PZTV2CanonicalizeIngredientsTests(unittest.TestCase):
    def test_chopped_tomatoes_preserves_base_and_state(self):
        candidate = extract_ingredient_candidate("2 cups chopped tomatoes")
        canonical = canonicalize_ingredient_candidate(candidate)

        self.assertEqual(canonical.quantity, "2")
        self.assertEqual(canonical.unit, "cups")
        self.assertEqual(canonical.canonical_ingredient_id, "ING_TOMATO")
        self.assertEqual(canonical.canonical_ingredient_label, "tomato")
        self.assertEqual([state.canonical_state_id for state in canonical.preparation_states], ["STATE_CHOPPED"])
        self.assertEqual(canonical.preparation_states[0].implied_operation_id, "OP_CHOP")

    def test_onion_diced_canonicalizes_state(self):
        canonical = canonicalize_ingredient_candidate(extract_ingredient_candidate("1 onion, diced"))

        self.assertEqual(canonical.canonical_ingredient_id, "ING_ONION")
        self.assertEqual([state.canonical_state_id for state in canonical.preparation_states], ["STATE_DICED"])

    def test_large_eggs_beaten_preserves_descriptor_and_state(self):
        canonical = canonicalize_ingredient_candidate(extract_ingredient_candidate("3 large eggs, beaten"))

        self.assertEqual(canonical.canonical_ingredient_id, "ING_EGG")
        self.assertEqual(canonical.descriptors, ["large"])
        self.assertEqual([state.canonical_state_id for state in canonical.preparation_states], ["STATE_BEATEN"])

    def test_melted_butter_canonicalizes_state(self):
        canonical = canonicalize_ingredient_candidate(extract_ingredient_candidate("1/2 cup melted butter"))

        self.assertEqual(canonical.quantity, "1/2")
        self.assertEqual(canonical.unit, "cup")
        self.assertEqual(canonical.canonical_ingredient_id, "ING_BUTTER")
        self.assertEqual([state.canonical_state_id for state in canonical.preparation_states], ["STATE_MELTED"])

    def test_chopped_tomato_is_not_collapsed_to_plain_tomato(self):
        canonical = canonicalize_ingredient_candidate(extract_ingredient_candidate("chopped tomato"))

        self.assertEqual(canonical.canonical_ingredient_id, "ING_TOMATO")
        self.assertTrue(canonical.preparation_states)
        self.assertIn("STATE_CHOPPED", [state.canonical_state_id for state in canonical.preparation_states])

    def test_unknown_ingredient_is_explicit_warning(self):
        canonical = canonicalize_ingredient_candidate(extract_ingredient_candidate("1 cup moon crystals"))

        self.assertIsNone(canonical.canonical_ingredient_id)
        self.assertIn("unknown_ingredient:moon crystals", canonical.warnings)


if __name__ == "__main__":
    unittest.main()
