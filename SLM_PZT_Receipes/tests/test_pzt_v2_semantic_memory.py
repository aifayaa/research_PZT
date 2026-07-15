from __future__ import annotations

import unittest

from pzt_v2.semantic_memory import SemanticMemory


class PZTV2SemanticMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = SemanticMemory()

    def test_ingredient_variants_map_to_same_concepts(self):
        self.assertEqual(self.memory.lookup_ingredient("tomato").canonical_id, "ING_TOMATO")
        self.assertEqual(self.memory.lookup_ingredient("tomatoes").canonical_id, "ING_TOMATO")
        self.assertEqual(self.memory.lookup_ingredient("zucchini").canonical_id, "ING_ZUCCHINI")
        self.assertEqual(self.memory.lookup_ingredient("courgette").canonical_id, "ING_ZUCCHINI")

    def test_operation_synonyms_map_to_same_concepts(self):
        self.assertEqual(self.memory.lookup_operation("mix").canonical_id, "OP_MIX")
        self.assertEqual(self.memory.lookup_operation("combine").canonical_id, "OP_MIX")
        self.assertEqual(self.memory.lookup_operation("stir together").canonical_id, "OP_MIX")

    def test_states_map_to_state_concepts(self):
        self.assertEqual(self.memory.lookup_state("chopped").canonical_id, "STATE_CHOPPED")
        self.assertEqual(self.memory.lookup_state("melted").canonical_id, "STATE_MELTED")
        self.assertEqual(self.memory.lookup_state("chopped").metadata["implied_operation_id"], "OP_CHOP")

    def test_unknown_returns_none(self):
        self.assertIsNone(self.memory.lookup_ingredient("dragonfruit dust"))
        self.assertIsNone(self.memory.lookup_operation("teleport"))
        self.assertIsNone(self.memory.lookup_state("invisible"))


if __name__ == "__main__":
    unittest.main()
