from __future__ import annotations

import unittest
from pathlib import Path

from pzt_v2.graph_schema import (
    canonical_ids,
    canonical_signature,
    load_fixture_case,
    validate_fixture_case,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pzt_graph_cases"


def load_cases():
    return [load_fixture_case(path) for path in sorted(FIXTURE_DIR.glob("*.json"))]


class PZTGraphFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.by_id = {case.case_id: case for case in cls.cases}

    def test_all_expected_fixtures_exist(self):
        self.assertEqual(
            set(self.by_id),
            {
                "01_lexical_variation_mix_combine",
                "02_tomato_vs_chopped_tomato",
                "03_chicken_stir_fry_vs_tofu_stir_fry",
                "04_boiled_egg_vs_scrambled_eggs",
                "05_identity_banana_bread",
                "06_banana_bread_vs_zucchini_bread",
            },
        )

    def test_all_fixtures_validate_against_schema(self):
        for case in self.cases:
            with self.subTest(case=case.case_id):
                validate_fixture_case(case)

    def test_node_ids_are_unique_and_edge_endpoints_exist(self):
        for case in self.cases:
            for graph in [case.source_recipe.graph, case.target_recipe.graph]:
                with self.subTest(case=case.case_id, graph=graph.recipe_id):
                    node_ids = [node.id for node in graph.nodes]
                    self.assertEqual(len(node_ids), len(set(node_ids)))
                    known_nodes = set(node_ids)
                    for edge in graph.edges:
                        self.assertIn(edge.source, known_nodes)
                        self.assertIn(edge.target, known_nodes)

    def test_all_nodes_have_canonical_ids(self):
        for case in self.cases:
            for graph in [case.source_recipe.graph, case.target_recipe.graph]:
                with self.subTest(case=case.case_id, graph=graph.recipe_id):
                    for node in graph.nodes:
                        self.assertTrue(node.canonical_id)

    def test_expected_must_match_units_exist_in_both_graphs(self):
        for case in self.cases:
            source_units = canonical_ids(case.source_recipe.graph)
            target_units = canonical_ids(case.target_recipe.graph)
            for unit in case.expected.must_match_units:
                with self.subTest(case=case.case_id, unit=unit):
                    self.assertIn(unit, source_units)
                    self.assertIn(unit, target_units)

    def test_expected_must_not_match_units_are_not_collapsed(self):
        for case in self.cases:
            all_units = canonical_ids(case.source_recipe.graph) | canonical_ids(case.target_recipe.graph)
            for left, right in case.expected.must_not_match_units:
                with self.subTest(case=case.case_id, left=left, right=right):
                    self.assertNotEqual(left, right)
                    self.assertIn(left, all_units)
                    self.assertIn(right, all_units)

    def test_lexical_variation_maps_stir_and_combine_to_op_mix(self):
        case = self.by_id["01_lexical_variation_mix_combine"]
        source_ops = [node for node in case.source_recipe.graph.nodes if node.kind == "operation"]
        target_ops = [node for node in case.target_recipe.graph.nodes if node.kind == "operation"]

        self.assertEqual(source_ops[0].label, "stir together")
        self.assertEqual(target_ops[0].label, "combine")
        self.assertEqual(source_ops[0].canonical_id, "OP_MIX")
        self.assertEqual(target_ops[0].canonical_id, "OP_MIX")

    def test_tomato_vs_chopped_tomato_preserves_preparation_state(self):
        case = self.by_id["02_tomato_vs_chopped_tomato"]
        source_units = canonical_ids(case.source_recipe.graph)
        target_units = canonical_ids(case.target_recipe.graph)

        self.assertIn("ING_TOMATO", source_units)
        self.assertIn("ING_TOMATO", target_units)
        self.assertIn("STATE_CHOPPED_TOMATO", target_units)
        self.assertNotIn("STATE_CHOPPED_TOMATO", source_units)

        state_nodes = [
            node for node in case.target_recipe.graph.nodes
            if node.canonical_id == "STATE_CHOPPED_TOMATO"
        ]
        self.assertEqual(len(state_nodes), 1)
        state = state_nodes[0]
        self.assertEqual(state.source, "ingredient_modifier")
        self.assertEqual(state.metadata.get("base_object"), "ING_TOMATO")
        self.assertEqual(state.metadata.get("preparation_state"), "STATE_CHOPPED")
        self.assertEqual(state.metadata.get("implied_operation"), "OP_CHOP")
        self.assertIs(state.metadata.get("explicit_in_instruction"), False)

    def test_chicken_to_tofu_preserves_structure_but_not_protein_identity(self):
        case = self.by_id["03_chicken_stir_fry_vs_tofu_stir_fry"]
        source_units = canonical_ids(case.source_recipe.graph)
        target_units = canonical_ids(case.target_recipe.graph)

        for unit in ["OP_CUT", "OP_STIR_FRY", "ING_VEGETABLE", "ING_SAUCE"]:
            self.assertIn(unit, source_units)
            self.assertIn(unit, target_units)
        self.assertIn("ING_CHICKEN", source_units)
        self.assertIn("ING_TOFU", target_units)
        self.assertNotEqual("ING_CHICKEN", "ING_TOFU")

    def test_boiled_egg_vs_scrambled_eggs_preserves_egg_but_not_operations(self):
        case = self.by_id["04_boiled_egg_vs_scrambled_eggs"]
        source_units = canonical_ids(case.source_recipe.graph)
        target_units = canonical_ids(case.target_recipe.graph)

        self.assertIn("ING_EGG", source_units)
        self.assertIn("ING_EGG", target_units)
        self.assertIn("OP_BOIL", source_units)
        self.assertIn("OP_BEAT", target_units)
        self.assertIn("OP_COOK_IN_PAN", target_units)
        self.assertNotIn("OP_BOIL", target_units)

    def test_identity_case_has_structurally_identical_graphs(self):
        case = self.by_id["05_identity_banana_bread"]
        self.assertEqual(
            canonical_signature(case.source_recipe.graph),
            canonical_signature(case.target_recipe.graph),
        )

    def test_banana_to_zucchini_bread_preserves_quick_bread_structure(self):
        case = self.by_id["06_banana_bread_vs_zucchini_bread"]
        source_units = canonical_ids(case.source_recipe.graph)
        target_units = canonical_ids(case.target_recipe.graph)

        for unit in ["ING_FLOUR", "ING_EGG", "ING_SUGAR", "OP_MIX", "OP_BAKE", "STATE_BATTER", "RESULT_QUICK_BREAD"]:
            self.assertIn(unit, source_units)
            self.assertIn(unit, target_units)
        self.assertIn("ING_BANANA", source_units)
        self.assertIn("ING_ZUCCHINI", target_units)
        self.assertIn("OP_MASH", source_units)
        self.assertIn("OP_GRATE", target_units)


if __name__ == "__main__":
    unittest.main()
