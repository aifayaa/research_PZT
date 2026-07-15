from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pzt_v2.vocabulary_mining import mine_vocabulary, write_vocabulary_artifacts


FIXTURE = Path(__file__).parent / "fixtures" / "pzt_v2_vocab_mining" / "sample_vocab_recipes.ndjson"
UNIT_REGRESSION_FIXTURE = (
    Path(__file__).parent / "fixtures" / "pzt_v2_vocab_mining" / "unit_regression_recipes.ndjson"
)


class PZTV2VocabularyMiningTests(unittest.TestCase):
    def test_mining_loads_fixture_and_counts_records(self):
        result = mine_vocabulary(FIXTURE, top_k=20)
        report = result.report

        self.assertEqual(report.records_seen, 2)
        self.assertEqual(report.parsed_valid, 2)
        self.assertEqual(report.parsed_invalid, 0)
        self.assertEqual(report.extraction_success, 2)
        self.assertEqual(report.canonicalization_success, 2)

    def test_mining_collects_ingredient_state_descriptor_and_unit_surfaces(self):
        result = mine_vocabulary(FIXTURE, top_k=20)
        ingredients = {row.surface: row for row in result.ingredient_vocab}
        states = {row.surface: row for row in result.state_vocab}
        descriptors = {row.surface: row for row in result.descriptor_vocab}
        units = {row.surface: row for row in result.unit_vocab}

        self.assertIn("tomatoes", ingredients)
        self.assertEqual(ingredients["tomatoes"].canonical_id, "ING_TOMATO")
        self.assertIn("nebula root", ingredients)
        self.assertFalse(ingredients["nebula root"].known)
        self.assertIn("chopped", states)
        self.assertEqual(states["chopped"].canonical_id, "STATE_CHOPPED")
        self.assertIn("large", descriptors)
        self.assertIn("cups", units)

    def test_mining_collects_operation_surfaces_without_fusing_report_rows(self):
        result = mine_vocabulary(FIXTURE, top_k=20)
        operations = {row.surface: row for row in result.operation_vocab}

        self.assertIn("mix", operations)
        self.assertIn("combine", operations)
        self.assertIn("stir together", operations)
        self.assertEqual(operations["mix"].canonical_id, "OP_MIX")
        self.assertEqual(operations["combine"].canonical_id, "OP_MIX")
        self.assertEqual(operations["stir together"].canonical_id, "OP_MIX")

    def test_mining_collects_unknown_operations_explicitly(self):
        result = mine_vocabulary(FIXTURE, top_k=20)
        unknown_operations = {row.surface: row for row in result.unknown_operation_vocab}

        self.assertIn("teleport batter", unknown_operations)
        self.assertFalse(unknown_operations["teleport batter"].known)
        self.assertIn("Teleport batter.", unknown_operations["teleport batter"].examples)

    def test_examples_are_preserved(self):
        result = mine_vocabulary(FIXTURE, top_k=20)
        ingredients = {row.surface: row for row in result.ingredient_vocab}

        self.assertIn("2 cups chopped tomatoes", ingredients["tomatoes"].examples)
        self.assertLessEqual(len(ingredients["tomatoes"].examples), 5)

    def test_report_top_unknowns_are_well_formed(self):
        result = mine_vocabulary(FIXTURE, top_k=20)
        report = result.report

        self.assertGreaterEqual(report.ingredient_surface_count, 1)
        self.assertGreaterEqual(report.operation_surface_count, 1)
        self.assertGreaterEqual(report.state_surface_count, 1)
        self.assertEqual(report.unknown_ingredient_surface_count, 1)
        self.assertGreaterEqual(report.unknown_operation_surface_count, 1)
        self.assertEqual(report.top_unknown_ingredient_surfaces[0].surface, "nebula root")

    def test_artifact_writer_outputs_expected_json_and_jsonl_files(self):
        result = mine_vocabulary(FIXTURE, top_k=20)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "vocab"
            report_path = Path(tmp) / "report.json"
            paths = write_vocabulary_artifacts(result, out_dir, output_report=report_path)

            expected = {
                out_dir / "vocab_report.json",
                out_dir / "ingredient_vocab.jsonl",
                out_dir / "operation_vocab.jsonl",
                out_dir / "state_vocab.jsonl",
                out_dir / "descriptor_vocab.jsonl",
                out_dir / "unit_vocab.jsonl",
                out_dir / "unknown_ingredient_vocab.jsonl",
                out_dir / "unknown_operation_vocab.jsonl",
                out_dir / "unknown_state_vocab.jsonl",
                report_path,
            }
            self.assertEqual(set(paths), expected)
            self.assertTrue(all(path.exists() for path in expected))
            report = json.loads((out_dir / "vocab_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["records_seen"], 2)
            first_row = json.loads((out_dir / "ingredient_vocab.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("surface", first_row)

    def test_unit_abbreviations_do_not_pollute_ingredient_surfaces(self):
        result = mine_vocabulary(UNIT_REGRESSION_FIXTURE, top_k=20)
        ingredient_surfaces = {row.surface for row in result.ingredient_vocab}
        unit_surfaces = {row.surface for row in result.unit_vocab}

        self.assertTrue({"salt", "sugar", "flour", "milk"}.issubset(ingredient_surfaces))
        self.assertFalse({"tsp salt", "c sugar", "c flour", "cup milk"} & ingredient_surfaces)
        self.assertTrue({"tsp", "c", "cup"}.issubset(unit_surfaces))

    def test_operation_surface_never_inherits_another_verbs_canonical_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operations.ndjson"
            path.write_text(
                json.dumps(
                    {
                        "input": (
                            "Ordered operations\n\nIngredients:\nwater\n\nInstructions:\n"
                            "Cook over medium heat, stirring constantly. Pour into a pan. Bake until firm."
                        )
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = mine_vocabulary(path)
        by_surface = {row.surface: row for row in result.operation_vocab}
        self.assertEqual(by_surface["cook"].canonical_id, "OP_COOK")
        self.assertEqual(by_surface["pour"].canonical_id, "OP_POUR")
        self.assertNotIn("canonical_conflict", by_surface["cook"].metadata)
        self.assertNotIn("canonical_conflict", by_surface["pour"].metadata)

    def test_contextual_cook_in_pan_does_not_conflict_with_generic_cook(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cook.ndjson"
            path.write_text(
                json.dumps(
                    {
                        "input": (
                            "Cook contexts\n\nIngredients:\negg\n\nInstructions:\n"
                            "Cook egg. Cook egg in a pan. Crush cookies into a pan."
                        )
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = mine_vocabulary(path)
        by_surface = {row.surface: row for row in result.operation_vocab}
        self.assertEqual(by_surface["cook"].canonical_id, "OP_COOK")
        self.assertEqual(by_surface["cook in pan"].canonical_id, "OP_COOK_IN_PAN")
        self.assertNotIn("canonical_conflict", by_surface["cook"].metadata)
        self.assertNotIn("canonical_conflict", by_surface["cook in pan"].metadata)


if __name__ == "__main__":
    unittest.main()
