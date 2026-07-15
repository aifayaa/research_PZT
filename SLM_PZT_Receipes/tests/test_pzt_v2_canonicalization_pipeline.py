from __future__ import annotations

import unittest
from pathlib import Path

from pzt_v2.canonicalization_pipeline import canonicalize_recipe_extraction
from pzt_v2.extraction_pipeline import extract_recipe_candidates
from pzt_v2.parsing import parse_ndjson


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pzt_v2_raw_recipes"


class PZTV2CanonicalizationPipelineTests(unittest.TestCase):
    def test_pipeline_canonicalizes_recipe_extraction(self):
        parsed = parse_ndjson(FIXTURE_DIR / "ingredient_modifiers.ndjson")[0]
        extraction = extract_recipe_candidates(parsed)
        canonicalized = canonicalize_recipe_extraction(extraction)

        self.assertEqual(canonicalized.recipe_id, extraction.recipe_id)
        self.assertEqual(canonicalized.raw_ingredients, extraction.raw_ingredients)
        self.assertEqual(canonicalized.raw_instructions, extraction.raw_instructions)
        self.assertEqual(canonicalized.canonicalization_audit.ingredient_candidates_total, 4)
        self.assertEqual(canonicalized.canonicalization_audit.canonical_ingredients_total, 4)
        self.assertEqual(canonicalized.canonicalization_audit.operation_candidates_total, 1)
        self.assertEqual(canonicalized.canonicalization_audit.canonical_operations_total, 1)
        self.assertEqual(canonicalized.canonicalization_audit.errors, [])

    def test_pipeline_preserves_chopped_tomato_state(self):
        parsed = parse_ndjson(FIXTURE_DIR / "valid_minimal.ndjson")[0]
        extraction = extract_recipe_candidates(parsed)
        canonicalized = canonicalize_recipe_extraction(extraction)

        tomato = canonicalized.canonical_ingredients[0]
        self.assertEqual(tomato.raw, "2 cups chopped tomatoes")
        self.assertEqual(tomato.canonical_ingredient_id, "ING_TOMATO")
        self.assertEqual([state.canonical_state_id for state in tomato.preparation_states], ["STATE_CHOPPED"])

    def test_pipeline_collects_unknowns(self):
        from pzt_v2.extraction_schema import (
            ExtractionAudit,
            IngredientCandidate,
            OperationCandidate,
            InstructionStepCandidate,
            RecipeExtraction,
        )

        extraction = RecipeExtraction(
            recipe_id="unknown_case",
            title="Unknown Case",
            raw_ingredients=["moon crystals"],
            raw_instructions=["Teleport the batter."],
            ingredient_candidates=[
                IngredientCandidate(
                    raw="moon crystals",
                    quantity=None,
                    unit=None,
                    base_object_candidate="moon crystals",
                )
            ],
            instruction_step_candidates=[
                InstructionStepCandidate(
                    raw="Teleport the batter.",
                    step_index=0,
                    operation_candidates=[
                        OperationCandidate(
                            raw_span="Teleport the batter",
                            operation_lemma_candidate="teleport",
                            input_candidates=["batter"],
                        )
                    ],
                )
            ],
            extraction_audit=ExtractionAudit(1, 1, 1, 1),
        )

        canonicalized = canonicalize_recipe_extraction(extraction)
        audit = canonicalized.canonicalization_audit
        self.assertEqual(audit.unknown_ingredients, ["moon crystals"])
        self.assertEqual(audit.unknown_operations, ["teleport"])
        self.assertTrue(any("unknown_ingredient:moon crystals" in warning for warning in audit.warnings))
        self.assertTrue(any("unknown_operation:teleport" in warning for warning in audit.warnings))


if __name__ == "__main__":
    unittest.main()
