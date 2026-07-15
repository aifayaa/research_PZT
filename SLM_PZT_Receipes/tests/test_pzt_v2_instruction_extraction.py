from __future__ import annotations

import unittest

from pzt_v2.instruction_extraction import extract_instruction_step


class PZTV2InstructionExtractionTests(unittest.TestCase):
    def test_chop_tomatoes_extracts_operation_input_and_output(self):
        step = extract_instruction_step("Chop the tomatoes.", step_index=0)

        self.assertEqual(len(step.operation_candidates), 1)
        op = step.operation_candidates[0]
        self.assertEqual(op.raw_span, "Chop the tomatoes")
        self.assertEqual(op.operation_lemma_candidate, "chop")
        self.assertEqual(op.input_candidates, ["tomatoes"])
        self.assertEqual(op.output_state_candidate, "chopped tomatoes")

    def test_mix_extracts_multiple_inputs(self):
        step = extract_instruction_step("Mix flour, eggs and milk.", step_index=0)

        self.assertEqual(len(step.operation_candidates), 1)
        op = step.operation_candidates[0]
        self.assertEqual(op.operation_lemma_candidate, "mix")
        self.assertEqual(op.input_candidates, ["flour", "eggs", "milk"])
        self.assertEqual(op.output_state_candidate, "mixture")

    def test_bake_extracts_batter_and_output_state(self):
        step = extract_instruction_step("Bake the batter for 40 minutes.", step_index=0)

        self.assertEqual(len(step.operation_candidates), 1)
        op = step.operation_candidates[0]
        self.assertEqual(op.operation_lemma_candidate, "bake")
        self.assertEqual(op.input_candidates, ["batter"])
        self.assertEqual(op.output_state_candidate, "baked batter")

    def test_beat_then_cook_extracts_two_operations(self):
        step = extract_instruction_step("Beat the eggs, then cook them in a pan.", step_index=0)

        self.assertEqual([op.operation_lemma_candidate for op in step.operation_candidates], ["beat", "cook"])
        self.assertEqual(step.operation_candidates[0].input_candidates, ["eggs"])
        self.assertEqual(step.operation_candidates[0].output_state_candidate, "beaten eggs")
        self.assertEqual(step.operation_candidates[1].input_candidates, ["eggs"])
        self.assertEqual(step.operation_candidates[1].output_state_candidate, "cooked eggs")

    def test_unrecognized_instruction_produces_warning(self):
        step = extract_instruction_step("Let happiness happen.", step_index=0)

        self.assertEqual(step.operation_candidates, [])
        self.assertTrue(step.warnings)
        self.assertIn("unrecognized_instruction_clause:Let happiness happen", step.warnings)


if __name__ == "__main__":
    unittest.main()
