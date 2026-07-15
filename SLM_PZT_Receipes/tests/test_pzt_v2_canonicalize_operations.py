from __future__ import annotations

import unittest

from pzt_v2.canonicalize_operations import canonicalize_operation_candidate
from pzt_v2.instruction_extraction import extract_instruction_step


class PZTV2CanonicalizeOperationsTests(unittest.TestCase):
    def _first_op(self, raw: str):
        return extract_instruction_step(raw, step_index=0).operation_candidates[0]

    def test_mix_canonicalizes_to_op_mix(self):
        canonical = canonicalize_operation_candidate(self._first_op("Mix flour, eggs and milk."))

        self.assertEqual(canonical.canonical_operation_id, "OP_MIX")
        self.assertEqual(canonical.input_candidates, ["flour", "eggs", "milk"])

    def test_combine_canonicalizes_to_op_mix(self):
        canonical = canonicalize_operation_candidate(self._first_op("Combine flour and eggs."))

        self.assertEqual(canonical.canonical_operation_id, "OP_MIX")

    def test_stir_together_canonicalizes_to_op_mix(self):
        canonical = canonicalize_operation_candidate(self._first_op("Stir flour and eggs together."))

        self.assertEqual(canonical.canonical_operation_id, "OP_MIX")

    def test_bake_canonicalizes_to_op_bake(self):
        canonical = canonicalize_operation_candidate(self._first_op("Bake the batter."))

        self.assertEqual(canonical.canonical_operation_id, "OP_BAKE")
        self.assertEqual(canonical.output_state_candidate, "baked batter")

    def test_beat_then_cook_canonicalizes_both_operations(self):
        step = extract_instruction_step("Beat the eggs, then cook them in a pan.", step_index=0)
        canonical_ops = [canonicalize_operation_candidate(op) for op in step.operation_candidates]

        self.assertEqual(canonical_ops[0].canonical_operation_id, "OP_BEAT")
        self.assertEqual(canonical_ops[1].canonical_operation_id, "OP_COOK_IN_PAN")

    def test_saucepan_does_not_trigger_cook_in_pan_context(self):
        canonical = canonicalize_operation_candidate(self._first_op("Cook sugar in a saucepan."))
        self.assertEqual(canonical.canonical_operation_id, "OP_COOK")

    def test_unknown_operation_is_explicit_warning(self):
        from pzt_v2.extraction_schema import OperationCandidate

        canonical = canonicalize_operation_candidate(
            OperationCandidate(
                raw_span="Teleport the batter",
                operation_lemma_candidate="teleport",
                input_candidates=["batter"],
                output_state_candidate=None,
                confidence=0.1,
            )
        )

        self.assertIsNone(canonical.canonical_operation_id)
        self.assertIn("unknown_operation:teleport", canonical.warnings)


if __name__ == "__main__":
    unittest.main()
