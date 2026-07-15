from __future__ import annotations

import unittest

from pzt_v2.canonicalize_states import canonicalize_state_candidate


class PZTV2CanonicalizeStatesTests(unittest.TestCase):
    def test_chopped_state_has_implied_chop_operation(self):
        state, warnings = canonicalize_state_candidate("chopped")

        self.assertEqual(warnings, [])
        self.assertEqual(state.canonical_state_id, "STATE_CHOPPED")
        self.assertEqual(state.canonical_state_label, "chopped")
        self.assertEqual(state.implied_operation_id, "OP_CHOP")

    def test_melted_state_has_implied_melt_operation(self):
        state, warnings = canonicalize_state_candidate("melted")

        self.assertEqual(warnings, [])
        self.assertEqual(state.canonical_state_id, "STATE_MELTED")
        self.assertEqual(state.implied_operation_id, "OP_MELT")

    def test_unknown_state_is_explicit_warning(self):
        state, warnings = canonicalize_state_candidate("sparkled")

        self.assertIsNone(state)
        self.assertEqual(warnings, ["unknown_state:sparkled"])


if __name__ == "__main__":
    unittest.main()
