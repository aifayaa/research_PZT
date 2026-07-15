from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from pzt_v2.alignment import align_recipe_graphs, validate_alignment_replay
from pzt_v2.scoring import score_alignment
from tests.test_pzt_v2_graph_builder import _build


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pzt_graph_cases"


class PZTV2AlignmentScoringTests(unittest.TestCase):
    def _case(self, name: str):
        data = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
        return _build(data["source_recipe"]), _build(data["target_recipe"])

    def test_edit_script_reconstructs_complete_target(self):
        for path in sorted(FIXTURE_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            source = _build(data["source_recipe"])
            target = _build(data["target_recipe"])
            alignment = align_recipe_graphs(source, target)
            validate_alignment_replay(alignment, source, target)
            self.assertEqual(
                alignment.total_edit_cost,
                sum(edit.cost for edit in alignment.edits),
            )

    def test_identity_has_maximum_raw_and_zero_useful_pzt(self):
        source, target = self._case("05_identity_banana_bread.json")
        transferability, pzt = score_alignment(align_recipe_graphs(source, target))
        self.assertEqual(transferability.raw_transferability, 1.0)
        self.assertEqual(transferability.normalized_edit_distance, 0.0)
        self.assertEqual(pzt.novelty, 0.0)
        self.assertEqual(pzt.pzt_score, 0.0)

    def test_lexical_operation_variation_has_zero_cognitive_edit(self):
        source, target = self._case("01_lexical_variation_mix_combine.json")
        alignment = align_recipe_graphs(source, target)
        transferability, pzt = score_alignment(alignment)
        self.assertEqual(alignment.total_edit_cost, 0.0)
        self.assertEqual(transferability.raw_transferability, 1.0)
        self.assertEqual(pzt.pzt_score, 0.0)

    def test_chopped_tomato_requires_an_edit(self):
        source, target = self._case("02_tomato_vs_chopped_tomato.json")
        transferability, _ = score_alignment(align_recipe_graphs(source, target))
        self.assertGreater(transferability.total_edit_cost, 0.0)
        self.assertLess(transferability.raw_transferability, 1.0)

    def test_adding_an_edit_cannot_increase_raw_transferability(self):
        source, target = self._case("05_identity_banana_bread.json")
        identity = align_recipe_graphs(source, target)
        baseline = score_alignment(identity)[0].raw_transferability
        degraded = replace(identity, total_edit_cost=identity.total_edit_cost + 1.0)
        self.assertLess(score_alignment(degraded)[0].raw_transferability, baseline)

    def test_direction_is_recorded_and_can_change_normalization(self):
        source, target = self._case("02_tomato_vs_chopped_tomato.json")
        forward = align_recipe_graphs(source, target)
        reverse = align_recipe_graphs(target, source)
        self.assertEqual(forward.source_recipe_id, source.recipe_id)
        self.assertEqual(reverse.source_recipe_id, target.recipe_id)
        self.assertNotEqual(
            score_alignment(forward)[0].normalized_edit_distance,
            score_alignment(reverse)[0].normalized_edit_distance,
        )


if __name__ == "__main__":
    unittest.main()
