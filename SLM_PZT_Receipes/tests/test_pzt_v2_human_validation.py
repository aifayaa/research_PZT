from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pzt_v2.human_analysis import analyze_human_scores, spearman
from pzt_v2.human_validation import select_blind_human_pairs, write_human_package


class PZTV2HumanValidationTests(unittest.TestCase):
    def _pair_records(self, count: int = 100) -> list[dict]:
        records = []
        for index in range(count):
            score = index / max(1, count - 1)
            records.append(
                {
                    "pair_id": f"pair-{index}",
                    "source": {"recipe_id": f"s-{index}", "title": f"S {index}", "ingredients": ["a"], "instructions": ["mix"]},
                    "target": {"recipe_id": f"t-{index}", "title": f"T {index}", "ingredients": ["b"], "instructions": ["bake"]},
                    "transferability": {"raw_transferability": score},
                    "pzt": {"pzt_score": score * (1 - score)},
                }
            )
        return records

    def test_blind_selection_covers_deciles_and_hides_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs_path = root / "pairs.jsonl"
            pairs_path.write_text("".join(json.dumps(row) + "\n" for row in self._pair_records()), encoding="utf-8")
            blind, key = select_blind_human_pairs(pairs_path, distinct_count=50, repeats=5)
            outputs = write_human_package(blind, key, root / "human")
        self.assertEqual(len(blind), 55)
        self.assertEqual(len({row["pair_id"] for row in key}), 50)
        self.assertEqual(sum(bool(row["duplicate_of"]) for row in key), 5)
        self.assertEqual({int(row["selection_decile"]) for row in key}, set(range(1, 11)))
        self.assertNotIn("raw_transferability", blind[0])
        self.assertEqual(len(outputs), 3)

    def test_spearman_handles_ordinal_ties(self):
        self.assertAlmostEqual(spearman([0.1, 0.2, 0.3, 0.4], [1, 2, 2, 3]), 0.948683298, places=6)

    def test_analysis_supports_a_strong_blind_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs_path = root / "pairs.jsonl"
            pairs_path.write_text("".join(json.dumps(row) + "\n" for row in self._pair_records()), encoding="utf-8")
            blind, key = select_blind_human_pairs(pairs_path, distinct_count=50, repeats=5)
            blind_path, key_path, _ = write_human_package(blind, key, root / "human")
            key_by_id = {row["presentation_id"]: row for row in key}
            rows = []
            with blind_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    model_score = float(key_by_id[row["presentation_id"]]["raw_transferability"])
                    row["source_known_yes_no"] = "yes"
                    row["human_score_1_to_6"] = str(min(6, max(1, round(1 + 5 * model_score))))
                    rows.append(row)
            with blind_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            report = analyze_human_scores(blind_path, key_path, bootstrap_samples=200)
        self.assertEqual(report["scientific_verdict"], "SUPPORTED")
        self.assertGreaterEqual(report["spearman_rho"], 0.5)


if __name__ == "__main__":
    unittest.main()
