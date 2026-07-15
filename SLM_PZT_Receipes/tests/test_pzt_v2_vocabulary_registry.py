from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pzt_v2.vocabulary_mining import mine_vocabulary, write_vocabulary_artifacts
from pzt_v2.vocabulary_registry import build_registry, singleton_id, validate_registry


FIXTURE = Path(__file__).parent / "fixtures" / "pzt_v2_vocab_mining" / "sample_vocab_recipes.ndjson"


class PZTV2VocabularyRegistryTests(unittest.TestCase):
    def test_unknown_surfaces_receive_stable_conservative_singletons(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_vocabulary_artifacts(mine_vocabulary(FIXTURE), output_dir)
            concepts, report = build_registry(output_dir)
        nebula = [concept for concept in concepts if "nebula root" in concept.aliases]
        self.assertEqual(len(nebula), 1)
        self.assertEqual(nebula[0].canonical_id, singleton_id("ingredient", "nebula root"))
        self.assertEqual(nebula[0].status, "provisional_singleton")
        self.assertEqual(report.represented_token_coverage, 1.0)
        self.assertEqual(report.conflicts, [])
        validate_registry(concepts)

    def test_validated_synonyms_keep_one_concept(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_vocabulary_artifacts(mine_vocabulary(FIXTURE), output_dir)
            concepts, _ = build_registry(output_dir)
        mix = [concept for concept in concepts if concept.canonical_id == "OP_MIX"]
        self.assertEqual(len(mix), 1)
        self.assertTrue({"mix", "combine", "stir together"}.issubset(set(mix[0].aliases)))


if __name__ == "__main__":
    unittest.main()
