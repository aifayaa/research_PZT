from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pzt_v2.evidence import EvidenceValidationError, create_manifest, verify_manifest, write_manifest


class PZTV2EvidenceTests(unittest.TestCase):
    def test_manifest_verifies_inputs_outputs_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "output.json"
            input_path.write_text('{"input": 1}\n', encoding="utf-8")
            output_path.write_text('{"output": 2}\n', encoding="utf-8")
            manifest = create_manifest(
                run_id="test-run",
                stage="parsing",
                inputs=[input_path],
                outputs=[output_path],
                configuration={"limit": 1},
                metrics={"parsed": 1},
                thresholds={"parsed": 1},
                root=root,
                command=["test"],
            )
            manifest_path = write_manifest(manifest, root / "manifest.json")
            loaded = verify_manifest(manifest_path, root=root)
            self.assertEqual(loaded.verdict, "PASS")
            self.assertEqual(loaded.configuration_sha256, manifest.configuration_sha256)

    def test_modified_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.txt"
            output_path = root / "output.txt"
            input_path.write_text("input", encoding="utf-8")
            output_path.write_text("before", encoding="utf-8")
            manifest_path = root / "manifest.json"
            write_manifest(
                create_manifest(
                    run_id="test-run",
                    stage="extraction",
                    inputs=[input_path],
                    outputs=[output_path],
                    configuration={},
                    metrics={},
                    thresholds={},
                    root=root,
                    command=["test"],
                ),
                manifest_path,
            )
            output_path.write_text("after", encoding="utf-8")
            with self.assertRaises(EvidenceValidationError):
                verify_manifest(manifest_path, root=root)

    def test_failed_manifest_is_a_hard_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            artifact.write_text(json.dumps({"ok": False}), encoding="utf-8")
            manifest_path = write_manifest(
                create_manifest(
                    run_id="test-run",
                    stage="vocabulary",
                    inputs=[artifact],
                    outputs=[artifact],
                    configuration={},
                    metrics={},
                    thresholds={},
                    failure_reasons=["coverage_below_threshold"],
                    root=root,
                    command=["test"],
                ),
                root / "manifest.json",
            )
            with self.assertRaises(EvidenceValidationError):
                verify_manifest(manifest_path, root=root)
            self.assertEqual(verify_manifest(manifest_path, root=root, require_pass=False).verdict, "FAIL")


if __name__ == "__main__":
    unittest.main()
