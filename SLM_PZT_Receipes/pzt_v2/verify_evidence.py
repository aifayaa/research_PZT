from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import EvidenceValidationError, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a PZT V2 evidence manifest and every referenced artifact.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    try:
        manifest = verify_manifest(args.manifest, root=args.root, require_pass=not args.allow_fail)
    except EvidenceValidationError as exc:
        parser.exit(2, f"evidence verification failed: {exc}\n")
    print(json.dumps({"stage": manifest.stage, "run_id": manifest.run_id, "verdict": manifest.verdict}, indent=2))


if __name__ == "__main__":
    main()
