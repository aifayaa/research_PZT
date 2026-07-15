from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import create_manifest, write_manifest
from .human_analysis import analyze_human_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed blinded human PZT scores.")
    parser.add_argument("--completed-blind", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    report = analyze_human_scores(
        args.completed_blind,
        args.key,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    data_failures = list(report["invalid_rows"])
    manifest = create_manifest(
        run_id=args.run_id,
        stage="human_validation_analysis",
        inputs=[args.completed_blind, args.key],
        outputs=[args.output],
        upstream_manifests=[args.upstream_manifest],
        configuration={"bootstrap_samples": args.bootstrap_samples, "human_scale": "integer_1_to_6"},
        metrics=report,
        thresholds=report["thresholds"],
        failure_reasons=data_failures,
        root=Path.cwd(),
    )
    write_manifest(manifest, args.manifest)
    print(json.dumps({"evidence_verdict": manifest.verdict, **report}, indent=2))
    if manifest.verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
