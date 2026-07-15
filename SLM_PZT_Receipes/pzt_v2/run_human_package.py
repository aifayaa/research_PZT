from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import create_manifest, write_manifest
from .human_validation import select_blind_human_pairs, write_human_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blinded 1-6 human validation package from scored PZT pairs.")
    parser.add_argument("--pair-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--distinct-count", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", default="pzt-human-v1")
    args = parser.parse_args()
    blind, key = select_blind_human_pairs(
        args.pair_scores,
        distinct_count=args.distinct_count,
        repeats=args.repeats,
        seed=args.seed,
    )
    outputs = write_human_package(blind, key, args.output_dir)
    metrics = {
        "distinct_pairs": args.distinct_count,
        "blind_repeats": args.repeats,
        "presentations": len(blind),
        "deciles": 10,
        "human_scale_min": 1,
        "human_scale_max": 6,
    }
    failures = []
    if len({row["pair_id"] for row in key}) != args.distinct_count:
        failures.append("distinct_pair_count_mismatch")
    if sum(bool(row["duplicate_of"]) for row in key) != args.repeats:
        failures.append("repeat_count_mismatch")
    manifest = create_manifest(
        run_id=args.run_id,
        stage="human_validation_package",
        inputs=[args.pair_scores],
        outputs=outputs,
        upstream_manifests=[args.upstream_manifest],
        configuration={
            "distinct_count": args.distinct_count,
            "repeats": args.repeats,
            "deciles": 10,
            "seed": args.seed,
            "score_hidden": True,
        },
        metrics=metrics,
        thresholds={"distinct_pairs": 50, "blind_repeats": 5},
        failure_reasons=failures,
        root=Path.cwd(),
    )
    write_manifest(manifest, args.manifest)
    print(json.dumps({"verdict": manifest.verdict, **metrics}, indent=2))
    if manifest.verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
