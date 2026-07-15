from __future__ import annotations

import argparse
import json
from pathlib import Path

from .vocabulary_mining import mine_vocabulary, write_vocabulary_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine PZT V2 typed vocabularies from recipe NDJSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-invalid", action="store_true", default=False)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()

    result = mine_vocabulary(
        args.input,
        limit=args.limit,
        skip_invalid=args.skip_invalid,
        top_k=args.top_k,
    )
    if args.output_dir is not None:
        write_vocabulary_artifacts(result, args.output_dir, output_report=args.output)
    elif args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.report.to_dict(), indent=2), encoding="utf-8")

    print(json.dumps(result.report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
