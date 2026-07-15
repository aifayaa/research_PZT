from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .alignment import align_recipe_graphs
from .canonicalization_pipeline import canonicalize_recipe_extraction
from .evidence import create_manifest, write_manifest
from .extraction_pipeline import extract_recipe_candidates
from .graph_builder import build_recipe_graph
from .parsing import parse_ndjson
from .scoring import score_alignment


SCORE_CONTRACT_VERSION = "pzt-v2-equal-unit-edits/v0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a deterministic set of directional PZT V2 recipe pairs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe-limit", type=int, default=200)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument("--seed", default="pzt-v2-pairs-v1")
    parser.add_argument("--upstream-manifest", type=Path, action="append", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parsed_recipes = parse_ndjson(args.input, limit=args.recipe_limit)
    graphs = []
    recipes = []
    graph_errors: List[str] = []
    for parsed in parsed_recipes:
        try:
            canonicalized = canonicalize_recipe_extraction(extract_recipe_candidates(parsed))
            graph = build_recipe_graph(canonicalized)
        except Exception as exc:
            graph_errors.append(f"{parsed.recipe_id}:{exc}")
            continue
        recipes.append(parsed)
        graphs.append(graph)

    available = len(graphs) * max(0, len(graphs) - 1)
    requested = args.pair_count
    selected_count = min(requested, available)
    pair_indices = _select_directional_pairs(graphs, selected_count, seed=args.seed)
    pairs_path = args.output_dir / "pair_scores.jsonl"
    raw_scores: List[float] = []
    useful_scores: List[float] = []
    edit_costs: List[float] = []
    pair_errors: List[str] = []
    written = 0
    with pairs_path.open("w", encoding="utf-8") as handle:
        for source_index, target_index in pair_indices:
            source_graph = graphs[source_index]
            target_graph = graphs[target_index]
            try:
                alignment = align_recipe_graphs(source_graph, target_graph)
                transferability, pzt = score_alignment(alignment)
            except Exception as exc:
                pair_errors.append(f"{source_graph.recipe_id}->{target_graph.recipe_id}:{exc}")
                continue
            source_recipe = recipes[source_index]
            target_recipe = recipes[target_index]
            pair_id = _pair_id(source_graph.recipe_id, target_graph.recipe_id)
            record = {
                "pair_id": pair_id,
                "direction": {"source_recipe_id": source_graph.recipe_id, "target_recipe_id": target_graph.recipe_id},
                "source": {
                    "recipe_id": source_graph.recipe_id,
                    "title": source_graph.title,
                    "ingredients": source_recipe.raw_ingredients,
                    "instructions": source_recipe.raw_instructions,
                },
                "target": {
                    "recipe_id": target_graph.recipe_id,
                    "title": target_graph.title,
                    "ingredients": target_recipe.raw_ingredients,
                    "instructions": target_recipe.raw_instructions,
                },
                "transferability": transferability.to_dict(),
                "pzt": pzt.to_dict(),
                "alignment": alignment.to_dict(),
                "score_contract_version": SCORE_CONTRACT_VERSION,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            raw_scores.append(transferability.raw_transferability)
            useful_scores.append(pzt.pzt_score)
            edit_costs.append(transferability.total_edit_cost)
            written += 1

    report = {
        "score_contract_version": SCORE_CONTRACT_VERSION,
        "recipe_limit": args.recipe_limit,
        "recipes_parsed": len(parsed_recipes),
        "graphs_built": len(graphs),
        "available_directional_pairs": available,
        "pairs_requested": requested,
        "pairs_selected": selected_count,
        "pairs_written": written,
        "graph_errors": graph_errors[:50],
        "pair_errors": pair_errors[:50],
        "raw_transferability": _distribution(raw_scores),
        "pzt_score": _distribution(useful_scores),
        "edit_cost": _distribution(edit_costs),
    }
    report_path = args.output_dir / "pair_model_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failures = []
    if graph_errors:
        failures.append(f"graph_errors:{len(graph_errors)}")
    if pair_errors:
        failures.append(f"pair_errors:{len(pair_errors)}")
    if requested > available:
        failures.append(f"insufficient_available_pairs:{available}<{requested}")
    if written != requested:
        failures.append(f"pair_count_mismatch:{written}!={requested}")
    manifest = create_manifest(
        run_id=args.run_id,
        stage="pair_scoring",
        inputs=[args.input],
        outputs=[pairs_path, report_path],
        upstream_manifests=args.upstream_manifest,
        configuration={
            "recipe_limit": args.recipe_limit,
            "pair_count": args.pair_count,
            "seed": args.seed,
            "score_contract_version": SCORE_CONTRACT_VERSION,
            "retrieval": "deterministic_exact_selected_pairs",
        },
        metrics=report,
        thresholds={"graph_errors": 0, "pair_errors": 0, "pairs_written": requested},
        failure_reasons=failures,
        root=Path.cwd(),
    )
    write_manifest(manifest, args.manifest)
    print(json.dumps({"verdict": manifest.verdict, **report}, indent=2))
    if manifest.verdict != "PASS":
        raise SystemExit(2)


def _select_directional_pairs(graphs: Sequence, count: int, *, seed: str) -> List[Tuple[int, int]]:
    candidates = [(source, target) for source in range(len(graphs)) for target in range(len(graphs)) if source != target]
    candidates.sort(
        key=lambda pair: hashlib.sha256(
            f"{seed}:{graphs[pair[0]].recipe_id}:{graphs[pair[1]].recipe_id}".encode("utf-8")
        ).digest()
    )
    return candidates[:count]


def _pair_id(source_recipe_id: str, target_recipe_id: str) -> str:
    return hashlib.sha256(f"{source_recipe_id}->{target_recipe_id}".encode("utf-8")).hexdigest()[:20]


def _distribution(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": _quantile(ordered, 0.10),
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _quantile(ordered: Sequence[float], probability: float) -> float:
    return ordered[min(len(ordered) - 1, int(probability * (len(ordered) - 1)))]


if __name__ == "__main__":
    main()
