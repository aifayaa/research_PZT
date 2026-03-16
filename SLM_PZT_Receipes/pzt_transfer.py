import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running this script directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pzt.pipeline import run_pipeline


def main():
    p = argparse.ArgumentParser(description="PZT transfer source recommendation (Sentence-Transformers)")
    p.add_argument("--ndjson", type=Path, required=True, help="NDJSON file path")
    p.add_argument("--limit", type=int, default=None, help="Max number of recipes")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--cache-dir", type=Path, default=Path("SLM_PZT_Receipes/cache"), help="Cache directory")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence-Transformer model")
    p.add_argument("--metric", type=str, choices=["section_max", "jaccard", "coverage", "hybrid"], default="coverage")
    p.add_argument("--retrieval", type=str, choices=["full", "ingr", "hybrid"], default="hybrid", help="Candidate retrieval space")
    p.add_argument("--hybrid-w", type=float, default=0.6, help="Weight for ingredients in hybrid retrieval (0..1)")
    p.add_argument("--alpha", type=float, default=1.0, help="Weight for instructions similarity in section_max")
    p.add_argument("--beta", type=float, default=1.0, help="Weight for ingredients similarity in section_max")
    p.add_argument("--wI", type=float, default=0.5, help="Weight for instruction coverage")
    p.add_argument("--wG", type=float, default=0.5, help="Weight for ingredient coverage")
    p.add_argument("--tau", type=float, default=0.8, help="Default cosine threshold if specific ones are not given")
    p.add_argument("--tau-ingr", type=float, default=None, help="Cosine threshold for ingredient line-level matches (default: --tau)")
    p.add_argument("--tau-instr", type=float, default=None, help="Cosine threshold for instruction line-level matches (default: --tau)")
    p.add_argument("--gap-threshold", type=float, default=0.5, help="Report target lines whose best match is below this cosine")
    p.add_argument("--cand-k", type=int, default=200, help="Candidate retrieval size")
    p.add_argument("--top-k", type=int, default=50, help="Top sources per target")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--strict", action="store_true", help="Fail on bad records (default)")
    group.add_argument("--skip-bad-records", action="store_true", help="Skip invalid recipes and continue")
    p.add_argument("--show-target", type=str, default=None, help="Recipe id or index to display top-10 sources")
    p.add_argument("--explain-normalized", action="store_true", help="Show normalized ingredient lines in explanations")
    p.add_argument("--rerank-by-sim-full", action="store_true", help="Tie-break by sim_full: sort by (score, sim_full)")
    p.add_argument("--no-tfidf-heads", action="store_true", help="Disable TF-IDF weighting over ingredient heads")

    args = p.parse_args()

    strict = True
    skip_bad = False
    if args.skip_bad_records:
        strict = False
        skip_bad = True

    run_pipeline(
        ndjson_path=args.ndjson,
        out_dir=args.out_dir,
        strict=strict,
        skip_bad=skip_bad,
        limit=args.limit,
        model_name=args.model,
        cache_dir=args.cache_dir,
        metric=args.metric,
        retrieval=args.retrieval,
        hybrid_w=args.hybrid_w,
        alpha=args.alpha,
        beta=args.beta,
        wI=args.wI,
        wG=args.wG,
        tau_ingr=(args.tau_ingr if args.tau_ingr is not None else args.tau),
        tau_instr=(args.tau_instr if args.tau_instr is not None else args.tau),
        cand_k=args.cand_k,
        top_k=args.top_k,
        show_target=args.show_target,
        explain_normalized=args.explain_normalized,
        gap_threshold=args.gap_threshold,
        rerank_by_sim_full=args.rerank_by_sim_full,
        use_tfidf_heads=(not args.no_tfidf_heads),
    )


if __name__ == "__main__":
    main()
