import argparse
import json
import sys
from pathlib import Path

# Ensure project root import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pztpp.build import build_pipeline
from pztpp.run import run_rankings


def apply_preset(p: argparse.ArgumentParser, preset: str, args: argparse.Namespace) -> None:
    # set defaults but do not override explicit flags (argparse's default behavior won't overwrite provided args)
    if preset == 'quality':
        if hasattr(args, 'retrieval'):
            setattr(args, 'retrieval', getattr(args, 'retrieval', None) or 'hybrid')
        if hasattr(args, 'hybrid_w'):
            setattr(args, 'hybrid_w', getattr(args, 'hybrid_w', None) if getattr(args, 'hybrid_w', None) is not None else 0.6)
        if hasattr(args, 'cand_k'):
            setattr(args, 'cand_k', getattr(args, 'cand_k', None) if getattr(args, 'cand_k', None) is not None else 500)
        if hasattr(args, 'top_k'):
            setattr(args, 'top_k', getattr(args, 'top_k', None) if getattr(args, 'top_k', None) is not None else 20)
        # scoring weights and gates
        for name, default in (
            ('a',0.60),('b',0.30),('c',0.10),
            ('min_ingr_cov',0.55),('min_ops_cov',0.45),('min_instr_sim',0.50),
            ('explain_top_n',30),('gap_threshold',0.50),
            # New defaults for PZT++-Edit
            ('min_ingr_precision',0.55),('min_ops_precision',0.30),
            ('min_switch_cost',0.02),('max_switch_cost',0.25),
            ('a_m',0.55),('a_f',0.15),('b_m',0.15),('b_f',0.05),
        ):
            if hasattr(args, name):
                setattr(args, name, getattr(args, name, None) if getattr(args, name, None) is not None else default)
        if hasattr(args, 'rerank_by_sim_full'):
            setattr(args, 'rerank_by_sim_full', True)
    elif preset == 'fast':
        if hasattr(args, 'cand_k'):
            setattr(args, 'cand_k', args.cand_k if args.cand_k is not None else 200)
        if hasattr(args, 'explain_top_n'):
            setattr(args, 'explain_top_n', args.explain_top_n if args.explain_top_n is not None else 10)
    elif preset == 'debug':
        if hasattr(args, 'limit'):
            setattr(args, 'limit', args.limit if args.limit is not None else 200)
        if hasattr(args, 'top_k'):
            setattr(args, 'top_k', args.top_k if args.top_k is not None else 10)


def main():
    ap = argparse.ArgumentParser(description='PZT++ transferability engine (build/run/query/bench)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--ndjson', type=Path, required=True)
    common.add_argument('--limit', type=int, default=None)
    common.add_argument('--out-dir', type=Path, required=True)
    common.add_argument('--model', type=str, default='all-MiniLM-L6-v2')
    common.add_argument('--strict', action='store_true', default=True)
    common.add_argument('--skip-bad-records', action='store_true', default=False)
    common.add_argument('--preset', type=str, choices=['quality','fast','debug'], default='quality')

    # build
    ap_build = sub.add_parser('build', parents=[common])
    ap_build.add_argument('--no-ann', action='store_true', default=False)
    ap_build.add_argument('--ann-engine', type=str, choices=['faiss','hnswlib'], default='faiss')
    ap_build.add_argument('--head-neighbors-k', type=int, default=20)
    ap_build.add_argument('--tau-sub', type=float, default=0.75)
    ap_build.add_argument('--no-head-neighbors', action='store_true', default=False)

    # run
    ap_run = sub.add_parser('run', parents=[common])
    ap_run.add_argument('--retrieval', type=str, choices=['hybrid','ingr','full'], default='hybrid')
    ap_run.add_argument('--hybrid-w', type=float, default=0.6)
    ap_run.add_argument('--cand-k', type=int, default=1000)
    ap_run.add_argument('--top-k', type=int, default=20)
    ap_run.add_argument('--a', type=float, default=0.60)
    ap_run.add_argument('--b', type=float, default=0.30)
    ap_run.add_argument('--c', type=float, default=0.10)
    ap_run.add_argument('--min-ingr-cov', type=float, default=0.55)
    ap_run.add_argument('--min-ops-cov', type=float, default=0.45)
    ap_run.add_argument('--min-instr-sim', type=float, default=0.50)
    # New precision gates
    ap_run.add_argument('--min-ingr-precision', type=float, default=0.55)
    ap_run.add_argument('--min-ops-precision', type=float, default=0.30)
    ap_run.add_argument('--no-gates', action='store_true', default=False)
    ap_run.add_argument('--explain-top-n', type=int, default=30)
    ap_run.add_argument('--gap-threshold', type=float, default=0.50)
    ap_run.add_argument('--rerank-by-sim-full', action='store_true', default=True)
    # PZT zone filtering and edit weights
    ap_run.add_argument('--mode', type=str, choices=['closest','few_switches'], default='few_switches')
    ap_run.add_argument('--min-switch-cost', type=float, default=0.02)
    ap_run.add_argument('--max-switch-cost', type=float, default=0.25)
    ap_run.add_argument('--a-m', dest='a_m', type=float, default=0.55)
    ap_run.add_argument('--a-f', dest='a_f', type=float, default=0.15)
    ap_run.add_argument('--b-m', dest='b_m', type=float, default=0.15)
    ap_run.add_argument('--b-f', dest='b_f', type=float, default=0.05)

    args = ap.parse_args()

    if args.cmd == 'build':
        apply_preset(ap, args.preset, args)
        if args.no_ann and (args.limit is None or args.limit > 50000):
            ap.error('--no-ann is not allowed when limit > 50k')
        build_pipeline(
            ndjson_path=args.ndjson,
            out_dir=args.out_dir,
            strict=args.strict,
            skip_bad=args.skip_bad_records,
            limit=args.limit,
            model_name=args.model,
            ann=(not args.no_ann),
            ann_engine=args.ann_engine,
            head_neighbors_k=args.head_neighbors_k,
            tau_sub=args.tau_sub,
            build_head_neighbors_flag=(not args.no_head_neighbors),
        )
        # Save run_config (build-time)
        run_cfg = vars(args).copy()
        for k in ['ndjson','out_dir','model']:
            if k in run_cfg and hasattr(run_cfg[k], '__fspath__'):
                run_cfg[k] = str(run_cfg[k])
        (args.out_dir / 'run_config.json').write_text(json.dumps(run_cfg, indent=2), encoding='utf-8')
    elif args.cmd == 'run':
        apply_preset(ap, args.preset, args)
        run_rankings(
            out_dir=args.out_dir,
            retrieval=args.retrieval,
            hybrid_w=args.hybrid_w,
            cand_k=args.cand_k,
            top_k=args.top_k,
            a=args.a, b=args.b, c=args.c,
            min_ingr_cov=args.min_ingr_cov,
            min_ops_cov=args.min_ops_cov,
            min_instr_sim=args.min_instr_sim,
            gates=(not args.no_gates),
            explain_top_n=args.explain_top_n,
            gap_threshold=args.gap_threshold,
            rerank_by_sim_full=args.rerank_by_sim_full,
            min_ingr_precision=args.min_ingr_precision,
            min_ops_precision=args.min_ops_precision,
            min_switch_cost=args.min_switch_cost,
            max_switch_cost=args.max_switch_cost,
            mode=args.mode,
            a_m=args.a_m, a_f=args.a_f, b_m=args.b_m, b_f=args.b_f,
        )
        # Save run-time args next to outputs without overwriting build's run_config
        run_args = vars(args).copy()
        for k in ['ndjson','out_dir','model']:
            if k in run_args and hasattr(run_args[k], '__fspath__'):
                run_args[k] = str(run_args[k])
        (args.out_dir / 'run_config_run.json').write_text(json.dumps(run_args, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
