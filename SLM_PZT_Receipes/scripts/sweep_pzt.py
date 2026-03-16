import csv
import itertools
from pathlib import Path
from typing import Dict, Any
import json

from pzt.pipeline import run_pipeline


def summarize(jsonl_path: Path) -> Dict[str, float]:
    soft_ingr=[]; hard_ingr=[]; soft_instr=[]; hard_instr=[]
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line)
            s=obj['sources'][0]
            cov=s['components']['coverage']
            soft_ingr.append(float(cov['soft_ingr']))
            soft_instr.append(float(cov['soft_instr']))
            hard_ingr.append(int(cov['hard_ingr_count']))
            hard_instr.append(int(cov['hard_instr_count']))
    import statistics as st
    return {
        'soft_ingr_mean': st.mean(soft_ingr),
        'soft_instr_mean': st.mean(soft_instr),
        'hard_ingr_mean': st.mean(hard_ingr),
        'hard_instr_mean': st.mean(hard_instr),
    }


def main():
    root = Path('SLM_PZT_Receipes')
    ndjson = root / 'all_recipes.ndjson'
    out_root = root / 'output_pzt_sweep'
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = root / 'cache'

    # Sweep
    hybrid_ws = [0.4, 0.5, 0.6, 0.7]
    tau_instrs = [0.6, 0.7, 0.8]
    tfidf_opts = [False, True]

    rows = []
    for hw, ti, tfidf in itertools.product(hybrid_ws, tau_instrs, tfidf_opts):
        out_dir = out_root / f'hw{hw}_ti{ti}_tfidf{int(tfidf)}'
        recs, audit = run_pipeline(
            ndjson_path=ndjson,
            out_dir=out_dir,
            strict=True,
            skip_bad=False,
            limit=1000,
            model_name='all-MiniLM-L6-v2',
            cache_dir=cache_dir,
            metric='coverage',
            retrieval='hybrid',
            hybrid_w=hw,
            alpha=1.0,
            beta=1.0,
            wI=0.5,
            wG=0.5,
            tau_ingr=0.8,
            tau_instr=ti,
            cand_k=200,
            top_k=20,
            show_target=None,
            explain_normalized=True,
            gap_threshold=0.5,
            rerank_by_sim_full=True,
            use_tfidf_heads=tfidf,
        )
        summary = summarize(out_dir / 'transfer_sources_coverage.jsonl')
        row = {
            'hybrid_w': hw,
            'tau_instr': ti,
            'use_tfidf_heads': int(tfidf),
            **{k: round(v, 3) for k,v in summary.items()}
        }
        rows.append(row)

    # Write CSV
    csv_path = out_root / 'sweep_results.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['hybrid_w','tau_instr','use_tfidf_heads','soft_ingr_mean','soft_instr_mean','hard_ingr_mean','hard_instr_mean']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {csv_path}')


if __name__ == '__main__':
    main()

