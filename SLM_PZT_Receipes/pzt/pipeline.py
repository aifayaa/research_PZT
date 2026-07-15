from __future__ import annotations

from pathlib import Path

from pztpp.build import build_pipeline
from pztpp.run import run_rankings


def run_pipeline(
    *,
    ndjson_path: Path,
    out_dir: Path,
    strict: bool = True,
    skip_bad: bool = False,
    limit: int | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: Path | None = None,
    metric: str = "coverage",
    retrieval: str = "hybrid",
    hybrid_w: float = 0.6,
    alpha: float = 1.0,
    beta: float = 1.0,
    wI: float = 0.5,
    wG: float = 0.5,
    tau_ingr: float = 0.8,
    tau_instr: float = 0.8,
    cand_k: int = 200,
    top_k: int = 50,
    show_target: str | None = None,
    explain_normalized: bool = False,
    gap_threshold: float = 0.5,
    rerank_by_sim_full: bool = False,
    use_tfidf_heads: bool = True,
) -> None:
    del cache_dir, metric, alpha, beta, wI, wG, tau_ingr, tau_instr, show_target, explain_normalized, use_tfidf_heads
    build_pipeline(
        ndjson_path=ndjson_path,
        out_dir=out_dir,
        strict=strict,
        skip_bad=skip_bad,
        limit=limit,
        model_name=model_name,
        ann=False,
        ann_engine="faiss",
        head_neighbors_k=0,
        tau_sub=0.75,
        build_head_neighbors_flag=False,
    )
    run_rankings(
        out_dir=out_dir,
        retrieval=retrieval,
        hybrid_w=hybrid_w,
        cand_k=cand_k,
        top_k=top_k,
        gap_threshold=gap_threshold,
        rerank_by_sim_full=rerank_by_sim_full,
        mode="closest",
    )
