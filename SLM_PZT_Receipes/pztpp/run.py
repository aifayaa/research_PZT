from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from pztpp.embeddings import l2_normalize_rows
from pztpp.score import (
    directional_ingr_coverage,
    directional_ingr_precision,
    directional_ops_coverage,
    directional_ops_precision,
    in_pzt_zone,
    normalized_weights,
    pzt_edit_costs,
    pzt_edit_score,
    pztpp_score,
)


CSV_FIELDS = [
    "query_id", "query_title", "other_id", "other_title",
    "pzt_score", "pzt_edit_score",
    "cov_ingr_soft", "cov_ops_soft",
    "ingr_recall", "ingr_precision", "ops_recall", "ops_precision",
    "sim_instr", "sim_full",
    "miss_ingr", "fa_ingr", "miss_ops", "fa_ops",
    "switch_cost", "edit_cost", "in_pzt_zone", "filtered_reason",
    "n_heads_T", "n_heads_S", "n_missing_heads", "n_extra_heads",
    "n_ops_T", "n_ops_S", "n_missing_ops", "n_extra_ops",
]


def run_rankings(
    *,
    out_dir: Path,
    retrieval: str = "hybrid",
    hybrid_w: float = 0.6,
    cand_k: int = 1000,
    top_k: int = 20,
    a: float = 0.60,
    b: float = 0.30,
    c: float = 0.10,
    min_ingr_cov: float = 0.55,
    min_ops_cov: float = 0.45,
    min_instr_sim: float = 0.50,
    gates: bool = True,
    explain_top_n: int = 30,
    gap_threshold: float = 0.50,
    rerank_by_sim_full: bool = True,
    min_ingr_precision: float = 0.55,
    min_ops_precision: float = 0.30,
    min_switch_cost: float = 0.02,
    max_switch_cost: float = 0.25,
    mode: str = "few_switches",
    a_m: float = 0.55,
    a_f: float = 0.15,
    b_m: float = 0.15,
    b_f: float = 0.05,
) -> None:
    del gap_threshold
    titles = _load_json(out_dir / "titles.json")
    recipe_ids = _load_json(out_dir / "recipe_ids.json")
    head_ids = _load_json(out_dir / "head_ids_per_recipe.json")
    op_ids = _load_json(out_dir / "op_ids_per_recipe.json")
    head_vocab = _load_json(out_dir / "head_vocab.json")
    op_vocab = _load_json(out_dir / "op_vocab.json")
    idf_heads = np.load(out_dir / "idf_heads.npy")
    idf_ops = np.load(out_dir / "idf_ops.npy")
    E_full = l2_normalize_rows(np.load(out_dir / "embeddings" / "E_full.f16.npy"))
    E_ingr = l2_normalize_rows(np.load(out_dir / "embeddings" / "E_ingr.f16.npy"))
    E_instr = l2_normalize_rows(np.load(out_dir / "embeddings" / "E_instr.f16.npy"))
    postings = _load_json(out_dir / "head_postings.json") if (out_dir / "head_postings.json").exists() else {}
    inv_head = {v: k for k, v in head_vocab.items()}
    inv_op = {v: k for k, v in op_vocab.items()}

    per_target: List[Dict[str, Any]] = []
    reverse: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(titles))}

    for target_idx in range(len(titles)):
        candidates = _candidate_indices(
            target_idx,
            E_full=E_full,
            E_ingr=E_ingr,
            retrieval=retrieval,
            hybrid_w=hybrid_w,
            cand_k=cand_k,
            target_heads=head_ids[target_idx],
            postings=postings,
        )
        scored: List[Dict[str, Any]] = []
        filtered_seen = 0
        for source_idx in candidates:
            entry = _score_pair(
                source_idx,
                target_idx,
                titles=titles,
                recipe_ids=recipe_ids,
                head_ids=head_ids,
                op_ids=op_ids,
                idf_heads=idf_heads,
                idf_ops=idf_ops,
                E_full=E_full,
                E_instr=E_instr,
                inv_head=inv_head,
                inv_op=inv_op,
                explain_top_n=explain_top_n,
                a=a,
                b=b,
                c=c,
                a_m=a_m,
                a_f=a_f,
                b_m=b_m,
                b_f=b_f,
                min_switch_cost=min_switch_cost,
                max_switch_cost=max_switch_cost,
            )
            reason = _filtered_reason(entry, gates, min_ingr_cov, min_ingr_precision, min_ops_cov, min_ops_precision, min_instr_sim, mode)
            entry["filtered_reason"] = reason
            if reason == "pass":
                scored.append(entry)
            else:
                filtered_seen += 1

        scored.sort(
            key=lambda row: (
                row["pzt_edit_score"],
                -row["switch_cost"],
                row["sim_full"] if rerank_by_sim_full else 0.0,
            ),
            reverse=True,
        )
        top = scored[:top_k]
        record = {
            "query_id": target_idx,
            "query_recipe_id": recipe_ids[target_idx],
            "query_title": titles[target_idx],
            "results": top,
            "candidate_count": len(candidates),
            "filtered_count": filtered_seen,
        }
        per_target.append(record)
        for entry in top:
            reverse[int(entry["other_id"])].append(_reverse_entry(entry, target_idx, recipe_ids[target_idx], titles[target_idx]))

    _write_outputs(out_dir, "targets_to_sources", per_target, query_kind="target")
    source_records = [
        {
            "query_id": idx,
            "query_recipe_id": recipe_ids[idx],
            "query_title": titles[idx],
            "results": sorted(rows, key=lambda row: row["pzt_edit_score"], reverse=True)[:top_k],
        }
        for idx, rows in reverse.items()
    ]
    _write_outputs(out_dir, "sources_to_targets", source_records, query_kind="source")
    _write_summary(out_dir, per_target)


def _candidate_indices(
    target_idx: int,
    *,
    E_full: np.ndarray,
    E_ingr: np.ndarray,
    retrieval: str,
    hybrid_w: float,
    cand_k: int,
    target_heads: Sequence[int],
    postings: Dict[str, List[int]],
) -> List[int]:
    n = E_full.shape[0]
    k = min(max(1, cand_k), max(1, n - 1))
    sims_full = E_full @ E_full[target_idx]
    sims_ingr = E_ingr @ E_ingr[target_idx]
    if retrieval == "full":
        sims = sims_full
    elif retrieval == "ingr":
        sims = sims_ingr
    else:
        sims = hybrid_w * sims_ingr + (1.0 - hybrid_w) * sims_full
    sims = sims.copy()
    sims[target_idx] = -np.inf
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    candidates = set(int(i) for i in idx if i != target_idx)
    for head in target_heads:
        for ridx in postings.get(str(head), [])[: max(k, 50)]:
            if ridx != target_idx:
                candidates.add(int(ridx))
    return list(candidates)


def _score_pair(
    source_idx: int,
    target_idx: int,
    *,
    titles: List[str],
    recipe_ids: List[str],
    head_ids: List[List[int]],
    op_ids: List[List[int]],
    idf_heads: np.ndarray,
    idf_ops: np.ndarray,
    E_full: np.ndarray,
    E_instr: np.ndarray,
    inv_head: Dict[int, str],
    inv_op: Dict[int, str],
    explain_top_n: int,
    a: float,
    b: float,
    c: float,
    a_m: float,
    a_f: float,
    b_m: float,
    b_f: float,
    min_switch_cost: float,
    max_switch_cost: float,
) -> Dict[str, Any]:
    S_heads = head_ids[source_idx]
    T_heads = head_ids[target_idx]
    S_ops = op_ids[source_idx]
    T_ops = op_ids[target_idx]
    w_T_heads = normalized_weights(T_heads, idf_heads)
    w_S_heads = normalized_weights(S_heads, idf_heads)
    w_T_ops = normalized_weights(T_ops, idf_ops)
    w_S_ops = normalized_weights(S_ops, idf_ops)
    ingr_recall, _, missing_heads = directional_ingr_coverage(S_heads, T_heads, w_T_heads, None)
    ingr_precision, extra_heads = directional_ingr_precision(S_heads, T_heads, w_S_heads, None)
    ops_recall, missing_ops = directional_ops_coverage(S_ops, T_ops, w_T_ops)
    ops_precision, extra_ops = directional_ops_precision(S_ops, T_ops, w_S_ops)
    sim_instr = float(np.dot(E_instr[source_idx], E_instr[target_idx]))
    sim_full = float(np.dot(E_full[source_idx], E_full[target_idx]))
    switch_cost, edit_cost, masses = pzt_edit_costs(
        ingr_recall=ingr_recall,
        ingr_precision=ingr_precision,
        ops_recall=ops_recall,
        ops_precision=ops_precision,
        sim_instr=sim_instr,
        a_m=a_m,
        a_f=a_f,
        b_m=b_m,
        b_f=b_f,
        c=c,
    )
    legacy = pztpp_score(ingr_recall, ops_recall, sim_instr, a=a, b=b, c=c)
    edit_score = pzt_edit_score(edit_cost)
    zone = in_pzt_zone(switch_cost, min_switch_cost, max_switch_cost)
    return {
        "recipe_id": recipe_ids[source_idx],
        "other_id": source_idx,
        "other_recipe_id": recipe_ids[source_idx],
        "other_title": titles[source_idx],
        "pzt_score": legacy,
        "pzt_edit_score": edit_score,
        "scores": {"pzt_score": legacy, "pzt_edit_score": edit_score},
        "cov_ingr_soft": ingr_recall,
        "cov_ops_soft": ops_recall,
        "ingr_recall": ingr_recall,
        "ingr_precision": ingr_precision,
        "ops_recall": ops_recall,
        "ops_precision": ops_precision,
        "sim_instr": sim_instr,
        "sim_full": sim_full,
        "miss_ingr": masses["miss_ingr"],
        "fa_ingr": masses["fa_ingr"],
        "miss_ops": masses["miss_ops"],
        "fa_ops": masses["fa_ops"],
        "switch_cost": switch_cost,
        "edit_cost": edit_cost,
        "in_pzt_zone": zone,
        "components": {
            "cov_ingr_soft": ingr_recall,
            "cov_ops_soft": ops_recall,
            "ingr_recall": ingr_recall,
            "ingr_precision": ingr_precision,
            "ops_recall": ops_recall,
            "ops_precision": ops_precision,
            "sim_instr": sim_instr,
            "sim_full": sim_full,
        },
        "masses": masses,
        "explanations": {
            "missing_heads_top": _name_residuals(missing_heads, inv_head, explain_top_n),
            "extra_heads_top": _name_residuals(extra_heads, inv_head, explain_top_n),
            "missing_ops_top": _name_residuals(missing_ops, inv_op, explain_top_n),
            "extra_ops_top": _name_residuals(extra_ops, inv_op, explain_top_n),
            "substitution_suggestions_top": [],
        },
        "n_heads_T": len(T_heads),
        "n_heads_S": len(S_heads),
        "n_missing_heads": sum(1 for _, residual in missing_heads if residual > 0.0),
        "n_extra_heads": sum(1 for _, residual in extra_heads if residual > 0.0),
        "n_ops_T": len(T_ops),
        "n_ops_S": len(S_ops),
        "n_missing_ops": sum(1 for _, residual in missing_ops if residual > 0.0),
        "n_extra_ops": sum(1 for _, residual in extra_ops if residual > 0.0),
    }


def _filtered_reason(
    row: Dict[str, Any],
    gates: bool,
    min_ingr_cov: float,
    min_ingr_precision: float,
    min_ops_cov: float,
    min_ops_precision: float,
    min_instr_sim: float,
    mode: str,
) -> str:
    if gates:
        if row["ingr_recall"] < min_ingr_cov:
            return "min_ingr_recall"
        if row["ingr_precision"] < min_ingr_precision:
            return "min_ingr_precision"
        if row["ops_recall"] < min_ops_cov:
            return "min_ops_recall"
        if row["ops_precision"] < min_ops_precision:
            return "min_ops_precision"
        if row["sim_instr"] < min_instr_sim:
            return "min_instr_sim"
    if mode == "few_switches" and not row["in_pzt_zone"]:
        return "pzt_zone"
    return "pass"


def _name_residuals(rows: List[tuple[int, float]], names: Dict[int, str], limit: int) -> List[List[Any]]:
    named = [[names.get(item, str(item)), float(residual)] for item, residual in rows if residual > 0.0]
    named.sort(key=lambda item: item[1], reverse=True)
    return named[:limit]


def _reverse_entry(entry: Dict[str, Any], target_idx: int, target_recipe_id: str, target_title: str) -> Dict[str, Any]:
    row = dict(entry)
    row["other_id"] = target_idx
    row["other_recipe_id"] = target_recipe_id
    row["other_title"] = target_title
    row["recipe_id"] = target_recipe_id
    return row


def _write_outputs(out_dir: Path, stem: str, records: List[Dict[str, Any]], *, query_kind: str) -> None:
    with (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (out_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            for entry in record["results"]:
                row = {field: entry.get(field, "") for field in CSV_FIELDS}
                row["query_id"] = record["query_id"]
                row["query_title"] = record["query_title"]
                row["filtered_reason"] = entry.get("filtered_reason", "pass")
                writer.writerow(row)


def _write_summary(out_dir: Path, records: List[Dict[str, Any]]) -> None:
    top_rows = [record["results"][0] for record in records if record["results"]]
    def dist(name: str) -> Dict[str, float]:
        vals = [float(row[name]) for row in top_rows]
        if not vals:
            return {"count": 0}
        arr = np.array(vals, dtype=np.float32)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    (out_dir / "analysis_summary.json").write_text(json.dumps({
        "n_targets": len(records),
        "n_targets_with_results": len(top_rows),
        "top1_pzt_score": dist("pzt_score"),
        "top1_pzt_edit_score": dist("pzt_edit_score"),
        "top1_switch_cost": dist("switch_cost"),
        "top1_sim_full": dist("sim_full"),
    }, indent=2), encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
