#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np

# Ensure project root import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pztpp.score import (
    directional_ingr_coverage,
    directional_ops_coverage,
    directional_ingr_precision,
    directional_ops_precision,
    pztpp_score,
    pzt_edit_costs,
    pzt_edit_score,
    in_pzt_zone,
)


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


def exact_topk_by_dot(E_all: np.ndarray, K: int, block: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Exact top-K per row by dot product against E_all (cosine if rows L2-normalized).

    Returns (indices, scores) both shape (N, K).
    """
    N = E_all.shape[0]
    top_idx = np.empty((N, K), dtype=np.int32)
    top_val = np.empty((N, K), dtype=np.float32)
    X = E_all.astype(np.float32)
    for i in range(0, N, block):
        Q = X[i:i+block]
        sims = Q @ X.T  # (b, N)
        b = sims.shape[0]
        # exclude self per row
        rows = np.arange(b)
        sims[rows, i + rows] = -np.inf
        # argpartition to get topK unsorted, then sort
        part = np.argpartition(-sims, K-1, axis=1)[:, :K]
        part_vals = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_vals, axis=1)
        idx_sorted = np.take_along_axis(part, order, axis=1)
        val_sorted = np.take_along_axis(part_vals, order, axis=1)
        top_idx[i:i+b] = idx_sorted
        top_val[i:i+b] = val_sorted
    return top_idx, top_val


def weighted_jaccard_topk(
    sets_per_recipe: List[List[int]],
    idf_weights: np.ndarray,
    K: int,
    *,
    rare_R: int = 6,
) -> Tuple[List[List[int]], List[List[float]]]:
    """Compute topK per target using weighted Jaccard with candidate pruning via rare features postings.

    Returns per-target lists of indices and scores.
    """
    N = len(sets_per_recipe)
    # Build postings
    postings: Dict[int, List[int]] = {}
    for ridx, feats in enumerate(sets_per_recipe):
        for f in feats:
            postings.setdefault(f, []).append(ridx)
    # Precompute sets for speed
    sets = [set(lst) for lst in sets_per_recipe]
    out_idx: List[List[int]] = [[] for _ in range(N)]
    out_val: List[List[float]] = [[] for _ in range(N)]
    for t in range(N):
        feats = sets_per_recipe[t]
        if not feats:
            out_idx[t] = []
            out_val[t] = []
            continue
        # choose rare features by highest IDF
        feats_sorted = sorted(feats, key=lambda f: idf_weights[f], reverse=True)
        chosen = feats_sorted[:min(rare_R, len(feats_sorted))]
        cand: Set[int] = set()
        for f in chosen:
            for ridx in postings.get(f, []):
                if ridx != t:
                    cand.add(ridx)
        # compute weighted Jaccard over candidates
        A = sets[t]
        scores: List[Tuple[int, float]] = []
        if not A:
            out_idx[t] = []
            out_val[t] = []
            continue
        wA = sum(idf_weights[list(A)])
        for s in cand:
            B = sets[s]
            if not B:
                continue
            inter = A & B
            union = A | B
            if not union:
                sim = 0.0
            else:
                w_int = float(np.sum(idf_weights[list(inter)])) if inter else 0.0
                # w_union = wA + wB - w_int
                wB = float(np.sum(idf_weights[list(B)]))
                w_union = wA + wB - w_int
                sim = w_int / w_union if w_union > 0 else 0.0
            scores.append((s, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:K]
        out_idx[t] = [i for (i, v) in top]
        out_val[t] = [v for (i, v) in top]
    return out_idx, out_val


@dataclass
class EvalDefaults:
    min_ingr_recall: float = 0.55
    min_ingr_precision: float = 0.55
    min_ops_recall: float = 0.45
    min_ops_precision: float = 0.30
    min_instr_sim: float = 0.50
    min_switch_cost: float = 0.02
    max_switch_cost: float = 0.25
    a_m: float = 0.55
    a_f: float = 0.15
    b_m: float = 0.15
    b_f: float = 0.05
    c: float = 0.10
    # legacy score weights
    a: float = 0.60
    b: float = 0.30
    c_legacy: float = 0.10


def compute_pzt_components(
    S_heads: List[int], T_heads: List[int], idf_heads: np.ndarray,
    S_ops: List[int], T_ops: List[int], idf_ops: np.ndarray,
    sim_instr: float,
    *,
    head_neighbors: Optional[List[List[Tuple[int, float]]]] = None,
    defaults: EvalDefaults,
) -> Dict[str, float]:
    # weights for target recall
    w_T_heads = [idf_heads[h] for h in T_heads]
    if w_T_heads:
        m = sum(w_T_heads)/len(w_T_heads)
        w_T_heads = [x/m if m>0 else x for x in w_T_heads]
    w_T_ops = [idf_ops[o] for o in T_ops]
    if w_T_ops:
        m2 = sum(w_T_ops)/len(w_T_ops)
        w_T_ops = [x/m2 if m2>0 else x for x in w_T_ops]
    # precision weights on S
    w_S_heads = [idf_heads[h] for h in S_heads]
    if w_S_heads:
        ms = sum(w_S_heads)/len(w_S_heads)
        w_S_heads = [x/ms if ms>0 else x for x in w_S_heads]
    w_S_ops = [idf_ops[o] for o in S_ops]
    if w_S_ops:
        ms2 = sum(w_S_ops)/len(w_S_ops)
        w_S_ops = [x/ms2 if ms2>0 else x for x in w_S_ops]

    hn = head_neighbors or [[] for _ in range(max(max(S_heads or [0]), max(T_heads or [0])) + 1)]
    ingr_recall, _, _ = directional_ingr_coverage(S_heads, T_heads, w_T_heads, hn, substitution_credit_exp=2.0)
    ops_recall, _ = directional_ops_coverage(S_ops, T_ops, w_T_ops)
    ingr_precision, _ = directional_ingr_precision(S_heads, T_heads, w_S_heads, hn, substitution_credit_exp=2.0)
    ops_precision, _ = directional_ops_precision(S_ops, T_ops, w_S_ops)

    switch_cost, edit_cost, masses = pzt_edit_costs(
        ingr_recall=ingr_recall,
        ingr_precision=ingr_precision,
        ops_recall=ops_recall,
        ops_precision=ops_precision,
        sim_instr=sim_instr,
        a_m=defaults.a_m, a_f=defaults.a_f, b_m=defaults.b_m, b_f=defaults.b_f, c=defaults.c,
    )
    return {
        "ingr_recall": float(ingr_recall),
        "ingr_precision": float(ingr_precision),
        "ops_recall": float(ops_recall),
        "ops_precision": float(ops_precision),
        "sim_instr": float(sim_instr),
        "switch_cost": float(switch_cost),
        "edit_cost": float(edit_cost),
        "pzt_edit_score": float(pzt_edit_score(edit_cost)),
        "in_pzt_zone": bool(in_pzt_zone(float(switch_cost), defaults.min_switch_cost, defaults.max_switch_cost)),
    }


def main():
    ap = argparse.ArgumentParser(description="10K-only measure comparison for PZT++ transferability")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--rare-R", type=int, default=6)
    args = ap.parse_args()

    run_dir = args.run_dir; out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    titles = load_json(run_dir / "titles.json")
    recipe_ids = load_json(run_dir / "recipe_ids.json") if (run_dir / "recipe_ids.json").exists() else [str(i) for i in range(len(titles))]
    head_ids = load_json(run_dir / "head_ids_per_recipe.json")
    op_ids = load_json(run_dir / "op_ids_per_recipe.json")
    idf_heads = np.load(run_dir / "idf_heads.npy")
    idf_ops = np.load(run_dir / "idf_ops.npy")
    hn_path = run_dir / "head_neighbors" / "head_neighbors.json"
    head_neighbors = load_json(hn_path)["neighbors"] if hn_path.exists() else None

    # Embeddings (float16 memmaps) -> float32 normalized
    E_full = l2_normalize_rows(np.load(run_dir / "embeddings" / "E_full.f16.npy").astype(np.float32))
    E_ingr = l2_normalize_rows(np.load(run_dir / "embeddings" / "E_ingr.f16.npy").astype(np.float32))
    E_instr = l2_normalize_rows(np.load(run_dir / "embeddings" / "E_instr.f16.npy").astype(np.float32))
    N = E_full.shape[0]

    defaults = EvalDefaults()

    # Load few_switches run outputs for best PZT source per target
    best_pzt: Dict[int, Tuple[int, float, float]] = {}
    with (run_dir / "targets_to_sources.csv").open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        last_q = None
        for row in r:
            q = int(row["query_id"]) if row.get("query_id","0").isdigit() else None
            if q is None:
                continue
            if last_q == q:
                continue
            s = int(row["other_id"]) if row.get("other_id","0").isdigit() else None
            if s is None:
                continue
            sc = float(row.get("switch_cost", "nan"))
            ec = float(row.get("edit_cost", "nan"))
            best_pzt[q] = (s, sc, ec)
            last_q = q

    # Compute embedding-based topK
    idx_full, val_full = exact_topk_by_dot(E_full, args.k, block=args.block)
    idx_ingr, val_ingr = exact_topk_by_dot(E_ingr, args.k, block=args.block)
    idx_instr, val_instr = exact_topk_by_dot(E_instr, args.k, block=args.block)

    # Structured Jaccards topK
    idx_heads, val_heads = weighted_jaccard_topk(head_ids, idf_heads, args.k, rare_R=args.rare_R)
    idx_ops, val_ops = weighted_jaccard_topk(op_ids, idf_ops, args.k, rare_R=args.rare_R)

    # PZT rankings (legacy MISS-only and Edit variants)
    def pzt_legacy_scores_for_t(t: int, cand: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        cand = cand or set(range(N))
        if t in cand:
            cand.discard(t)
        T_heads = head_ids[t]; T_ops = op_ids[t]
        w_T_heads = [idf_heads[h] for h in T_heads]
        if w_T_heads:
            m = sum(w_T_heads)/len(w_T_heads)
            w_T_heads = [x/m if m>0 else x for x in w_T_heads]
        w_T_ops = [idf_ops[o] for o in T_ops]
        if w_T_ops:
            m2 = sum(w_T_ops)/len(w_T_ops)
            w_T_ops = [x/m2 if m2>0 else x for x in w_T_ops]
        scores: List[Tuple[int, float]] = []
        for s in cand:
            cov_ingr, _, _ = directional_ingr_coverage(head_ids[s], T_heads, w_T_heads, head_neighbors or [[] for _ in range(len(idf_heads))], substitution_credit_exp=2.0)
            cov_ops, _ = directional_ops_coverage(op_ids[s], T_ops, w_T_ops)
            sim_i = float(np.dot(E_instr[s], E_instr[t]))
            sc = pztpp_score(cov_ingr, cov_ops, sim_i, a=defaults.a, b=defaults.b, c=defaults.c_legacy)
            scores.append((s, sc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def pzt_edit_scores_for_t(t: int, cand: Optional[Set[int]] = None) -> List[Tuple[int, float, Dict[str, float]]]:
        cand = cand or set(range(N))
        if t in cand:
            cand.discard(t)
        T_heads = head_ids[t]; T_ops = op_ids[t]
        out: List[Tuple[int, float, Dict[str, float]]] = []
        for s in cand:
            comps = compute_pzt_components(
                head_ids[s], T_heads, idf_heads, op_ids[s], T_ops, idf_ops,
                sim_instr=float(np.dot(E_instr[s], E_instr[t])),
                head_neighbors=head_neighbors,
                defaults=defaults,
            )
            out.append((s, comps["pzt_edit_score"], comps))
        out.sort(key=lambda x: (x[1], -x[2]["switch_cost"]), reverse=True)
        return out

    # For scalability, reuse candidates from union of methods for Edit scores
    all_cands: List[Set[int]] = []
    for t in range(N):
        cand = set(idx_full[t]) | set(idx_ingr[t]) | set(idx_instr[t]) | set(idx_heads[t]) | set(idx_ops[t])
        all_cands.append(cand)

    # Collect per-target top1 comparisons
    methods = [
        ("sim_full", idx_full, val_full),
        ("sim_ingr_embed", idx_ingr, val_ingr),
        ("sim_instr", idx_instr, val_instr),
        ("sim_ingr_heads_wjaccard", idx_heads, val_heads),
        ("sim_ops_wjaccard", idx_ops, val_ops),
    ]

    out_csv = (out_dir / "per_target_top1_comparison.csv").open("w", encoding="utf-8", newline="")
    fieldnames = [
        "target_id","target_title","best_pzt_source_id","best_pzt_switch_cost","best_pzt_edit_cost",
    ]
    for name, _, _ in methods:
        fieldnames += [
            f"{name}_top1_source_id", f"{name}_top1_source_title", f"{name}_top1_metric_value",
            f"{name}_top1_switch_cost", f"{name}_top1_edit_cost", f"{name}_top1_in_pzt_zone",
            f"{name}_top1_ingr_recall", f"{name}_top1_ingr_precision", f"{name}_top1_ops_recall", f"{name}_top1_ops_precision",
            f"{name}_top1_sim_instr", f"{name}_top1_sim_full",
        ]
    for name in ["pzt_score_legacy","pzt_edit_closest","pzt_edit_few_switches"]:
        fieldnames += [
            f"{name}_top1_source_id", f"{name}_top1_source_title", f"{name}_top1_metric_value",
            f"{name}_top1_switch_cost", f"{name}_top1_edit_cost", f"{name}_top1_in_pzt_zone",
            f"{name}_top1_ingr_recall", f"{name}_top1_ingr_precision", f"{name}_top1_ops_recall", f"{name}_top1_ops_precision",
            f"{name}_top1_sim_instr", f"{name}_top1_sim_full",
        ]
    w = csv.DictWriter(out_csv, fieldnames=fieldnames)
    w.writeheader()

    # For agreement@K
    topK_by_method: Dict[str, List[List[int]]] = {name: [lst for lst in idx] for (name, idx, _) in methods}
    # Fill for PZT variants below

    # Summaries accumulators
    sum_stats: Dict[str, List[float]] = {k: [] for k in [
        "sim_full","sim_ingr_embed","sim_instr","sim_ingr_heads_wjaccard","sim_ops_wjaccard",
        "pzt_score_legacy","pzt_edit_closest","pzt_edit_few_switches"
    ]}
    zone_hits: Dict[str, int] = {k: 0 for k in sum_stats}
    dup_hits: Dict[str, int] = {k: 0 for k in sum_stats}
    regrets_switch: Dict[str, List[float]] = {k: [] for k in sum_stats}
    regrets_edit: Dict[str, List[float]] = {k: [] for k in sum_stats}

    for t in range(N):
        row: Dict[str, Any] = {
            "target_id": recipe_ids[t],
            "target_title": titles[t] if t < len(titles) else "",
        }
        # best PZT source from run CSV
        best_s, best_sc, best_ec = best_pzt.get(t, (None, math.nan, math.nan))
        row.update({
            "best_pzt_source_id": str(best_s) if best_s is not None else "",
            "best_pzt_switch_cost": best_sc,
            "best_pzt_edit_cost": best_ec,
        })

        # For each similarity method, evaluate top1
        for name, idx_mat, val_mat in methods:
            if len(idx_mat[t]) == 0:
                continue
            s = int(idx_mat[t][0])
            sim_full_val = float(np.dot(E_full[s], E_full[t]))
            comps = compute_pzt_components(
                head_ids[s], head_ids[t], idf_heads, op_ids[s], op_ids[t], idf_ops,
                sim_instr=float(np.dot(E_instr[s], E_instr[t])),
                head_neighbors=head_neighbors,
                defaults=defaults,
            )
            row.update({
                f"{name}_top1_source_id": recipe_ids[s],
                f"{name}_top1_source_title": titles[s] if s < len(titles) else "",
                f"{name}_top1_metric_value": float(val_mat[t][0]) if len(val_mat[t])>0 else float("nan"),
                f"{name}_top1_switch_cost": comps["switch_cost"],
                f"{name}_top1_edit_cost": comps["edit_cost"],
                f"{name}_top1_in_pzt_zone": comps["in_pzt_zone"],
                f"{name}_top1_ingr_recall": comps["ingr_recall"],
                f"{name}_top1_ingr_precision": comps["ingr_precision"],
                f"{name}_top1_ops_recall": comps["ops_recall"],
                f"{name}_top1_ops_precision": comps["ops_precision"],
                f"{name}_top1_sim_instr": comps["sim_instr"],
                f"{name}_top1_sim_full": sim_full_val,
            })
            # Summaries
            sum_stats[name].append(comps["switch_cost"])
            if comps["in_pzt_zone"]:
                zone_hits[name] += 1
            if comps["switch_cost"] < defaults.min_switch_cost:
                dup_hits[name] += 1
            if best_s is not None and best_sc == best_sc:
                regrets_switch[name].append(comps["switch_cost"] - best_sc)
                if best_ec == best_ec:
                    regrets_edit[name].append(comps["edit_cost"] - best_ec)

        # Legacy PZT MISS-only top1 from union candidates
        cand = set(all_cands[t])
        leg = pzt_legacy_scores_for_t(t, cand)
        if leg:
            s, val = leg[0]
            comps = compute_pzt_components(
                head_ids[s], head_ids[t], idf_heads, op_ids[s], op_ids[t], idf_ops,
                sim_instr=float(np.dot(E_instr[s], E_instr[t])), head_neighbors=head_neighbors, defaults=defaults,
            )
            row.update({
                "pzt_score_legacy_top1_source_id": recipe_ids[s],
                "pzt_score_legacy_top1_source_title": titles[s] if s < len(titles) else "",
                "pzt_score_legacy_top1_metric_value": float(val),
                "pzt_score_legacy_top1_switch_cost": comps["switch_cost"],
                "pzt_score_legacy_top1_edit_cost": comps["edit_cost"],
                "pzt_score_legacy_top1_in_pzt_zone": comps["in_pzt_zone"],
                "pzt_score_legacy_top1_ingr_recall": comps["ingr_recall"],
                "pzt_score_legacy_top1_ingr_precision": comps["ingr_precision"],
                "pzt_score_legacy_top1_ops_recall": comps["ops_recall"],
                "pzt_score_legacy_top1_ops_precision": comps["ops_precision"],
                "pzt_score_legacy_top1_sim_instr": comps["sim_instr"],
                "pzt_score_legacy_top1_sim_full": float(np.dot(E_full[s], E_full[t])),
            })
            sum_stats["pzt_score_legacy"].append(comps["switch_cost"])
            if comps["in_pzt_zone"]:
                zone_hits["pzt_score_legacy"] += 1
            if comps["switch_cost"] < defaults.min_switch_cost:
                dup_hits["pzt_score_legacy"] += 1
            if best_s is not None and best_sc == best_sc:
                regrets_switch["pzt_score_legacy"].append(comps["switch_cost"] - best_sc)
                if best_ec == best_ec:
                    regrets_edit["pzt_score_legacy"].append(comps["edit_cost"] - best_ec)

        # PZT Edit: closest vs few_switches
        ped = pzt_edit_scores_for_t(t, all_cands[t])
        if ped:
            # closest (no zone filter), we already sorted by pzt_edit and fewer switch
            s, val, comps = ped[0]
            row.update({
                "pzt_edit_closest_top1_source_id": recipe_ids[s],
                "pzt_edit_closest_top1_source_title": titles[s] if s < len(titles) else "",
                "pzt_edit_closest_top1_metric_value": float(val),
                "pzt_edit_closest_top1_switch_cost": comps["switch_cost"],
                "pzt_edit_closest_top1_edit_cost": comps["edit_cost"],
                "pzt_edit_closest_top1_in_pzt_zone": comps["in_pzt_zone"],
                "pzt_edit_closest_top1_ingr_recall": comps["ingr_recall"],
                "pzt_edit_closest_top1_ingr_precision": comps["ingr_precision"],
                "pzt_edit_closest_top1_ops_recall": comps["ops_recall"],
                "pzt_edit_closest_top1_ops_precision": comps["ops_precision"],
                "pzt_edit_closest_top1_sim_instr": comps["sim_instr"],
                "pzt_edit_closest_top1_sim_full": float(np.dot(E_full[s], E_full[t])),
            })
            sum_stats["pzt_edit_closest"].append(comps["switch_cost"])
            if comps["in_pzt_zone"]:
                zone_hits["pzt_edit_closest"] += 1
            if comps["switch_cost"] < defaults.min_switch_cost:
                dup_hits["pzt_edit_closest"] += 1
            if best_s is not None and best_sc == best_sc:
                regrets_switch["pzt_edit_closest"].append(comps["switch_cost"] - best_sc)
                if best_ec == best_ec:
                    regrets_edit["pzt_edit_closest"].append(comps["edit_cost"] - best_ec)

            # few_switches: first candidate inside band, else None
            s2 = None; comps2 = None; val2 = None
            for s_i, val_i, c_i in ped:
                if in_pzt_zone(c_i["switch_cost"], defaults.min_switch_cost, defaults.max_switch_cost):
                    s2, val2, comps2 = s_i, val_i, c_i
                    break
            if s2 is not None:
                row.update({
                    "pzt_edit_few_switches_top1_source_id": recipe_ids[s2],
                    "pzt_edit_few_switches_top1_source_title": titles[s2] if s2 < len(titles) else "",
                    "pzt_edit_few_switches_top1_metric_value": float(val2),
                    "pzt_edit_few_switches_top1_switch_cost": comps2["switch_cost"],
                    "pzt_edit_few_switches_top1_edit_cost": comps2["edit_cost"],
                    "pzt_edit_few_switches_top1_in_pzt_zone": comps2["in_pzt_zone"],
                    "pzt_edit_few_switches_top1_ingr_recall": comps2["ingr_recall"],
                    "pzt_edit_few_switches_top1_ingr_precision": comps2["ingr_precision"],
                    "pzt_edit_few_switches_top1_ops_recall": comps2["ops_recall"],
                    "pzt_edit_few_switches_top1_ops_precision": comps2["ops_precision"],
                    "pzt_edit_few_switches_top1_sim_instr": comps2["sim_instr"],
                    "pzt_edit_few_switches_top1_sim_full": float(np.dot(E_full[s2], E_full[t])),
                })
                sum_stats["pzt_edit_few_switches"].append(comps2["switch_cost"])
                if comps2["in_pzt_zone"]:
                    zone_hits["pzt_edit_few_switches"] += 1
                if comps2["switch_cost"] < defaults.min_switch_cost:
                    dup_hits["pzt_edit_few_switches"] += 1
                if best_s is not None and best_sc == best_sc:
                    regrets_switch["pzt_edit_few_switches"].append(comps2["switch_cost"] - best_sc)
                    if best_ec == best_ec:
                        regrets_edit["pzt_edit_few_switches"].append(comps2["edit_cost"] - best_ec)

            # record topK for agreement
            topK_by_method["pzt_edit_closest"] = topK_by_method.get("pzt_edit_closest", [[] for _ in range(N)])
            topK_by_method["pzt_edit_few_switches"] = topK_by_method.get("pzt_edit_few_switches", [[] for _ in range(N)])
            topK_by_method["pzt_edit_closest"][t] = [s_i for (s_i, _, _) in ped[:args.k]]
            # few_switches topK constrained to those in band (in order)
            few_ids = [s_i for (s_i, _, c_i) in ped if in_pzt_zone(c_i["switch_cost"], defaults.min_switch_cost, defaults.max_switch_cost)]
            topK_by_method["pzt_edit_few_switches"][t] = few_ids[:args.k]

        w.writerow(row)

    out_csv.close()

    # Summaries
    def quantiles(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"count": 0}
        arr = np.array(xs, dtype=np.float32)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {}
    for name in sum_stats:
        summary[name] = {
            "zone_hit_rate_top1": zone_hits[name] / max(1, len(sum_stats[name])),
            "switch_cost_top1": quantiles(sum_stats[name]),
            "duplicate_rate_top1": dup_hits[name] / max(1, len(sum_stats[name])),
            "switch_regret": quantiles(regrets_switch[name]),
            "edit_regret": quantiles(regrets_edit[name]),
        }
    (out_dir / "measure_compare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # agreement@K
    method_names = list(topK_by_method.keys())
    with (out_dir / "agreement_at_k.csv").open("w", encoding="utf-8", newline="") as f:
        wA = csv.writer(f)
        wA.writerow(["method_a","method_b","mean_jaccard_topK"])
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                a = method_names[i]; b = method_names[j]
                vals = []
                for t in range(N):
                    A = set(topK_by_method[a][t])
                    B = set(topK_by_method[b][t])
                    if not A and not B:
                        continue
                    inter = len(A & B)
                    union = len(A | B)
                    vals.append(inter / union if union > 0 else 0.0)
                mean_j = float(np.mean(vals)) if vals else 0.0
                wA.writerow([a,b,mean_j])

    # Disagreement examples (~50)
    # Load JSONL explanations to extract missing/extra lists
    ex_jsonl = run_dir / "targets_to_sources.jsonl"
    ex_map: Dict[int, Dict[str, Any]] = {}
    if ex_jsonl.exists():
        with ex_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qid = int(obj.get("query_id")) if str(obj.get("query_id","0")).isdigit() else None
                if qid is not None:
                    ex_map[qid] = obj

    out_ex = (out_dir / "disagreement_examples.jsonl").open("w", encoding="utf-8")
    picked = 0
    for t in range(N):
        # choose cases where sim_full top1 has high switch and best_pzt (from run) has lower switch
        # Use earlier computed row info by recomputing for sim_full and best
        if len(idx_full[t]) == 0 or t not in best_pzt:
            continue
        s_sim = int(idx_full[t][0]); s_best, sc_best, _ = best_pzt[t]
        comps_sim = compute_pzt_components(
            head_ids[s_sim], head_ids[t], idf_heads, op_ids[s_sim], op_ids[t], idf_ops,
            sim_instr=float(np.dot(E_instr[s_sim], E_instr[t])), head_neighbors=head_neighbors, defaults=defaults,
        )
        if comps_sim["switch_cost"] > sc_best and comps_sim["switch_cost"] >= defaults.max_switch_cost * 0.9:
            # build exemplar
            ex = {
                "target_id": recipe_ids[t],
                "target_title": titles[t] if t < len(titles) else "",
                "candidate_id": recipe_ids[s_sim],
                "candidate_title": titles[s_sim] if s_sim < len(titles) else "",
                "sim_full": float(np.dot(E_full[s_sim], E_full[t])),
                "switch_cost_candidate": comps_sim["switch_cost"],
                "switch_cost_best_pzt": sc_best,
            }
            # attach missing/extra lists if available in run JSONL
            obj = ex_map.get(t)
            if obj and obj.get("results"):
                res = obj["results"][0]
                ex["missing_heads_top"] = res.get("explanations",{}).get("missing_heads_top", [])
                ex["extra_heads_top"] = res.get("explanations",{}).get("extra_heads_top", [])
                ex["missing_ops_top"] = res.get("explanations",{}).get("missing_ops_top", [])
                ex["extra_ops_top"] = res.get("explanations",{}).get("extra_ops_top", [])
            out_ex.write(json.dumps(ex) + "\n")
            picked += 1
            if picked >= 50:
                break
    out_ex.close()

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
