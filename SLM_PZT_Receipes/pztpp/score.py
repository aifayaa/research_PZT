from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


def directional_ingr_coverage(
    source_heads: Sequence[int],
    target_heads: Sequence[int],
    target_weights: Sequence[float],
    head_neighbors: Sequence[Sequence[Tuple[int, float]]] | None = None,
    *,
    substitution_credit_exp: float = 2.0,
) -> Tuple[float, List[Tuple[int, float]], List[Tuple[int, float]]]:
    score, residuals = _directional_weighted_match(
        target_heads, source_heads, target_weights, head_neighbors, substitution_credit_exp=substitution_credit_exp
    )
    matched = [(h, 1.0 - r) for h, r in residuals if r < 1.0]
    missing = [(h, r) for h, r in residuals if r > 0.0]
    return score, matched, missing


def directional_ingr_precision(
    source_heads: Sequence[int],
    target_heads: Sequence[int],
    source_weights: Sequence[float],
    head_neighbors: Sequence[Sequence[Tuple[int, float]]] | None = None,
    *,
    substitution_credit_exp: float = 2.0,
) -> Tuple[float, List[Tuple[int, float]]]:
    score, residuals = _directional_weighted_match(
        source_heads, target_heads, source_weights, head_neighbors, substitution_credit_exp=substitution_credit_exp
    )
    extra = [(h, r) for h, r in residuals if r > 0.0]
    return score, extra


def directional_ops_coverage(
    source_ops: Sequence[int],
    target_ops: Sequence[int],
    target_weights: Sequence[float],
) -> Tuple[float, List[Tuple[int, float]]]:
    score, residuals = _directional_exact_match(target_ops, source_ops, target_weights)
    missing = [(op, r) for op, r in residuals if r > 0.0]
    return score, missing


def directional_ops_precision(
    source_ops: Sequence[int],
    target_ops: Sequence[int],
    source_weights: Sequence[float],
) -> Tuple[float, List[Tuple[int, float]]]:
    score, residuals = _directional_exact_match(source_ops, target_ops, source_weights)
    extra = [(op, r) for op, r in residuals if r > 0.0]
    return score, extra


def pztpp_score(cov_ingr: float, cov_ops: float, sim_instr: float, *, a: float = 0.60, b: float = 0.30, c: float = 0.10) -> float:
    return a * _clamp01(cov_ingr) + b * _clamp01(cov_ops) + c * _clamp01(sim_instr)


def pzt_edit_costs(
    *,
    ingr_recall: float,
    ingr_precision: float,
    ops_recall: float,
    ops_precision: float,
    sim_instr: float,
    a_m: float = 0.55,
    a_f: float = 0.15,
    b_m: float = 0.15,
    b_f: float = 0.05,
    c: float = 0.10,
) -> Tuple[float, float, dict]:
    miss_ingr = 1.0 - _clamp01(ingr_recall)
    fa_ingr = 1.0 - _clamp01(ingr_precision)
    miss_ops = 1.0 - _clamp01(ops_recall)
    fa_ops = 1.0 - _clamp01(ops_precision)
    instr_gap = 1.0 - _clamp01(sim_instr)
    switch_cost = a_m * miss_ingr + a_f * fa_ingr + b_m * miss_ops + b_f * fa_ops
    edit_cost = switch_cost + c * instr_gap
    masses = {
        "miss_ingr": miss_ingr,
        "fa_ingr": fa_ingr,
        "miss_ops": miss_ops,
        "fa_ops": fa_ops,
        "instr_gap": instr_gap,
    }
    return switch_cost, edit_cost, masses


def pzt_edit_score(edit_cost: float) -> float:
    return math.exp(-max(0.0, float(edit_cost)))


def in_pzt_zone(switch_cost: float, min_switch_cost: float = 0.02, max_switch_cost: float = 0.25) -> bool:
    return min_switch_cost <= float(switch_cost) <= max_switch_cost


def normalized_weights(ids: Sequence[int], idf_weights) -> List[float]:
    weights = [float(idf_weights[i]) for i in ids]
    if not weights:
        return []
    mean = sum(weights) / len(weights)
    if mean <= 0:
        return [1.0 for _ in weights]
    return [w / mean for w in weights]


def _directional_exact_match(
    query_items: Sequence[int],
    candidate_items: Sequence[int],
    query_weights: Sequence[float],
) -> Tuple[float, List[Tuple[int, float]]]:
    if not query_items:
        return 1.0, []
    weights = list(query_weights) if query_weights else [1.0 for _ in query_items]
    candidate = set(candidate_items)
    denom = sum(weights) or 1.0
    matched = 0.0
    residuals: List[Tuple[int, float]] = []
    for item, weight in zip(query_items, weights):
        residual = 0.0 if item in candidate else 1.0
        matched += weight * (1.0 - residual)
        residuals.append((item, residual))
    return matched / denom, residuals


def _directional_weighted_match(
    query_items: Sequence[int],
    candidate_items: Sequence[int],
    query_weights: Sequence[float],
    neighbors: Sequence[Sequence[Tuple[int, float]]] | None,
    *,
    substitution_credit_exp: float,
) -> Tuple[float, List[Tuple[int, float]]]:
    if not query_items:
        return 1.0, []
    weights = list(query_weights) if query_weights else [1.0 for _ in query_items]
    candidate = set(candidate_items)
    denom = sum(weights) or 1.0
    matched = 0.0
    residuals: List[Tuple[int, float]] = []
    for item, weight in zip(query_items, weights):
        credit = 1.0 if item in candidate else _neighbor_credit(item, candidate, neighbors, substitution_credit_exp)
        residual = 1.0 - credit
        matched += weight * credit
        residuals.append((item, residual))
    return matched / denom, residuals


def _neighbor_credit(
    item: int,
    candidate: set[int],
    neighbors: Sequence[Sequence[Tuple[int, float]]] | None,
    substitution_credit_exp: float,
) -> float:
    if not neighbors or item >= len(neighbors):
        return 0.0
    best = 0.0
    for nb, sim in neighbors[item]:
        if nb in candidate:
            best = max(best, float(sim) ** substitution_credit_exp)
    return _clamp01(best)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
