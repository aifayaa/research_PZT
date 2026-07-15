from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def analyze_human_scores(
    completed_blind_path: Path,
    key_path: Path,
    *,
    bootstrap_samples: int = 2000,
    seed: str = "pzt-human-analysis-v1",
) -> dict:
    completed = {row["presentation_id"]: row for row in _read_csv(completed_blind_path)}
    key_rows = _read_csv(key_path)
    scored: List[tuple[float, float]] = []
    invalid_rows: List[str] = []
    unavailable_rows: List[str] = []
    repeats: List[dict] = []
    score_by_presentation: Dict[str, float] = {}

    for key in key_rows:
        presentation_id = key["presentation_id"]
        row = completed.get(presentation_id)
        if row is None:
            invalid_rows.append(f"missing_presentation:{presentation_id}")
            continue
        known = row.get("source_known_yes_no", "").strip().lower()
        if known in {"no", "non", "n", "0"}:
            unavailable_rows.append(presentation_id)
            continue
        if known not in {"yes", "oui", "y", "o", "1"}:
            invalid_rows.append(f"invalid_source_known:{presentation_id}:{known}")
            continue
        try:
            human = float(row.get("human_score_1_to_6", ""))
        except ValueError:
            invalid_rows.append(f"invalid_score:{presentation_id}")
            continue
        if human not in {1, 2, 3, 4, 5, 6}:
            invalid_rows.append(f"score_out_of_range:{presentation_id}:{human}")
            continue
        score_by_presentation[presentation_id] = human
        if not key.get("duplicate_of"):
            scored.append((float(key["raw_transferability"]), human))

    for key in key_rows:
        duplicate_of = key.get("duplicate_of", "").strip()
        if not duplicate_of:
            continue
        duplicate_score = score_by_presentation.get(key["presentation_id"])
        original_score = score_by_presentation.get(duplicate_of)
        repeats.append(
            {
                "presentation_id": key["presentation_id"],
                "duplicate_of": duplicate_of,
                "original_score": original_score,
                "duplicate_score": duplicate_score,
                "absolute_difference": (
                    abs(original_score - duplicate_score)
                    if original_score is not None and duplicate_score is not None
                    else None
                ),
            }
        )

    model = [item[0] for item in scored]
    human = [item[1] for item in scored]
    rho = spearman(model, human) if len(scored) >= 2 else math.nan
    mae = (
        sum(abs((1.0 + 5.0 * model_score) - human_score) for model_score, human_score in scored) / len(scored)
        if scored
        else math.nan
    )
    bias = (
        sum((1.0 + 5.0 * model_score) - human_score for model_score, human_score in scored) / len(scored)
        if scored
        else math.nan
    )
    lower = bootstrap_spearman_lower_bound(scored, samples=bootstrap_samples, seed=seed) if len(scored) >= 3 else math.nan
    stable_repeats = sum(
        repeat["absolute_difference"] is not None and repeat["absolute_difference"] <= 1.0 for repeat in repeats
    )
    repeat_stability_pass = len(repeats) >= 5 and stable_repeats >= 4

    if len(scored) < 30 or invalid_rows:
        verdict = "INCONCLUSIVE"
        reasons = ["insufficient_valid_scores_or_invalid_rows"]
    elif rho <= 0:
        verdict = "REFUTED"
        reasons = ["non_positive_rank_correlation"]
    elif rho >= 0.50 and lower > 0.20 and mae <= 1.0 and repeat_stability_pass:
        verdict = "SUPPORTED"
        reasons = []
    else:
        verdict = "INCONCLUSIVE"
        reasons = ["pre_registered_support_thresholds_not_met"]

    return {
        "scientific_verdict": verdict,
        "verdict_reasons": reasons,
        "distinct_valid_scores": len(scored),
        "unavailable_source_count": len(unavailable_rows),
        "invalid_rows": invalid_rows,
        "spearman_rho": rho,
        "bootstrap_95pct_lower_bound": lower,
        "mean_absolute_error_fixed_1_to_6_projection": mae,
        "mean_bias_fixed_1_to_6_projection": bias,
        "repeat_count": len(repeats),
        "stable_repeat_count": stable_repeats,
        "repeat_stability_pass": repeat_stability_pass,
        "repeats": repeats,
        "thresholds": {
            "minimum_valid_scores": 30,
            "support_spearman_rho": 0.50,
            "support_bootstrap_lower_bound": 0.20,
            "support_mae": 1.0,
            "stable_repeats_required": "4_of_5",
            "refutation_spearman_rho_max": 0.0,
        },
    }


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires equal vectors of length >= 2")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    return _pearson(left_ranks, right_ranks)


def bootstrap_spearman_lower_bound(
    pairs: Sequence[tuple[float, float]], *, samples: int, seed: str
) -> float:
    rng = random.Random(seed)
    correlations: List[float] = []
    for _ in range(samples):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        try:
            value = spearman([item[0] for item in draw], [item[1] for item in draw])
        except ValueError:
            continue
        if not math.isnan(value):
            correlations.append(value)
    if not correlations:
        return math.nan
    correlations.sort()
    return correlations[int(0.025 * (len(correlations) - 1))]


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator else math.nan


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
