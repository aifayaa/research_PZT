#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_top1(csv_path: Path):
    """Return dict: query_id -> row (dict) picking max by pzt_score if present else pzt_edit_score."""
    top = {}
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = row["query_id"]
            # prefer old pzt_score if present for old file; else use pzt_edit_score
            pzt = float(row.get("pzt_score", 0.0) or 0.0)
            ped = float(row.get("pzt_edit_score", 0.0) or 0.0)
            key_score = pzt if pzt > 0 else ped
            prev = top.get(q)
            if (prev is None) or (key_score > float(prev[1])):
                top[q] = (row, key_score)
    return {k: v[0] for k, v in top.items()}


def main():
    ap = argparse.ArgumentParser(description="Compare old vs new PZT++ scores and summarize duplicates and distributions")
    ap.add_argument("--old-csv", type=Path, required=True, help="Old run targets_to_sources.csv (MISS-only)")
    ap.add_argument("--new-csv", type=Path, required=True, help="New run targets_to_sources.csv (with edit metrics)")
    ap.add_argument("--new-jsonl", type=Path, required=False, help="New run targets_to_sources.jsonl (for extra heads aggregation)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to write analysis_summary_edit.json")
    ap.add_argument("--dup-thresh", type=float, default=0.02, help="Switch-cost threshold to consider duplicates/near-identical")
    args = ap.parse_args()

    old_top1 = load_top1(args.old_csv)
    new_top1 = load_top1(args.new_csv)

    # Map pair -> switch_cost from new CSV
    switch_cost_map = {}
    pzt_edit_scores = []
    switch_costs = []
    with args.new_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["query_id"], row["other_id"])  # pair identity
            try:
                sc = float(row.get("switch_cost", "nan"))
                pe = float(row.get("pzt_edit_score", "nan"))
            except Exception:
                sc = float("nan"); pe = float("nan")
            switch_cost_map[key] = sc
            if sc == sc:
                switch_costs.append(sc)
            if pe == pe:
                pzt_edit_scores.append(pe)

    # Duplicates in old top-1 under new switch-cost
    dup_count = 0
    tot_old = 0
    for q, row in old_top1.items():
        key = (row["query_id"], row["other_id"])
        sc = switch_cost_map.get(key, float("nan"))
        if sc == sc:
            tot_old += 1
            if sc <= args.dup_thresh:
                dup_count += 1

    # How many such duplicates removed by few-switches mode (i.e., not present as new top-1)
    removed_dups = 0
    for q, row in old_top1.items():
        key = (row["query_id"], row["other_id"])
        sc = switch_cost_map.get(key, float("nan"))
        if sc == sc and sc <= args.dup_thresh:
            # compare with new top-1 pair
            new_row = new_top1.get(q)
            if new_row is None or (new_row["other_id"],) != (row["other_id"],):
                removed_dups += 1

    # Aggregate top extra heads across new top-1 via JSONL if provided
    top_extra_heads = []
    extra_head_counts = Counter()
    if args.new_jsonl and args.new_jsonl.exists():
        # collate pairs for top-1 from new JSONL
        new_pairs = {(row["query_id"], row["other_id"]) for row in new_top1.values()}
        with args.new_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qid = obj.get("query_id")
                # find the top-1 result only
                if not obj.get("results"):
                    continue
                top = obj["results"][0]
                oid = top.get("other_id")
                if (qid, oid) not in new_pairs:
                    continue
                ex = (top.get("explanations") or {}).get("extra_heads_top")
                if isinstance(ex, list):
                    for h, w in ex:
                        extra_head_counts[h] += 1
        top_extra_heads = extra_head_counts.most_common(20)

    # Simple distribution summaries
    def dist(xs):
        if not xs:
            return {}
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        def q(p):
            i = int(p*(n-1))
            return xs_sorted[i]
        return {
            "count": n,
            "mean": sum(xs_sorted)/n,
            "p05": q(0.05), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p95": q(0.95),
            "min": xs_sorted[0], "max": xs_sorted[-1],
        }

    summary = {
        "old_top1_total": tot_old,
        "old_top1_duplicates_pct": (dup_count / tot_old * 100.0) if tot_old else 0.0,
        "removed_duplicates_count": removed_dups,
        "pzt_edit_score_dist": dist(pzt_edit_scores),
        "switch_cost_dist": dist(switch_costs),
        "top_extra_heads": top_extra_heads,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis_summary_edit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

