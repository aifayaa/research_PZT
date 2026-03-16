#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple


def load_vocab(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    size = max(obj.values()) + 1 if obj else 0
    arr = [None] * size
    for tok, idx in obj.items():
        arr[idx] = tok
    return arr


def fmt_list(items: List[str], max_n: int) -> str:
    items = [x for x in items if x]
    if not items:
        return ""
    items = items[:max_n]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_sentence(qtitle: str, otitle: str, comps: dict, expl: dict, inv_head: List[str], inv_op: List[str], top_n: int = 3) -> str:
    miss_heads = [inv_head[h] if 0 <= h < len(inv_head) else str(h) for (h, w) in expl.get("missing_heads_top", [])]
    extra_heads = [inv_head[h] if 0 <= h < len(inv_head) else str(h) for (h, w) in expl.get("extra_heads_top", [])]
    miss_ops = [inv_op[o] if 0 <= o < len(inv_op) else str(o) for (o, w) in expl.get("missing_ops_top", [])]
    extra_ops = [inv_op[o] if 0 <= o < len(inv_op) else str(o) for (o, w) in expl.get("extra_ops_top", [])]
    swaps = []
    for (th, nh, sim) in expl.get("substitution_suggestions_top", [])[:top_n]:
        tname = inv_head[th] if 0 <= th < len(inv_head) else str(th)
        nname = inv_head[nh] if 0 <= nh < len(inv_head) else str(nh)
        if tname and nname and tname != nname:
            swaps.append(f"swap {nname} for {tname}")

    miss_h_txt = fmt_list(miss_heads, top_n)
    extra_h_txt = fmt_list(extra_heads, top_n)
    miss_o_txt = fmt_list(miss_ops, top_n)
    extra_o_txt = fmt_list(extra_ops, top_n)
    swap_txt = fmt_list(swaps, top_n)

    switch_cost = comps.get("switch_cost")
    pzt_edit = comps.get("pzt_edit_score")

    parts: List[str] = []
    if miss_h_txt:
        parts.append(f"add {miss_h_txt}")
    if swap_txt:
        parts.append(swap_txt)
    if extra_h_txt:
        parts.append(f"remove/skip {extra_h_txt}")
    if miss_o_txt or extra_o_txt:
        op_bits = []
        if miss_o_txt:
            op_bits.append(f"add op(s): {miss_o_txt}")
        if extra_o_txt:
            op_bits.append(f"remove op(s): {extra_o_txt}")
        parts.append("; ".join(op_bits))

    edits = "; ".join(parts) if parts else "no major edits"
    if switch_cost is not None and pzt_edit is not None:
        return (f"From {otitle} to {qtitle}: {edits}. "
                f"(switch_cost={switch_cost:.2f}, pzt_edit={pzt_edit:.3f}).")
    return f"From {otitle} to {qtitle}: {edits}."


def main():
    ap = argparse.ArgumentParser(description="Generate natural-language edit sentences for PZT++ results")
    ap.add_argument("--jsonl", type=Path, required=True, help="targets_to_sources.jsonl path")
    ap.add_argument("--titles", type=Path, required=True, help="titles.json path")
    ap.add_argument("--head-vocab", type=Path, required=True, help="head_vocab.json path")
    ap.add_argument("--op-vocab", type=Path, required=True, help="op_vocab.json path")
    ap.add_argument("--out-jsonl", type=Path, required=True, help="output JSONL with nl_edit added")
    ap.add_argument("--out-csv", type=Path, required=True, help="output CSV with a sentence per result")
    ap.add_argument("--top-n", type=int, default=3)
    args = ap.parse_args()

    titles = json.loads(args.titles.read_text(encoding="utf-8"))
    inv_head = load_vocab(args.head_vocab)
    inv_op = load_vocab(args.op_vocab)

    import csv
    out_rows = []
    with args.jsonl.open("r", encoding="utf-8") as f, args.out_jsonl.open("w", encoding="utf-8") as wj:
        for line in f:
            obj = json.loads(line)
            qid = int(obj.get("query_id")) if obj.get("query_id","0").isdigit() else None
            qtitle = titles[qid] if (qid is not None and 0 <= qid < len(titles)) else obj.get("query_title", "")
            new_results = []
            for res in obj.get("results", []):
                oid = int(res.get("other_id")) if str(res.get("other_id","0")).isdigit() else None
                otitle = titles[oid] if (oid is not None and 0 <= oid < len(titles)) else res.get("other_title", "")
                comps = dict(res.get("components", {}))
                # Pull masses/costs/scores if present at either top-level or nested
                if "switch_cost" in res:
                    comps["switch_cost"] = res["switch_cost"]
                if "scores" in res and "pzt_edit_score" in res["scores"]:
                    comps["pzt_edit_score"] = res["scores"]["pzt_edit_score"]
                expl = dict(res.get("explanations", {}))
                sent = build_sentence(qtitle, otitle, comps, expl, inv_head, inv_op, top_n=args.top_n)
                res2 = dict(res)
                res2["nl_edit"] = sent
                new_results.append(res2)
                out_rows.append({
                    "query_id": obj.get("query_id"),
                    "query_title": qtitle,
                    "other_id": res.get("other_id"),
                    "other_title": otitle,
                    "nl_edit": sent,
                })
            obj["results"] = new_results
            wj.write(json.dumps(obj) + "\n")

    with args.out_csv.open("w", encoding="utf-8", newline="") as wc:
        w = csv.DictWriter(wc, fieldnames=["query_id","query_title","other_id","other_title","nl_edit"])
        w.writeheader()
        for row in out_rows:
            w.writerow(row)


if __name__ == "__main__":
    main()

