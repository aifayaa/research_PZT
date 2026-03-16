#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_vocab(path: Path) -> List[str]:
    obj = _load_json(path)
    arr = [None] * (max(obj.values()) + 1 if obj else 0)
    for tok, idx in obj.items():
        arr[idx] = tok
    return arr


def map_ids(ids: List[int], inv: List[str]) -> List[str]:
    out = []
    for i in ids:
        if 0 <= i < len(inv):
            out.append(inv[i])
    return out


def build_prompt(
    *,
    qtitle: str,
    otitle: str,
    T_heads: List[str],
    S_heads: List[str],
    T_ops: List[str],
    S_ops: List[str],
    missing_heads: List[str],
    extra_heads: List[str],
    missing_ops: List[str],
    extra_ops: List[str],
    swaps: List[str],
    pzt: float,
    pzt_edit: float | None,
    switch_cost: float | None,
) -> List[Dict[str, str]]:
    sys = (
        "You are a concise culinary assistant. The cook knows the SOURCE recipe well. "
        "Explain how to get to the TARGET using the SOURCE, in simple home-cook language. "
        "Focus on CHANGES ONLY (add, remove/skip, swap, small technique tweaks); do not restate the whole source recipe. "
        "Use only the provided ingredient heads (treat them as ingredient names) and operations (techniques). "
        "Do NOT invent ingredients that are not in the TARGET heads; for amounts, use plain phrases like 'a pinch' or 'to taste' if helpful."
    )
    usr = {
        "role": "user",
        "content": (
            f"TARGET recipe: {qtitle}\n"
            f"SOURCE recipe: {otitle}\n\n"
            f"TARGET ingredient heads (names): {', '.join(T_heads)}\n"
            f"SOURCE ingredient heads (names): {', '.join(S_heads)}\n"
            f"TARGET ops (techniques): {', '.join(T_ops)}\n"
            f"SOURCE ops (techniques): {', '.join(S_ops)}\n\n"
            f"Computed changes — INGREDIENTS: add {', '.join(missing_heads) or 'none'}; remove/skip {', '.join(extra_heads) or 'none'};"
            f" possible swaps: {', '.join(swaps) or 'none'}.\n"
            f"Computed changes — OPERATIONS: add {', '.join(missing_ops) or 'none'}; remove {', '.join(extra_ops) or 'none'}.\n\n"
            f"Scores (context only): pzt={pzt:.3f}" + (
                f", pzt_edit={pzt_edit:.3f}, switch_cost={switch_cost:.2f}" if (pzt_edit is not None and switch_cost is not None) else ""
            ) + "\n\n"
            "Write ONE short paragraph (3–5 sentences). Start exactly with: "
            f"'Starting from {otitle}, to make {qtitle},' then describe the changes and any simple technique tweaks. "
            "Keep it straightforward like guidance from a good home cook."
        ),
    }
    return [
        {"role": "system", "content": sys},
        usr,
    ]


def call_openai(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, api_key: str) -> str:
    """Call OpenAI Chat API.

    Prefers openai>=1.0 interface; falls back to legacy ChatCompletion if available.
    """
    try:
        # New client (openai>=1.0)
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Legacy fallback
        try:
            import openai  # type: ignore
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e2:
            raise RuntimeError("OpenAI API call failed; ensure openai package is installed and API key is valid.") from e2


def main():
    ap = argparse.ArgumentParser(description="Generate natural recipe adaptation descriptions using an LLM")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory with pztpp outputs (jsonl, vocab, ids)")
    ap.add_argument("--jsonl", type=Path, default=None, help="Optional explicit path to targets_to_sources.jsonl")
    ap.add_argument("--top-k-per-target", type=int, default=1)
    ap.add_argument("--sample-queries", type=int, default=50)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    ap.add_argument("--dry-run", action="store_true", help="Print prompts without calling the API")
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir
    jsonl_path = args.jsonl or (out_dir / "targets_to_sources.jsonl")
    titles = _load_json(out_dir / "titles.json")
    inv_head = load_vocab(out_dir / "head_vocab.json")
    inv_op = load_vocab(out_dir / "op_vocab.json")
    head_ids = _load_json(out_dir / "head_ids_per_recipe.json")
    op_ids = _load_json(out_dir / "op_ids_per_recipe.json")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("ERROR: Provide --api-key or set OPENAI_API_KEY in the environment.")

    out_jsonl = args.out_jsonl or (out_dir / "targets_to_sources_llm.jsonl")
    out_csv = args.out_csv or (out_dir / "targets_to_sources_llm.csv")

    import csv
    wc = out_csv.open("w", encoding="utf-8", newline="")
    wcsv = csv.DictWriter(wc, fieldnames=["query_id","query_title","other_id","other_title","adaptation"])
    wcsv.writeheader()

    written = 0
    with jsonl_path.open("r", encoding="utf-8") as f_in, out_jsonl.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            obj = json.loads(line)
            qid = int(obj.get("query_id"))
            qtitle = titles[qid] if 0 <= qid < len(titles) else obj.get("query_title","")
            T_heads = map_ids(head_ids[qid], inv_head)
            T_ops = map_ids(op_ids[qid], inv_op)
            results = obj.get("results", [])[: args.top_k_per_target]
            new_results: List[Dict[str, Any]] = []
            for res in results:
                oid = int(res.get("other_id"))
                otitle = titles[oid] if 0 <= oid < len(titles) else res.get("other_title","")
                S_heads = map_ids(head_ids[oid], inv_head)
                S_ops = map_ids(op_ids[oid], inv_op)
                comps = res.get("components", {})
                scores = res.get("scores", {})
                # New: pzt_score is the edit-sensitive score; fall back gracefully
                pzt = float(scores.get("pzt_score", res.get("pzt_score", 0.0)))
                pzt_edit = scores.get("pzt_score", scores.get("pzt_edit_score"))
                switch_cost = res.get("switch_cost")
                expl = res.get("explanations", {})
                missing_heads = [inv_head[h] for (h, w) in expl.get("missing_heads_top", [])[:5]]
                extra_heads = [inv_head[h] for (h, w) in expl.get("extra_heads_top", [])[:5]]
                missing_ops = [inv_op[o] for (o, w) in expl.get("missing_ops_top", [])[:3]]
                extra_ops = [inv_op[o] for (o, w) in expl.get("extra_ops_top", [])[:3]]
                swaps = []
                for th, nh, sim in expl.get("substitution_suggestions_top", [])[:3]:
                    thn = inv_head[th] if 0 <= th < len(inv_head) else str(th)
                    nhn = inv_head[nh] if 0 <= nh < len(inv_head) else str(nh)
                    if thn and nhn and thn != nhn:
                        swaps.append(f"swap {nhn} for {thn}")

                messages = build_prompt(
                    qtitle=qtitle,
                    otitle=otitle,
                    T_heads=T_heads,
                    S_heads=S_heads,
                    T_ops=T_ops,
                    S_ops=S_ops,
                    missing_heads=missing_heads,
                    extra_heads=extra_heads,
                    missing_ops=missing_ops,
                    extra_ops=extra_ops,
                    swaps=swaps,
                    pzt=pzt,
                    pzt_edit=pzt_edit,
                    switch_cost=switch_cost,
                )
                if args.dry_run:
                    text = json.dumps(messages, ensure_ascii=False)
                else:
                    text = call_openai(messages, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, api_key=api_key)

                res2 = dict(res)
                res2["nl_adaptation"] = text
                new_results.append(res2)
                wcsv.writerow({
                    "query_id": obj.get("query_id"),
                    "query_title": qtitle,
                    "other_id": res.get("other_id"),
                    "other_title": otitle,
                    "adaptation": text,
                })
                written += 1
                if written >= args.sample_queries:
                    break

            obj["results"] = new_results
            f_out.write(json.dumps(obj) + "\n")
            if written >= args.sample_queries:
                break

    wc.close()
    print(f"Wrote {written} adaptation sentences to {out_jsonl} and {out_csv}")


if __name__ == "__main__":
    main()
