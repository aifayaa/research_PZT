from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from pzt.norm import extract_heads, normalized_ingredients_text
from pzt.ops import extract_ops
from pzt.parser import parse_ndjson
from pztpp.embeddings import encode_texts


def build_pipeline(
    *,
    ndjson_path: Path,
    out_dir: Path,
    strict: bool = True,
    skip_bad: bool = False,
    limit: int | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    ann: bool = False,
    ann_engine: str = "faiss",
    head_neighbors_k: int = 20,
    tau_sub: float = 0.75,
    build_head_neighbors_flag: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    recipes, audit = parse_ndjson(ndjson_path, strict=strict, skip_bad=skip_bad, limit=limit)

    head_names_per_recipe = [extract_heads(r.ingredients_lines_raw) for r in recipes]
    op_names_per_recipe = [extract_ops(r.instructions_lines_raw) for r in recipes]
    head_vocab = _vocab(head_names_per_recipe)
    op_vocab = _vocab(op_names_per_recipe)
    head_ids_per_recipe = [_ids(names, head_vocab) for names in head_names_per_recipe]
    op_ids_per_recipe = [_ids(names, op_vocab) for names in op_names_per_recipe]
    idf_heads = _idf(head_ids_per_recipe, len(head_vocab))
    idf_ops = _idf(op_ids_per_recipe, len(op_vocab))

    full_texts = [r.full_text for r in recipes]
    ingredient_texts = [normalized_ingredients_text(r.ingredients_lines_raw) for r in recipes]
    instruction_texts = ["\n".join(r.instructions_lines_raw) for r in recipes]
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    E_full, meta_full = encode_texts(full_texts, model_name=model_name, cache_dir=Path("cache"))
    E_ingr, meta_ingr = encode_texts(ingredient_texts, model_name=model_name, cache_dir=Path("cache"))
    E_instr, meta_instr = encode_texts(instruction_texts, model_name=model_name, cache_dir=Path("cache"))
    np.save(emb_dir / "E_full.f16.npy", E_full.astype(np.float16))
    np.save(emb_dir / "E_ingr.f16.npy", E_ingr.astype(np.float16))
    np.save(emb_dir / "E_instr.f16.npy", E_instr.astype(np.float16))
    (emb_dir / "embeddings_meta.json").write_text(json.dumps({
        "full": meta_full,
        "ingredients": meta_ingr,
        "instructions": meta_instr,
    }, indent=2), encoding="utf-8")

    _write_json(out_dir / "parse_audit.json", audit)
    _write_json(out_dir / "recipe_ids.json", [r.recipe_id for r in recipes])
    _write_json(out_dir / "titles.json", [r.title for r in recipes])
    _write_json(out_dir / "head_vocab.json", head_vocab)
    _write_json(out_dir / "op_vocab.json", op_vocab)
    _write_json(out_dir / "head_ids_per_recipe.json", head_ids_per_recipe)
    _write_json(out_dir / "op_ids_per_recipe.json", op_ids_per_recipe)
    _write_json(out_dir / "head_postings.json", _postings(head_ids_per_recipe))
    np.save(out_dir / "idf_heads.npy", idf_heads.astype(np.float32))
    np.save(out_dir / "idf_ops.npy", idf_ops.astype(np.float32))
    _write_json(out_dir / "build_config.json", {
        "model_name": model_name,
        "ann": ann,
        "ann_engine": ann_engine,
        "head_neighbors_k": head_neighbors_k,
        "tau_sub": tau_sub,
        "build_head_neighbors": build_head_neighbors_flag,
        "n_recipes": len(recipes),
    })


def _vocab(items_per_recipe: Iterable[Iterable[str]]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for items in items_per_recipe:
        for item in items:
            if item not in vocab:
                vocab[item] = len(vocab)
    return vocab


def _ids(names: Iterable[str], vocab: Dict[str, int]) -> List[int]:
    return sorted({vocab[name] for name in names})


def _idf(ids_per_recipe: List[List[int]], vocab_size: int) -> np.ndarray:
    if vocab_size == 0:
        return np.ones((0,), dtype=np.float32)
    df = Counter()
    for ids in ids_per_recipe:
        df.update(set(ids))
    n = max(1, len(ids_per_recipe))
    return np.array([math.log(1.0 + n / (1.0 + df.get(i, 0))) for i in range(vocab_size)], dtype=np.float32)


def _postings(ids_per_recipe: List[List[int]]) -> Dict[str, List[int]]:
    postings: dict[int, list[int]] = defaultdict(list)
    for idx, ids in enumerate(ids_per_recipe):
        for item in ids:
            postings[item].append(idx)
    return {str(k): v for k, v in sorted(postings.items())}


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
