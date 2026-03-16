import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


def read_ndjson(path: Path, limit: int) -> List[Dict[str, Any]]:
    docs = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                obj = json.loads(line)
                docs.append(obj)
            except Exception:
                # Skip malformed lines
                continue
    return docs


def split_recipe(text: str) -> Tuple[str, List[str], List[str]]:
    text = (text or '').strip()
    if not text:
        return '', [], []

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    title = parts[0] if parts else ''

    ingredients_section = parts[1] if len(parts) > 1 else ''
    instructions_section = parts[2] if len(parts) > 2 else ''

    low_ing = ingredients_section.lower()
    if low_ing.startswith('ingredients:'):
        ingredients_section = ingredients_section[len('ingredients:'):].strip()

    low_instr = instructions_section.lower()
    if low_instr.startswith('directions:'):
        instructions_section = instructions_section[len('directions:'):].strip()
    elif low_instr.startswith('instructions:'):
        instructions_section = instructions_section[len('instructions:'):].strip()

    ingredients_lines = [l.strip() for l in ingredients_section.split('\n') if l.strip()]
    instructions_lines = [l.strip() for l in instructions_section.split('\n') if l.strip()]
    return title, ingredients_lines, instructions_lines


def compute_embeddings(
    model: SentenceTransformer,
    documents: List[Dict[str, Any]],
) -> Tuple[List[str], torch.Tensor, List[Dict[str, Any]]]:
    titles: List[str] = []
    full_texts: List[str] = []
    ingredients_full: List[str] = []
    instructions_full: List[str] = []
    ingredients_lines_all: List[List[str]] = []
    instructions_lines_all: List[List[str]] = []

    for obj in documents:
        full = obj.get('input', '')
        title, ingr_lines, instr_lines = split_recipe(full)
        titles.append(title)
        full_texts.append(full)
        ingredients_full.append('\n'.join(ingr_lines))
        instructions_full.append('\n'.join(instr_lines))
        ingredients_lines_all.append(ingr_lines)
        instructions_lines_all.append(instr_lines)

    # Encode full recipe texts as tensors (GPU if available)
    full_embeddings = model.encode(full_texts, convert_to_tensor=True, show_progress_bar=True)

    # Encode full ingredients/instructions (may be empty strings)
    ingr_full_emb = model.encode(ingredients_full, convert_to_tensor=True, show_progress_bar=False)
    instr_full_emb = model.encode(instructions_full, convert_to_tensor=True, show_progress_bar=False)

    # Encode line-level embeddings
    ingr_lines_emb: List[List[torch.Tensor]] = []
    instr_lines_emb: List[List[torch.Tensor]] = []

    for ingr_lines in ingredients_lines_all:
        if ingr_lines:
            embs = model.encode(ingr_lines, convert_to_tensor=True, show_progress_bar=False)
            # Ensure list of tensors per line
            if isinstance(embs, torch.Tensor) and embs.ndim == 2:
                ingr_lines_emb.append([embs[i] for i in range(embs.size(0))])
            else:
                ingr_lines_emb.append([])
        else:
            ingr_lines_emb.append([])

    for instr_lines in instructions_lines_all:
        if instr_lines:
            embs = model.encode(instr_lines, convert_to_tensor=True, show_progress_bar=False)
            if isinstance(embs, torch.Tensor) and embs.ndim == 2:
                instr_lines_emb.append([embs[i] for i in range(embs.size(0))])
            else:
                instr_lines_emb.append([])
        else:
            instr_lines_emb.append([])

    per_doc: List[Dict[str, Any]] = []
    for i in range(len(documents)):
        per_doc.append({
            'title': titles[i],
            'ingredients': {
                'full_emb': ingr_full_emb[i],
                'lines': ingr_lines_emb[i],
            },
            'instructions': {
                'full_emb': instr_full_emb[i],
                'lines': instr_lines_emb[i],
            },
        })

    return titles, full_embeddings, per_doc


def build_similarity_matrix(full_embeddings: torch.Tensor) -> np.ndarray:
    # cosine similarity on-device, then bring to CPU numpy
    sim = util.cos_sim(full_embeddings, full_embeddings)  # [N, N]
    return sim.detach().cpu().numpy()


def compute_transferability(doc_i: Dict[str, Any], doc_j: Dict[str, Any], alpha: float, beta: float) -> float:
    ingr_sim = None
    instr_sim = None

    ingr_i = doc_i['ingredients']['full_emb']
    ingr_j = doc_j['ingredients']['full_emb']
    if isinstance(ingr_i, torch.Tensor) and isinstance(ingr_j, torch.Tensor) and ingr_i.numel() > 0 and ingr_j.numel() > 0:
        ingr_sim = util.cos_sim(ingr_i, ingr_j).item()

    instr_i = doc_i['instructions']['full_emb']
    instr_j = doc_j['instructions']['full_emb']
    if isinstance(instr_i, torch.Tensor) and isinstance(instr_j, torch.Tensor) and instr_i.numel() > 0 and instr_j.numel() > 0:
        instr_sim = util.cos_sim(instr_i, instr_j).item()

    # If both missing, fall back to 0
    if ingr_sim is None and instr_sim is None:
        return 0.0

    candidates = []
    if instr_sim is not None:
        candidates.append(alpha * instr_sim)
    if ingr_sim is not None:
        candidates.append(beta * ingr_sim)
    return float(max(candidates)) if candidates else 0.0


def jaccard_line_level(doc_i: Dict[str, Any], doc_j: Dict[str, Any], threshold: float) -> Tuple[float, float, float]:
    # Ingredients
    ingr_i = doc_i['ingredients']['lines']
    ingr_j = doc_j['ingredients']['lines']
    inter_ingr = 0
    for e1 in ingr_i:
        matched = False
        for e2 in ingr_j:
            if util.cos_sim(e1, e2).item() >= threshold:
                matched = True
                break
        if matched:
            inter_ingr += 1
    union_ingr = len(ingr_i) + len(ingr_j) - inter_ingr
    j_ingr = inter_ingr / union_ingr if union_ingr > 0 else 0.0

    # Instructions
    instr_i = doc_i['instructions']['lines']
    instr_j = doc_j['instructions']['lines']
    inter_instr = 0
    for e1 in instr_i:
        matched = False
        for e2 in instr_j:
            if util.cos_sim(e1, e2).item() >= threshold:
                matched = True
                break
        if matched:
            inter_instr += 1
    union_instr = len(instr_i) + len(instr_j) - inter_instr
    j_instr = inter_instr / union_instr if union_instr > 0 else 0.0

    overall = (j_ingr + j_instr) / 2.0
    return j_ingr, j_instr, overall


def main():
    p = argparse.ArgumentParser(description='Build similarity matrix and pairwise scores with Sentence-Transformers (MiniLM).')
    p.add_argument('--ndjson', type=Path, default=Path('SLM_PZT_Receipes/all_recipes.ndjson'), help='Path to NDJSON file with an "input" field per line.')
    p.add_argument('--limit', type=int, default=500, help='Max number of recipes to process.')
    p.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='Sentence-Transformer model name.')
    p.add_argument('--alpha', type=float, default=1.0, help='Weight for instructions similarity in transferability.')
    p.add_argument('--beta', type=float, default=1.0, help='Weight for ingredients similarity in transferability.')
    p.add_argument('--jaccard-threshold', type=float, default=0.8, help='Cosine threshold for line-level Jaccard matches.')
    p.add_argument('--top-k', type=int, default=50, help='How many top pairs to save for each metric.')
    p.add_argument('--out-prefix', type=Path, default=Path('SLM_PZT_Receipes/output'), help='Output prefix (directory or file stem).')

    args = p.parse_args()

    out_dir = args.out_prefix if args.out_prefix.suffix == '' else args.out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model: {args.model} ...')
    model = SentenceTransformer(args.model)
    print('Model loaded.')

    print(f'Reading NDJSON: {args.ndjson} (limit={args.limit}) ...')
    raw_docs = read_ndjson(args.ndjson, args.limit)
    print(f'Loaded {len(raw_docs)} recipes.')

    print('Encoding documents and sections ...')
    titles, full_emb, per_doc = compute_embeddings(model, raw_docs)
    print('Encodings done.')

    print('Building full similarity matrix ...')
    sim_mat = build_similarity_matrix(full_emb)
    sim_path = out_dir / 'similarity_matrix_st.csv'
    np.savetxt(sim_path, sim_mat, delimiter=',')
    print(f'Saved similarity matrix to {sim_path}')

    print('Computing pairwise scores (transferability, Jaccard) ...')
    n = len(per_doc)
    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            T = compute_transferability(per_doc[i], per_doc[j], args.alpha, args.beta)
            j_ingr, j_instr, j_overall = jaccard_line_level(per_doc[i], per_doc[j], args.jaccard_threshold)
            pair_rows.append({
                'i': i,
                'j': j,
                'title_i': titles[i],
                'title_j': titles[j],
                'transferability': T,
                'jaccard_ingredients': j_ingr,
                'jaccard_instructions': j_instr,
                'jaccard_overall': j_overall,
            })

    # Save top-k by transferability
    pair_rows_sorted_T = sorted(pair_rows, key=lambda r: r['transferability'], reverse=True)[: args.top_k]
    out_T = out_dir / 'top_pairs_by_transferability.csv'
    if pair_rows_sorted_T:
        import csv
        with out_T.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows_sorted_T[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows_sorted_T)
        print(f'Saved top {len(pair_rows_sorted_T)} pairs by transferability to {out_T}')
    else:
        print('No pair rows to save for transferability.')

    # Save top-k by Jaccard overall
    pair_rows_sorted_J = sorted(pair_rows, key=lambda r: r['jaccard_overall'], reverse=True)[: args.top_k]
    out_J = out_dir / 'top_pairs_by_jaccard.csv'
    if pair_rows_sorted_J:
        import csv
        with out_J.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows_sorted_J[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows_sorted_J)
        print(f'Saved top {len(pair_rows_sorted_J)} pairs by Jaccard to {out_J}')
    else:
        print('No pair rows to save for Jaccard.')

    print('Done.')


if __name__ == '__main__':
    main()

