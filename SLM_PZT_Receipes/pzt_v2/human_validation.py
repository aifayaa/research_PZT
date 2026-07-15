from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


HUMAN_SCALE = {
    1: "Pas transférable : une nouvelle procédure est nécessaire.",
    2: "Très peu d'éléments de la source sont réutilisables.",
    3: "Transfert partiel avec plusieurs changements majeurs.",
    4: "Transfert raisonnable avec plusieurs adaptations explicites.",
    5: "Très transférable avec quelques changements localisés.",
    6: "Transformation presque directe avec un effort cognitif minimal.",
}


def select_blind_human_pairs(
    pair_scores_path: Path,
    *,
    distinct_count: int = 50,
    deciles: int = 10,
    repeats: int = 5,
    seed: str = "pzt-human-v1",
) -> tuple[List[dict], List[dict]]:
    records = [json.loads(line) for line in pair_scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) < distinct_count:
        raise ValueError(f"need at least {distinct_count} scored pairs; got {len(records)}")
    records.sort(key=lambda row: (row["transferability"]["raw_transferability"], row["pair_id"]))
    rng = random.Random(seed)
    per_decile = distinct_count // deciles
    remainder = distinct_count % deciles
    selected: List[dict] = []
    for decile in range(deciles):
        start = int(decile * len(records) / deciles)
        end = int((decile + 1) * len(records) / deciles)
        bucket = list(records[start:end])
        rng.shuffle(bucket)
        take = per_decile + (1 if decile < remainder else 0)
        for record in bucket[:take]:
            selected.append({**record, "selection_decile": decile + 1})
    rng.shuffle(selected)

    blind_rows: List[dict] = []
    key_rows: List[dict] = []
    for index, record in enumerate(selected, start=1):
        presentation_id = f"P{index:03d}"
        blind_rows.append(_blind_row(record, presentation_id))
        key_rows.append(_key_row(record, presentation_id, duplicate_of=""))
    repeated = rng.sample(selected, min(repeats, len(selected)))
    for offset, record in enumerate(repeated, start=1):
        presentation_id = f"R{offset:03d}"
        original = next(row["presentation_id"] for row in key_rows if row["pair_id"] == record["pair_id"])
        blind_rows.append(_blind_row(record, presentation_id))
        key_rows.append(_key_row(record, presentation_id, duplicate_of=original))
    rng.shuffle(blind_rows)
    return blind_rows, key_rows


def write_human_package(blind_rows: Sequence[dict], key_rows: Sequence[dict], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "human_scores_blind.csv"
    key_path = output_dir / "human_scores_key.csv"
    rubric_path = output_dir / "human_scoring_rubric.json"
    _write_csv(blind_path, blind_rows)
    _write_csv(key_path, key_rows)
    rubric_path.write_text(
        json.dumps(
            {
                "question": "En connaissant la recette source, à quel point est-elle transférable vers la recette cible ?",
                "directional": True,
                "scale": HUMAN_SCALE,
                "instructions": [
                    "Ne pas consulter les scores du modèle.",
                    "Mettre source_known=no si la recette source n'est pas suffisamment connue.",
                    "Donner une note entière de 1 à 6 uniquement si source_known=yes.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return [blind_path, key_path, rubric_path]


def _blind_row(record: dict, presentation_id: str) -> dict:
    return {
        "presentation_id": presentation_id,
        "source_title": record["source"]["title"],
        "source_ingredients": "\n".join(record["source"]["ingredients"]),
        "source_instructions": "\n".join(record["source"]["instructions"]),
        "target_title": record["target"]["title"],
        "target_ingredients": "\n".join(record["target"]["ingredients"]),
        "target_instructions": "\n".join(record["target"]["instructions"]),
        "source_known_yes_no": "",
        "human_score_1_to_6": "",
        "notes": "",
    }


def _key_row(record: dict, presentation_id: str, *, duplicate_of: str) -> dict:
    return {
        "presentation_id": presentation_id,
        "pair_id": record["pair_id"],
        "source_recipe_id": record["source"]["recipe_id"],
        "target_recipe_id": record["target"]["recipe_id"],
        "selection_decile": record["selection_decile"],
        "raw_transferability": record["transferability"]["raw_transferability"],
        "pzt_score": record["pzt"]["pzt_score"],
        "duplicate_of": duplicate_of,
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
