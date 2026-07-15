from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class ParsedRecipe:
    recipe_id: str
    title: str
    ingredients_lines_raw: List[str]
    instructions_lines_raw: List[str]
    full_text: str
    parse_warnings: List[str] = field(default_factory=list)
    source_line_number: int = 0


_SECTION_RE = re.compile(r"^\s*(ingredients|instructions|directions)\s*:\s*$", re.I)


def _clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[\-\*\u2022]\s+", "", line)
    line = re.sub(r"^(?:step\s*)?\d+[\.\)]\s*", "", line, flags=re.I)
    return line.strip()


def _recipe_id(obj: Dict[str, Any], full_text: str) -> str:
    for key in ("id", "_id", "recipe_id"):
        val = obj.get(key)
        if val:
            return str(val)
    normalized = "\n".join(line.rstrip() for line in full_text.replace("\r\n", "\n").splitlines()).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def parse_recipe_object(obj: Dict[str, Any], line_number: int) -> ParsedRecipe:
    full_text = str(obj.get("input", "")).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not full_text:
        raise ValueError(f"line {line_number}: missing non-empty input field")

    raw_lines = full_text.splitlines()
    title = next((line.strip() for line in raw_lines if line.strip()), "")
    section = "title"
    ingredients: List[str] = []
    instructions: List[str] = []
    warnings: List[str] = []

    for raw in raw_lines[1:]:
        match = _SECTION_RE.match(raw)
        if match:
            name = match.group(1).lower()
            section = "instructions" if name in {"instructions", "directions"} else "ingredients"
            continue
        cleaned = _clean_line(raw)
        if not cleaned:
            continue
        if section == "ingredients":
            ingredients.append(cleaned)
        elif section == "instructions":
            instructions.append(cleaned)

    if not ingredients:
        warnings.append("missing_ingredients")
    if not instructions:
        warnings.append("missing_instructions")
    if not ingredients or not instructions:
        raise ValueError(f"line {line_number}: recipe must contain Ingredients and Instructions sections")

    return ParsedRecipe(
        recipe_id=_recipe_id(obj, full_text),
        title=title,
        ingredients_lines_raw=ingredients,
        instructions_lines_raw=instructions,
        full_text=full_text,
        parse_warnings=warnings,
        source_line_number=line_number,
    )


def parse_ndjson(path: Path, *, strict: bool = True, skip_bad: bool = False, limit: int | None = None) -> Tuple[List[ParsedRecipe], Dict[str, Any]]:
    recipes: List[ParsedRecipe] = []
    errors: List[Dict[str, Any]] = []
    total = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if limit is not None and total >= limit:
                break
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
                recipes.append(parse_recipe_object(obj, line_number))
            except Exception as exc:
                errors.append({"line": line_number, "error": str(exc)})
                if strict and not skip_bad:
                    raise

    audit = {
        "total_lines": total,
        "parsed_success": len(recipes),
        "skipped": len(errors),
        "missing_ingredients": sum(1 for e in errors if "Ingredients" in e.get("error", "")),
        "missing_instructions": sum(1 for e in errors if "Instructions" in e.get("error", "")),
        "errors": errors[:100],
        "ingredients_line_count_hist": _hist(len(r.ingredients_lines_raw) for r in recipes),
        "instructions_line_count_hist": _hist(len(r.instructions_lines_raw) for r in recipes),
    }
    return recipes, audit


def _hist(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out
