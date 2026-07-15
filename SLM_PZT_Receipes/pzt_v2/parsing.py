from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ParseAudit:
    has_input: bool
    has_title: bool
    has_ingredients_block: bool
    has_instructions_block: bool
    ingredient_count: int
    instruction_count: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ParsedRecipe:
    recipe_id: str
    title: str
    raw_ingredients: List[str]
    raw_instructions: List[str]
    parse_audit: ParseAudit
    raw_input: str
    source_line_number: int = 0


class RecipeParseError(ValueError):
    def __init__(self, message: str, audit: ParseAudit):
        super().__init__(message)
        self.audit = audit


_INGREDIENTS_RE = re.compile(r"^\s*ingredients\s*:\s*$", re.IGNORECASE)
_INSTRUCTIONS_RE = re.compile(r"^\s*(instructions|directions)\s*:\s*$", re.IGNORECASE)


def parse_ndjson(path: Path, *, skip_invalid: bool = False, limit: int | None = None) -> List[ParsedRecipe]:
    recipes: List[ParsedRecipe] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if limit is not None and len(recipes) >= limit:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                recipes.append(parse_recipe_object(obj, source_line_number=line_number))
            except Exception as exc:
                if skip_invalid:
                    continue
                if isinstance(exc, RecipeParseError):
                    raise
                audit = ParseAudit(False, False, False, False, 0, 0, errors=[str(exc)])
                raise RecipeParseError(f"line {line_number}: invalid NDJSON recipe: {exc}", audit) from exc
    return recipes


def parse_recipe_object(obj: Dict[str, Any], *, source_line_number: int = 0) -> ParsedRecipe:
    input_value = obj.get("input")
    if not isinstance(input_value, str) or not input_value.strip():
        audit = ParseAudit(
            has_input=False,
            has_title=False,
            has_ingredients_block=False,
            has_instructions_block=False,
            ingredient_count=0,
            instruction_count=0,
            errors=["missing_input"],
        )
        raise RecipeParseError("recipe is missing a non-empty input field", audit)

    raw_input = input_value.replace("\r\n", "\n").replace("\r", "\n")
    title, raw_ingredients, raw_instructions, audit = parse_recipe_blocks(raw_input)
    if not audit.is_valid:
        raise RecipeParseError("; ".join(audit.errors), audit)

    recipe_id = _recipe_id(obj, raw_input)
    return ParsedRecipe(
        recipe_id=recipe_id,
        title=title,
        raw_ingredients=raw_ingredients,
        raw_instructions=raw_instructions,
        parse_audit=audit,
        raw_input=raw_input,
        source_line_number=source_line_number,
    )


def parse_recipe_blocks(raw_input: str) -> tuple[str, List[str], List[str], ParseAudit]:
    raw_lines = raw_input.splitlines()
    non_empty = [(idx, line.strip()) for idx, line in enumerate(raw_lines) if line.strip()]
    title = ""
    ingredients: List[str] = []
    instructions: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []

    if non_empty:
        title = non_empty[0][1]
    else:
        errors.append("missing_title")

    section = "title"
    has_ingredients = False
    has_instructions = False
    for line in raw_lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if _INGREDIENTS_RE.match(stripped):
            section = "ingredients"
            has_ingredients = True
            continue
        if _INSTRUCTIONS_RE.match(stripped):
            section = "instructions"
            has_instructions = True
            continue
        if section == "ingredients":
            ingredients.append(stripped)
        elif section == "instructions":
            instructions.append(stripped)
        else:
            warnings.append(f"ignored_line_before_first_section:{stripped}")

    has_title = bool(title)
    if not has_title and "missing_title" not in errors:
        errors.append("missing_title")
    if not has_ingredients:
        errors.append("missing_ingredients_block")
    if not has_instructions:
        errors.append("missing_instructions_block")
    if has_ingredients and not ingredients:
        errors.append("empty_ingredients_block")
    if has_instructions and not instructions:
        errors.append("empty_instructions_block")

    audit = ParseAudit(
        has_input=bool(raw_input.strip()),
        has_title=has_title,
        has_ingredients_block=has_ingredients,
        has_instructions_block=has_instructions,
        ingredient_count=len(ingredients),
        instruction_count=len(instructions),
        warnings=warnings,
        errors=errors,
    )
    return title, ingredients, instructions, audit


def _recipe_id(obj: Dict[str, Any], raw_input: str) -> str:
    for key in ("id", "_id", "recipe_id"):
        value = obj.get(key)
        if value:
            return str(value)
    normalized = "\n".join(line.rstrip() for line in raw_input.splitlines()).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
