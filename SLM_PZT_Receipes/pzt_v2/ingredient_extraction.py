from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .extraction_schema import IngredientCandidate


_QUANTITY_RE = re.compile(
    r"^\s*(?P<quantity>\d+\s*[- ]\s*\d+[\\\/]\d+|\d+[\\\/]\d+|[¼½¾]|\.\d+|\d+(?:\.\d+)?)"
    r"(?=\s|[A-Za-z]|$)"
)
_UNITS = {
    "c", "c.", "cup", "cups",
    "t", "t.", "tsp", "tsp.", "teaspoon", "teaspoons",
    "tbsp", "tbsp.", "T", "T.", "tablespoon", "tablespoons",
    "oz", "oz.", "ounce", "ounces",
    "lb", "lb.", "lbs", "pound", "pounds",
    "g", "kg", "ml", "l", "liter", "liters",
    "pinch", "pinches", "dash", "dashes",
    "clove", "cloves", "can", "cans",
    "package", "packages", "pkg", "pkg.",
    "pt", "pt.", "pint", "pints", "qt", "qt.", "qts", "qts.", "quart", "quarts",
}
_UNIT_RE = re.compile(
    r"^\s*(?P<unit>tablespoons?|teaspoons?|packages?|pinches|dashes|cloves|ounces?|pounds?|"
    r"quarts?|pints?|cups?|tbsp\.?|tsp\.?|pkg\.?|qts?\.?|pts?\.?|oz\.?|lb\.?|"
    r"c\.?|t\.?|T\.?|kg|ml|g|l|can|cans|dash|pinch)"
    r"(?=\s|$)",
    re.IGNORECASE,
)
_PREPARATION_STATES = {
    "chopped", "diced", "beaten", "melted", "minced", "sliced", "grated", "shredded",
    "crushed", "mashed", "softened", "drained", "peeled", "roasted", "toasted",
    "whisked", "boiled", "cooked", "baked", "fried", "steamed", "simmered", "mixed",
    "kneaded", "risen", "cooled",
}
_DESCRIPTORS = {
    "large", "small", "medium", "fresh", "ripe", "cold", "warm", "hot", "dry", "fine",
    "finely", "thinly", "roughly", "ground",
}
_FORM_WORDS = {
    "whole", "halved", "wedges", "strips", "cubes", "pieces", "slices",
}


def extract_ingredient_candidate(raw: str) -> IngredientCandidate:
    original = raw.strip()
    warnings: List[str] = []
    if not original:
        return IngredientCandidate(
            raw=raw,
            quantity=None,
            unit=None,
            base_object_candidate=None,
            confidence=0.0,
            warnings=["empty_ingredient_line"],
        )

    working = _prepare_leading_measurement_text(_strip_leading_list_marker(original))
    quantity, working = _take_quantity(working)
    unit, working = _take_unit(working)

    working = _strip_parentheticals(working)
    before_comma, comma_modifiers = _split_comma_modifiers(working)
    tokens = _word_tokens(before_comma)
    modifier_tokens = _word_tokens(" ".join(comma_modifiers))

    preparation_states: List[str] = []
    descriptors: List[str] = []
    forms: List[str] = []
    base_tokens: List[str] = []

    for token in tokens:
        _classify_token(token, preparation_states, descriptors, forms, base_tokens)
    for token in modifier_tokens:
        _classify_token(token, preparation_states, descriptors, forms, base_tokens=None)

    base_object = " ".join(base_tokens).strip() or None
    if base_object is None:
        warnings.append("missing_base_object_candidate")

    confidence = 0.95 if base_object else 0.35
    return IngredientCandidate(
        raw=original,
        quantity=quantity,
        unit=unit,
        base_object_candidate=base_object,
        preparation_state_candidates=_dedupe(preparation_states),
        descriptors=_dedupe(descriptors),
        form_candidates=_dedupe(forms),
        confidence=confidence,
        warnings=warnings,
    )


def extract_ingredient_candidates(raw_lines: List[str]) -> List[IngredientCandidate]:
    return [extract_ingredient_candidate(line) for line in raw_lines]


def _take_quantity(text: str) -> Tuple[Optional[str], str]:
    match = _QUANTITY_RE.match(text)
    if not match:
        return None, text
    quantity = " ".join(match.group("quantity").split())
    quantity = re.sub(r"\s*-\s*", "-", quantity)
    return quantity, text[match.end():].strip()


def _strip_leading_list_marker(text: str) -> str:
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, count=1)


def _prepare_leading_measurement_text(text: str) -> str:
    return re.sub(r"\b(c|t|tsp|tbsp|oz|lb|pt|qt|qts|pkg)\.([A-Za-z])", r"\1. \2", text, count=1, flags=re.IGNORECASE)


def _take_unit(text: str) -> Tuple[Optional[str], str]:
    match = _UNIT_RE.match(text)
    if not match:
        return None, text
    unit = match.group("unit")
    if unit not in _UNITS and unit.lower() not in _UNITS:
        return None, text
    return unit, text[match.end():].strip()


def _strip_parentheticals(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text).strip()


def _split_comma_modifiers(text: str) -> tuple[str, List[str]]:
    parts = [part.strip() for part in text.split(",")]
    return parts[0], parts[1:]


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", text.lower())


def _classify_token(
    token: str,
    preparation_states: List[str],
    descriptors: List[str],
    forms: List[str],
    base_tokens: Optional[List[str]],
) -> None:
    if token in _PREPARATION_STATES:
        preparation_states.append(token)
    elif token in _DESCRIPTORS:
        descriptors.append(token)
    elif token in _FORM_WORDS:
        forms.append(token)
    elif base_tokens is not None:
        base_tokens.append(token)


def _dedupe(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
