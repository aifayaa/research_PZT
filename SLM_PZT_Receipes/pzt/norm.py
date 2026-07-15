from __future__ import annotations

import re
from typing import Iterable, List


_UNITS = {
    "c", "cup", "cups", "tsp", "teaspoon", "teaspoons", "tbsp", "tablespoon", "tablespoons",
    "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "g", "kg", "ml", "l",
    "can", "cans", "jar", "jars", "package", "packages", "pkg", "box", "boxes",
}
_PREP = {
    "chopped", "minced", "sliced", "diced", "drained", "fresh", "ground", "grated", "shredded",
    "beaten", "melted", "softened", "crushed", "optional", "taste", "large", "small", "medium",
}
_STOP = {"of", "and", "with", "in", "for", "the", "a", "an", "to"}
_MULTI_HEADS = [
    "evaporated milk", "condensed milk", "mushroom soup", "chicken soup", "tomato soup",
    "bell pepper", "olive oil", "brown sugar", "cream cheese", "sour cream", "peanut butter",
]
_CANONICAL = [
    (re.compile(r"\bgarbanzo(?: beans?)?\b", re.I), "chickpea"),
    (re.compile(r"\bchickpeas?\b", re.I), "chickpea"),
    (re.compile(r"\bcapsicum\b", re.I), "bell pepper"),
    (re.compile(r"\bcheddar cheese\b", re.I), "cheddar"),
    (re.compile(r"\bmozzarella cheese\b", re.I), "mozzarella"),
    (re.compile(r"\ball[- ]purpose flour\b", re.I), "flour"),
]


def normalize_ingredient(line: str) -> str:
    text = line.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b\d+(?:[\/\.\-]\d+)?\b", " ", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = text.replace("-", " ")
    for pattern, replacement in _CANONICAL:
        text = pattern.sub(replacement, text)
    tokens = [t for t in text.split() if t not in _UNITS and t not in _PREP]
    return " ".join(tokens).strip()


def ingredient_head(line: str) -> str:
    text = normalize_ingredient(line)
    if not text:
        return "unknown"
    for head in _MULTI_HEADS:
        if re.search(rf"\b{re.escape(head)}\b", text):
            return "sugar" if head == "brown sugar" else head
    tokens = [t for t in text.split() if t not in _STOP]
    if not tokens:
        return "unknown"
    return _singularize(tokens[-1])


def extract_heads(lines: Iterable[str]) -> List[str]:
    heads = sorted({ingredient_head(line) for line in lines if ingredient_head(line) != "unknown"})
    return heads


def normalized_ingredients_text(lines: Iterable[str]) -> str:
    return "\n".join(normalize_ingredient(line) for line in lines)


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
