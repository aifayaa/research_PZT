from __future__ import annotations

import re
from typing import Iterable, List


_MULTI = {
    "bring_to_boil": re.compile(r"\bbring(?:s|ing)?\s+(?:it\s+)?to\s+(?:a\s+)?boil\b", re.I),
    "preheat_oven": re.compile(r"\bpreheat(?:s|ed|ing)?\s+(?:the\s+)?oven\b", re.I),
}
_SINGLE = {
    "bake": r"bake|baked|baking",
    "boil": r"boil|boiled|boiling",
    "mix": r"mix|mixed|mixing|combine|combined|stir|stirred|stirring|blend|blended",
    "simmer": r"simmer|simmered|simmering",
    "chop": r"chop|chopped|chopping",
    "fold": r"fold|folded|folding",
    "fry": r"fry|fried|frying|saute|sauté|sauteed",
    "dice": r"dice|diced|dicing",
    "marinate": r"marinate|marinated|marinating",
    "grill": r"grill|grilled|grilling",
    "roast": r"roast|roasted|roasting",
    "serve": r"serve|served|serving",
    "chill": r"chill|chilled|chilling|refrigerate|refrigerated",
}


def extract_ops(lines: Iterable[str]) -> List[str]:
    text = " ".join(lines).lower()
    ops = set()
    for name, pattern in _MULTI.items():
        if pattern.search(text):
            ops.add(name)
    for name, words in _SINGLE.items():
        if re.search(rf"\b(?:{words})\b", text, flags=re.I):
            ops.add(name)
    return sorted(ops)
