from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

from .semantic_schema import CanonicalConcept


def normalize_alias(value: str) -> str:
    normalized = value.strip().lower().replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class SemanticMemory:
    def __init__(self) -> None:
        self._by_kind_alias: Dict[str, Dict[str, CanonicalConcept]] = {
            "ingredient": {},
            "operation": {},
            "state": {},
            "form": {},
            "descriptor": {},
        }
        self._register_defaults()

    def lookup(self, kind: str, raw: str | None) -> Optional[CanonicalConcept]:
        if not raw:
            return None
        return self._by_kind_alias.get(kind, {}).get(normalize_alias(raw))

    def lookup_ingredient(self, raw: str | None) -> Optional[CanonicalConcept]:
        return self.lookup("ingredient", raw)

    def lookup_operation(self, raw: str | None) -> Optional[CanonicalConcept]:
        return self.lookup("operation", raw)

    def lookup_state(self, raw: str | None) -> Optional[CanonicalConcept]:
        return self.lookup("state", raw)

    def _register(self, *, kind: str, canonical_id: str, label: str, aliases: Iterable[str], metadata: dict | None = None) -> None:
        concept = CanonicalConcept(
            canonical_id=canonical_id,
            kind=kind,
            label=label,
            aliases=[normalize_alias(alias) for alias in aliases],
            metadata=dict(metadata or {}),
        )
        for alias in concept.aliases:
            self._by_kind_alias[kind][alias] = concept

    def _register_defaults(self) -> None:
        self._register(kind="ingredient", canonical_id="ING_TOMATO", label="tomato", aliases=["tomato", "tomatoes"])
        self._register(kind="ingredient", canonical_id="ING_ZUCCHINI", label="zucchini", aliases=["zucchini", "courgette"])
        self._register(kind="ingredient", canonical_id="ING_BANANA", label="banana", aliases=["banana", "bananas"])
        self._register(kind="ingredient", canonical_id="ING_FLOUR", label="flour", aliases=["flour"])
        self._register(kind="ingredient", canonical_id="ING_EGG", label="egg", aliases=["egg", "eggs"])
        self._register(kind="ingredient", canonical_id="ING_SUGAR", label="sugar", aliases=["sugar"])
        self._register(kind="ingredient", canonical_id="ING_CHICKEN", label="chicken", aliases=["chicken"])
        self._register(kind="ingredient", canonical_id="ING_TOFU", label="tofu", aliases=["tofu"])
        self._register(kind="ingredient", canonical_id="ING_VEGETABLE", label="vegetable", aliases=["vegetable", "vegetables"])
        self._register(kind="ingredient", canonical_id="ING_SAUCE", label="sauce", aliases=["sauce"])
        self._register(kind="ingredient", canonical_id="ING_ONION", label="onion", aliases=["onion", "onions"])
        self._register(kind="ingredient", canonical_id="ING_BUTTER", label="butter", aliases=["butter"])
        self._register(kind="ingredient", canonical_id="ING_MILK", label="milk", aliases=["milk"])
        self._register(kind="ingredient", canonical_id="ING_CARROT", label="carrot", aliases=["carrot", "carrots"])
        self._register(kind="ingredient", canonical_id="ING_CREAM", label="cream", aliases=["cream"])
        self._register(kind="ingredient", canonical_id="ING_FISH", label="fish", aliases=["fish"])
        self._register(kind="ingredient", canonical_id="ING_RICE", label="rice", aliases=["rice"])
        self._register(kind="ingredient", canonical_id="ING_PASTA", label="pasta", aliases=["pasta", "spaghetti", "noodles"])
        self._register(kind="ingredient", canonical_id="ING_POTATO", label="potato", aliases=["potato", "potatoes"])
        self._register(kind="ingredient", canonical_id="ING_GARLIC", label="garlic", aliases=["garlic", "garlic clove", "garlic cloves"])
        self._register(kind="ingredient", canonical_id="ING_MUSHROOM", label="mushroom", aliases=["mushroom", "mushrooms"])
        self._register(kind="ingredient", canonical_id="ING_LENTIL", label="lentil", aliases=["lentil", "lentils"])
        self._register(kind="ingredient", canonical_id="ING_CHEESE", label="cheese", aliases=["cheese"])
        self._register(kind="ingredient", canonical_id="ING_OIL", label="oil", aliases=["oil", "olive oil", "vegetable oil"])
        self._register(kind="ingredient", canonical_id="ING_VINEGAR", label="vinegar", aliases=["vinegar"])
        self._register(kind="ingredient", canonical_id="ING_SALT", label="salt", aliases=["salt"])
        self._register(kind="ingredient", canonical_id="ING_PEPPER", label="pepper", aliases=["pepper", "black pepper"])
        self._register(kind="ingredient", canonical_id="ING_STOCK", label="stock", aliases=["stock", "vegetable stock", "chicken stock", "broth"])
        self._register(kind="ingredient", canonical_id="ING_CHOCOLATE", label="chocolate", aliases=["chocolate"])
        self._register(kind="ingredient", canonical_id="ING_COCOA", label="cocoa", aliases=["cocoa", "cocoa powder"])
        self._register(kind="ingredient", canonical_id="ING_YEAST", label="yeast", aliases=["yeast"])
        self._register(kind="ingredient", canonical_id="ING_WATER", label="water", aliases=["water"])
        self._register(kind="ingredient", canonical_id="ING_BEAN", label="bean", aliases=["bean", "beans", "green beans"])
        self._register(kind="ingredient", canonical_id="ING_CABBAGE", label="cabbage", aliases=["cabbage"])
        self._register(kind="ingredient", canonical_id="ING_LETTUCE", label="lettuce", aliases=["lettuce"])
        self._register(kind="ingredient", canonical_id="ING_AVOCADO", label="avocado", aliases=["avocado", "avocados"])
        self._register(kind="ingredient", canonical_id="ING_LEMON", label="lemon", aliases=["lemon", "lemon juice"])
        self._register(kind="ingredient", canonical_id="ING_VANILLA", label="vanilla", aliases=["vanilla"])
        self._register(kind="state", canonical_id="STATE_BATTER", label="batter", aliases=["batter"])

        self._register(kind="operation", canonical_id="OP_MIX", label="mix", aliases=["mix", "combine", "stir", "stir together", "incorporate"])
        self._register(kind="operation", canonical_id="OP_CHOP", label="chop", aliases=["chop", "chopped", "chopping"])
        self._register(kind="operation", canonical_id="OP_DICE", label="dice", aliases=["dice", "diced", "dicing"])
        self._register(kind="operation", canonical_id="OP_MASH", label="mash", aliases=["mash", "mashed"])
        self._register(kind="operation", canonical_id="OP_GRATE", label="grate", aliases=["grate", "grated"])
        self._register(kind="operation", canonical_id="OP_BAKE", label="bake", aliases=["bake", "baked", "baking"])
        self._register(kind="operation", canonical_id="OP_BOIL", label="boil", aliases=["boil", "boiled"])
        self._register(kind="operation", canonical_id="OP_BEAT", label="beat", aliases=["beat", "beaten"])
        self._register(kind="operation", canonical_id="OP_COOK", label="cook", aliases=["cook", "cooked"])
        self._register(kind="operation", canonical_id="OP_COOK_IN_PAN", label="cook in pan", aliases=["cook in pan", "pan cook", "cook them in a pan"])
        self._register(kind="operation", canonical_id="OP_STIR_FRY", label="stir-fry", aliases=["stir fry", "stir-fry", "fry"])
        self._register(kind="operation", canonical_id="OP_MELT", label="melt", aliases=["melt", "melted"])
        self._register(kind="operation", canonical_id="OP_SLICE", label="slice", aliases=["slice", "sliced"])
        self._register(kind="operation", canonical_id="OP_MINCE", label="mince", aliases=["mince", "minced"])
        self._register(kind="operation", canonical_id="OP_WHISK", label="whisk", aliases=["whisk", "whisked", "whisk together"])
        self._register(kind="operation", canonical_id="OP_FOLD", label="fold", aliases=["fold", "folded"])
        self._register(kind="operation", canonical_id="OP_SIMMER", label="simmer", aliases=["simmer", "simmered"])
        self._register(kind="operation", canonical_id="OP_STEAM", label="steam", aliases=["steam", "steamed"])
        self._register(kind="operation", canonical_id="OP_ROAST", label="roast", aliases=["roast", "roasted"])
        self._register(kind="operation", canonical_id="OP_SAUTE", label="saute", aliases=["saute", "sauté", "sauteed", "sautéed"])
        self._register(kind="operation", canonical_id="OP_KNEAD", label="knead", aliases=["knead", "kneaded"])
        self._register(kind="operation", canonical_id="OP_REST", label="rest", aliases=["rest", "rested"])
        self._register(kind="operation", canonical_id="OP_RISE", label="rise", aliases=["rise", "risen"])
        self._register(kind="operation", canonical_id="OP_POUR", label="pour", aliases=["pour", "poured"])
        self._register(kind="operation", canonical_id="OP_ADD", label="add", aliases=["add", "added"])
        self._register(kind="operation", canonical_id="OP_SEASON", label="season", aliases=["season", "seasoned"])
        self._register(kind="operation", canonical_id="OP_BLEND", label="blend", aliases=["blend", "blended"])
        self._register(kind="operation", canonical_id="OP_SERVE", label="serve", aliases=["serve", "served"])
        self._register(kind="operation", canonical_id="OP_DRAIN", label="drain", aliases=["drain", "drained"])
        self._register(kind="operation", canonical_id="OP_PEEL", label="peel", aliases=["peel", "peeled"])
        self._register(kind="operation", canonical_id="OP_CRUSH", label="crush", aliases=["crush", "crushed"])

        self._register(kind="state", canonical_id="STATE_CHOPPED", label="chopped", aliases=["chopped"], metadata={"implied_operation_id": "OP_CHOP"})
        self._register(kind="state", canonical_id="STATE_DICED", label="diced", aliases=["diced"], metadata={"implied_operation_id": "OP_DICE"})
        self._register(kind="state", canonical_id="STATE_MELTED", label="melted", aliases=["melted"], metadata={"implied_operation_id": "OP_MELT"})
        self._register(kind="state", canonical_id="STATE_BEATEN", label="beaten", aliases=["beaten"], metadata={"implied_operation_id": "OP_BEAT"})
        self._register(kind="state", canonical_id="STATE_MASHED", label="mashed", aliases=["mashed"], metadata={"implied_operation_id": "OP_MASH"})
        self._register(kind="state", canonical_id="STATE_GRATED", label="grated", aliases=["grated"], metadata={"implied_operation_id": "OP_GRATE"})
        self._register(kind="state", canonical_id="STATE_BAKED", label="baked", aliases=["baked"], metadata={"implied_operation_id": "OP_BAKE"})
        self._register(kind="state", canonical_id="STATE_BOILED", label="boiled", aliases=["boiled"], metadata={"implied_operation_id": "OP_BOIL"})
        self._register(kind="state", canonical_id="STATE_COOKED", label="cooked", aliases=["cooked"], metadata={"implied_operation_id": "OP_COOK"})
        self._register(kind="state", canonical_id="STATE_MIXTURE", label="mixture", aliases=["mixture"])
        self._register(kind="state", canonical_id="STATE_SLICED", label="sliced", aliases=["sliced", "thinly sliced"], metadata={"implied_operation_id": "OP_SLICE"})
        self._register(kind="state", canonical_id="STATE_MINCED", label="minced", aliases=["minced"], metadata={"implied_operation_id": "OP_MINCE"})
        self._register(kind="state", canonical_id="STATE_WHISKED", label="whisked", aliases=["whisked"], metadata={"implied_operation_id": "OP_WHISK"})
        self._register(kind="state", canonical_id="STATE_ROASTED", label="roasted", aliases=["roasted"], metadata={"implied_operation_id": "OP_ROAST"})
        self._register(kind="state", canonical_id="STATE_FRIED", label="fried", aliases=["fried"], metadata={"implied_operation_id": "OP_STIR_FRY"})
        self._register(kind="state", canonical_id="STATE_STEAMED", label="steamed", aliases=["steamed"], metadata={"implied_operation_id": "OP_STEAM"})
        self._register(kind="state", canonical_id="STATE_SIMMERED", label="simmered", aliases=["simmered"], metadata={"implied_operation_id": "OP_SIMMER"})
        self._register(kind="state", canonical_id="STATE_MIXED", label="mixed", aliases=["mixed"], metadata={"implied_operation_id": "OP_MIX"})
        self._register(kind="state", canonical_id="STATE_KNEADED", label="kneaded", aliases=["kneaded"], metadata={"implied_operation_id": "OP_KNEAD"})
        self._register(kind="state", canonical_id="STATE_RISEN", label="risen", aliases=["risen"], metadata={"implied_operation_id": "OP_RISE"})
        self._register(kind="state", canonical_id="STATE_COOLED", label="cooled", aliases=["cooled"])
        self._register(kind="state", canonical_id="STATE_PEELED", label="peeled", aliases=["peeled"], metadata={"implied_operation_id": "OP_PEEL"})
        self._register(kind="state", canonical_id="STATE_CRUSHED", label="crushed", aliases=["crushed"], metadata={"implied_operation_id": "OP_CRUSH"})


DEFAULT_SEMANTIC_MEMORY = SemanticMemory()
