from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .extraction_schema import InstructionStepCandidate, OperationCandidate


_VERB_PATTERNS = [
    ("cut", re.compile(r"\b(cut|cuts|cutting)\b", re.IGNORECASE)),
    ("chop", re.compile(r"\b(chop|chops|chopped|chopping)\b", re.IGNORECASE)),
    ("dice", re.compile(r"\b(dice|dices|diced|dicing)\b", re.IGNORECASE)),
    ("slice", re.compile(r"\b(slice|slices|sliced|slicing)\b", re.IGNORECASE)),
    ("mince", re.compile(r"\b(mince|minces|minced|mincing)\b", re.IGNORECASE)),
    ("grate", re.compile(r"\b(grate|grates|grated|grating)\b", re.IGNORECASE)),
    ("mash", re.compile(r"\b(mash|mashes|mashed|mashing)\b", re.IGNORECASE)),
    ("mix", re.compile(r"\b(mix|mixes|mixed|mixing|combine|combines|combined|stir|stirs|stirred|stirring)\b", re.IGNORECASE)),
    ("whisk", re.compile(r"\b(whisk|whisks|whisked|whisking)\b", re.IGNORECASE)),
    ("fold", re.compile(r"\b(fold|folds|folded|folding)\b", re.IGNORECASE)),
    ("melt", re.compile(r"\b(melt|melts|melted|melting)\b", re.IGNORECASE)),
    ("boil", re.compile(r"\b(boil|boils|boiled|boiling)\b", re.IGNORECASE)),
    ("simmer", re.compile(r"\b(simmer|simmers|simmered|simmering)\b", re.IGNORECASE)),
    ("steam", re.compile(r"\b(steam|steams|steamed|steaming)\b", re.IGNORECASE)),
    ("bake", re.compile(r"\b(bake|bakes|baked|baking)\b", re.IGNORECASE)),
    ("roast", re.compile(r"\b(roast|roasts|roasted|roasting)\b", re.IGNORECASE)),
    ("stir_fry", re.compile(r"\b(stir[- ]fry|stir[- ]fries|stir[- ]fried|fry|fries|fried|frying)\b", re.IGNORECASE)),
    ("saute", re.compile(r"\b(saute|sauté|sautes|sautés|sauteed|sautéed|sauteing|sautéing)\b", re.IGNORECASE)),
    ("beat", re.compile(r"\b(beat|beats|beaten|beating)\b", re.IGNORECASE)),
    ("cook", re.compile(r"\b(cook|cooks|cooked|cooking)\b", re.IGNORECASE)),
    ("knead", re.compile(r"\b(knead|kneads|kneaded|kneading)\b", re.IGNORECASE)),
    ("rest", re.compile(r"\b(rest|rests|rested|resting)\b", re.IGNORECASE)),
    ("rise", re.compile(r"\b(rise|rises|rose|risen|rising)\b", re.IGNORECASE)),
    ("pour", re.compile(r"\b(pour|pours|poured|pouring)\b", re.IGNORECASE)),
    ("add", re.compile(r"\b(add|adds|added|adding)\b", re.IGNORECASE)),
    ("season", re.compile(r"\b(season|seasons|seasoned|seasoning)\b", re.IGNORECASE)),
    ("blend", re.compile(r"\b(blend|blends|blended|blending)\b", re.IGNORECASE)),
    ("serve", re.compile(r"\b(serve|serves|served|serving)\b", re.IGNORECASE)),
    ("drain", re.compile(r"\b(drain|drains|drained|draining)\b", re.IGNORECASE)),
    ("peel", re.compile(r"\b(peel|peels|peeled|peeling)\b", re.IGNORECASE)),
    ("crush", re.compile(r"\b(crush|crushes|crushed|crushing)\b", re.IGNORECASE)),
]
_STOP_INPUTS = {"the", "a", "an", "with", "in", "for", "to", "then", "together"}
_TIME_OR_TEMP_RE = re.compile(r"\b(?:for\s+)?\d+\s*(?:minutes?|mins?|hours?|degrees?)\b.*$", re.IGNORECASE)


def extract_instruction_step(raw: str, *, step_index: int) -> InstructionStepCandidate:
    original = raw.strip()
    warnings: List[str] = []
    if not original:
        return InstructionStepCandidate(raw=raw, step_index=step_index, warnings=["empty_instruction_step"])

    clauses = _split_clauses(original)
    operations: List[OperationCandidate] = []
    last_inputs: List[str] = []
    for clause in clauses:
        clause_operations = _extract_operations_from_clause(clause, last_inputs=last_inputs)
        if not clause_operations:
            warnings.append(f"unrecognized_instruction_clause:{clause}")
            continue
        operations.extend(clause_operations)
        for op in clause_operations:
            if op.input_candidates:
                last_inputs = op.input_candidates

    if not operations and not warnings:
        warnings.append("unrecognized_instruction")
    return InstructionStepCandidate(
        raw=original,
        step_index=step_index,
        operation_candidates=operations,
        warnings=warnings,
    )


def extract_instruction_steps(raw_lines: List[str]) -> List[InstructionStepCandidate]:
    return [extract_instruction_step(line, step_index=i) for i, line in enumerate(raw_lines)]


def _split_clauses(text: str) -> List[str]:
    normalized = text.strip().rstrip(".")
    parts = re.split(r"\bthen\b|;|(?<=[.!?])\s+", normalized, flags=re.IGNORECASE)
    return [part.strip(" .") for part in parts if part.strip(" .")]


def _operation_matches(clause: str) -> List[Tuple[int, int, str, re.Match[str]]]:
    """Return non-overlapping verb matches in textual order.

    Pattern declaration order must never decide which operation a sentence
    contains.  At the same offset the longest surface wins, which keeps
    ``stir-fry`` from being reduced to ``stir``.
    """
    matches: List[Tuple[int, int, str, re.Match[str]]] = []
    for lemma, pattern in _VERB_PATTERNS:
        for match in pattern.finditer(clause):
            matches.append((match.start(), match.end(), lemma, match))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

    selected: List[Tuple[int, int, str, re.Match[str]]] = []
    for candidate in matches:
        start, end, _, _ = candidate
        if selected and start < selected[-1][1]:
            continue
        selected.append(candidate)
    return selected


def _extract_operations_from_clause(clause: str, *, last_inputs: List[str]) -> List[OperationCandidate]:
    matches = _operation_matches(clause)
    operations: List[OperationCandidate] = []
    current_inputs = list(last_inputs)
    for index, (start, end, lemma, _match) in enumerate(matches):
        next_start = matches[index + 1][0] if index + 1 < len(matches) else len(clause)
        raw_span = clause[start:next_start].strip(" ,-.")
        tail = clause[end:next_start].strip(" ,-.\t")
        inputs = _extract_inputs(tail, fallback=current_inputs)
        output = _output_state(lemma, inputs)
        operation = OperationCandidate(
            raw_span=raw_span or clause[start:end],
            operation_lemma_candidate=lemma,
            input_candidates=inputs,
            output_state_candidate=output,
            confidence=0.85 if inputs else 0.60,
            warnings=[] if inputs else ["missing_input_candidates"],
        )
        operations.append(operation)
        if inputs:
            current_inputs = inputs
    return operations


def _extract_operation_from_clause(clause: str, *, last_inputs: List[str]) -> Optional[OperationCandidate]:
    """Compatibility wrapper for callers that expect one operation."""
    operations = _extract_operations_from_clause(clause, last_inputs=last_inputs)
    return operations[0] if operations else None


def _extract_inputs(text: str, *, fallback: List[str]) -> List[str]:
    cleaned = _TIME_OR_TEMP_RE.sub("", text.lower())
    cleaned = re.sub(r"\b(in|into|on|over|until|with)\b.*$", "", cleaned).strip()
    cleaned = cleaned.replace(",", " and ")
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned)
    tokens = [tok for tok in cleaned.split() if tok not in _STOP_INPUTS]
    if tokens in (["them"], ["it"]):
        return list(fallback)
    tokens = [tok for tok in tokens if tok not in {"them", "it"}]
    if not tokens:
        return list(fallback)
    return _join_input_tokens(tokens)


def _join_input_tokens(tokens: List[str]) -> List[str]:
    inputs: List[str] = []
    current: List[str] = []
    for token in tokens:
        if token == "and":
            if current:
                inputs.append(" ".join(current))
                current = []
        else:
            current.append(token)
    if current:
        inputs.append(" ".join(current))

    split_inputs: List[str] = []
    for item in inputs:
        split_inputs.extend(part.strip() for part in item.split("  ") if part.strip())
    return split_inputs or inputs


def _output_state(lemma: str, inputs: List[str]) -> Optional[str]:
    primary = inputs[0] if inputs else None
    if lemma == "chop" and primary:
        return f"chopped {primary}"
    if lemma == "cut" and primary:
        return f"cut {primary}"
    if lemma == "dice" and primary:
        return f"diced {primary}"
    if lemma == "slice" and primary:
        return f"sliced {primary}"
    if lemma == "mince" and primary:
        return f"minced {primary}"
    if lemma == "grate" and primary:
        return f"grated {primary}"
    if lemma == "mash" and primary:
        return f"mashed {primary}"
    if lemma == "mix":
        return "mixture"
    if lemma == "whisk":
        return "whisked mixture"
    if lemma == "fold":
        return "folded mixture"
    if lemma == "melt" and primary:
        return f"melted {primary}"
    if lemma == "boil" and primary:
        return f"boiled {primary}"
    if lemma == "bake" and primary:
        return f"baked {primary}"
    if lemma == "roast" and primary:
        return f"roasted {primary}"
    if lemma == "beat" and primary:
        return f"beaten {primary}"
    if lemma == "cook" and primary:
        return f"cooked {primary}"
    if lemma == "simmer" and primary:
        return f"simmered {primary}"
    if lemma == "steam" and primary:
        return f"steamed {primary}"
    if lemma in {"stir_fry", "saute"} and primary:
        return f"cooked {primary}"
    if lemma == "knead" and primary:
        return f"kneaded {primary}"
    if lemma == "rest" and primary:
        return f"rested {primary}"
    if lemma == "rise" and primary:
        return f"risen {primary}"
    if lemma == "pour" and primary:
        return f"poured {primary}"
    if lemma == "add":
        return "combined ingredients"
    if lemma == "season" and primary:
        return f"seasoned {primary}"
    if lemma == "blend" and primary:
        return f"blended {primary}"
    if lemma == "serve" and primary:
        return f"served {primary}"
    if lemma == "drain" and primary:
        return f"drained {primary}"
    if lemma == "peel" and primary:
        return f"peeled {primary}"
    if lemma == "crush" and primary:
        return f"crushed {primary}"
    return None
