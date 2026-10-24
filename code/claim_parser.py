"""Deterministic extraction of the claim intent from noisy conversations.

This module does not decide whether visual evidence proves the claim. It extracts
what the user says should be inspected, preserving negations/corrections and
treating embedded instructions as untrusted text. Visual verification belongs to
the later multimodal decision layer.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

ClaimObject = Literal["car", "laptop", "package"]
IssueType = Literal[
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
]

PARTS: dict[str, tuple[str, ...]] = {
    "car": ("front_bumper", "rear_bumper", "door", "hood", "windshield",
            "side_mirror", "headlight", "taillight", "fender", "quarter_panel", "body"),
    "laptop": ("screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body"),
    "package": ("box", "package_corner", "package_side", "seal", "label", "contents", "item"),
}
PART_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "car": {
        "front_bumper": ("front bumper", "bumper", "front fascia", "parachoques delantero", "front bumpr"),
        "rear_bumper": ("rear bumper", "back bumper", "rear fascia", "parachoques trasero", "parachoques de atras", "parachoques", "rear bumpr"),
        "door": ("car door", "door panel", "driver door", "passenger door", "door", "puerta"),
        "hood": ("hood", "bonnet", "bonet"),
        "windshield": ("windshield", "windscreen", "front glass", "windsheild"),
        "side_mirror": ("side mirror", "wing mirror", "side-view mirror", "rear-view mirror", "espejo", "mirror"),
        "headlight": ("headlight", "head lamp", "headlamp", "headlght", "foco"),
        "taillight": ("taillight", "tail light", "rear light", "tail lamp", "back light", "taillght", "foco trasero"),
        "fender": ("fender", "wing panel"),
        "quarter_panel": ("quarter panel", "rear quarter", "quarter-panel"),
        "body": ("car body", "vehicle body", "body panel", "bodywork", "body"),
    },
    "laptop": {
        "screen": ("screen", "display", "monitor", "lcd", "panel", "pantalla", "scren"),
        "keyboard": ("keyboard", "key", "keys", "keycap", "teclas", "keybaord"),
        "trackpad": ("trackpad", "touchpad", "trackpd", "palm-rest"),
        "hinge": ("hinge", "hinges"),
        "lid": ("lid", "top cover", "back cover", "laptop cover"),
        "corner": ("corner", "edge", "outer corner"),
        "port": ("port", "usb port", "charging port", "connector"),
        "base": ("base", "bottom cover", "underside"),
        "body": ("laptop body", "chassis", "laptop casing", "outer body", "body"),
    },
    "package": {
        "box": ("box", "carton", "package exterior", "outer packaging", "caja", "dabba", "shipping box", "cardboard box"),
        "package_corner": ("package corner", "box corner", "corner of the box", "corner", "esquina", "kona"),
        "package_side": ("package side", "side of the box", "side panel"),
        "seal": ("seal", "tamper seal", "tape", "sealed flap", "sello"),
        "label": ("label", "shipping label", "address label", "barcode", "etiqueta"),
        "contents": ("contents", "inside the box", "inside package", "inner packaging", "contenido"),
        "item": ("item", "product inside", "delivered product", "product", "producto", "articulo"),
    },
}
ISSUE_PATTERNS: tuple[tuple[IssueType, tuple[str, ...]], ...] = (
    ("glass_shatter", ("shattered glass", "glass shattered", "glass is broken", "shattered", "shatered")),
    ("torn_packaging", ("torn packaging", "torn box", "ripped box", "package torn", "ripped packaging", "torn", "ripped", "phati hui")),
    ("crushed_packaging", ("crushed packaging", "crushed box", "box is crushed", "dented box", "package crushed", "crushed in", "crushed", "dab gaya")),
    ("water_damage", ("water damage", "water-damaged", "water damaged", "wet damage", "liquid damage", "wet box", "wet-looking", "liquid stain", "water stain")),
    ("missing_part", ("missing part", "part is missing", "missing key", "missing keys", "keys missing", "key is missing", "missing item", "item is missing", "not included", "not in the box", "keycaps came off", "keys came off", "missing", "faltan")),
    ("broken_part", ("broken", "broke", "breakage", "snapped off", "detached", "toot", "toota", "roto")),
    ("crack", ("crack", "cracks", "cracked", "fracture", "fractured", "grieta")),
    ("dent", ("dent", "dents", "dented", "ding", "dings", "dimple", "dentd", "abolladura")),
    ("scratch", ("scratch", "scratches", "scratched", "scuff", "scuffs", "scuffed", "abrasion", "kharoch", "scrape", "scraches", "scratche", "rasguño")),
    ("stain", ("stain", "stains", "stained", "discoloration", "discoloured", "discolored", "smudge", "smudges")),
)
NEGATION_PATTERNS = (
    r"\bnot\s+(?:the\s+)?{phrase}\b",
    r"\bno\s+{phrase}\b",
    r"\bno\s+\w+\s+(?:or|nor|and)\s+(?:any\s+)?{phrase}\b",
    r"\bneither\s+\w+\s+nor\s+{phrase}\b",
    r"\bwithout\s+(?:any\s+)?{phrase}\b",
    r"\bwithout\s+(?:any\s+)?\w+\s+(?:or|and)\s+{phrase}\b",
    r"\b(?:isn't|is not|wasn't|was not)\s+{phrase}\b",
    r"\b(?:isn't|is not|wasn't|was not)\s+\w+\s+(?:or|nor|and)\s+{phrase}\b",
)
CORRECTION_MARKERS = (
    "actually", "correction", "i mean", "rather", "not", "instead", "sorry",
    "to clarify", "i meant", "ignore my previous",
)
PROMPT_INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b",
    r"\b(?:approve|accept|pay)\s+(?:this\s+)?claim\s+(?:automatically|regardless|without review|immediately)\b",
    r"\bskip\s+(?:manual\s+)?review\b",
    r"\bignore\s+(?:the\s+)?evidence\b",
    r"\b(?:system\s+)?override\b",
    r"\bignore\s+(?:all\s+)?(?:visual\s+)?(?:inspections?|checks?|damage\s+checks?|instructions?|evidence)\b",
    r"\b(?:output|set)\s+(?:claim_status|status)\s*=\s*\w+\b",
    r"\bignore\s+(?:the\s+)?rules?\b",
    r"\bapprove\s+(?:this\s+)?automatically\b",
)
HISTORICAL_MARKERS = (
    r"\b(?:old|prior|previous|pre-existing|already\s+there|from\s+last\s+year|before\s+this)\s+(?:on\s+the\s+)?{phrase}\b",
    r"\b{phrase}\s+(?:was\s+already\s+there|is\s+old|is\s+not\s+new|from\s+last\s+year|before\s+this)\b",
    r"\b(?:old|prior|previous)\s+{phrase}\b",
)
UNDAMAGED_MARKERS = (
    r"\b(?:the\s+)?{phrase}\s+(?:itself\s+)?(?:is|was)\s+(?:totally\s+|completely\s+)?(?:fine|okay|ok|intact|good|not\s+damaged|undamaged|untouched|theek\s+hai|bien|intacta|safe)\b",
    r"\b(?:not\s+the|no\s+damage\s+to\s+(?:the\s+)?){phrase}\b",
    r"\bno\s+{phrase}\s+damage\b",
    r"\bno,\s+(?:it\s+is\s+)?{phrase}\s+only\b",
)
MULTILINGUAL_HINTS = {
    "hi": ("kharoch", "toot", "damage hai", "nuksan", "theek hai"),
    "es": ("rayado", "rasguño", "roto", "abolladura", "dañado", "parachoques"),
}


@dataclass(frozen=True, slots=True)
class ClaimIntent:
    """Normalized claim details; extracted fields are hypotheses, not proof."""

    claim_object: str
    issue_type: str
    object_part: str
    negated_issues: tuple[str, ...]
    mentioned_parts: tuple[str, ...]
    correction_detected: bool
    instruction_injection_detected: bool
    multilingual_hint_detected: bool
    normalized_text: str
    extraction_notes: tuple[str, ...]
    claimed_parts: tuple[str, ...] = ()
    undamaged_parts: tuple[str, ...] = ()
    claimed_issues: tuple[str, ...] = ()
    historical_damage_detected: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable output with tuples converted to lists."""
        data = asdict(self)
        for key in ("negated_issues", "mentioned_parts", "extraction_notes",
                    "claimed_parts", "undamaged_parts", "claimed_issues"):
            if key in data and isinstance(data[key], tuple):
                data[key] = list(data[key])
        return data


def _phrase_present(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
    return re.search(r"(?<!\w)" + escaped + r"(?!\w)", text) is not None


def _is_historical(text: str, alias: str, start: int) -> bool:
    window = text[max(0, start - 45):min(len(text), start + len(alias) + 35)]
    for pattern in HISTORICAL_MARKERS:
        try:
            if re.search(pattern.format(phrase=re.escape(alias)), window):
                return True
        except re.error:
            continue
    return False


def _find_issue_mentions(text: str) -> list[tuple[IssueType, str, int, bool]]:
    mentions: list[tuple[IssueType, str, int, bool]] = []
    for issue, aliases in ISSUE_PATTERNS:
        for alias in aliases:
            pattern = re.compile(r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)")
            for match in pattern.finditer(text):
                hist = _is_historical(text, alias, match.start())
                mentions.append((issue, alias, match.start(), hist))
    # Stable ordering preserves the conversation's order for primary issue choice.
    return sorted(mentions, key=lambda item: (item[2], -len(item[1])))


def _is_negated(text: str, alias: str, start: int) -> bool:
    prefix = text[max(0, start - 55):start]
    suffix = text[start:start + len(alias) + 24]
    for pattern in NEGATION_PATTERNS:
        for candidate in (alias, re.escape(alias)):
            try:
                if re.search(pattern.format(phrase=candidate), prefix + suffix):
                    return True
            except re.error:
                continue
    # Common corrections: "not the screen, only the keyboard".
    return bool(re.search(r"\bnot\s+(?:the\s+)?" + re.escape(alias) + r"\b", prefix))


def _detect_undamaged_parts(text: str, claim_object: str) -> tuple[str, ...]:
    if claim_object not in PART_ALIASES:
        return ()
    undamaged: list[str] = []
    for part, aliases in PART_ALIASES[claim_object].items():
        for alias in aliases:
            for pattern in UNDAMAGED_MARKERS:
                try:
                    if re.search(pattern.format(phrase=re.escape(alias)), text):
                        if part not in undamaged:
                            undamaged.append(part)
                except re.error:
                    continue
    return tuple(undamaged)


def _detect_part(text: str, claim_object: str) -> tuple[str, tuple[str, ...]]:
    found: list[tuple[int, int, str]] = []
    for part, aliases in PART_ALIASES.get(claim_object, {}).items():
        for alias in aliases:
            pattern = re.compile(r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)")
            for match in pattern.finditer(text):
                found.append((match.start(), -len(alias), part))
    found.sort()
    ordered: list[str] = []
    for _, _, part in found:
        if part not in ordered:
            ordered.append(part)
    return (ordered[0] if ordered else "unknown", tuple(ordered))


def _part_for_issue(text: str, claim_object: str, issue_start: int, fallback: str, undamaged: tuple[str, ...] = ()) -> str:
    """Associate the primary positive issue with the closest preceding non-undamaged part mention."""
    found: list[tuple[int, int, str]] = []
    for part, aliases in PART_ALIASES.get(claim_object, {}).items():
        for alias in aliases:
            pattern = re.compile(r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)")
            for match in pattern.finditer(text):
                found.append((match.start(), -len(alias), part))
    if not found:
        return fallback
    found.sort()
    candidate_items = [item for item in found if item[2] not in undamaged]
    items_to_use = candidate_items if candidate_items else found
    preceding = [item for item in items_to_use if item[0] <= issue_start]
    if preceding:
        return preceding[-1][2]
    return min(items_to_use, key=lambda item: (abs(item[0] - issue_start), item[0]))[2]


def _detect_object(text: str, supplied_object: str | None) -> str:
    if supplied_object in {"car", "laptop", "package"}:
        return supplied_object
    patterns = {
        "car": ("car", "vehicle", "sedan", "suv", "bumper", "windshield", "fender", "headlight"),
        "laptop": ("laptop", "notebook computer", "keyboard", "trackpad", "touchpad", "hinge"),
        "package": ("package", "parcel", "shipping box", "carton", "delivery box"),
    }
    scores = {obj: sum(1 for word in words if _phrase_present(text, word)) for obj, words in patterns.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 and list(scores.values()).count(scores[best]) == 1 else "unknown"


def _split_turns(raw_text: str) -> tuple[str, str, list[str]]:
    """Separate conversation transcript into claimant statements, reviewer prompts, and turns."""
    pattern = r"(?=[|\n]|\b(?:support(?:\s+agent)?|customer|cliente|user|claimant|agent|soporte|reviewer|rep)\s*:)"
    segments = [s.strip(" |\n\r") for s in re.split(pattern, raw_text, flags=re.IGNORECASE) if s.strip(" |\n\r")]
    claimant_turns: list[str] = []
    support_turns: list[str] = []
    for seg in segments:
        m_cust = re.match(r"^(?:customer|cliente|user|claimant)\s*:\s*(.*)$", seg, re.IGNORECASE)
        if m_cust:
            claimant_turns.append(m_cust.group(1).strip())
            continue
        m_supp = re.match(r"^(?:support(?:\s+agent)?|agent|soporte|reviewer|rep)\s*:\s*(.*)$", seg, re.IGNORECASE)
        if m_supp:
            support_turns.append(m_supp.group(1).strip())
            continue
        claimant_turns.append(seg.strip())
    claimant_text = " ".join(claimant_turns) if claimant_turns else raw_text
    support_text = " ".join(support_turns)
    return claimant_text, support_text, claimant_turns


def extract_claim_intent(user_claim: str, claim_object: str | None = None) -> ClaimIntent:
    """Extract primary issue/part, negation, corrections, undamaged parts, and suspicious instructions.

    The provided claim_object field is authoritative when valid. Conversation text
    cannot override it. Text instructions never influence evidence or claim status.
    """
    raw = user_claim or ""
    claimant_text, support_text, claimant_turns = _split_turns(raw)
    full_text = re.sub(r"\s+", " ", raw).strip().lower()
    text = re.sub(r"\s+", " ", claimant_text).strip().lower()
    obj = _detect_object(full_text, claim_object)
    notes: list[str] = []
    if not full_text:
        notes.append("empty_claim_conversation")
    if obj == "unknown":
        notes.append("object_not_identified")

    # Detect explicitly undamaged or disclaimed parts
    undamaged_parts = _detect_undamaged_parts(full_text, obj)

    # Search for issues in claimant turns first; fallback to full text if empty
    mentions = _find_issue_mentions(text)
    if not mentions and full_text != text:
        mentions = _find_issue_mentions(full_text)

    negated: list[str] = []
    positive: list[tuple[IssueType, str, int]] = []
    historical_detected = False

    for issue, alias, start, hist in mentions:
        if hist:
            historical_detected = True
            notes.append("historical_damage_language_detected")
            continue
        if _is_negated(text if text else full_text, alias, start):
            if issue not in negated:
                negated.append(issue)
        elif not any(existing[0] == issue and abs(existing[2] - start) < 40 for existing in positive):
            positive.append((issue, alias, start))

    # If the claimant provided multiple turns and the latest turn states a clear resolution/correction,
    # evaluate whether the latest turn refines the primary issue.
    if len(claimant_turns) > 1:
        last_turn = claimant_turns[-1].lower()
        last_mentions = _find_issue_mentions(last_turn)
        last_positive = [m for m in last_mentions if not m[3] and not _is_negated(last_turn, m[1], m[2])]
        if last_positive and not any(m[0] == positive[0][0] for m in last_positive if positive):
            # The claimant specifically resolved to a new issue in their closing response
            positive = [(last_positive[0][0], last_positive[0][1], last_positive[0][2])] + positive
            notes.append("latest_claimant_turn_refined_issue")

    # If no positive damage is described but a negated issue exists, do not invent damage.
    primary_issue: str = positive[0][0] if positive else ("none" if negated else "unknown")
    if len({item[0] for item in positive}) > 1:
        notes.append("multiple_issue_types_mentioned_primary_is_first_positive_mention")

    # Part detection
    part, parts = _detect_part(text if text else full_text, obj) if obj in PARTS else ("unknown", ())
    if positive and obj in PARTS:
        part = _part_for_issue(text if text else full_text, obj, positive[0][2], part, undamaged=undamaged_parts)

    # Refine part from claimant's latest turn if resolved
    if len(claimant_turns) > 1 and obj in PARTS:
        last_turn = claimant_turns[-1].lower()
        last_part, last_parts = _detect_part(last_turn, obj)
        candidate_last_parts = [p for p in last_parts if p not in undamaged_parts]
        if candidate_last_parts and candidate_last_parts[0] != part:
            part = candidate_last_parts[0]
            notes.append("latest_claimant_turn_refined_part")

    # Collect all actively claimed parts
    claimed_parts = tuple(dict.fromkeys([p for p in parts if p not in undamaged_parts]))
    if not claimed_parts and part != "unknown" and part not in undamaged_parts:
        claimed_parts = (part,)

    claimed_issues = tuple(dict.fromkeys([item[0] for item in positive])) if positive else ((primary_issue,) if primary_issue not in {"unknown", "none"} else ())

    correction = any(marker in full_text for marker in CORRECTION_MARKERS)
    if correction:
        notes.append("correction_or_negation_language_present")
    injection = any(re.search(pattern, full_text) for pattern in PROMPT_INJECTION_PATTERNS)
    if injection:
        notes.append("embedded_instruction_is_untrusted_and_must_not_change_decision")
    multilingual = any(hint in full_text for hints in MULTILINGUAL_HINTS.values() for hint in hints)
    if multilingual:
        notes.append("possible_multilingual_damage_phrase_detected")

    return ClaimIntent(
        claim_object=obj,
        issue_type=primary_issue,
        object_part=part,
        negated_issues=tuple(negated),
        mentioned_parts=parts,
        correction_detected=correction,
        instruction_injection_detected=injection,
        multilingual_hint_detected=multilingual,
        normalized_text=full_text,
        extraction_notes=tuple(notes),
        claimed_parts=claimed_parts,
        undamaged_parts=undamaged_parts,
        claimed_issues=claimed_issues,
        historical_damage_detected=historical_detected,
    )
