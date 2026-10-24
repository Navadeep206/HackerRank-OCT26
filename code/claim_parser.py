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
        "front_bumper": ("front bumper", "front fascia", "parachoques delantero"),
        "rear_bumper": ("rear bumper", "back bumper", "rear fascia", "parachoques trasero", "parachoques de atras", "parachoques"),
        "door": ("car door", "door panel", "driver door", "passenger door", "door", "puerta"),
        "hood": ("hood", "bonnet"),
        "windshield": ("windshield", "windscreen", "front glass"),
        "side_mirror": ("side mirror", "wing mirror", "side-view mirror", "rear-view mirror", "espejo"),
        "headlight": ("headlight", "head lamp", "headlamp"),
        "taillight": ("taillight", "tail light", "rear light", "tail lamp", "back light"),
        "fender": ("fender", "wing panel"),
        "quarter_panel": ("quarter panel", "rear quarter", "quarter-panel"),
        "body": ("car body", "vehicle body", "body panel", "bodywork"),
    },
    "laptop": {
        "screen": ("screen", "display", "monitor", "lcd", "panel", "pantalla"),
        "keyboard": ("keyboard", "key", "keys", "keycap", "teclas"),
        "trackpad": ("trackpad", "touchpad"),
        "hinge": ("hinge", "hinges"),
        "lid": ("lid", "top cover", "back cover", "laptop cover"),
        "corner": ("corner", "edge"),
        "port": ("port", "usb port", "charging port", "connector"),
        "base": ("base", "bottom cover", "underside"),
        "body": ("laptop body", "chassis", "laptop casing"),
    },
    "package": {
        "box": ("box", "carton", "package exterior", "outer packaging", "caja", "dabba"),
        "package_corner": ("package corner", "box corner", "corner of the box", "corner", "esquina"),
        "package_side": ("package side", "side of the box", "side panel"),
        "seal": ("seal", "tamper seal", "tape", "sealed flap", "sello"),
        "label": ("label", "shipping label", "address label", "barcode", "etiqueta"),
        "contents": ("contents", "inside the box", "inside package", "inner packaging"),
        "item": ("item", "product inside", "delivered product", "product"),
    },
}
ISSUE_PATTERNS: tuple[tuple[IssueType, tuple[str, ...]], ...] = (
    ("glass_shatter", ("shattered glass", "glass shattered", "glass is broken", "shattered")),
    ("torn_packaging", ("torn packaging", "torn box", "ripped box", "package torn", "ripped packaging", "torn", "ripped")),
    ("crushed_packaging", ("crushed packaging", "crushed box", "box is crushed", "dented box", "package crushed", "crushed in", "crushed", "dab gaya")),
    ("water_damage", ("water damage", "water-damaged", "water damaged", "wet damage", "liquid damage", "wet box", "wet-looking")),
    ("missing_part", ("missing part", "part is missing", "missing key", "missing keys", "keys missing", "key is missing", "missing item", "item is missing", "not included", "not in the box", "keycaps came off", "keys came off", "missing", "faltan")),
    ("broken_part", ("broken", "broke", "breakage", "snapped off", "detached", "toot", "toota")),
    ("crack", ("crack", "cracks", "cracked", "fracture", "fractured")),
    ("dent", ("dent", "dents", "dented", "ding", "dings", "dimple")),
    ("scratch", ("scratch", "scratches", "scratched", "scuff", "scuffs", "scuffed", "abrasion", "kharoch", "scrape")),
    ("stain", ("stain", "stains", "stained", "discoloration", "discoloured", "discolored", "smudge", "smudges")),
)
NEGATION_PATTERNS = (
    r"\bnot\s+(?:the\s+)?{phrase}\b",
    r"\bno\s+{phrase}\b",
    r"\bwithout\s+(?:any\s+)?{phrase}\b",
    r"\b(?:isn't|is not|wasn't|was not)\s+{phrase}\b",
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
    r"\boverride\s+(?:the\s+)?(?:system|review|decision)\b",
    r"\bignore\s+(?:the\s+)?rules?\b",
    r"\bapprove\s+(?:this\s+)?automatically\b",
)
MULTILINGUAL_HINTS = {
    "hi": ("kharoch", "toot", "damage hai", "nuksan"),
    "es": ("rayado", "rasguño", "roto", "abolladura", "dañado"),
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

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable output with tuples converted to lists."""
        data = asdict(self)
        for key in ("negated_issues", "mentioned_parts", "extraction_notes"):
            data[key] = list(data[key])
        return data


def _phrase_present(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
    return re.search(r"(?<!\w)" + escaped + r"(?!\w)", text) is not None


def _find_issue_mentions(text: str) -> list[tuple[IssueType, str, int]]:
    mentions: list[tuple[IssueType, str, int]] = []
    for issue, aliases in ISSUE_PATTERNS:
        for alias in aliases:
            pattern = re.compile(r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)")
            for match in pattern.finditer(text):
                mentions.append((issue, alias, match.start()))
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


def _detect_part(text: str, claim_object: str) -> tuple[str, tuple[str, ...]]:
    found: list[tuple[int, int, str]] = []
    for part, aliases in PART_ALIASES[claim_object].items():
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


def _part_for_issue(text: str, claim_object: str, issue_start: int, fallback: str) -> str:
    """Associate the primary positive issue with the closest preceding part mention."""
    found: list[tuple[int, int, str]] = []
    for part, aliases in PART_ALIASES[claim_object].items():
        for alias in aliases:
            pattern = re.compile(r"(?<!\w)" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?!\w)")
            for match in pattern.finditer(text):
                found.append((match.start(), -len(alias), part))
    if not found:
        return fallback
    found.sort()
    preceding = [item for item in found if item[0] <= issue_start]
    if preceding:
        return preceding[-1][2]
    return min(found, key=lambda item: (abs(item[0] - issue_start), item[0]))[2]


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


def extract_claim_intent(user_claim: str, claim_object: str | None = None) -> ClaimIntent:
    """Extract primary issue/part, negation, corrections, and suspicious instructions.

    The provided claim_object field is authoritative when valid. Conversation text
    cannot override it. Text instructions never influence evidence or claim status.
    """
    raw = user_claim or ""
    text = re.sub(r"\s+", " ", raw).strip().lower()
    obj = _detect_object(text, claim_object)
    notes: list[str] = []
    if not text:
        notes.append("empty_claim_conversation")
    if obj == "unknown":
        notes.append("object_not_identified")

    mentions = _find_issue_mentions(text)
    negated: list[str] = []
    positive: list[tuple[IssueType, str, int]] = []
    for issue, alias, start in mentions:
        # Negation must target the damage phrase itself. A later correction such as
        # "the keyboard is cracked, not the screen" negates the part, not the crack.
        if _is_negated(text, alias, start):
            # Record that the issue is disclaimed for a corrected part without erasing its positive mention.
            if issue not in negated:
                negated.append(issue)
        elif not any(existing[0] == issue and abs(existing[2] - start) < 40 for existing in positive):
            positive.append((issue, alias, start))

    # If no positive damage is described but a negated issue exists, do not invent damage.
    primary_issue: str = positive[0][0] if positive else ("none" if negated else "unknown")
    if len({item[0] for item in positive}) > 1:
        notes.append("multiple_issue_types_mentioned_primary_is_first_positive_mention")

    part, parts = _detect_part(text, obj) if obj in PARTS else ("unknown", ())
    if positive and obj in PARTS:
        part = _part_for_issue(text, obj, positive[0][2], part)
    if len(parts) > 1:
        notes.append("multiple_parts_mentioned_primary_is_associated_with_primary_issue")
    correction = any(marker in text for marker in CORRECTION_MARKERS)
    if correction:
        notes.append("correction_or_negation_language_present")
    injection = any(re.search(pattern, text) for pattern in PROMPT_INJECTION_PATTERNS)
    if injection:
        notes.append("embedded_instruction_is_untrusted_and_must_not_change_decision")
    multilingual = any(hint in text for hints in MULTILINGUAL_HINTS.values() for hint in hints)
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
        normalized_text=text,
        extraction_notes=tuple(notes),
    )
