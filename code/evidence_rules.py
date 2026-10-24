"""Evidence sufficiency rules derived from dataset/evidence_requirements.csv.

This module assesses whether submitted evidence is reviewable for the requested
claim. It intentionally does not decide claim_status; visual inspection and the
decision layer must do that separately.
"""
from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from code.claim_parser import ClaimIntent, extract_claim_intent


@dataclass(frozen=True, slots=True)
class EvidenceRule:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    applicable_requirement_ids: tuple[str, ...]
    required_evidence: tuple[str, ...]
    reviewable: bool
    missing_or_unusable_paths: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("applicable_requirement_ids", "required_evidence",
                    "missing_or_unusable_paths", "notes"):
            result[key] = list(result[key])
        return result


def load_evidence_rules(csv_path: Path) -> list[EvidenceRule]:
    """Read and validate the challenge's evidence checklist."""
    required = {"requirement_id", "claim_object", "applies_to", "minimum_image_evidence"}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not required.issubset(fields):
                raise ValueError(f"Evidence requirements missing columns: {sorted(required - fields)}")
            rules = [
                EvidenceRule(
                    requirement_id=(row.get("requirement_id") or "").strip(),
                    claim_object=(row.get("claim_object") or "").strip().lower(),
                    applies_to=(row.get("applies_to") or "").strip(),
                    minimum_image_evidence=(row.get("minimum_image_evidence") or "").strip(),
                )
                for row in reader
            ]
    except OSError as exc:
        raise ValueError(f"Cannot read evidence requirements: {csv_path}") from exc
    if not rules:
        raise ValueError("Evidence requirements file contains no rules")
    ids = [r.requirement_id for r in rules]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Evidence requirement IDs must be non-empty and unique")
    if any(not r.minimum_image_evidence for r in rules):
        raise ValueError("Each evidence rule must define minimum image evidence")
    return rules


def _issue_family(intent: ClaimIntent) -> str:
    issue = intent.issue_type
    if issue in {"dent", "scratch"}:
        return "body_surface"
    if issue in {"crack", "glass_shatter", "broken_part", "missing_part"}:
        return "breakage_or_missing"
    if issue in {"torn_packaging", "crushed_packaging"}:
        return "package_exterior"
    if issue in {"water_damage", "stain"}:
        return "surface_or_water"
    return "unknown"


def _rule_matches(rule: EvidenceRule, intent: ClaimIntent) -> bool:
    if rule.claim_object not in {"all", intent.claim_object}:
        return False
    applies = rule.applies_to.lower()
    if applies in {"general claim review", "multi-image rows", "reviewability"}:
        return True
    family = _issue_family(intent)
    if intent.claim_object == "car":
        if "dent or scratch" in applies:
            return family == "body_surface"
        if "crack, broken, or missing part" in applies:
            return family == "breakage_or_missing"
        if "vehicle identity or orientation" in applies:
            return bool(re.search(r"\b(left|right|driver|passenger|same vehicle|different vehicle|side|orientation)\b", intent.normalized_text))
    elif intent.claim_object == "laptop":
        if "screen, keyboard, or trackpad" in applies:
            return intent.object_part in {"screen", "keyboard", "trackpad"} or family in {"body_surface", "breakage_or_missing", "surface_or_water"}
        if "hinge, lid, corner, body, or port" in applies:
            return intent.object_part in {"hinge", "lid", "corner", "body", "base", "port"}
    elif intent.claim_object == "package":
        if "crushed, torn, or seal damage" in applies:
            return family == "package_exterior" or intent.object_part in {"seal", "package_corner", "package_side", "box"}
        if "water, stain, or label damage" in applies:
            return family == "surface_or_water" or intent.object_part == "label"
        if "contents or inner item" in applies:
            return intent.object_part in {"contents", "item"} or family == "breakage_or_missing"
    return False


def assess_evidence(
    *,
    claim_object: str,
    user_claim: str,
    image_paths: Iterable[str],
    dataset_dir: Path,
    rules: list[EvidenceRule] | None = None,
    readable_paths: Iterable[str] | None = None,
    relevant_paths: Iterable[str] | None = None,
    visual_assessment_complete: bool = False,
) -> EvidenceAssessment:
    """Assess minimum evidence using paths/readability plus object/issue-specific rules.

    `readable_paths` is optional. When supplied, it should contain the original
    CSV-relative paths that were successfully decoded and passed quality checks.
    Without it, file existence is checked here; actual blur/occlusion assessment
    belongs to image inspection.
    """
    intent = extract_claim_intent(user_claim, claim_object)
    submitted = tuple(dict.fromkeys(p.strip() for p in image_paths if p.strip()))
    if rules is None:
        rules = load_evidence_rules(dataset_dir.parent / "dataset" / "evidence_requirements.csv")
    applicable = [r for r in rules if _rule_matches(r, intent)]
    # Keep generic and object-specific guidance, but de-duplicate identical text.
    requirements = tuple(dict.fromkeys(r.minimum_image_evidence for r in applicable))
    ids = tuple(r.requirement_id for r in applicable)

    if readable_paths is None:
        readable = []
        unusable = []
        for raw in submitted:
            candidate = (dataset_dir / raw).resolve()
            try:
                candidate.relative_to(dataset_dir.resolve())
                safe = True
            except ValueError:
                safe = False
            if safe and candidate.is_file() and candidate.stat().st_size > 0:
                readable.append(raw)
            else:
                unusable.append(raw)
    else:
        readable_set = set(readable_paths)
        readable = [p for p in submitted if p in readable_set]
        unusable = [p for p in submitted if p not in readable_set]

    notes: list[str] = []
    if not submitted:
        notes.append("no_images_submitted")
    if unusable:
        notes.append("one_or_more_images_missing_unreadable_or_failed_quality_check")
    if intent.claim_object == "unknown":
        notes.append("claim_object_unknown")
    if intent.issue_type == "unknown":
        notes.append("issue_type_unknown_or_not_explicitly_described")
    if intent.object_part == "unknown":
        notes.append("object_part_not_identified")
    if not requirements:
        notes.append("no_specific_requirement_matched; general reviewability still applies")

    # Existing files are only candidates. The challenge defines sufficiency in
    # visual terms, so paths alone must never produce a positive evidence label.
    enough_candidates = bool(readable)
    relevant = set(relevant_paths or ())
    has_specific_target = intent.object_part != "unknown" and (intent.issue_type != "unknown" or bool(relevant))
    if not visual_assessment_complete:
        met = False
        notes.append("awaiting_visual_assessment")
        reason = (
            "File paths were checked, but image relevance, visibility, and the required view "
            "have not yet been visually assessed; evidence sufficiency remains unconfirmed."
        )
    else:
        met = (
            enough_candidates
            and bool(relevant.intersection(readable))
            and has_specific_target
            and not unusable
            and intent.claim_object != "unknown"
        )
        if met:
            reason = (
                "Visual assessment found at least one readable image relevant to the claimed "
                "object and part; the required evidence criteria were judged satisfied."
            )
        else:
            reasons = []
            if not enough_candidates:
                reasons.append("no readable image files were found")
            if unusable:
                reasons.append(f"{len(unusable)} submitted image path(s) are missing or unusable")
            if not relevant.intersection(readable):
                reasons.append("no readable image was confirmed relevant to the claimed object and part")
            if not has_specific_target:
                reasons.append("the issue or claimed part is not specific enough")
            if intent.claim_object == "unknown":
                reasons.append("the object type is unknown")
            reason = "; ".join(reasons) or "The visual evidence requirements were not met."
    reviewable = enough_candidates and intent.claim_object != "unknown"
    return EvidenceAssessment(
        evidence_standard_met=met,
        evidence_standard_met_reason=reason,
        applicable_requirement_ids=ids,
        required_evidence=requirements,
        reviewable=reviewable,
        missing_or_unusable_paths=tuple(unusable),
        notes=tuple(notes),
    )
