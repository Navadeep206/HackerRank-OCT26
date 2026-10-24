"""Claim-level decision policy over parsed intent and validated vision findings.

The policy is deliberately conservative: risk history cannot prove or disprove
damage, and contradictory/low-confidence images lead to manual review or
not-enough-information rather than forced conclusions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from code.claim_parser import ClaimIntent, extract_claim_intent
from code.evidence_rules import EvidenceAssessment, EvidenceRule, assess_evidence
from code.vision_provider import ALLOWED_FLAGS, VisionFinding


ALLOWED_STATUSES = {"supported", "contradicted", "not_enough_information"}
ALLOWED_SEVERITIES = {"none", "low", "medium", "high", "unknown"}
SEVERITY_BY_ISSUE = {
    "none": "none",
    "stain": "low",
    "scratch": "low",
    "torn_packaging": "low",
    "dent": "medium",
    "crack": "medium",
    "water_damage": "medium",
    "crushed_packaging": "medium",
    "broken_part": "high",
    "missing_part": "high",
    "glass_shatter": "high",
    "unknown": "unknown",
}
QUALITY_FLAG_MAP = {
    "low_resolution": "blurry_image",
    "low_light": "low_light_or_glare",
    "possible_overexposure": "low_light_or_glare",
    "low_contrast": "blurry_image",
    "possible_blur_or_low_detail": "blurry_image",
    "highlight_or_shadow_clipping": "low_light_or_glare",
}


@dataclass(frozen=True, slots=True)
class ClaimDecision:
    claim_status: str
    claim_status_justification: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: tuple[str, ...]
    issue_type: str
    object_part: str
    supporting_image_ids: tuple[str, ...]
    valid_image: bool
    severity: str
    evidence_requirement_ids: tuple[str, ...]
    decision_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("risk_flags", "supporting_image_ids", "evidence_requirement_ids", "decision_notes"):
            result[key] = list(result[key])
        return result


def _flags_from_findings(findings: Sequence[VisionFinding]) -> list[str]:
    flags: list[str] = []
    for finding in findings:
        flags.extend(flag for flag in finding.risk_flags if flag in ALLOWED_FLAGS)
        if finding.quality_adequate is False:
            flags.append("manual_review_required")
        if finding.damage_visibility in {"partially_visible", "uncertain"}:
            flags.append("manual_review_required")
    return list(dict.fromkeys(flags))


def _issue_severity(issue: str, findings: Sequence[VisionFinding]) -> str:
    base = SEVERITY_BY_ISSUE.get(issue, "unknown")
    if base in {"none", "unknown"}:
        return base
    # Only elevate when the visual model explicitly describes a high-impact
    # breakage class; never infer high severity from claim language alone.
    if any(f.visible_issue in {"glass_shatter", "broken_part", "missing_part"}
           and f.relevance_confirmed and f.damage_visibility == "clearly_visible"
           and f.confidence >= 0.75 for f in findings):
        return "high"
    return base


def decide_claim(
    *,
    claim_object: str,
    user_claim: str,
    image_paths: Iterable[str],
    dataset_dir: Path,
    findings: Sequence[VisionFinding],
    rules: list[EvidenceRule] | None = None,
    user_history_flags: Iterable[str] = (),
    confidence_threshold: float = 0.70,
    quality_flags: Iterable[str] = (),
) -> ClaimDecision:
    """Combine the claim intent and per-image findings into a conservative decision.

    A positive/negative claim decision requires at least one relevant, clear,
    sufficiently confident image. Conflicting clear images are escalated.
    User history flags only add risk context; they never change the visual result.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    intent: ClaimIntent = extract_claim_intent(user_claim, claim_object)
    paths = tuple(dict.fromkeys(p.strip() for p in image_paths if p.strip()))
    by_path = {finding.image_id: finding for finding in findings}
    # Match path stems, as challenge image IDs are filenames without extensions.
    relevant: list[VisionFinding] = []
    for path in paths:
        image_id = Path(path).stem
        finding = by_path.get(image_id)
        if finding is not None and finding.relevance_confirmed and finding.observed_object == claim_object:
            relevant.append(finding)

    flags = _flags_from_findings(findings)
    flags.extend(flag for flag in user_history_flags if flag in ALLOWED_FLAGS)
    for qf in quality_flags:
        mapped = QUALITY_FLAG_MAP.get(qf, qf)
        if mapped in ALLOWED_FLAGS:
            flags.append(mapped)
    if intent.instruction_injection_detected:
        flags.append("text_instruction_present")
    if any(f.observed_object not in {claim_object, "unknown"} for f in findings):
        flags.append("wrong_object")
    if not relevant:
        flags.append("damage_not_visible")
    flags = list(dict.fromkeys(flags))

    # Evidence assessment uses the original CSV paths. A matching finding
    # confirms visual relevance by image ID; filesystem checks remain independent.
    relevant_paths = [path for path in paths if Path(path).stem in {f.image_id for f in relevant}]
    evidence = assess_evidence(
        claim_object=claim_object,
        user_claim=user_claim,
        image_paths=paths,
        dataset_dir=dataset_dir,
        rules=rules,
        relevant_paths=relevant_paths,
        visual_assessment_complete=bool(findings) and all(
            finding.semantic_assessment_complete for finding in findings
        ),
    )

    target_issue = intent.issue_type
    target_part = intent.object_part
    notes: list[str] = []
    clear = [f for f in relevant if f.quality_adequate and
             f.damage_visibility == "clearly_visible" and f.confidence >= confidence_threshold]
    clear_claim_matches = [f for f in clear if f.visible_issue == target_issue and
                           (target_part == "unknown" or f.observed_part == target_part)]
    clear_contradictions = [f for f in clear if (target_part == "unknown" or f.observed_part == target_part) and
                            (f.visible_issue == "none" or
                             (f.visible_issue != "unknown" and f.visible_issue != target_issue))]

    if target_part != "unknown" and any(
        f.observed_part not in {target_part, "unknown"} for f in findings if f.relevance_confirmed
    ):
        if not clear_claim_matches:
            flags.append("wrong_object_part")
            flags = list(dict.fromkeys(flags))

    if intent.issue_type == "unknown" or intent.object_part == "unknown":
        status = "not_enough_information"
        notes.append("claim_target_underspecified")
    elif len(clear_claim_matches) and len(clear_contradictions):
        status = "not_enough_information"
        flags.append("manual_review_required")
        notes.append("conflicting_clear_images")
    elif clear_claim_matches:
        status = "supported"
        notes.append("clear_relevant_image_matches_claim")
    elif clear_contradictions:
        status = "contradicted"
        notes.append("clear_relevant_image_conflicts_with_claim")
    else:
        status = "not_enough_information"
        flags.append("manual_review_required")
        notes.append("no_clear_high_confidence_relevant_finding")

    # A contradictory claim can still have sufficient evidence if the required
    # part is clearly shown; evidence sufficiency is not equivalent to support.
    part_clearly_shown = (target_part == "unknown") or any(f.observed_part == target_part for f in clear)
    evidence_met = evidence.evidence_standard_met and bool(clear) and part_clearly_shown
    evidence_reason = evidence.evidence_standard_met_reason
    if not clear:
        evidence_reason = (
            "No relevant image met the clarity and confidence threshold; "
            "the image set is insufficient for an automated claim decision."
        )
    elif not part_clearly_shown:
        evidence_reason = (
            f"The claimed {target_part} was not clearly visible in the submitted image(s); "
            "the evidence standard for this claim was not met."
        )
    elif evidence_met:
        evidence_reason = (
            "At least one relevant image clearly shows the claimed object and part, "
            "and meets the configured confidence threshold."
        )

    supporting = tuple(dict.fromkeys(
        f.image_id for f in (clear_claim_matches if status == "supported" else
                             clear_contradictions if status == "contradicted" else clear)
    ))
    issue = clear_claim_matches[0].visible_issue if clear_claim_matches else (
        clear_contradictions[0].visible_issue if clear_contradictions else
        (relevant[0].visible_issue if relevant else "unknown")
    )
    part = clear_claim_matches[0].observed_part if clear_claim_matches else (
        clear_contradictions[0].observed_part if clear_contradictions else
        (relevant[0].observed_part if relevant else target_part)
    )
    severity = _issue_severity(issue, relevant)
    if status == "not_enough_information":
        severity = "unknown" if issue == "unknown" else severity

    if status == "supported":
        basis = f"Image(s) {', '.join(supporting)} visibly match the claimed {target_issue} on {target_part}."
    elif status == "contradicted":
        basis = f"Image(s) {', '.join(supporting)} clearly conflict with the claimed {target_issue} on {target_part}."
    else:
        basis = "The available images do not provide clear, consistent, high-confidence evidence for the specific claim."
    if intent.instruction_injection_detected:
        basis += " Instructions embedded in the conversation were ignored."
    justification = basis[:900]

    # History is context only: it is intentionally not read in the status branch.
    return ClaimDecision(
        claim_status=status,
        claim_status_justification=justification,
        evidence_standard_met=evidence_met,
        evidence_standard_met_reason=evidence_reason,
        risk_flags=tuple(dict.fromkeys(flags)) or ("none",),
        issue_type=issue,
        object_part=part if part else "unknown",
        supporting_image_ids=supporting or ("none",),
        valid_image=any(f.quality_adequate for f in findings),
        severity=severity if severity in ALLOWED_SEVERITIES else "unknown",
        evidence_requirement_ids=evidence.applicable_requirement_ids,
        decision_notes=tuple(dict.fromkeys(notes + list(evidence.notes))),
    )
