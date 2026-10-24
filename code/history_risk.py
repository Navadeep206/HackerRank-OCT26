"""User-history risk context, strictly separated from visual claim decisions."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ALLOWED_HISTORY_FLAGS = {"user_history_risk", "manual_review_required"}


@dataclass(frozen=True, slots=True)
class HistoryRiskAssessment:
    user_id: str
    found: bool
    past_claim_count: int
    accepted_claim_count: int
    manual_review_claim_count: int
    rejected_claim_count: int
    last_90_days_claim_count: int
    risk_flags: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["risk_flags"] = list(self.risk_flags)
        result["reasons"] = list(self.reasons)
        return result


def _integer(row: dict[str, str], field: str) -> int:
    try:
        value = int(row.get(field, "0") or 0)
        return max(value, 0)
    except (TypeError, ValueError):
        return 0


def assess_user_history(user_id: str, history_csv: Path) -> HistoryRiskAssessment:
    """Load one user row; history can add risk flags but never decide claim status.

    Explicit dataset flags are retained only if allowed. The summary is treated
    as descriptive context, not executable instructions or visual evidence.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if not history_csv.is_file():
        raise FileNotFoundError(f"User history file not found: {history_csv}")

    with history_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "past_claim_count", "accept_claim", "manual_review_claim",
                    "rejected_claim", "last_90_days_claim_count", "history_flags"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"User history CSV is missing columns: {', '.join(sorted(missing))}")
        matched = next((row for row in reader if row.get("user_id", "").strip() == user_id.strip()), None)

    if matched is None:
        return HistoryRiskAssessment(user_id.strip(), False, 0, 0, 0, 0, 0,
                                     ("manual_review_required",), ("user_history_not_found",))

    past = _integer(matched, "past_claim_count")
    accepted = _integer(matched, "accept_claim")
    reviewed = _integer(matched, "manual_review_claim")
    rejected = _integer(matched, "rejected_claim")
    recent = _integer(matched, "last_90_days_claim_count")
    flags: list[str] = []
    reasons: list[str] = []
    raw_flags = {part.strip() for part in (matched.get("history_flags") or "").split(";") if part.strip()}
    for flag in raw_flags.intersection(ALLOWED_HISTORY_FLAGS):
        flags.append(flag)
        reasons.append(f"dataset_flag:{flag}")

    # Conservative deterministic indicators: avoid treating small histories as
    # a fraud verdict. These are review signals only.
    if past >= 5 and rejected / max(past, 1) >= 0.40:
        flags.append("user_history_risk")
        reasons.append("high_rejection_ratio_in_recorded_history")
    if reviewed >= 2:
        flags.append("manual_review_required")
        reasons.append("multiple_prior_manual_reviews")
    if recent >= 5:
        flags.append("manual_review_required")
        reasons.append("high_recent_claim_volume")

    if past == 0:
        reasons.append("no_prior_claims_recorded")
    if not flags:
        reasons.append("no_history_risk_threshold_triggered")
    return HistoryRiskAssessment(
        user_id=user_id.strip(), found=True, past_claim_count=past,
        accepted_claim_count=accepted, manual_review_claim_count=reviewed,
        rejected_claim_count=rejected, last_90_days_claim_count=recent,
        risk_flags=tuple(dict.fromkeys(flags)) or ("none",),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def history_flags_for_user(user_id: str, history_csv: Path) -> tuple[str, ...]:
    """Convenience adapter for the decision engine's user_history_flags argument."""
    assessment = assess_user_history(user_id, history_csv)
    return tuple(flag for flag in assessment.risk_flags if flag in ALLOWED_HISTORY_FLAGS)
