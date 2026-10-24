"""End-to-end Gemini-first claim processing pipeline.

This module connects safe image loading, Gemini findings, evidence rules,
history risk context, and the deterministic claim decision engine.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from code.config import AppConfig
from code.decision_engine import decide_claim
from code.evidence_rules import load_evidence_rules
from code.history_risk import assess_user_history
from code.image_inspector import inspect_one_image
from code.image_utils import ImageLoadError, normalize_dataset_image
from code.vision_provider import VisionProviderError, analyze_with_retries, build_vision_provider
from code.vision_efficiency import EfficientVisionProvider

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def split_image_paths(value: str) -> list[str]:
    """Split semicolon-delimited paths, preserving order and removing duplicates."""
    return list(dict.fromkeys(item.strip() for item in value.split(";") if item.strip()))


def _history_context(user_id: str, dataset_dir: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        assessment = assess_user_history(user_id, dataset_dir / "user_history.csv")
        flags = tuple(flag for flag in assessment.risk_flags
                      if flag in {"user_history_risk", "manual_review_required"})
        return assessment.to_dict(), flags
    except (OSError, ValueError) as exc:
        flags = ("manual_review_required",)
        return {"user_id": user_id, "found": False,
                "risk_flags": list(flags),
                "reasons": [f"history_unavailable:{type(exc).__name__}"]}, flags


def process_claim(
    row: dict[str, str],
    config: AppConfig,
    provider: Any,
    rules: list[Any],
    *,
    row_number: int = 1,
) -> dict[str, Any]:
    """Process one claim while isolating failures to individual images."""
    started = time.monotonic()
    user_id = (row.get("user_id") or "").strip()
    user_claim = (row.get("user_claim") or "").strip()
    claim_object = (row.get("claim_object") or "unknown").strip().lower()
    paths = split_image_paths(row.get("image_paths") or "")
    history, history_flags = _history_context(user_id, config.dataset_dir)
    findings = []
    image_records: list[dict[str, Any]] = []
    errors: list[str] = []
    quality_flags: list[str] = []
    model_calls = 0
    model_attempts = 0
    provider_requests_start = getattr(provider, 'provider_requests', 0)
    cache_hits_start = getattr(provider, 'cache_hits', 0)

    def _increment_attempts() -> None:
        nonlocal model_attempts
        model_attempts += 1

    for relative_path in paths:
        observation = inspect_one_image(config.dataset_dir, relative_path, config)
        record = observation.to_dict()
        if observation.quality is not None:
            quality_flags.extend(observation.quality.quality_flags)
        if not observation.load_ok or observation.quality is None:
            errors.append(f"{observation.image_id}:image_unusable")
            image_records.append(record)
            continue
        if not observation.quality.usable_for_review:
            errors.append(f"{observation.image_id}:quality_below_threshold")
            image_records.append(record)
            continue
        try:
            normalized = normalize_dataset_image(
                config.dataset_dir, relative_path,
                max_bytes=config.max_image_bytes,
                max_pixels=config.max_image_pixels,
                max_dimension=config.max_image_dimension,
            )
            finding = analyze_with_retries(
                provider, normalized, claim_object=claim_object,
                user_claim=user_claim, max_retries=config.max_retries,
                on_attempt=lambda _attempt: _increment_attempts(),
            )
            findings.append(finding)
            record.update(finding.to_dict())
            model_calls += 1
        except (VisionProviderError, ImageLoadError, ValueError, RuntimeError) as exc:
            errors.append(f"{observation.image_id}:{type(exc).__name__}")
            record["semantic_assessment_complete"] = False
            record["error_code"] = "vision_analysis_failed"
            record["error_message"] = str(exc)[:200]
        image_records.append(record)

    decision = decide_claim(
        claim_object=claim_object,
        user_claim=user_claim,
        image_paths=paths,
        dataset_dir=config.dataset_dir,
        findings=findings,
        rules=rules,
        user_history_flags=history_flags,
        quality_flags=quality_flags,
    ).to_dict()
    if errors:
        current_flags = [f for f in decision["risk_flags"] if f not in {"none", ""}]
        decision["risk_flags"] = list(dict.fromkeys(current_flags + ["manual_review_required"])) or ["manual_review_required"]
        decision["decision_notes"] = list(decision["decision_notes"]) + ["image_or_model_processing_errors"]
        if decision["claim_status"] == "not_enough_information":
            decision["claim_status_justification"] = (
                "One or more images could not be analyzed; available evidence is incomplete."
            )
    return {
        "row_number": row_number,
        "user_id": user_id,
        "image_paths": ";".join(paths),
        "user_claim": user_claim,
        "claim_object": claim_object,
        "decision": decision,
        "image_findings": image_records,
        "history_assessment": history,
        "processing_seconds": round(time.monotonic() - started, 4),
        "model_calls": model_calls,
        "model_attempts": model_attempts,
        "provider_requests": getattr(provider, "provider_requests", 0) - provider_requests_start,
        "cache_hits": getattr(provider, "cache_hits", 0) - cache_hits_start,
        "error": ";".join(errors) if errors else None,
    }


def read_claims(path: Path) -> list[dict[str, str]]:
    """Read and validate the assignment's input CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "image_paths", "user_claim", "claim_object"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Claims CSV missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def _output_row(result: dict[str, Any]) -> dict[str, str]:
    decision = result["decision"]
    rf = [f for f in decision["risk_flags"] if f not in {"none", ""}]
    return {
        "user_id": result["user_id"],
        "image_paths": result["image_paths"],
        "user_claim": result["user_claim"],
        "claim_object": result["claim_object"],
        "evidence_standard_met": str(bool(decision["evidence_standard_met"])).lower(),
        "evidence_standard_met_reason": decision["evidence_standard_met_reason"],
        "risk_flags": ";".join(rf) if rf else "none",
        "issue_type": decision["issue_type"],
        "object_part": decision["object_part"],
        "claim_status": decision["claim_status"],
        "claim_status_justification": decision["claim_status_justification"],
        "supporting_image_ids": ";".join(decision["supporting_image_ids"]),
        "valid_image": str(bool(decision["valid_image"])).lower(),
        "severity": decision["severity"],
    }


def write_output(results: list[dict[str, Any]], path: Path) -> None:
    """Write exact output schema atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_output_row(item) for item in results)
    temporary.replace(path)


def write_audit(results: list[dict[str, Any]], path: Path) -> None:
    """Write detailed local audit records; never include API credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_pipeline(
    config: AppConfig,
    *,
    claims_csv: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run claims through configured Gemini provider and write CSV plus JSONL audit."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    config.validate_credentials()
    base_provider = build_vision_provider(config)
    provider = EfficientVisionProvider(
        base_provider,
        cache_path=config.repo_root / ".cache" / "vision_cache.sqlite3",
        cache_enabled=config.cache_enabled,
        requests_per_minute=config.request_rate_per_minute,
        provider_name=config.provider,
        model_name=config.model_name,
    )
    source = claims_csv or config.dataset_dir / "claims.csv"
    rows = read_claims(source)
    if limit is not None:
        rows = rows[:limit]
    rules = load_evidence_rules(config.dataset_dir / "evidence_requirements.csv")
    results = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{total}] Processing claim {row.get('user_id', 'unknown')} ({row.get('claim_object', 'unknown')})...", flush=True)
        res = process_claim(row, config, provider, rules, row_number=index)
        print(f"    -> Claim {index} status: {res['decision']['claim_status']} (valid_image={res['decision']['valid_image']})", flush=True)
        results.append(res)
    write_output(results, config.output_csv)
    write_audit(results, config.output_csv.with_name("audit.jsonl"))
    return results
