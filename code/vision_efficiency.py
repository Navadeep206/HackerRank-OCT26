"""Persistent response caching and conservative request pacing for vision providers.

Cache keys include the model/provider, normalized image digest, object type, and
claim text. Cache contents may contain sensitive claim-derived data; keep the
cache local and excluded from version control.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from code.image_utils import NormalizedImage
from code.vision_provider import VisionProviderError, validate_finding


CACHE_VERSION = "vision-finding-v1"


class EfficientVisionProvider:
    """Wrap a provider with SQLite response caching and per-process RPM pacing."""

    def __init__(
        self,
        provider: Any,
        *,
        cache_path: Path,
        cache_enabled: bool,
        requests_per_minute: int,
        provider_name: str,
        model_name: str,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self._provider = provider
        self._cache_path = cache_path
        self._cache_enabled = cache_enabled
        self._interval = 60.0 / requests_per_minute
        self._provider_name = provider_name
        self._model_name = model_name
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_request_at: float | None = None
        self.provider_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _key(self, image: NormalizedImage, claim_object: str, user_claim: str) -> str:
        payload = {
            "version": CACHE_VERSION,
            "provider": self._provider_name,
            "model": self._model_name,
            "image_sha256": image.sha256,
            "claim_object": claim_object,
            "user_claim": user_claim,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if not self._cache_enabled or not self._cache_path.is_file():
            return None
        try:
            with sqlite3.connect(self._cache_path, timeout=5) as connection:
                row = connection.execute("SELECT payload FROM findings WHERE cache_key = ?", (key,)).fetchone()
            if row is None:
                return None
            value = json.loads(row[0])
            return value if isinstance(value, dict) else None
        except (sqlite3.Error, OSError, json.JSONDecodeError):
            # A cache is an optimization, never a source of truth or a reason to fail a claim.
            return None

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        if not self._cache_enabled:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._cache_path, timeout=5) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS findings (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT OR REPLACE INTO findings(cache_key, payload) VALUES (?, ?)",
                    (key, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
                )
        except (sqlite3.Error, OSError, TypeError, ValueError):
            # Disk/cache failures should fall back to the live provider.
            return

    def _pace(self) -> None:
        # Hold the lock through sleep and reservation so concurrent callers cannot
        # burst past the configured per-process request interval.
        with self._lock:
            now = self._clock()
            if self._last_request_at is not None:
                wait = self._interval - (now - self._last_request_at)
                if wait > 0:
                    self._sleep(wait)
                    now = self._clock()
            self._last_request_at = now
            self.provider_requests += 1

    def analyze(self, image: NormalizedImage, *, claim_object: str, user_claim: str) -> dict[str, Any]:
        key = self._key(image, claim_object, user_claim)
        cached = self._read_cache(key)
        if cached is not None:
            try:
                validate_finding(cached, expected_object=claim_object, image_id=image.image_id)
            except (VisionProviderError, ValueError, TypeError):
                cached = None
            else:
                self.cache_hits += 1
                return cached
        self.cache_misses += 1
        self._pace()
        payload = self._provider.analyze(image, claim_object=claim_object, user_claim=user_claim)
        # Never cache malformed model output: retries must be able to recover.
        validate_finding(payload, expected_object=claim_object, image_id=image.image_id)
        self._write_cache(key, payload)
        return payload
