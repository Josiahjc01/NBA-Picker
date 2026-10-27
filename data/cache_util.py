"""Simple JSON/CSV cache with age checks and fingerprint invalidation."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config

logger = logging.getLogger(__name__)


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def content_fingerprint(payload: Any) -> str:
    """Stable short hash of JSON-serializable payload for cache invalidation."""
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_fresh(
    path: Path,
    max_age_hours: float | None = None,
    *,
    expected_fingerprint: str | None = None,
) -> bool:
    max_age = max_age_hours if max_age_hours is not None else config.CACHE_MAX_AGE_HOURS
    meta = _meta_path(path)
    if not path.exists() or not meta.exists():
        return False
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(info["fetched_at"])
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0
        if age_h >= max_age:
            return False
        if expected_fingerprint is not None:
            stored = info.get("fingerprint")
            if stored and stored != expected_fingerprint:
                logger.info(
                    "Cache fingerprint mismatch for %s (%s -> %s)",
                    path.name,
                    stored,
                    expected_fingerprint,
                )
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bad cache meta for %s: %s", path, exc)
        return False


def write_json_cache(
    path: Path,
    payload: Any,
    *,
    fingerprint: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    fp = fingerprint if fingerprint is not None else content_fingerprint(payload)
    meta: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fp,
    }
    if extra_meta:
        meta.update(extra_meta)
    _meta_path(path).write_text(json.dumps(meta), encoding="utf-8")


def read_json_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading cache %s: %s", path, exc)
        return None


def read_cache_meta(name: str) -> dict[str, Any] | None:
    path = _meta_path(config.CACHE_DIR / f"{name}.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def invalidate_cache(name: str) -> None:
    """Drop a named cache payload + meta so the next load refetches."""
    path = config.CACHE_DIR / f"{name}.json"
    meta = _meta_path(path)
    for p in (path, meta):
        try:
            if p.exists():
                p.unlink()
                logger.info("Invalidated cache: %s", p.name)
        except OSError as exc:
            logger.warning("Could not invalidate %s: %s", p, exc)


def load_or_fetch_json(
    name: str,
    fetcher,
    max_age_hours: float | None = None,
    force: bool = False,
    *,
    fingerprint_fn: Callable[[Any], str] | None = None,
    validate: Callable[[Any], bool] | None = None,
) -> Any:
    """Load from cache/<name>.json if fresh, else call fetcher() and store."""
    path = config.CACHE_DIR / f"{name}.json"
    if not force and cache_fresh(path, max_age_hours):
        data = read_json_cache(path)
        if data is not None and (validate is None or validate(data)):
            logger.info("Cache hit: %s", name)
            return data
        if data is not None and validate is not None:
            logger.info("Cache failed validation for %s — refreshing", name)
    try:
        data = fetcher()
        if validate is not None and not validate(data):
            raise RuntimeError(f"Fetched {name} failed validation")
        fp = fingerprint_fn(data) if fingerprint_fn else content_fingerprint(data)
        write_json_cache(path, data, fingerprint=fp)
        logger.info("Cache refresh: %s (fp=%s)", name, fp)
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fetch failed for %s (%s); trying stale cache", name, exc)
        stale = read_json_cache(path)
        if stale is not None and (validate is None or validate(stale)):
            return stale
        raise
