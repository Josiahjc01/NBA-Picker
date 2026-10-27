"""Simple JSON/CSV cache with age checks."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def cache_fresh(path: Path, max_age_hours: float | None = None) -> bool:
    max_age = max_age_hours if max_age_hours is not None else config.CACHE_MAX_AGE_HOURS
    meta = _meta_path(path)
    if not path.exists() or not meta.exists():
        return False
    try:
        info = json.loads(meta.read_text())
        fetched = datetime.fromisoformat(info["fetched_at"])
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0
        return age_h < max_age
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bad cache meta for %s: %s", path, exc)
        return False


def write_json_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    _meta_path(path).write_text(
        json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat()})
    )


def read_json_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading cache %s: %s", path, exc)
        return None


def load_or_fetch_json(
    name: str,
    fetcher,
    max_age_hours: float | None = None,
    force: bool = False,
) -> Any:
    """Load from cache/<name>.json if fresh, else call fetcher() and store."""
    path = config.CACHE_DIR / f"{name}.json"
    if not force and cache_fresh(path, max_age_hours):
        data = read_json_cache(path)
        if data is not None:
            logger.info("Cache hit: %s", name)
            return data
    try:
        data = fetcher()
        write_json_cache(path, data)
        logger.info("Cache refresh: %s", name)
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fetch failed for %s (%s); trying stale cache", name, exc)
        stale = read_json_cache(path)
        if stale is not None:
            return stale
        raise
