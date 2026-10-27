"""Windows-safe text helpers (avoid cp1252 / charmap UnicodeEncodeError)."""

from __future__ import annotations

import sys
from typing import Any

# Common glyphs that break Windows console / default locale file writes
_REPLACEMENTS = (
    ("\u0394", "d"),  # Greek capital Delta
    ("\u03b4", "d"),  # Greek small delta
    ("\u2192", "->"),  # right arrow
    ("\u2190", "<-"),
    ("\u2022", "*"),
    ("\u2248", "~"),
    ("\u2265", ">="),
    ("\u2264", "<="),
    ("\u03c3", "sigma"),
)


def ascii_safe(text: Any, *, errors: str = "replace") -> str:
    """Return a string that encodes cleanly on Windows cp1252 / ascii consoles."""
    s = "" if text is None else str(text)
    for bad, good in _REPLACEMENTS:
        s = s.replace(bad, good)
    # Drop anything still outside Latin-1/cp1252-ish printable by encoding round-trip
    try:
        s.encode("cp1252")
        return s
    except UnicodeEncodeError:
        return s.encode("cp1252", errors=errors).decode("cp1252")


def write_text(path: Any, text: str) -> None:
    """Always write UTF-8 so Greek/symbols never hit the Windows locale codec."""
    from pathlib import Path

    Path(path).write_text(text if isinstance(text, str) else str(text), encoding="utf-8")


def safe_print(*args: Any, **kwargs: Any) -> None:
    """print() that won't raise UnicodeEncodeError on cp1252 stdout."""
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file = kwargs.get("file", sys.stdout)
    flush = kwargs.get("flush", False)
    line = sep.join(ascii_safe(a) for a in args) + end
    try:
        file.write(line)
    except UnicodeEncodeError:
        enc = getattr(file, "encoding", None) or "cp1252"
        file.buffer.write(line.encode(enc, errors="replace"))
        if end and not end.endswith("\n"):
            pass
    if flush:
        try:
            file.flush()
        except Exception:  # noqa: BLE001
            pass
