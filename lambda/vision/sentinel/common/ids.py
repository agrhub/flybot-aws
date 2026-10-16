"""ULIDs (protocol 3: `id` is a ULID made by the sender) and small time helpers."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(ts_ms: int | None = None) -> str:
    """26-character Crockford base32 ULID: 48-bit ms timestamp + 80 random bits."""
    ts = int(time.time() * 1000) if ts_ms is None else int(ts_ms)
    value = (ts << 80) | int.from_bytes(os.urandom(10), "big")
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_dt(value) -> datetime | None:
    """ISO string, unix seconds/ms or datetime -> aware UTC datetime (None stays None)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        secs = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(secs, UTC)
    text = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
