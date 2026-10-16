"""Load agent/policy.yaml into PolicyConfig; quiet-hours and local-day helpers."""

from __future__ import annotations

import os
from datetime import datetime, time
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from sentinel.common.schemas import PolicyConfig

POLICY_PATH = Path(__file__).with_name("policy.yaml")


@lru_cache(maxsize=8)
def _load(path: str) -> PolicyConfig:
    with open(path, encoding="utf-8") as f:
        return PolicyConfig.model_validate(yaml.safe_load(f))


def load_policy(path: str | Path | None = None) -> PolicyConfig:
    """Policy from `path`, $SENTINEL_POLICY or the packaged policy.yaml (a fresh copy each call)."""
    p = str(path or os.environ.get("SENTINEL_POLICY") or POLICY_PATH)
    return _load(p).model_copy(deep=True)


def policy_text(path: str | Path | None = None) -> str:
    p = Path(path or os.environ.get("SENTINEL_POLICY") or POLICY_PATH)
    return p.read_text(encoding="utf-8")


def _hm(text: str) -> time:
    h, m = text.split(":")
    return time(int(h), int(m))


def local(dt: datetime, policy: PolicyConfig) -> datetime:
    return dt.astimezone(ZoneInfo(policy.timezone))


def local_day(dt: datetime, policy: PolicyConfig) -> str:
    return local(dt, policy).date().isoformat()


def in_quiet_hours(dt: datetime, policy: PolicyConfig) -> bool:
    q = policy.budgets.quiet_hours
    start, end, now = _hm(q.start), _hm(q.end), local(dt, policy).time()
    if start <= end:
        return start <= now < end
    return now >= start or now < end
