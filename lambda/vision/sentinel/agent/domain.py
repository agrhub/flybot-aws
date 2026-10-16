"""Domain rules mirrored from FlyBot (ezviz-manager/server/src/services) so the agent and FlyBot agree.

- `detect_surge`: surgeService.detectSurge (today vs SURGE_DAYS earlier, ratio / +N / min count, skipped
  after a card change), branched on the trap kind: `sticky` compares daily peaks, `lure` compares the
  day's average per photo (total / scans), which is what FlyBot thresholds use for bait traps.
- `spray_suggest`, `dry_day`: sprayPlanService.suggest / dryDay.
- `day_degrees`, `next_wave`: degreeDayService (B. dorsalis, base 12 C, cutoff 35 C, 350 DD per generation).
- `disagreement`: NPU / OpenCV / production spread rule (policy.disagreement).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sentinel.common.schemas import PolicyConfig

# degreeDayService.js constants
BASE_TEMP = 12.0
UPPER_TEMP = 35.0
GENERATION_DD = 350.0
MIN_WAVE = 5
ANCHOR_DAYS = 30
WAVE_GRACE_DAYS = 2


def _d(key: str) -> date:
    return date.fromisoformat(key)


def add_days(key: str, n: int) -> str:
    return (_d(key) + timedelta(days=n)).isoformat()


def day_value(row: dict | None, basis: str) -> float | None:
    """A daily rollup row {peak, total, scans} -> the value the surge rule compares."""
    if not row:
        return None
    if basis == "daily_average":
        scans = row.get("scans") or 0
        if scans <= 0 or row.get("total") is None:
            return None
        return float(row["total"]) / float(scans)
    peak = row.get("peak")
    return None if peak is None else float(peak)


def detect_surge(daily: dict[str, dict], today: str, policy: PolicyConfig, trap_kind: str = "sticky",
                 card_replaced_days: list[str] | None = None) -> dict | None:
    """surgeService.detectSurge on a {day: {peak, total, scans}} map. Returns the rise or None."""
    s = policy.surge
    basis = policy.kind(trap_kind).count_basis
    base_key = add_days(today, -s.days)
    cur, base = day_value(daily.get(today), basis), day_value(daily.get(base_key), basis)
    if cur is None or base is None:
        return None
    if cur < s.min_count or cur - base < s.min_increase or cur < base * s.ratio:
        return None
    if policy.kind(trap_kind).card_changes and card_replaced_days:
        earliest = add_days(today, -s.card_replaced_skip_days)
        if any(earliest <= d <= today for d in card_replaced_days):
            return None
    pct = round((cur - base) / base * 100) if base > 0 else 100
    return {"previous": round(base, 2), "count": round(cur, 2), "days": s.days, "pct": pct, "basis": basis,
            "rule": f"count>={s.min_count} and rise>={s.min_increase} and count>={s.ratio}x {s.days}d ago"}


def with_today(daily: dict[str, dict], today: str, count: int | None) -> dict[str, dict]:
    """Fold the current photo's count into today's rollup (peak = max, total += count, scans += 1)."""
    out = {k: dict(v) for k, v in daily.items()}
    if count is None:
        return out
    row = out.setdefault(today, {"peak": None, "total": 0, "scans": 0})
    row["peak"] = count if row.get("peak") is None else max(row["peak"], count)
    row["total"] = (row.get("total") or 0) + count
    row["scans"] = (row.get("scans") or 0) + 1
    return out


def disagreement(counts: dict[str, int | None], policy: PolicyConfig) -> dict:
    """Spread of the available counts vs max(abs_min, rel x max)."""
    vals = {k: v for k, v in counts.items() if v is not None}
    if len(vals) < 2:
        return {"disagree": False, "sources": sorted(vals), "spread": None, "limit": None}
    hi, lo = max(vals.values()), min(vals.values())
    limit = max(policy.disagreement.abs_min, policy.disagreement.rel * hi)
    return {"disagree": (hi - lo) > limit, "sources": sorted(vals), "spread": hi - lo, "limit": round(limit, 2),
            "counts": vals}


def dry_day(from_key: str, outlook_days: list[dict] | None, policy: PolicyConfig) -> dict:
    """sprayPlanService.dryDay: first day from `from_key` not forecast >= rain_mm (at most 7 tries)."""
    wet = {d["day"] for d in (outlook_days or []) if d.get("rain") is not None and d["rain"] >= policy.treatment.rain_mm}
    day = from_key
    for _ in range(7):
        if day not in wet:
            break
        day = add_days(day, 1)
    return {"day": day, "moved": day != from_key}


def spray_suggest(*, at_threshold: int, last_treatment: dict | None, sprays: list[dict], wave: dict | None,
                  outlook: dict | None, today: str, now: datetime, policy: PolicyConfig) -> dict:
    """sprayPlanService.suggest: treat_now | wait_check | bait_before_wave | sanitation | monitor."""
    t = policy.treatment
    action, day, wait_until = "monitor", None, None
    if at_threshold > 0:
        since = (now - last_treatment["at"]).total_seconds() / 86400 if last_treatment else float("inf")
        if since < t.min_interval_days:
            action = "wait_check"
            wait_until = (last_treatment["at"] + timedelta(days=t.min_interval_days)).date().isoformat()
        else:
            action, day = "treat_now", today
    elif wave and wave.get("inDays") is not None and wave["inDays"] <= t.wave_window_days:
        action = "bait_before_wave"
        lead = add_days(wave["date"], -t.wave_lead_days)
        day = max(today, lead)
    elif outlook and outlook.get("level") not in (None, "low"):
        action, day = "sanitation", today
    rain_moved = False
    if day and action in ("treat_now", "bait_before_wave"):
        dry = dry_day(day, (outlook or {}).get("days"), policy)
        day, rain_moved = dry["day"], dry["moved"]
    rotate_from = None
    if len(sprays) >= 2 and action in ("treat_now", "wait_check"):
        a, b = sprays[0].get("activeIngredient"), sprays[1].get("activeIngredient")
        if a and b and a.strip().lower() == b.strip().lower():
            rotate_from = a
    return {"action": action, "date": day, "waitUntil": wait_until, "rainMoved": rain_moved, "rotateFrom": rotate_from}


def day_degrees(tmin: float, tmax: float) -> float:
    mean = (min(tmin, UPPER_TEMP) + min(tmax, UPPER_TEMP)) / 2
    return max(0.0, mean - BASE_TEMP)


def next_wave(temps: list[dict], catches_by_day: dict[str, float], today: str) -> dict:
    """degreeDayService.computeDegreeDays, reduced to the anchor and the next wave.
    temps: [{day, min, max}] oldest first (past + forecast)."""
    if not temps:
        return {"nextWave": None, "anchor": None, "avgDailyDd": None}
    dd = {t["day"]: day_degrees(t["min"], t["max"]) for t in temps}
    upcoming = [dd[t["day"]] for t in temps if t["day"] >= today][:7]
    avg = sum(upcoming) / len(upcoming) if upcoming else 0.0
    future = [dd[t["day"]] for t in temps if t["day"] > today]
    extra = sum(future) / len(future) if future else avg
    earliest = add_days(today, -ANCHOR_DAYS)
    anchor = None
    for day, value in sorted(catches_by_day.items()):
        if day < earliest or day > today or not value or value < MIN_WAVE:
            continue
        if anchor is None or value >= anchor["catches"]:
            anchor = {"day": day, "catches": value}
    wave = None
    if anchor:
        total, gen, day = 0.0, 1, add_days(anchor["day"], 1)
        while True:
            total += dd.get(day, extra)
            while total >= gen * GENERATION_DD:
                if day >= add_days(today, -WAVE_GRACE_DAYS):
                    wave = {"date": day, "inDays": (_d(day) - _d(today)).days, "generation": gen}
                    break
                gen += 1
            if wave or gen > 6 or (_d(day) - _d(today)).days > 120:
                break
            day = add_days(day, 1)
    return {"nextWave": wave, "anchor": anchor, "avgDailyDd": round(avg, 1),
            "generationDays": round(GENERATION_DD / avg) if avg > 0 else None,
            "model": {"species": "bactrocera_dorsalis", "baseTemp": BASE_TEMP, "upperTemp": UPPER_TEMP,
                      "generationDd": GENERATION_DD}}
