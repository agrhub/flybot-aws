"""What the read tools know about cameras, history and weather.

`AgentData` keeps the agent self-contained: camera facts come from a seed list (`SENTINEL_CAMERAS_FILE`,
JSON list) and are learned from FlyBot webhooks (farm, plot, GPS, trap kind); daily rollups
{peak, total, scans} per camera are recorded by the agent itself as photos are analyzed (state store
`daily`, 40 days), card changes and treatments from approved proposals and from FlyBot's
`cardReplacedAt`. Weather is a pluggable function: `OpenMeteoWeather` (real, httpx) or a fake.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta

import httpx

from sentinel.agent import domain
from sentinel.agent.stores import StateStore
from sentinel.common.ids import iso, parse_dt
from sentinel.common.schemas import PolicyConfig

KEEP_DAYS = 40


def haversine_m(a: dict, b: dict) -> float | None:
    if None in (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon")):
        return None
    r = 6371000.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp, dl = p2 - p1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


class OpenMeteoWeather:
    """Daily min/max temperature and rain (past 31 days + 7-day forecast) from api.open-meteo.com
    (no key). The same source family FlyBot's weatherService uses; cached per rounded location/day."""

    URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, http: httpx.Client | None = None):
        self._http = http or httpx.Client(timeout=10.0)
        self._cache: dict[tuple, list[dict]] = {}

    def __call__(self, lat: float, lon: float, today: str) -> list[dict] | None:
        key = (round(lat, 2), round(lon, 2), today)
        if key not in self._cache:
            resp = self._http.get(self.URL, params={
                "latitude": lat, "longitude": lon, "timezone": "auto", "past_days": 31, "forecast_days": 7,
                "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum"})
            resp.raise_for_status()
            d = resp.json()["daily"]
            self._cache[key] = [{"day": day, "min": lo, "max": hi, "rain": rain, "forecast": day > today}
                                for day, lo, hi, rain in zip(d["time"], d["temperature_2m_min"],
                                                             d["temperature_2m_max"], d["precipitation_sum"], strict=False)
                                if lo is not None and hi is not None]
        return self._cache[key]


class AgentData:
    def __init__(self, state: StateStore, policy: PolicyConfig, cameras: list[dict] | None = None, weather=None):
        self.state, self.policy = state, policy
        self._seed = {c["serial"]: dict(c) for c in (cameras or [])}
        self.weather_fn = weather

    @classmethod
    def seed_from_env(cls) -> list[dict]:
        """$SENTINEL_CAMERAS_FILE (JSON list of camera facts), else nothing: FlyBot sends the camera with each photo."""
        path = os.environ.get("SENTINEL_CAMERAS_FILE")
        if not path:
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------ cameras
    def camera(self, serial: str) -> dict:
        """learned facts (this run) < seed list; FlyBot's request is the source of truth for the trap kind."""
        st = self.state.get(serial)
        return {**(st.get("device") or {}), **(st.get("info") or {}), **self._seed.get(serial, {}), "serial": serial}

    def trap_kind_for(self, serial: str, event_kind: str, explicit: bool, source: str = "ezviz") -> tuple[str, dict | None]:
        """Trap kind to use for a photo, and the override detail when it differs from the event's. The event's
        kind wins when it was explicit (FlyBot's Camera.trapKind); the seed list fills in when it was not."""
        kind = self.camera(serial).get("trap_kind")
        if explicit or not kind or kind == event_kind:
            return event_kind, None
        return kind, {"event": event_kind, "registry": kind, "why": "event has no trap kind"}

    def cameras(self) -> list[dict]:
        serials = set(self._seed) | set(self.state.cameras())
        return [self.camera(s) for s in sorted(serials)]

    def learn(self, serial: str, facts: dict) -> None:
        facts = {k: v for k, v in facts.items() if v is not None}
        if facts:
            self.state.update(serial, lambda s: s.setdefault("info", {}).update(facts))

    def neighbors(self, serial: str, k: int) -> list[dict]:
        me = self.camera(serial)
        out = []
        for c in self.cameras():
            if c["serial"] == serial or c.get("excluded"):
                continue
            same_farm = me.get("farm_id") and c.get("farm_id") == me.get("farm_id")
            same_plot = me.get("plot_id") and c.get("plot_id") == me.get("plot_id")
            if not (same_farm or same_plot):
                continue
            dist = haversine_m(me, c)
            out.append({"serial": c["serial"], "source": c.get("source"), "trap_kind": c.get("trap_kind", "sticky"),
                        "plot_id": c.get("plot_id"), "distance_m": None if dist is None else round(dist, 1)})
        out.sort(key=lambda c: (c["plot_id"] != me.get("plot_id"), c["distance_m"] is None, c["distance_m"] or 0))
        return out[:k]

    # ------------------------------------------------------------ history
    def record_count(self, serial: str, at: datetime | None, count: int | None, image: str | None) -> bool:
        if count is None or at is None:
            return False
        day = domain_day(at, self.policy)
        recorded = {"done": False}

        def fn(s: dict):
            seen = s.setdefault("counted", [])
            if image and image in seen:
                return
            daily = s.setdefault("daily", {})
            row = daily.setdefault(day, {"peak": None, "total": 0, "scans": 0})
            row["peak"] = count if row.get("peak") is None else max(row["peak"], count)
            row["total"] = (row.get("total") or 0) + count
            row["scans"] = (row.get("scans") or 0) + 1
            for old in sorted(daily)[:-KEEP_DAYS]:
                daily.pop(old)
            if image:
                seen.append(image)
                del seen[:-200]
            recorded["done"] = True

        self.state.update(serial, fn)
        return recorded["done"]

    def note_card_replaced(self, serial: str, at: datetime | None) -> None:
        if at is None:
            return
        day = domain_day(at, self.policy)

        def fn(s: dict):
            days = s.setdefault("card_replaced_days", [])
            if day not in days:
                days.append(day)

        self.state.update(serial, fn)

    def note_treatment(self, serials: list[str], at: datetime, payload: dict) -> None:
        for serial in serials:
            self.state.update(serial, lambda s: s.setdefault("treatments", []).insert(
                0, {"at": iso(at), "type": payload.get("type", "spray"), "product": payload.get("product"),
                    "activeIngredient": payload.get("activeIngredient")}))

    def history(self, serial: str, days: int, today: str) -> dict:
        s = self.state.get(serial)
        first = domain.add_days(today, -days)
        daily = {d: v for d, v in (s.get("daily") or {}).items() if first <= d <= today}
        treatments = [{**t, "at": parse_dt(t["at"])} for t in (s.get("treatments") or [])]
        return {"daily": daily, "card_replaced_days": sorted(s.get("card_replaced_days") or []),
                "treatments": treatments}

    # ------------------------------------------------------------ weather / degree days
    def weather(self, serial: str, today: str) -> list[dict] | None:
        cam = self.camera(serial)
        if self.weather_fn is None or cam.get("lat") is None or cam.get("lon") is None:
            return None
        return self.weather_fn(cam["lat"], cam["lon"], today)

    def degree_day_outlook(self, serial: str, today: str) -> dict:
        temps = self.weather(serial, today)
        me = self.camera(serial)
        farm = [c for c in self.cameras() if c.get("farm_id") and c.get("farm_id") == me.get("farm_id")] or [me]
        catches: dict[str, float] = {}
        for c in farm:
            basis = self.policy.kind(c.get("trap_kind", "sticky")).count_basis
            daily = self.history(c["serial"], domain.ANCHOR_DAYS + 1, today)["daily"]
            prev = None
            for day in sorted(daily):
                v = domain.day_value(daily[day], basis)
                if v is None:
                    continue
                # sticky: a day's catch is the rise of the peak (cumulative card); lure: the day's average
                value = max(0.0, v - prev) if basis == "peak" and prev is not None else v
                prev = v if basis == "peak" else prev
                catches[day] = catches.get(day, 0.0) + value
        if not temps:
            return {"available": False, "reason": "no weather (no GPS or weather source)", "catches_by_day": catches}
        out = domain.next_wave(temps, catches, today)
        rainy = [d for d in temps if d["day"] >= today and (d.get("rain") or 0) >= self.policy.treatment.rain_mm]
        return {"available": True, **out, "days": [d for d in temps if d["day"] >= today][:7],
                "rainy_days": [d["day"] for d in rainy]}


def domain_day(at: datetime, policy: PolicyConfig) -> str:
    from sentinel.agent.policy import local_day

    return local_day(at, policy)


def last_treatment_within(history: dict, now: datetime, days: int) -> dict | None:
    for t in history.get("treatments") or []:
        if t.get("at") and now - t["at"] < timedelta(days=days):
            return t
    return None
