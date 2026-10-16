"""Fakes for tests: devices, analyzer, a FlyBot stand-in, clock.

Nothing here touches the network. `fake_runtime(...)` wires a full Runtime from them.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

from sentinel.agent.data import AgentData
from sentinel.agent.guardrails import StaticKillSwitch
from sentinel.agent.policy import load_policy
from sentinel.agent.runtime import Runtime
from sentinel.agent.stores import MemoryApprovalStore, MemoryStateStore
from sentinel.agent.trace import MemoryTraceSink
from sentinel.agent.vision import quality_score
from sentinel.common.ids import ulid
from sentinel.common.schemas import Counts, DeviceAck, NewCatch, Observation, PhotoEvent, QualityOut
from sentinel.devices.base import Device, DeviceRegistry, not_supported, validate_image_params


class FakeClock:
    def __init__(self, start: datetime):
        self.t = start if start.tzinfo else start.replace(tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw) -> datetime:
        self.t = self.t + timedelta(**kw)
        return self.t


class FakeDevice(Device):
    """Scripted camera. `captures`: list of {"image": ref, "ai": {"n": int}} (or {"fail": code}) returned in
    order; when exhausted, a new `fake://<serial>/capture-<n>.jpg` is made up. Records every call."""

    def __init__(self, serial: str, source: str = "camos", captures: list[dict] | None = None,
                 image_settings: dict | None = None, battery: float | None = None):
        self.serial, self.source = serial, source
        self.battery = battery
        self.capabilities = frozenset({"capture", "capture_max", "set_image", "set_detection_roi", "set_capture_schedule"}
                                      if source == "camos" else {"capture", "set_capture_schedule"})
        self.captures = list(captures or [])
        self.image_settings = dict(image_settings or {"brightness": 0, "contrast": 50, "wdr": 0, "flicker": 3})
        self.calls: list[tuple[str, dict]] = []
        self.n = 0

    def _ack(self, method, **kw) -> DeviceAck:
        transport = "fake_iot" if self.source == "camos" else "fake_flybot"
        return DeviceAck(serial=self.serial, method=method, transport=transport, **kw)

    def capture(self, resolution="main", detection_id="trap-count") -> DeviceAck:
        self.calls.append(("capture", {"resolution": resolution}))
        self.n += 1
        if self.source == "ezviz":
            return self._ack("capture", ok=True, pending=True, request_id=f"job-{self.n}", result={"jobId": f"job-{self.n}"},
                             evidence=[f"flybot://sentinel/capture/{self.serial}"])
        spec = self.captures.pop(0) if self.captures else {}
        if spec.get("fail"):
            return self._ack("capture", ok=False, request_id=ulid(), error={"code": spec["fail"]})
        ref = spec.get("image") or f"fake://{self.serial}/capture-{self.n}.jpg"
        rid = ulid()
        return self._ack("capture", ok=True, request_id=rid, result={
            "image": {"key": ref, "w": 2560, "h": 1440}, "ai": spec.get("ai") or {}, "image_ref": ref,
            "resolution": resolution}, evidence=[f"ddb://sentinel-rpc/{rid}", ref])

    def set_image(self, params) -> DeviceAck:
        self.calls.append(("set_image", dict(params)))
        if self.source != "camos":
            return not_supported(self.serial, "set_image", "fake_flybot", "ezviz")
        clean, errors = validate_image_params(params)
        if errors:
            return self._ack("set_image", ok=False, error={"code": "BAD_VALUE", "errors": errors})
        self.image_settings.update(clean)
        return self._ack("set_image", ok=True, applied=True, request_id=f"shadow-v{len(self.calls)}",
                         result={"reported": dict(self.image_settings)},
                         evidence=[f"iot://camos/{self.serial}/shadow/desired#v{len(self.calls)}"])

    def get_image_settings(self):
        return dict(self.image_settings) if self.source == "camos" else None

    def battery_percent(self):
        return self.battery

    def set_detection_roi(self, polygon, detection_id="trap-count") -> DeviceAck:
        self.calls.append(("set_detection_roi", {"polygon": polygon}))
        return self._ack("set_detection_roi", ok=True, applied=True)

    def set_capture_schedule(self, every_min, from_, to, detection_id="trap-count") -> DeviceAck:
        self.calls.append(("set_capture_schedule", {"every_min": every_min, "from": from_, "to": to}))
        return self._ack("set_capture_schedule", ok=True, applied=True)

    def actions(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeAnalyzer:
    """image ref -> scripted vision outputs:
    {"issues": [...], "opencv": int, "npu": int, "glare_ratio": float, "new_count": int, "saturation": float,
     "card_change": bool, "quad_norm": [[x, y] x4], "motion": {op 8 dict}, "scores": {full op 2 scores}}. Unknown images: quality ok, opencv = production/npu count."""

    name = "fake"

    def __init__(self, specs: dict[str, dict] | None = None):
        self.specs = dict(specs or {})
        self.calls: list[str] = []

    def analyze(self, event: PhotoEvent, previous_image=None) -> Observation:
        self.calls.append(event.image)
        s = self.specs.get(event.image, {})
        issues = list(s.get("issues", []))
        scores = s.get("scores") or {"glare_ratio": s.get("glare_ratio", 0.0), "clip_high": s.get("clip_high", 0.0),
                                     "brightness": s.get("brightness", 128.0)}
        q = QualityOut(ok=not issues, issues=issues, scores=scores)
        q.score = quality_score(q)
        default = event.production_count if event.production_count is not None else event.npu_count
        nc = None
        if "new_count" in s or s.get("card_change") or previous_image:  # op 4 for both kinds (lure = activity)
            sticky = event.trap_kind == "sticky"
            nc = NewCatch(new_count=s.get("new_count", 0), persisted_count=s.get("persisted", 0),
                          disappeared_count=s.get("departed", 0), meaning="catches" if sticky else "activity",
                          inliers=s.get("inliers", 120), registration_ok=True,
                          card_change_suspected=sticky and bool(s.get("card_change")),
                          card_change_reasons=["disappeared>>persisted"] if sticky and s.get("card_change") else [])
        stem = (event.image or "x").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return Observation(
            event_id=event.event_id, camera=event.camera, source=event.source, trap_kind=event.trap_kind,
            captured_at=event.captured_at, image=event.image, previous_image=previous_image,
            resolution=event.resolution, width=event.width, height=event.height, quality=q,
            counts=Counts(opencv=s.get("opencv", default), npu=s.get("npu", event.npu_count),
                          production=event.production_count),
            new_catch=nc, saturation=s.get("saturation"),
            card={"found": True, "quad_norm": s["quad_norm"]} if s.get("quad_norm") else None,
            motion=s.get("motion"), privacy=s.get("privacy"),
            snapshot_id=event.snapshot_id, battery_percent=event.battery_percent, analyzer=self.name, vision_ms=1.0,
            evidence=[f"evidence://{event.camera}/{stem}_quality.jpg"])

    def compare(self, prev, cur, trap_kind):
        s = self.specs.get(cur, {})
        return {"new_count": s.get("new_count", 0), "card_change_suspected": bool(s.get("card_change")),
                "evidence": [f"evidence://compare/{cur.rsplit('/', 1)[-1]}_heatmap.jpg"]}


class FakeFlyBot:
    """Stands in for FlyBotClient; records every POST body."""

    def __init__(self, fail: set[str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail = set(fail or ())
        self.n = 0

    def _post(self, path, body):
        self.calls.append((path, deepcopy(body)))
        if path in self.fail:
            raise RuntimeError(f"FlyBot {path}: HTTP 503")
        self.n += 1
        return {"ok": True, "id": f"{path}-{self.n}", "proposalId": f"prop-{self.n}" if path == "proposals" else None}

    def capture(self, device_serial):
        return self._post("capture", {"deviceSerial": device_serial})

    def capture_interval(self, device_serial, interval_minutes, until_iso):
        return self._post("capture-interval", {"deviceSerial": device_serial, "intervalMinutes": interval_minutes,
                                               "untilISO": until_iso})

    def proposal(self, type_, device_serials, payload, evidence):
        return self._post("proposals", {"type": type_, "deviceSerials": device_serials, "payload": payload,
                                        "evidence": evidence})

    def review_request(self, snapshot_id, reason, **extra):
        return self._post("review-request", {"snapshotId": snapshot_id, "reason": reason, **extra})

    def analysis(self, snapshot_id, counts, boxes, quality, evidence):
        return self._post("analysis", {"snapshotId": snapshot_id, "counts": counts, "boxes": boxes,
                                       "quality": quality, "evidence": evidence})

    def paths(self) -> list[str]:
        return [c[0] for c in self.calls]


def fake_weather(days_rain: dict[str, float] | None = None, tmin: float = 24.0, tmax: float = 33.0):
    """weather_fn(lat, lon, today) -> 31 past + 7 forecast days."""
    rain = dict(days_rain or {})

    def fn(lat, lon, today):
        from sentinel.agent.domain import add_days

        return [{"day": add_days(today, i), "min": tmin, "max": tmax, "rain": rain.get(add_days(today, i), 0.0),
                 "forecast": i > 0} for i in range(-31, 8)]

    return fn


def fake_runtime(*, cameras: list[dict], devices: list[FakeDevice] | None = None, analyzer: FakeAnalyzer | None = None,
                 flybot: FakeFlyBot | None = None, llm=None, planner: str = "rule", kill_switch=None, clock=None,
                 policy=None, weather=None) -> Runtime:
    policy = policy or load_policy()
    state = MemoryStateStore()
    return Runtime(policy=policy, devices=DeviceRegistry({d.serial: d for d in devices or []}),
                   analyzer=analyzer or FakeAnalyzer(), data=AgentData(state, policy, cameras, weather or fake_weather()),
                   state=state, approvals=MemoryApprovalStore(), traces=MemoryTraceSink(),
                   kill_switch=kill_switch or StaticKillSwitch(False), flybot=flybot or FakeFlyBot(), llm=llm,
                   planner=planner, clock=clock or FakeClock(datetime(2026, 10, 15, 2, 30, tzinfo=UTC)))
