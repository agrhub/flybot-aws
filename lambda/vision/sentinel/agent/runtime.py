"""Runtime (the agent's dependencies) and RunContext (one run: an event, its trace, guarded tool calls).

`RunContext.call(name, args)` is the only way a planner touches the world:
validate args -> dedup -> guardrails.check -> tool -> guardrails.after -> trace (+ evidence index).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import ValidationError

from sentinel.agent import qbaseline
from sentinel.agent.guardrails import Guardrails, KillSwitch, sanitize_texts
from sentinel.agent.tools import TOOLS
from sentinel.agent.vision import quality_score
from sentinel.common.ids import iso, ulid, utcnow
from sentinel.common.schemas import (
    Decision,
    DeviceAck,
    Observation,
    PhotoEvent,
    PolicyConfig,
    ToolCall,
    Trace,
    TraceEvent,
)


@dataclass
class Runtime:
    policy: PolicyConfig
    devices: object  # DeviceRegistry
    analyzer: object  # vision.Analyzer
    data: object  # data.AgentData
    state: object  # stores.StateStore
    approvals: object | None  # stores.ApprovalStore
    traces: object  # trace.TraceSink
    kill_switch: KillSwitch
    flybot: object | None = None  # flybot.client.FlyBotClient
    llm: object | None = None  # llm.LLMClient
    planner: str = "rule"  # rule | llm
    clock: Callable[[], datetime] = utcnow
    post_analysis: bool = True  # POST /api/sentinel/analysis for photos FlyBot knows (snapshot_id)
    new_id: Callable[[], str] = ulid  # replay passes a deterministic counter
    timings: bool = True  # wall-clock ms in tool calls (off in replay: byte-identical traces)
    # Optional: hand each new follow-up to an external scheduler (e.g. EventBridge Scheduler, a one-off schedule at
    # item["at"] whose target calls loops.run_followup(item)). Called with the item dict; its return value is kept
    # as a reference. None = the follow-up is only stored and loops.run_followups() picks it up when due.
    followup_scheduler: Callable[[dict], object] | None = None
    guard: Guardrails = field(init=False)

    def __post_init__(self):
        self.guard = Guardrails(self.policy, self.state, self.kill_switch, self.devices)

    def now(self) -> datetime:
        return self.clock()

    def set_followup_scheduler(self, callback: Callable[[dict], object] | None) -> None:
        """Inject (or remove, with None) the external follow-up scheduler. See `followup_scheduler`."""
        self.followup_scheduler = callback

    def schedule_followup(self, item: dict) -> dict:
        """Store a follow-up {id, at, kind, camera, reason, event_id} and, when a scheduler is injected, hand it
        over. Externally scheduled items stay stored (the record and the de-duplication key); run_followups()
        leaves them to the scheduler unless they are long overdue. A scheduler error keeps the polling path."""
        record = dict(item)
        if self.followup_scheduler is not None:
            try:
                ref = self.followup_scheduler(dict(item))
                record["scheduler"] = {"external": True, "ref": None if ref is None else str(ref)}
            except Exception as e:  # noqa: BLE001 - fall back to the polling path
                record["scheduler"] = {"external": False, "error": f"{type(e).__name__}: {e}"[:200]}
        self.state.update(item["camera"], lambda s: s.setdefault("followups", []).append(record))
        return record


class RunContext:
    def __init__(self, runtime: Runtime, trace: Trace, event: PhotoEvent | None = None):
        self.runtime, self.trace, self.event = runtime, trace, event
        self.policy = runtime.policy
        self.guard = runtime.guard
        self.observations: dict[str, Observation] = {}
        self.photos: dict[str, PhotoEvent] = {}
        self.evidence: set[str] = set()
        self.neighbour_captures = 0
        self.recaptures = 0
        self._cache: dict[str, dict] = {}
        self._kill: dict[str | None, tuple[bool, str | None]] = {}
        self._battery: dict[str, float | None] = {}
        self._battery_noted: set[str] = set()
        if event is not None and event.image:
            self.photos[event.image] = event
            self.evidence.add(event.image)

    # ------------------------------------------------------------ basics
    def now(self) -> datetime:
        return self.runtime.now()

    @property
    def state(self) -> str:
        return self.trace.state

    def set_state(self, state: str) -> None:
        self.trace.state = state

    def note(self, kind: str, detail: dict | None = None) -> None:
        self.trace.events.append(TraceEvent(at=self.now(), kind=kind, detail=detail or {}))

    def kill_switch(self, camera: str | None) -> tuple[bool, str | None]:
        if camera not in self._kill:
            self._kill[camera] = self.runtime.kill_switch.check(camera)
            if self._kill[camera][0]:
                self.trace.kill_switch = True
        return self._kill[camera]

    def battery(self, camera: str) -> float | None:
        """Battery % of `camera`: this run's photo, the device, or the last known value; None = mains / unknown."""
        if camera not in self._battery:
            v = None
            obs = self.observations.get(camera)
            if obs is not None and obs.battery_percent is not None:
                v = obs.battery_percent
            elif self.event is not None and self.event.camera == camera and self.event.battery_percent is not None:
                v = self.event.battery_percent
            else:
                dev = self.runtime.devices.get(camera)
                v = dev.battery_percent() if dev is not None else None
                if v is None:
                    v = self.runtime.data.camera(camera).get("battery_percent")
            self._battery[camera] = v
        return self._battery[camera]

    def trap_kind(self, camera: str) -> str:
        obs = self.observations.get(camera)
        if obs is not None:
            return obs.trap_kind
        if self.event is not None and self.event.camera == camera:
            return self.event.trap_kind
        return self.runtime.data.camera(camera).get("trap_kind") or "sticky"

    def ack(self, ack: DeviceAck) -> None:
        self.trace.device_ack.append(ack)

    # ------------------------------------------------------------ photos / observations
    def register_photo(self, ev: PhotoEvent) -> None:
        self.photos[ev.image] = ev
        self.evidence.add(ev.image)

    def photo_event(self, camera: str, image: str | None) -> PhotoEvent | None:
        if image is None:
            if self.event is not None and self.event.camera == camera:
                return self.event
            return None
        if image in self.photos:
            return self.photos[image]
        cam = self.runtime.data.camera(camera)
        return PhotoEvent(event_id=f"{camera}:{abs(hash(image)) % 10**10}", camera=camera,
                          source=cam.get("source") or "camos", trap_kind=cam.get("trap_kind") or "sticky",
                          captured_at=self.now(), image=image)

    def last_image(self, camera: str, exclude: str | None = None) -> str | None:
        ref = self.runtime.state.get(camera).get("last_image")
        return None if ref == exclude else ref

    def observe(self, obs: Observation, ev: PhotoEvent) -> Observation:
        """Store an analysis: sanitize untrusted text, score quality, record the count, settle ISP checks."""
        texts, flags = sanitize_texts({**ev.untrusted_text, **obs.untrusted_text}, self.policy)
        obs.untrusted_text = texts
        if flags:
            self.note("untrusted_text_flagged", {"camera": obs.camera, "fields": flags})
        if obs.battery_percent is None:
            obs.battery_percent = ev.battery_percent if ev.battery_percent is not None else self.battery(obs.camera)
        rows = self.runtime.state.get(obs.camera).get("qbaseline") or []
        added = qbaseline.rejudge(obs, rows, self.policy)
        if added:
            self.note("quality_baseline_issues", {"camera": obs.camera, "image": obs.image, "added": added,
                                                  "baseline": obs.quality.baseline})
        if obs.quality.score is None or added:
            obs.quality.score = quality_score(obs.quality)
        self.observations[obs.camera] = obs
        key = obs.camera if obs.camera not in self.trace.opencv_outputs else f"{obs.camera}@{len(self.trace.opencv_outputs)}"
        self.trace.opencv_outputs[key] = obs.model_dump(mode="json")
        self.evidence.update(e for e in obs.evidence if e)
        if obs.image:
            self.evidence.add(obs.image)
            self.runtime.state.update(obs.camera, lambda s: s.__setitem__("last_image", obs.image))
        why_not = qbaseline.not_countable(obs, self.policy)
        if why_not:
            self.note("photo_not_counted", {"camera": obs.camera, "image": obs.image, "why": why_not})
        else:
            self.runtime.data.record_count(obs.camera, obs.captured_at or self.now(), best_count(obs), obs.image)
        row = qbaseline.accepted_row(obs)
        if row is not None:
            size = self.policy.quality.baseline_size

            def keep(s, row=row, size=size):
                s["qbaseline"] = [*(s.get("qbaseline") or []), row][-size:]

            self.runtime.state.update(obs.camera, keep)
        self.guard.verify_isp(self, obs)
        return obs

    # ------------------------------------------------------------ the guarded call
    def call(self, name: str, args: dict, *, by: str = "planner", internal: bool = False) -> dict:
        idx = len(self.trace.tool_calls)
        started, t0 = self.now(), time.perf_counter()
        before = self.state
        self._decisions_before = len(self.trace.decisions)
        if name not in TOOLS:
            return self._record(idx, name, args, "blocked", {"blocked": True, "reason": f"unknown tool {name}"},
                                before, started, t0, by, "unknown tool")
        model = TOOLS[name][0]
        try:
            parsed = model.model_validate(args)
        except ValidationError as e:
            errs = [{"loc": list(x["loc"]), "msg": x["msg"]} for x in e.errors()][:5]
            return self._record(idx, name, args, "error", {"error": "invalid arguments", "details": errs, "evidence": []},
                                before, started, t0, by)
        clean = parsed.model_dump(by_alias=True, exclude_none=True)
        key = name + json.dumps(clean, sort_keys=True, default=str)
        if key in self._cache and not internal:
            return self._record(idx, name, clean, "dedup", {**self._cache[key], "dedup_of": self._cache[key].get("_call")},
                                before, started, t0, by)
        verdict = self.guard.check(self, name, clean, internal=internal, by=by)
        if not verdict.allowed:
            self.note("guardrail_block", {"tool": name, "code": verdict.code, "reason": verdict.reason})
            res = {"blocked": True, "code": verdict.code, "reason": verdict.reason, "evidence": []}
            rec = self._record(idx, name, clean, "blocked", res, before, started, t0, by, verdict.reason)
            if verdict.code == "downgraded":
                rec["downgraded_to"] = self._downgrade(name, clean, verdict.reason)
                call = next(c for c in self.trace.tool_calls if c.index == idx)
                call.result = {**call.result, "downgraded_to": rec["downgraded_to"]}
                decision = Decision(rule="guardrail.llm_ungrounded", action="request_human_review",
                                    reasons=[verdict.reason], evidence=[e for e in clean.get("evidence") or []
                                                                        if e in self.evidence])
                self.trace.decisions.append(decision)
            return rec
        try:
            result = TOOLS[name][1](self, parsed)
        except Exception as e:  # a tool failure is data for the planner and the trace  # noqa: BLE001 - a tool failure is data for the planner
            return self._record(idx, name, clean, "error",
                                {"error": f"{type(e).__name__}: {e}"[:500], "evidence": []}, before, started, t0, by)
        failed = result.get("error") is not None and result.get("ok") is not True or result.get("ok") is False
        if not failed:
            self.guard.after(self, name, clean, result, internal=internal)
        result.setdefault("evidence", [])
        result["evidence"] = [*result["evidence"], f"tool:{idx}"]
        self.evidence.update(e for e in result["evidence"] if isinstance(e, str))
        result["_call"] = idx
        rec = self._record(idx, name, clean, "error" if failed else "ok", result, before, started, t0, by)
        if not failed:
            self._cache[key] = result
            if name == "set_image" and not internal:
                self._verification(clean["camera"], result)
        return rec

    def _downgrade(self, name: str, args: dict, why: str) -> dict:
        """An ungrounded LLM proposal becomes a human review of the same traps (the LLM's text is not forwarded)."""
        camera = args.get("camera") or (args.get("cameras") or [None])[0]
        kind = "treatment" if name == "propose_treatment" else "card change"
        ev = [e for e in args.get("evidence") or [] if e in self.evidence]
        r = self.call("request_human_review", {"camera": camera, "evidence": ev,
                                               "reason": f"The AI planner wanted a {kind} proposal that the photos do not "
                                                         f"support. Please look at this trap. ({why[:200]})"},
                      by="guardrail")
        return {k: r.get(k) for k in ("ok", "review_id", "blocked", "reason", "error") if k in r}

    def _verification(self, camera: str, result: dict) -> None:
        """Guardrail: an ISP change is always followed by a verification capture (or a scheduled one)."""
        cap = self.call("capture", {"camera": camera, "resolution": "main", "purpose": "verify_isp",
                                    "reason": "verify ISP change"}, by="guardrail")
        if cap.get("blocked") or cap.get("error"):
            at = iso(self.now().replace(microsecond=0))
            fu = self.call("schedule_followup", {"camera": camera, "at": at, "kind": "verify_isp",
                                                 "reason": f"verification capture blocked: {cap.get('reason') or cap.get('error')}"},
                           by="guardrail", internal=True)
            result["verification_followup"] = fu.get("followup")
        result["verification_capture"] = {k: cap.get(k) for k in ("ok", "image", "pending", "blocked", "reason", "error")
                                          if k in cap}

    def _record(self, idx, name, args, status, result, before, started, t0, by, reason=None) -> dict:
        decisions_before = getattr(self, "_decisions_before", None)
        self.trace.tool_calls.append(ToolCall(index=idx, name=name, args=args, status=status, blocked_reason=reason,
                                              result=_compact(result), state_before=before, state_after=self.state,
                                              started_at=started, by=by, decisions_before=decisions_before,
                                              ms=round((time.perf_counter() - t0) * 1000, 2) if self.runtime.timings else None))
        return result


def best_count(obs: Observation) -> int | None:
    c = obs.counts
    return c.opencv if c.opencv is not None else (c.production if c.production is not None else c.npu)


def _compact(result: dict) -> dict:
    """Tool results in the trace without bulky box lists."""
    out = {}
    for k, v in result.items():
        if k == "_call":
            continue
        if k == "observation" and isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk not in ("boxes",)}
        out[k] = v
    return json.loads(json.dumps(out, default=str))
