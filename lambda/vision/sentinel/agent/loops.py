"""Agent loops, called by the vision Lambda handler (sentinel/handler.py) and the tests.

    per_photo(event, runtime)            one trap photo: analyze -> plan -> guarded actions -> trace
    daily_per_plot(day, runtime)         approvals sweep, due follow-ups, per-plot capture schedule
    sweep_approvals(runtime)             expire proposals older than policy.approval.timeout_h (no-op)
    run_followups(runtime)               execute due follow-ups (verification / recheck captures)
    record_approval_decision(...)        FlyBot told us a person approved / rejected a proposal

`runtime` defaults to `wiring.runtime_from_env()`; the handler and tests pass their own.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sentinel.agent.events import normalize_event
from sentinel.agent.planner import LLMPlanner, RulePlanner, decide
from sentinel.agent.policy import local_day
from sentinel.agent.runtime import RunContext, Runtime, best_count
from sentinel.agent.stores import due
from sentinel.common.ids import iso, parse_dt
from sentinel.common.schemas import PhotoEvent, Trace, TraceEvent

_default: Runtime | None = None


def default_runtime() -> Runtime:
    global _default
    if _default is None:
        from sentinel.agent.wiring import runtime_from_env

        _default = runtime_from_env()
    return _default


def make_planner(rt: Runtime):
    if rt.planner == "llm" and rt.llm is not None:
        return LLMPlanner(rt.llm)
    return RulePlanner()


def _finish(rc: RunContext) -> Trace:
    t = rc.trace
    if t.state == "ACTED" and (t.followup or rc.runtime.state.get(t.camera or "").get("pending_isp")):
        t.state = "FOLLOWUP"
    if t.kill_switch and not t.decisions:
        decide(rc, "kill_switch", "no_op", ["kill switch on: no actions"])
    rules = list(dict.fromkeys(d.rule for d in t.decisions))
    if t.kill_switch and all(c.status != "ok" or c.name.startswith(("get_", "analyze", "compare")) for c in t.tool_calls):
        t.outcome = "kill_switch_no_action"
    elif t.approval is not None and t.approval.status == "pending":
        t.outcome = f"waiting_approval:{t.approval.kind} ({', '.join(rules)})"
    else:
        t.outcome = ", ".join(rules) or "no_op"
    blocked = [f"{c.name}({c.result.get('code')})" for c in t.tool_calls if c.status == "blocked"]
    if blocked and t.outcome != "kill_switch_no_action":
        t.outcome += f"; blocked: {', '.join(blocked)}"
    if not t.reasoning_summary:
        t.reasoning_summary = " ".join(f"[{d.rule}] {'; '.join(d.reasons)}" for d in t.decisions)[:2000]
    t.finished_at = rc.now()
    rc.runtime.traces.write(t)
    return t


def per_photo(event, runtime=None) -> Trace:
    """One trap photo. `runtime`: a Runtime, or None (wiring.runtime_from_env)."""
    return _per_photo(event, runtime or default_runtime())


def on_approval(proposal: dict, rt: Runtime) -> Trace | None:
    """A proposal was approved outside (FlyBot): record it on the run that made it."""
    for a in rt.approvals.pending():
        if a.flybot_proposal_id == proposal.get("proposal_id"):
            return record_approval_decision(a.approval_id, "approved", rt)
    return None


def _per_photo(event, rt: Runtime) -> Trace:
    ev = normalize_event(event, rt.data.camera)
    existing = rt.traces.get(ev.event_id)
    if existing is not None and existing.finished_at is not None and existing.loop == "per_photo":
        return existing  # webhook retry / SQS redelivery: the same photo is never acted on twice
    info = rt.data.camera(ev.camera)
    if info.get("excluded"):  # not a trap camera (e.g. AC6527557 films a building): never analyzed or acted on
        t = Trace(event_id=ev.event_id, camera=ev.camera, source=ev.source, created_at=rt.now(), state="INGESTED",
                  outcome="excluded_camera", finished_at=rt.now(), policy_version=rt.policy.version)
        rt.traces.write(t)
        return t
    # FlyBot's request carries Camera.trapKind (explicit); the seed list only fills a missing kind
    ev.trap_kind, override = rt.data.trap_kind_for(ev.camera, ev.trap_kind, bool(ev.raw.get("trapKind")), ev.source)
    if ev.battery_percent is None and info.get("battery_percent") is not None:
        ev.battery_percent = info["battery_percent"]
    rt.data.learn(ev.camera, {"source": ev.source, "trap_kind": ev.trap_kind, "battery_percent": ev.battery_percent,
                              "farm_id": ev.raw.get("farmId"), "plot_id": ev.raw.get("plotId"),
                              "lat": ev.raw.get("latitude"), "lon": ev.raw.get("longitude")})
    if ev.card_replaced_at:
        rt.data.note_card_replaced(ev.camera, ev.card_replaced_at)
    trace = Trace(event_id=ev.event_id, camera=ev.camera, source=ev.source, trap_kind=ev.trap_kind,
                  created_at=rt.now(), policy_version=rt.policy.version)
    rc = RunContext(rt, trace, ev)
    if override:
        rc.note("trap_kind_override", {"camera": ev.camera, **override})
    on, why = rc.kill_switch(ev.camera)
    if on:
        rc.note("kill_switch", {"source": why, "effect": "analysis only, planner not run, no actions"})
        rc.call("analyze_trap_photo", {"camera": ev.camera})
        return _finish(rc)
    make_planner(rt).plan(rc)
    _post_analysis(rc, ev)
    return _finish(rc)


def _post_analysis(rc: RunContext, ev: PhotoEvent) -> None:
    """Store Sentinel's analysis on the FlyBot snapshot (aiResults.sentinel) for photos FlyBot knows."""
    rt = rc.runtime
    obs = rc.observations.get(ev.camera)
    if not (rt.post_analysis and rt.flybot is not None and ev.snapshot_id and obs is not None):
        return
    try:
        res = rt.flybot.analysis(ev.snapshot_id, {**obs.counts.model_dump(), "best": best_count(obs),
                                                  "new": obs.new_catch.new_count if obs.new_catch else None},
                                 obs.boxes[:200], obs.quality.model_dump(mode="json"), obs.evidence)
        rc.note("flybot_analysis_posted", {"snapshot_id": ev.snapshot_id, "response": res})
    except Exception as e:  # noqa: BLE001 - a FlyBot failure must not fail the run
        rc.note("flybot_analysis_failed", {"snapshot_id": ev.snapshot_id, "error": f"{type(e).__name__}: {e}"[:300]})


# ---------------------------------------------------------------- approvals


def _update_trace(rt: Runtime, event_id: str, approval, kind: str, outcome_suffix: str, detail: dict) -> Trace | None:
    t = rt.traces.get(event_id)
    if t is None:
        return None
    t.approval = approval
    t.state = "FOLLOWUP"  # the person (or the timeout) answered: the run is in its follow-up phase
    t.events.append(TraceEvent(at=rt.now(), kind=kind, detail=detail))
    t.outcome = f"{t.outcome} -> {outcome_suffix}"
    rt.traces.write(t)
    return t


def sweep_approvals(runtime: Runtime | None = None) -> list[Trace]:
    rt = runtime or default_runtime()
    if rt.approvals is None:
        return []
    now, out = rt.now(), []
    for a in rt.approvals.pending():
        if a.expires_at > now:
            continue
        a.status, a.decided_at = "expired", now
        rt.approvals.put(a)
        t = _update_trace(rt, a.event_id, a, "approval_expired",
                          f"approval_expired_{a.on_timeout}",
                          {"approval_id": a.approval_id, "kind": a.kind, "cameras": a.cameras,
                           "age_h": round((now - a.created_at).total_seconds() / 3600, 1),
                           "effect": "no action taken (policy.approval.on_timeout = no_op)"})
        if t is not None:
            out.append(t)
    return out


def record_approval_decision(approval_id: str, status: str, runtime: Runtime | None = None,
                             at: datetime | None = None) -> Trace | None:
    """A person decided in FlyBot. Approved treatment -> remembered + post-treatment verification follow-up;
    approved card change -> card day remembered (restarts the sticky surge baseline). Never writes FlyBot."""
    rt = runtime or default_runtime()
    a = rt.approvals.get(approval_id) if rt.approvals else None
    if a is None or a.status != "pending" or status not in ("approved", "rejected"):
        return None
    now = at or rt.now()
    a.status, a.decided_at = status, now
    rt.approvals.put(a)
    detail = {"approval_id": a.approval_id, "kind": a.kind, "status": status}
    if status == "approved" and a.kind == "treatment":
        rt.data.note_treatment(a.cameras, now, {"type": "spray", **a.payload})
        check = now + timedelta(days=rt.policy.treatment.post_treatment_check_days)
        for cam in a.cameras:
            item = {"id": f"{a.approval_id}:{cam}", "at": iso(check), "kind": "post_treatment_check", "camera": cam,
                    "reason": "verify the approved treatment's effect", "event_id": a.event_id}
            rt.schedule_followup(item)
        detail["followup_at"] = iso(check)
    if status == "approved" and a.kind == "card_change":
        for cam in a.cameras:
            rt.data.note_card_replaced(cam, now)
    return _update_trace(rt, a.event_id, a, f"approval_{status}", f"approval_{status}", detail)


# ---------------------------------------------------------------- follow-ups and the daily loop


EXTERNAL_GRACE = timedelta(hours=1)  # an externally scheduled follow-up this overdue is run by the poller


def _execute_followup(rt: Runtime, cam: str, f: dict) -> list[Trace]:
    """A guarded capture for follow-up `f`; the captured photo then goes through per_photo."""
    trace = Trace(event_id=f"followup:{f['id']}", loop="followup", camera=cam, created_at=rt.now(),
                  policy_version=rt.policy.version, state="FOLLOWUP", followup={**f, "of": f.get("event_id")})
    rc = RunContext(rt, trace)
    purpose = {"post_treatment_check": "post_treatment_check", "verify_isp": "verify_isp"}.get(f["kind"], "followup")
    cap = rc.call("capture", {"camera": cam, "purpose": purpose, "reason": f.get("reason") or f["kind"]},
                  by="followup")
    decide(rc, f"followup.{f['kind']}", "capture", [f.get("reason") or f["kind"]])
    out = [_finish(rc)]
    if cap.get("image") and cap["image"] in rc.photos:
        out.append(_per_photo(rc.photos[cap["image"]], rt))
    return out


def run_followups(runtime: Runtime | None = None) -> list[Trace]:
    """Polling path: run every due follow-up. Follow-ups handed to an external scheduler
    (Runtime.followup_scheduler) are left to it, unless they are more than EXTERNAL_GRACE overdue."""
    rt = runtime or default_runtime()
    now, out = rt.now(), []
    for cam in rt.state.cameras():
        ready, later = due(rt.state.get(cam).get("followups") or [], now)
        mine = [f for f in ready if not (f.get("scheduler") or {}).get("external")
                or now - (parse_dt(f.get("at")) or now) > EXTERNAL_GRACE]
        if not mine:
            continue
        keep = later + [f for f in ready if f not in mine]
        rt.state.update(cam, lambda s, keep=keep: s.__setitem__("followups", keep))
        for f in mine:
            out.extend(_execute_followup(rt, cam, f))
    return out


def run_followup(item: dict | str, runtime: Runtime | None = None, camera: str | None = None) -> list[Trace]:
    """Scheduler path: run ONE follow-up (the item the scheduler was given, or its id plus `camera`). It is
    removed from the store first, so a repeated delivery or the poller never runs it twice ([] = no longer
    pending)."""
    rt = runtime or default_runtime()
    fid = item if isinstance(item, str) else item.get("id")
    cam = camera or (None if isinstance(item, str) else item.get("camera"))
    if not fid or not cam:
        raise ValueError("run_followup needs the follow-up id and its camera")
    found: dict = {}

    def pop(s):
        rest = []
        for f in s.get("followups") or []:
            if f.get("id") == fid and not found:
                found.update(f)
            else:
                rest.append(f)
        s["followups"] = rest

    rt.state.update(cam, pop)
    return _execute_followup(rt, cam, found) if found else []


def daily_per_plot(day: str | None = None, runtime: Runtime | None = None) -> list[Trace]:
    rt = runtime or default_runtime()
    now = rt.now()
    day = day or local_day(now, rt.policy)
    out = sweep_approvals(rt) + run_followups(rt)
    plots: dict[str, list[dict]] = {}
    for c in rt.data.cameras():
        if c.get("excluded"):
            continue
        plots.setdefault(c.get("plot_id") or c.get("farm_id") or f"camera:{c['serial']}", []).append(c)
    sp = rt.policy.schedule
    for plot, cams in sorted(plots.items()):
        trace = Trace(event_id=f"daily:{plot}:{day}", loop="daily_per_plot", plot=plot, created_at=now,
                      policy_version=rt.policy.version)
        rc = RunContext(rt, trace)
        on, why = rc.kill_switch(cams[0]["serial"])
        dd = rc.call("get_degree_day_outlook", {"camera": cams[0]["serial"]})
        rc.set_state("ANALYZED")  # the plot outlook is this loop's analysis
        wave = dd.get("nextWave")
        if on:
            rc.note("kill_switch", {"source": why})
        elif wave and wave.get("inDays") is not None and wave["inDays"] <= sp.wave_soon_days:
            decide(rc, "schedule.wave_soon", "set_capture_schedule",
                   [f"fly wave due {wave['date']} (in {wave['inDays']} d): photograph every {sp.wave_every_min} min"],
                   dd.get("evidence", []))
            for c in cams:
                st = rt.state.get(c["serial"])
                if st.get("dense_schedule_until", "") >= wave["date"]:
                    continue
                r = rc.call("set_capture_schedule", {"camera": c["serial"], "every_min": sp.wave_every_min,
                                                     "from": sp.from_, "to": sp.to, "reason": "fly wave due"})
                if r.get("ok"):
                    rt.state.update(c["serial"], lambda s, d=wave["date"]: s.__setitem__("dense_schedule_until", d))
        else:
            decide(rc, "schedule.keep", "no_op", [f"no wave within {sp.wave_soon_days} days ({wave})"], dd.get("evidence", []))
        out.append(_finish(rc))
    return out
