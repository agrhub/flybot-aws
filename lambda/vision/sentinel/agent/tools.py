"""The 14 Sentinel tools. One definition serves three callers:

- RulePlanner and LLMPlanner call them through `RunContext.call(name, args)` (guardrails + trace);
- `TOOL_SPECS` gives name / description / JSON schema to the LLM backends (Bedrock Converse toolSpec,
  Anthropic tools) and to the MCP server (mcp_server/server.py).

Every result is a JSON dict with an `evidence` list (S3 / FlyBot / IoT refs, evidence images, plus
`tool:<n>` for the call itself) that a later proposal can cite.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.agent import domain
from sentinel.agent.guardrails import clean_reason
from sentinel.agent.policy import local_day
from sentinel.common.ids import iso, parse_dt
from sentinel.common.schemas import Approval, PhotoEvent


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


Camera = Field(description="Camera serial (EZVIZ serial or CamOS AI camera serial)")
Evidence = Field(default_factory=list, description="Evidence refs returned by earlier tool calls in this run "
                                                   "(image refs, evidence paths, or 'tool:<n>')")


class AnalyzeArgs(_Args):
    camera: str = Camera
    image: str | None = Field(None, description="Image ref; default: the photo that triggered this run")
    previous_image: str | None = Field(None, description="Previous photo of this trap; default: the last one seen")


class CompareArgs(_Args):
    camera: str = Camera
    image: str = Field(description="Current image ref")
    previous_image: str | None = Field(None, description="Previous image ref; default: the last one seen")


class HistoryArgs(_Args):
    camera: str = Camera
    days: int = Field(7, ge=1, le=40, description="Days of daily rollups to return")


class NeighborsArgs(_Args):
    camera: str = Camera
    k: int = Field(3, ge=1, le=10, description="How many nearest traps on the same farm/plot")


class CameraArgs(_Args):
    camera: str = Camera


class CaptureArgs(_Args):
    camera: str = Camera
    resolution: Literal["main", "max"] = Field("main", description="'max' = 4 MP full sensor (AI camera only)")
    purpose: Literal["recapture", "verify_isp", "neighbour_check", "closer_look", "followup",
                     "post_treatment_check"] = "recapture"
    reason: str = ""


class SetImageArgs(_Args):
    camera: str = Camera
    brightness: int | None = Field(None, ge=-64, le=64)
    contrast: int | None = Field(None, ge=0, le=100)
    wdr: int | None = Field(None, ge=0, le=2, description="0 off, 1 manual, 2 auto")
    flicker: int | None = Field(None, ge=0, le=3, description="0 off, 1 50 Hz, 2 60 Hz, 3 auto")
    evidence: list[str] = Evidence
    reason: str = ""


class RoiArgs(_Args):
    camera: str = Camera
    polygon: list[list[float]] = Field(description="ROI polygon, normalized 0-1 [[x, y], ...] (>= 3 points)")
    evidence: list[str] = Evidence
    reason: str = ""


class ScheduleArgs(_Args):
    camera: str = Camera
    every_min: int = Field(ge=5, le=720, description="Minutes between scheduled captures")
    from_: str = Field("06:00", alias="from", pattern=r"^\d{2}:\d{2}$", description="Farm-local HH:MM")
    to: str = Field("18:00", pattern=r"^\d{2}:\d{2}$", description="Farm-local HH:MM")
    reason: str = ""


class ProposeCardArgs(_Args):
    camera: str = Camera
    kind: Literal["card_full", "card_replaced_detected"] = "card_full"
    reason: str
    evidence: list[str] = Evidence


class ProposeTreatmentArgs(_Args):
    cameras: list[str] = Field(min_length=1, description="Traps (serials) the treatment is for")
    action: Literal["treat_now", "bait_before_wave", "sanitation"]
    date: str | None = Field(None, description="Suggested farm-local date YYYY-MM-DD")
    reason: str
    rotate_from: str | None = Field(None, description="Active ingredient to rotate away from")
    evidence: list[str] = Evidence


class ReviewArgs(_Args):
    camera: str = Camera
    reason: str
    snapshot_id: str | None = None
    image: str | None = None
    evidence: list[str] = Evidence


class FollowupArgs(_Args):
    camera: str = Camera
    at: str = Field(description="ISO 8601 time to act")
    kind: Literal["capture", "recheck_surge", "post_treatment_check", "verify_isp"] = "capture"
    reason: str = ""


# ---------------------------------------------------------------- implementations


def _obs_summary(obs) -> dict:
    d = obs.model_dump(mode="json")
    d["boxes_count"] = len(d.pop("boxes") or [])
    return d


def t_analyze(rc, a: AnalyzeArgs) -> dict:
    ev = rc.photo_event(a.camera, a.image)
    if ev is None or not ev.image:
        return {"error": f"no image for camera {a.camera}", "evidence": []}
    prev = a.previous_image or rc.last_image(a.camera, exclude=ev.image)
    obs = rc.runtime.analyzer.analyze(ev, prev)
    obs = rc.observe(obs, ev)
    return {"observation": _obs_summary(obs), "evidence": list(obs.evidence)}


def t_compare(rc, a: CompareArgs) -> dict:
    prev = a.previous_image or rc.last_image(a.camera, exclude=a.image)
    if not prev:
        return {"error": "no previous photo for this trap", "evidence": []}
    fn = getattr(rc.runtime.analyzer, "compare", None)
    if fn is None:
        return {"error": "analyzer has no compare()", "evidence": []}
    res = fn(prev, a.image, rc.trap_kind(a.camera))
    return {**res, "evidence": [a.image, prev, *res.get("evidence", [])]}


def t_history(rc, a: HistoryArgs) -> dict:
    today = local_day(rc.now(), rc.policy)
    kind = rc.trap_kind(a.camera)
    h = rc.runtime.data.history(a.camera, a.days, today)
    surge = domain.detect_surge(h["daily"], today, rc.policy, kind, h["card_replaced_days"])
    last = h["treatments"][0] if h["treatments"] else None
    return {"camera": a.camera, "trap_kind": kind, "basis": rc.policy.kind(kind).count_basis, "today": today,
            "daily": h["daily"], "card_replaced_days": h["card_replaced_days"],
            "last_treatment": {**last, "at": iso(last["at"])} if last else None,
            "treatments": [{**t, "at": iso(t["at"])} for t in h["treatments"][:5]],
            "surge": surge, "evidence": [f"history://{a.camera}/{today}/{a.days}d"]}


def t_neighbors(rc, a: NeighborsArgs) -> dict:
    nb = rc.runtime.data.neighbors(a.camera, a.k)
    return {"camera": a.camera, "neighbors": nb, "evidence": [f"registry://{a.camera}/neighbors"]}


def t_degree_day(rc, a: CameraArgs) -> dict:
    today = local_day(rc.now(), rc.policy)
    out = rc.runtime.data.degree_day_outlook(a.camera, today)
    return {"camera": a.camera, **out, "evidence": [f"degree-day://{a.camera}/{today}"]}


def t_weather(rc, a: CameraArgs) -> dict:
    today = local_day(rc.now(), rc.policy)
    temps = rc.runtime.data.weather(a.camera, today)
    if temps is None:
        return {"camera": a.camera, "available": False, "evidence": []}
    days = [d for d in temps if d["day"] >= domain.add_days(today, -3)][:10]
    return {"camera": a.camera, "available": True, "days": days,
            "rainy_days": [d["day"] for d in days if (d.get("rain") or 0) >= rc.policy.treatment.rain_mm],
            "evidence": [f"weather://{a.camera}/{today}"]}


def t_capture(rc, a: CaptureArgs) -> dict:
    dev = rc.runtime.devices.get(a.camera)
    ack = dev.capture(a.resolution)
    rc.ack(ack)
    ref = (ack.result or {}).get("image_ref")
    out = {"ok": ack.ok, "camera": a.camera, "purpose": a.purpose, "resolution": a.resolution, "image": ref,
           "pending": ack.pending, "transport": ack.transport, "request_id": ack.request_id,
           "npu_count": ((ack.result or {}).get("ai") or {}).get("n"), "error": ack.error,
           "evidence": list(ack.evidence)}
    if ack.ok and ack.result.get("photo_event"):  # replay: the device returns the full next photo event
        pev = PhotoEvent.model_validate(ack.result["photo_event"])
        pev.trap_kind, override = rc.runtime.data.trap_kind_for(a.camera, pev.trap_kind, bool(pev.raw.get("trapKind")),
                                                                pev.source)
        if override:
            rc.note("trap_kind_override", {"camera": a.camera, "image": pev.image, **override})
        rc.register_photo(pev)
    elif ack.ok and ref:
        image = ack.result.get("image") or {}
        rc.register_photo(PhotoEvent(
            event_id=f"{a.camera}:{ack.request_id or rc.runtime.new_id()}", camera=a.camera, source=dev.source,
            trap_kind=rc.trap_kind(a.camera), captured_at=rc.now(), image=ref, width=image.get("w"),
            height=image.get("h"), resolution=a.resolution, trigger=f"agent:{a.purpose}",
            npu_count=out["npu_count"]))
    elif ack.ok and ack.pending:
        out["note"] = "photo will arrive through FlyBot's snapshot_analyzed webhook (new per-photo run)"
    return out


def t_set_image(rc, a: SetImageArgs) -> dict:
    dev = rc.runtime.devices.get(a.camera)
    current = {**rc.policy.quality.isp_defaults, **(dev.get_image_settings() or {})}
    wanted = {k: getattr(a, k) for k in ("brightness", "contrast", "wdr", "flicker") if getattr(a, k) is not None}
    if not wanted:
        return {"ok": False, "error": "no ISP field given", "evidence": []}
    ack = dev.set_image(wanted)
    rc.ack(ack)
    return {"ok": ack.ok, "camera": a.camera, "before_settings": {k: current.get(k) for k in wanted},
            "applied_settings": wanted, "applied": ack.applied, "pending": ack.pending, "error": ack.error,
            "reason": clean_reason(a.reason), "evidence": list(ack.evidence)}


def t_roi(rc, a: RoiArgs) -> dict:
    ack = rc.runtime.devices.get(a.camera).set_detection_roi(a.polygon)
    rc.ack(ack)
    return {"ok": ack.ok, "camera": a.camera, "polygon": a.polygon, "applied": ack.applied, "error": ack.error,
            "evidence": list(ack.evidence)}


def t_schedule(rc, a: ScheduleArgs) -> dict:
    ack = rc.runtime.devices.get(a.camera).set_capture_schedule(a.every_min, a.from_, a.to)
    rc.ack(ack)
    return {"ok": ack.ok, "camera": a.camera, "every_min": a.every_min, "from": a.from_, "to": a.to,
            "applied": ack.applied, "error": ack.error, "evidence": list(ack.evidence)}


def _approval(rc, kind: str, cameras: list[str], payload: dict, evidence: list[str]) -> dict:
    fb = rc.runtime.flybot
    if fb is None:
        return {"ok": False, "error": "FlyBot Sentinel API not configured (FLYBOT_API_URL, FLYBOT_SENTINEL_SECRET)",
                "evidence": []}
    res = fb.proposal(kind, cameras, payload, evidence)
    now = rc.now()
    appr = Approval(approval_id=rc.runtime.new_id(), event_id=rc.trace.event_id, kind=kind, cameras=cameras, payload=payload,
                    evidence=evidence, created_at=now, expires_at=now + timedelta(hours=rc.policy.approval.timeout_h),
                    flybot_proposal_id=str(res.get("proposalId") or res.get("id") or "") or None,
                    on_timeout=rc.policy.approval.on_timeout)
    if rc.runtime.approvals is not None:
        rc.runtime.approvals.put(appr)
    rc.trace.approval = appr
    return {"ok": True, "approval_id": appr.approval_id, "status": "pending", "kind": kind,
            "flybot_proposal_id": appr.flybot_proposal_id, "expires_at": iso(appr.expires_at),
            "note": "a person confirms or rejects this in FlyBot; nothing is logged until then",
            "evidence": [f"flybot://proposals/{appr.flybot_proposal_id or appr.approval_id}"]}


def t_propose_card(rc, a: ProposeCardArgs) -> dict:
    payload = {"kind": a.kind, "reason": clean_reason(a.reason),
               "observation": _brief(rc.observations.get(a.camera))}
    return _approval(rc, "card_change", [a.camera], payload, a.evidence)


def t_propose_treatment(rc, a: ProposeTreatmentArgs) -> dict:
    payload = {"action": a.action, "date": a.date, "reason": clean_reason(a.reason), "rotateFrom": a.rotate_from,
               "basis": "sprayPlanService.suggest (mirrored in sentinel/agent/domain.py)",
               "observations": {c: _brief(rc.observations.get(c)) for c in a.cameras}}
    return _approval(rc, "treatment", list(a.cameras), payload, a.evidence)


def t_review(rc, a: ReviewArgs) -> dict:
    fb = rc.runtime.flybot
    if fb is None:
        return {"ok": False, "error": "FlyBot Sentinel API not configured", "evidence": []}
    obs = rc.observations.get(a.camera)
    snap = a.snapshot_id or (obs.snapshot_id if obs else None) or (rc.event.snapshot_id if rc.event else None)
    image = a.image or (obs.image if obs else None)
    res = fb.review_request(snap, clean_reason(a.reason), device_serial=a.camera, image=image, evidence=a.evidence)
    return {"ok": True, "camera": a.camera, "snapshot_id": snap, "review_id": res.get("id"),
            "evidence": [f"flybot://review/{res.get('id') or snap}"]}


def t_followup(rc, a: FollowupArgs) -> dict:
    at = parse_dt(a.at)
    if at is None:
        return {"ok": False, "error": "bad time", "evidence": []}
    item = {"id": rc.runtime.new_id(), "at": iso(at), "kind": a.kind, "camera": a.camera, "reason": clean_reason(a.reason),
            "event_id": rc.trace.event_id}
    item = rc.runtime.schedule_followup(item)
    rc.trace.followup = item
    return {"ok": True, "followup": item, "evidence": [f"followup://{item['id']}"]}


def _brief(obs) -> dict | None:
    if obs is None:
        return None
    return {"image": obs.image, "counts": obs.counts.model_dump(), "quality_issues": obs.quality.issues,
            "new_count": obs.new_catch.new_count if obs.new_catch else None, "saturation": obs.saturation}


TOOLS: dict[str, tuple[type[_Args], object, str]] = {
    "analyze_trap_photo": (AnalyzeArgs, t_analyze,
                           ("Run the OpenCV 5 pipeline on a trap photo: quality gate, fly count, new catches vs the "
                           "previous photo, card state. Must be called before any action.")),
    "compare_with_previous": (CompareArgs, t_compare,
                              ("Register two photos of the same trap and diff detections (new / persisted / "
                              "disappeared, card change suspected).")),
    "get_trap_history": (HistoryArgs, t_history,
                         ("Daily rollups {peak, total, scans} of a trap, card replacements, treatments, and the "
                         "surge check (surgeService rule; sticky = peak, lure = daily average).")),
    "get_neighbors": (NeighborsArgs, t_neighbors, "Nearest traps on the same farm/plot, by GPS distance."),
    "get_degree_day_outlook": (CameraArgs, t_degree_day,
                               "Degree-day model (B. dorsalis, 350 DD/generation): next adult wave date and rainy days."),
    "get_weather": (CameraArgs, t_weather, "Daily min/max temperature and rain around the trap (past 3 + 7 days)."),
    "capture": (CaptureArgs, t_capture,
                ("Take a photo now. AI camera: synchronous, returns the image ref ('max' = 4 MP). EZVIZ: asynchronous "
                "through FlyBot. Counts against the daily capture budget.")),
    "set_image": (SetImageArgs, t_set_image,
                  ("Change the AI camera ISP (video.image: brightness -64..64, contrast 0..100, wdr 0..2, "
                  "flicker 0..3). Needs an analyzed before photo; a verification capture follows automatically and "
                  "the change is reverted if the next photo's quality is not better.")),
    "set_detection_roi": (RoiArgs, t_roi, "Set the AI camera's trap-count detection ROI (normalized polygon)."),
    "set_capture_schedule": (ScheduleArgs, t_schedule,
                             "Change how often the trap is photographed (AI camera shadow trigger; EZVIZ via FlyBot)."),
    "propose_card_change": (ProposeCardArgs, t_propose_card,
                            ("Ask a person (FlyBot Confirm card) to replace / log the sticky card. Sticky traps only. "
                            "Must cite evidence.")),
    "propose_treatment": (ProposeTreatmentArgs, t_propose_treatment,
                          ("Ask a person (FlyBot Confirm card) to approve a treatment. Creates a pending approval only; "
                          "expires after 24 h with no action. Must cite evidence.")),
    "request_human_review": (ReviewArgs, t_review, "Ask a person to review a photo's count in FlyBot."),
    "schedule_followup": (FollowupArgs, t_followup, "Schedule a later check (capture, surge recheck, ...)."),
}


def tool_schema(name: str) -> dict:
    model = TOOLS[name][0]
    schema = model.model_json_schema(by_alias=True)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


TOOL_SPECS: list[dict] = [{"name": n, "description": d, "input_schema": tool_schema(n)} for n, (_, _, d) in TOOLS.items()]
