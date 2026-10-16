"""Guardrails: what any planner (rule or LLM) is allowed to do, enforced outside the planner.

1. Tool whitelist by run state:  INGESTED -> ANALYZED -> DECIDED -> ACTED | WAITING_APPROVAL -> FOLLOWUP.
   Nothing acts before a photo is analyzed; once a proposal waits for a person, no further device actions
   or proposals happen in that run.
2. Budgets (policy.budgets): extra captures per camera per farm-local day, neighbour captures per surge,
   ISP changes per camera per rolling hour, quiet hours (no device actions).
3. Proposals need evidence: `propose_*` must cite >= 1 evidence ref that an earlier tool result in the same
   run produced. propose_* only creates a FlyBot Confirm card + a pending approval; nothing here writes a
   logbook. Expired approvals are a no-op.
4. ISP changes need a before image (an analyzed photo of that camera in this run), trigger a verification
   capture right away, and are reverted automatically when the next photo's quality score is not better.
5. Prompt-injection filter for text from image metadata / OSD / device names / user notes: control
   characters stripped, length capped, wrapped as quoted untrusted data, suspicious patterns flagged. The
   rule planner never reads that text; the LLM planner only sees it quoted.
6. Kill switch: policy flag or $SENTINEL_KILL_SWITCH=1 (the Lambda has no store: FlyBot keeps the switch and
   sets the env / stops invoking).
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from sentinel.agent.policy import in_quiet_hours, local_day
from sentinel.common.ids import iso, parse_dt
from sentinel.common.schemas import PolicyConfig

READ_TOOLS = frozenset({"analyze_trap_photo", "compare_with_previous", "get_trap_history", "get_neighbors",
                        "get_degree_day_outlook", "get_weather"})
DEVICE_TOOLS = frozenset({"capture", "set_image", "set_detection_roi", "set_capture_schedule"})
PROPOSAL_TOOLS = frozenset({"propose_card_change", "propose_treatment"})
HUMAN_TOOLS = frozenset({"request_human_review"})
FOLLOWUP_TOOLS = frozenset({"schedule_followup"})
ALL_TOOLS = READ_TOOLS | DEVICE_TOOLS | PROPOSAL_TOOLS | HUMAN_TOOLS | FOLLOWUP_TOOLS
ACTION_TOOLS = ALL_TOOLS - READ_TOOLS

WHITELIST: dict[str, frozenset[str]] = {
    "INGESTED": frozenset({"analyze_trap_photo", "get_trap_history", "get_neighbors", "get_weather",
                           "get_degree_day_outlook"}),
    "ANALYZED": ALL_TOOLS,
    "DECIDED": ALL_TOOLS,
    "ACTED": ALL_TOOLS,
    "WAITING_APPROVAL": READ_TOOLS | HUMAN_TOOLS | FOLLOWUP_TOOLS,
    "FOLLOWUP": READ_TOOLS | frozenset({"capture", "set_image"}) | HUMAN_TOOLS | FOLLOWUP_TOOLS,
}

CAPABILITY = {"set_image": "set_image", "set_detection_roi": "set_detection_roi",
              "set_capture_schedule": "set_capture_schedule", "capture": "capture"}

ISP_PENDING_MAX_H = 6


@dataclass
class Verdict:
    allowed: bool
    code: str = "ok"
    reason: str = ""


ALLOW = Verdict(True)


def block(code: str, reason: str) -> Verdict:
    return Verdict(False, code, reason)


# ---------------------------------------------------------------- prompt injection

_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")


def sanitize_untrusted(text: str | None, policy: PolicyConfig, source: str = "metadata") -> dict:
    """-> {text (quoted, safe to show an LLM), flagged, patterns, truncated}. Never interpreted."""
    raw = "" if text is None else str(text)
    clean = _CTRL.sub("", raw).replace("\r", " ").replace("\n", " ")
    hits = [p for p in policy.injection.patterns if re.search(p, clean)]
    truncated = len(clean) > policy.injection.max_len
    clean = clean[: policy.injection.max_len]
    quoted = f'<untrusted_text source="{html.escape(source)}" flagged="{str(bool(hits)).lower()}">' \
             f"{html.escape(clean, quote=True)}</untrusted_text>"
    return {"text": quoted, "flagged": bool(hits), "patterns": hits, "truncated": truncated}


def sanitize_texts(texts: dict[str, str], policy: PolicyConfig) -> tuple[dict[str, str], list[dict]]:
    out, flags = {}, []
    for k, v in (texts or {}).items():
        s = sanitize_untrusted(v, policy, k)
        out[k] = s["text"]
        if s["flagged"]:
            flags.append({"field": k, "patterns": s["patterns"]})
    return out, flags


def clean_reason(text: str | None, limit: int = 500) -> str:
    """Planner-written free text (reasons) forwarded to FlyBot: plain, bounded, no control characters."""
    return _CTRL.sub("", str(text or "")).strip()[:limit]


# ---------------------------------------------------------------- kill switch


class KillSwitch:
    def check(self, camera: str | None) -> tuple[bool, str | None]:
        raise NotImplementedError


class StaticKillSwitch(KillSwitch):
    def __init__(self, on: bool = False, cameras: set[str] | None = None):
        self.on, self.cameras = on, set(cameras or ())

    def check(self, camera):
        if self.on:
            return True, "static:global"
        if camera and camera in self.cameras:
            return True, f"static:{camera}"
        return False, None


def kill_switch_from(policy: PolicyConfig, base: KillSwitch) -> KillSwitch:
    env_on = os.environ.get("SENTINEL_KILL_SWITCH", "").strip().lower() in ("1", "true", "yes", "on")
    if policy.kill_switch.enabled or env_on:
        return StaticKillSwitch(True)
    return base


# ---------------------------------------------------------------- guardrails


class Guardrails:
    def __init__(self, policy: PolicyConfig, state_store, kill_switch: KillSwitch, devices):
        self.policy, self.state, self.kill, self.devices = policy, state_store, kill_switch, devices

    # ---------------------------------------------------------- checks
    def check(self, rc, name: str, args: dict, *, internal: bool = False, by: str = "planner") -> Verdict:
        if name not in ALL_TOOLS:
            return block("unknown_tool", f"{name} is not a Sentinel tool")
        camera = args.get("camera") or (args.get("cameras") or [None])[0]
        if name in ACTION_TOOLS:
            on, why = rc.kill_switch(camera)
            if on:
                return block("kill_switch", f"kill switch on ({why}): no actions")
        if not internal and name not in WHITELIST[rc.state]:
            return block("state", f"{name} not allowed in state {rc.state}")
        now = rc.now()
        if name in DEVICE_TOOLS:
            dev = self.devices.get(args.get("camera"))
            if dev is None:
                return block("unknown_camera", f"no device for camera {args.get('camera')!r}")
            if not dev.supports(CAPABILITY[name]):
                return block("unsupported", f"{dev.source} camera {dev.serial} cannot {name}")
            if name == "capture" and args.get("resolution") == "max" and not dev.supports("capture_max"):
                return block("unsupported", f"{dev.source} camera {dev.serial} has no resolution control")
            if in_quiet_hours(now, self.policy) and not internal:  # an auto-revert may run at night
                q = self.policy.budgets.quiet_hours
                return block("quiet_hours", f"quiet hours {q.start}-{q.end} {self.policy.timezone}: no device actions")
        b = self.policy.budgets
        if name in ("capture", "set_image") and not internal:
            limit, why = self.battery_budget(rc, args["camera"])
            if limit == 0:
                return block("battery", f"{why}: no extra captures or ISP changes (scheduled captures continue)")
        if name == "capture":
            used = self.captures_today(args["camera"], now)
            limit, why = self.battery_budget(rc, args["camera"])
            if used >= limit:
                return block("budget", f"capture budget used: {used}/{limit} today on {args['camera']}"
                                       + (f" ({why})" if why else ""))
            if args.get("purpose") == "neighbour_check" and rc.neighbour_captures >= b.neighbour_captures_per_surge:
                return block("budget", f"neighbour captures per surge: {rc.neighbour_captures}/{b.neighbour_captures_per_surge}")
            if args.get("purpose") == "recapture" and rc.recaptures >= self.policy.quality.max_recaptures_per_event:
                return block("budget", "one recapture per event")
        if name == "set_image" and not internal:
            used = self.isp_changes_last_hour(args["camera"], now)
            if used >= b.isp_changes_per_camera_per_hour:
                return block("budget", f"ISP change budget: {used}/{b.isp_changes_per_camera_per_hour} in the last hour")
            obs = rc.observations.get(args["camera"])
            if obs is None or not obs.image:
                return block("evidence", "set_image needs a before image: analyze a photo of this camera first")
            bad = [e for e in args.get("evidence") or [] if e not in rc.evidence]
            if bad:
                return block("evidence", f"unknown evidence refs: {bad[:3]}")
        if name in PROPOSAL_TOOLS:
            ev = args.get("evidence") or []
            if not ev:
                return block("evidence", f"{name} needs evidence: cite refs returned by earlier tools")
            bad = [e for e in ev if e not in rc.evidence]
            if bad:
                return block("evidence", f"{name} cites evidence no tool returned in this run: {bad[:3]}")
            if name == "propose_card_change":
                kind = rc.trap_kind(args["camera"])
                if not self.policy.kind(kind).card_changes:
                    return block("trap_kind", f"{args['camera']} is a {kind} trap: no card to change")
            if by == "llm" and self.policy.llm_ground_proposals:
                problem = self.ungrounded(rc, name, args)
                if problem:
                    return block("downgraded", f"LLM {name} not grounded in an observation ({problem}): "
                                               "a person reviews instead")
            if name == "propose_treatment" and rc.runtime.approvals is not None:
                for a in rc.runtime.approvals.pending():
                    if a.kind == "treatment" and set(a.cameras) & set(args.get("cameras") or []):
                        return block("duplicate", f"treatment proposal {a.approval_id} still pending for these traps")
        return ALLOW

    # ---------------------------------------------------------- LLM proposal grounding
    def ungrounded(self, rc, name: str, args: dict) -> str | None:
        """Why an LLM proposal is not backed by this run's observations of its traps (None = grounded).
        1. at least one cited ref is an observation's own evidence (its image or vision evidence files);
        2. the observations / history support the kind: treatment = a surge from get_trap_history or a count at
           the trap's alert threshold; card change = card_full / saturation >= full / card change suspected."""
        cams = set(args.get("cameras") or [args.get("camera")])
        obs = [o for o in rc.trace.opencv_outputs.values() if o.get("camera") in cams]
        refs = {r for o in obs for r in [o.get("image"), *(o.get("evidence") or [])] if r}
        if not refs & set(args.get("evidence") or []):
            return "cites no evidence of an observation of these traps"
        if name == "propose_card_change":
            full = self.policy.card.saturation_full
            for o in obs:
                q = o.get("quality") or {}
                nc = o.get("new_catch") or {}
                if ("card_full" in (q.get("issues") or []) or (o.get("saturation") or 0) >= full
                        or nc.get("card_change_suspected")):
                    return None
            if rc.event is not None and rc.event.camera in cams and                     "card_full" in ((rc.event.production_quality or {}).get("issues") or []):
                return None
            return "no observation shows a full or replaced card"
        surge = any(c.name == "get_trap_history" and c.status == "ok" and c.args.get("camera") in cams
                    and (c.result or {}).get("surge") for c in rc.trace.tool_calls)
        if surge:
            return None
        for o in obs:
            counts = [v for v in (o.get("counts") or {}).values() if isinstance(v, int | float)]
            thr = rc.runtime.data.camera(o["camera"]).get("alert_threshold")
            if thr is None and rc.event is not None and rc.event.camera == o["camera"]:
                thr = rc.event.alert_threshold
            if thr is not None and counts and max(counts) >= thr:
                return None
        return "no surge in the trap history and no count at the alert threshold"

    # ---------------------------------------------------------- budgets
    def battery_budget(self, rc, camera: str) -> tuple[int, str | None]:
        """(extra-capture limit today, reason) after the battery rule; the reason is noted once per run."""
        bp, base = self.policy.budgets.battery, self.policy.budgets.extra_captures_per_camera_per_day
        pct = rc.battery(camera)
        if pct is None:
            return base, None
        if pct < bp.critical_percent:
            limit, why = 0, f"battery {pct:g}% < {bp.critical_percent:g}%"
        elif pct < bp.low_percent:
            limit = int(base * bp.low_factor)
            why = f"battery {pct:g}% < {bp.low_percent:g}%: extra-capture budget {base} -> {limit}"
        else:
            return base, None
        if camera not in rc._battery_noted:
            rc._battery_noted.add(camera)
            rc.note("battery_budget", {"camera": camera, "battery_percent": pct, "limit": limit, "reason": why})
        return limit, why

    def captures_today(self, camera: str, now: datetime) -> int:
        return int((self.state.get(camera).get("captures") or {}).get(local_day(now, self.policy), 0))

    def isp_changes_last_hour(self, camera: str, now: datetime) -> int:
        times = [parse_dt(t) for t in self.state.get(camera).get("isp_changes") or []]
        return sum(1 for t in times if t and now - t < timedelta(hours=1))

    # ---------------------------------------------------------- after a successful call
    def after(self, rc, name: str, args: dict, result: dict, *, internal: bool = False) -> None:
        now = rc.now()
        if name == "analyze_trap_photo":
            if rc.state == "INGESTED":
                rc.set_state("ANALYZED")
            return
        if name in READ_TOOLS:
            return
        if name == "capture":
            day = local_day(now, self.policy)

            def bump(s):
                caps = s.setdefault("captures", {})
                caps[day] = caps.get(day, 0) + 1
                for old in sorted(caps)[:-7]:
                    caps.pop(old)

            self.state.update(args["camera"], bump)
            if args.get("purpose") == "neighbour_check":
                rc.neighbour_captures += 1
            if args.get("purpose") == "recapture":
                rc.recaptures += 1
        if name == "set_image" and not internal:
            obs = rc.observations[args["camera"]]
            before = result.get("before_settings") or {}
            pending = {"at": iso(now), "event_id": rc.trace.event_id, "before_image": obs.image,
                       "before_score": obs.quality.score, "before_issues": obs.quality.issues,
                       "before_settings": before, "after_settings": result.get("applied_settings") or {}}

            def note(s):
                s.setdefault("isp_changes", []).append(iso(now))
                s["isp_changes"] = s["isp_changes"][-10:]
                s["pending_isp"] = pending

            self.state.update(args["camera"], note)
        if name in PROPOSAL_TOOLS:
            rc.set_state("WAITING_APPROVAL")
        elif rc.state in ("ANALYZED", "DECIDED"):
            rc.set_state("ACTED")

    # ---------------------------------------------------------- ISP verification / auto-revert
    def verify_isp(self, rc, obs) -> dict | None:
        """Called after each analysis: settles a pending ISP change of `obs.camera`."""
        pending = self.state.get(obs.camera).get("pending_isp")
        if not pending or obs.image == pending.get("before_image"):
            return None
        before, after = pending.get("before_score"), obs.quality.score
        better = before is not None and after is not None and after > before
        verdict = {"camera": obs.camera, "before_score": before, "after_score": after,
                   "before_image": pending.get("before_image"), "after_image": obs.image,
                   "before_issues": pending.get("before_issues"), "after_issues": obs.quality.issues,
                   "changed_at": pending.get("at"), "change_event": pending.get("event_id")}
        if better:
            verdict["result"] = "kept"
            rc.note("isp_verified", verdict)
        else:
            verdict["result"] = "reverted"
            restore = {k: v for k, v in (pending.get("before_settings") or {}).items()
                       if k in ("brightness", "contrast", "wdr", "flicker")}
            res = rc.call("set_image", {"camera": obs.camera, **restore,
                                        "reason": "auto-revert: quality not better after ISP change"},
                          by="guardrail", internal=True) if restore else {"error": "no before settings to restore"}
            verdict["revert"] = {k: res.get(k) for k in ("ok", "error", "blocked", "reason") if k in res}
            rc.note("isp_reverted", verdict)
        self.state.update(obs.camera, lambda s: s.pop("pending_isp", None))
        return verdict

    def stale_isp(self, camera: str, now: datetime) -> bool:
        pending = self.state.get(camera).get("pending_isp")
        at = parse_dt((pending or {}).get("at"))
        return bool(at and now - at > timedelta(hours=ISP_PENDING_MAX_H))
