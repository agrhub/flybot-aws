"""FlyBot proposals: the agent's actions become contract proposals instead of device / FlyBot API calls.

The vision Lambda cannot act on a camera itself (docs/CONTRACTS.md section 1): every device tool and every
FlyBot-side tool the planners call is answered here with a *pending* acknowledgement, and after the run
`proposals_from_trace` turns the successful action tool calls into the response's `proposals[]`:

    tool call (status ok)       -> action          params
    capture                     -> capture_now     {purpose, resolution}
    set_image                   -> set_isp         {brightness?, contrast?, wdr: bool?, flicker?}
    set_capture_schedule        -> set_interval    {minutes, from, to}
    request_human_review        -> review_request  {kind: "count"|"quality"|..., snapshotId?}
    propose_card_change         -> review_request  {kind: "card_change", cardKind}
    propose_treatment           -> review_request  {kind: "treatment", action, date, rotateFrom}
    schedule_followup           -> notify          {kind, at, message}

Guardrails (state machine, budgets, evidence, kill switch) still run in front of every call, so a proposal
here is one the policy allowed. `requiresApproval` and `confidence` come from policy.yaml
(`proposal_approval`, `proposal_confidence`): policy choices, not measurements. `reason` is written in the
request's locale (vi / en) from the decision rule; the planner's English detail stays in `detail`.
"""

from __future__ import annotations

from sentinel.common.schemas import DeviceAck, PolicyConfig, Trace
from sentinel.devices.base import Device, not_supported

ACTION_OF = {"capture": "capture_now", "set_image": "set_isp", "set_capture_schedule": "set_interval",
             "request_human_review": "review_request", "propose_card_change": "review_request",
             "propose_treatment": "review_request", "schedule_followup": "notify"}

# Vietnamese / English sentences per decision rule (prefix match, longest first). Free text from the planner
# (English, with the measured values) is kept beside them in `detail`.
REASONS: dict[str, dict[str, str]] = {
    "quality.recapture": {"vi": "Ảnh không đạt chất lượng (tối, chói, mờ hoặc thiếu tương phản): chụp lại",
                          "en": "Photo quality is not good enough (dark, glare, blur or low contrast): capture again"},
    "quality.isp_fix": {"vi": "Ảnh chói hoặc tối: đổi cài đặt hình ảnh của camera rồi chụp lại để kiểm tra",
                        "en": "Glare or dark photo: change the camera's image settings, then capture again to verify"},
    "quality.still_bad": {"vi": "Ảnh vẫn chưa đạt sau khi chụp lại: hẹn kiểm tra lại sau",
                          "en": "Still a bad photo after one retry: check again later"},
    "quality.clean_lens": {"vi": "Ống kính hoặc nắp bẫy có vết bẩn: cần người lau",
                           "en": "Lens or trap cover looks dirty: a person should clean it"},
    "quality.postpone": {"vi": "Ảnh hồng ngoại ban đêm: không đếm, chờ ảnh tiếp theo",
                         "en": "Night IR frame: not counted, wait for the next photo"},
    "trap_missing": {"vi": "Không thấy bẫy trong ảnh (bẫy mất hoặc camera lệch): chụp lại, rồi nhờ người kiểm tra",
                     "en": "No trap in view (trap missing or camera misaimed): capture again, then a person checks"},
    "occlusion": {"vi": "Có vật lớn che bẫy: ảnh không đếm, chụp lại",
                  "en": "A large object covers the trap: photo not counted, capture again"},
    "motion.camera_moved": {"vi": "Camera bị dịch chuyển hoặc lệch góc: cần người kiểm tra giá lắp",
                            "en": "The camera moved or was re-aimed: please check its mounting"},
    "disagreement": {"vi": "Các cách đếm cho kết quả khác nhau nhiều: nhờ người xem lại số ruồi",
                     "en": "The counts disagree: a person should review the fly count"},
    "card": {"vi": "Tấm dính đã đầy hoặc vừa được thay: xác nhận thay tấm", "en": "Sticky card full or replaced: confirm the card change"},
    "surge.unconfirmed": {"vi": "Số ruồi tăng đột biến nhưng chưa có bẫy lân cận xác nhận: kiểm tra lại sau",
                          "en": "Fly count surged but no neighbour trap confirmed it: recheck later"},
    "surge": {"vi": "Số ruồi tăng đột biến so với 3 ngày trước", "en": "Fly count surged compared with 3 days ago"},
    "treatment.wait_check": {"vi": "Vừa xử lý chưa đủ 7 ngày: hẹn kiểm tra hiệu quả",
                             "en": "Treated less than 7 days ago: schedule a check of its effect"},
    "treatment": {"vi": "Đề xuất xử lý vườn theo lịch phun (cần bạn xác nhận)",
                  "en": "Treatment suggested by the spray plan (needs your confirmation)"},
    "followup": {"vi": "Theo dõi lại theo lịch đã hẹn", "en": "Scheduled follow-up"},
    "guardrail.llm_ungrounded": {"vi": "Đề xuất của AI chưa có bằng chứng từ ảnh: nhờ người xem lại",
                                 "en": "The AI planner's proposal is not backed by the photos: a person reviews instead"},
    "llm": {"vi": "Đề xuất của trợ lý AI (xem phần giải thích)", "en": "Proposed by the AI planner (see the explanation)"},
    "verify_isp": {"vi": "Chụp lại để kiểm tra cài đặt hình ảnh mới", "en": "Capture again to verify the new image settings"},
}


def reason_for(rule: str | None, locale: str, fallback: str) -> str:
    lang = "vi" if locale == "vi" else "en"
    if rule:
        for key in sorted(REASONS, key=len, reverse=True):
            if rule.startswith(key):
                return REASONS[key][lang]
    return fallback


def confidence_for(rule: str | None, policy: PolicyConfig) -> float:
    table = policy.proposal_confidence or {}
    if rule:
        for key in sorted(table, key=len, reverse=True):
            if key != "default" and rule.startswith(key):
                return float(table[key])
    return float(table.get("default", 0.7))


# ---------------------------------------------------------------- stand-ins the tools call


class ProposalDevice(Device):
    """The FlyBot camera as the tools see it: every action is accepted as pending (FlyBot executes it once the
    proposal is confirmed or auto-applied). `capture_max` and `set_detection_roi` are not offered: EZVIZ and
    ESP32 cameras have no resolution switch / detection ROI."""

    def __init__(self, serial: str, source: str = "ezviz", isp: dict | None = None, battery: float | None = None):
        self.serial, self.source = serial, source
        # Only CamOS cameras take image settings (shadow video.image): FlyBot refuses set_isp for EZVIZ, and the
        # kinesis platform (a plain KVS stream) has no control channel, so the tool is not offered to them
        self.capabilities = frozenset({"capture", "set_capture_schedule"} | ({"set_image"} if source == "camos" else set()))
        self.isp = dict(isp or {})
        self.battery = battery
        self.n = 0

    def _ack(self, method: str, **kw) -> DeviceAck:
        self.n += 1
        rid = f"prop-{method}-{self.n}"
        return DeviceAck(ok=True, serial=self.serial, method=method, transport="flybot_proposal", request_id=rid,
                         pending=True, applied=False, evidence=[f"proposal://{self.serial}/{rid}"], **kw)

    def capture(self, resolution: str = "main", detection_id: str | None = "trap-count") -> DeviceAck:
        return self._ack("capture", result={"resolution": resolution})

    def set_image(self, params: dict) -> DeviceAck:
        return self._ack("set_image", result={"requested": dict(params)})

    def get_image_settings(self) -> dict | None:
        return dict(self.isp) if self.isp else None

    def battery_percent(self) -> float | None:
        return self.battery

    def set_detection_roi(self, polygon, detection_id: str = "trap-count") -> DeviceAck:
        return not_supported(self.serial, "set_detection_roi", "flybot_proposal", "no detection ROI on this camera")

    def set_capture_schedule(self, every_min: int, from_: str, to: str, detection_id: str = "trap-count") -> DeviceAck:
        return self._ack("set_capture_schedule", result={"every_min": every_min, "from": from_, "to": to})


class ProposalFlyBot:
    """Stands in for the first build's FlyBotClient: proposals and review requests are returned to FlyBot in
    the response instead of being POSTed to it."""

    def __init__(self):
        self.n = 0

    def _id(self, kind: str) -> str:
        self.n += 1
        return f"{kind}-{self.n}"

    def proposal(self, type_: str, device_serials: list[str], payload: dict, evidence: list[str]) -> dict:
        return {"ok": True, "proposalId": self._id(type_)}

    def review_request(self, snapshot_id, reason, **extra) -> dict:
        return {"ok": True, "id": self._id("review")}

    def analysis(self, *a, **kw) -> dict:  # never called: Runtime.post_analysis is False in the handler
        return {"ok": True}


# ---------------------------------------------------------------- trace -> proposals


def _decision_for(trace: Trace, call) -> object | None:
    """The decision this call executes: the last one taken before the call with the same action, or the
    `llm.<tool>` decision recorded right after an LLM tool call."""
    before = call.decisions_before
    decisions = trace.decisions
    if before is not None:
        if 0 < before <= len(decisions) and decisions[before - 1].action == call.name:
            return decisions[before - 1]
        if before < len(decisions) and decisions[before].rule == f"llm.{call.name}":
            return decisions[before]
        for d in reversed(decisions[:before]):
            if d.action == call.name:
                return d
    return next((d for d in decisions if d.action == call.name), None)


def _params(call) -> dict:
    a, r = call.args or {}, call.result or {}
    if call.name == "capture":
        return {"purpose": a.get("purpose", "recapture"), "resolution": a.get("resolution", "main")}
    if call.name == "set_image":
        applied = r.get("applied_settings") or {k: a[k] for k in ("brightness", "contrast", "wdr", "flicker") if a.get(k) is not None}
        out = {k: v for k, v in applied.items() if k in ("brightness", "contrast", "flicker")}
        if applied.get("wdr") is not None:
            out["wdr"] = bool(applied["wdr"])
        return out
    if call.name == "set_capture_schedule":
        return {"minutes": a.get("every_min"), "from": a.get("from"), "to": a.get("to")}
    if call.name == "request_human_review":
        return {"kind": "review", "snapshotId": a.get("snapshot_id")}
    if call.name == "propose_card_change":
        return {"kind": "card_change", "cardKind": a.get("kind")}
    if call.name == "propose_treatment":
        return {"kind": "treatment", "action": a.get("action"), "date": a.get("date"), "rotateFrom": a.get("rotate_from"),
                "cameras": a.get("cameras")}
    if call.name == "schedule_followup":
        fu = r.get("followup") or {}
        return {"kind": a.get("kind"), "at": fu.get("at") or a.get("at"), "message": a.get("reason") or a.get("kind")}
    return {}


def proposals_from_trace(trace: Trace, policy: PolicyConfig, locale: str = "vi") -> list[dict]:
    out = []
    for call in trace.tool_calls:
        if call.status != "ok" or call.name not in ACTION_OF:
            continue
        action = ACTION_OF[call.name]
        d = _decision_for(trace, call)
        rule = d.rule if d else ("verify_isp" if (call.args or {}).get("purpose") == "verify_isp" else call.name)
        detail = "; ".join(d.reasons) if d else str((call.args or {}).get("reason") or "")
        approval = (policy.proposal_approval or {}).get(action, True)
        out.append({
            "id": f"p{len(out) + 1}", "action": action, "params": _params(call),
            "reason": reason_for(rule, locale, detail or rule), "detail": detail[:500], "rule": rule,
            "confidence": confidence_for(rule, policy), "requiresApproval": bool(approval),
            "evidence": [e for e in (call.result or {}).get("evidence", []) if isinstance(e, str) and not e.startswith("tool:")][:10],
            "toolCall": call.index, "by": call.by,
        })
    return out


def summary_text(obs, trace: Trace, proposals: list[dict], locale: str = "vi") -> str:
    """Short markdown summary for `answer` when the LLM planner did not write one (rule planner / fallback)."""
    vi = locale == "vi"
    c = obs.counts.opencv if obs else None
    lines = []
    if obs is not None:
        lines.append((f"**Đếm được {c} ruồi** trên bẫy {obs.camera}." if vi else f"**{c} flies counted** on trap {obs.camera}.")
                     if c is not None else ("Không có kết quả đếm (thiếu mô hình)." if vi else "No count (detector unavailable)."))
        if obs.quality.issues:
            lines.append(("Chất lượng ảnh: " if vi else "Photo quality: ") + ", ".join(obs.quality.issues))
        nc = obs.new_catch
        if nc and nc.new_count is not None:
            lines.append(f"So với ảnh trước: +{nc.new_count} mới, -{nc.disappeared_count or 0} biến mất." if vi
                          else f"Since the previous photo: +{nc.new_count} new, -{nc.disappeared_count or 0} gone.")
    rules = list(dict.fromkeys(d.rule for d in trace.decisions))
    if rules:
        lines.append(("Quy tắc đã áp dụng: " if vi else "Rules applied: ") + ", ".join(rules))
    for p in proposals:
        lines.append(f"- {p['action']}: {p['reason']}")
    if trace.planner.fallback:
        lines.append("(Trợ lý AI không khả dụng, dùng quy tắc.)" if vi else "(AI planner unavailable; rule planner decided.)")
    return "\n".join(lines)
