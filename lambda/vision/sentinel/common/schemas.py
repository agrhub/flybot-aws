"""Shared data models (pydantic v2) for the agent, devices, FlyBot client and replay.

Everything here is JSON-serializable with `.model_dump(mode="json")`. Image references are plain
strings: `s3://bucket/key`, `https://...` (FlyBot presigned link), `file://...`/a local path, or
`fake://...` in tests and replay.

Trace shape: {event_id, camera, source, opencv_outputs, planner,
reasoning_summary, tool_calls[], device_ack, approval, outcome, followup}.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["ezviz", "camos", "kinesis"]
TrapKind = Literal["sticky", "lure"]
RunStateName = Literal["INGESTED", "ANALYZED", "DECIDED", "ACTED", "WAITING_APPROVAL", "FOLLOWUP"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------- input events


class PhotoEvent(_Model):
    """One trap photo entering the per-photo loop, normalized from a FlyBot `snapshot_analyzed`
    webhook (EZVIZ or CamOS cameras known to FlyBot), a CamOS IoT event (SQS), or replay."""

    event_id: str
    camera: str
    source: Source
    trap_kind: TrapKind = "sticky"
    captured_at: datetime | None = None
    image: str | None = None
    width: int | None = None
    height: int | None = None
    resolution: Literal["main", "max"] | None = None
    snapshot_id: str | None = None
    previous_snapshot_id: str | None = None
    trigger: str | None = None  # schedule | rpc | agent:<purpose>
    production_count: int | None = None  # FlyBot cloud AI (Roboflow / Vertex)
    production_boxes: list[list[float]] = Field(default_factory=list)  # xyxy pixels
    production_quality: dict | None = None  # FlyBot imageQuality result
    npu_count: int | None = None  # AI camera on-device count (app_detection n)
    npu_boxes: list[list[float]] = Field(default_factory=list)  # normalized 0-1 [x1,y1,x2,y2,cls,conf]
    alert_threshold: int | None = None
    card_replaced_at: datetime | None = None
    battery_percent: float | None = None  # battery-powered cameras (ESP32 trap, EZVIZ CS-EB8); None = mains / unknown
    # Free text from the device / users. UNTRUSTED: only ever shown to an LLM quoted, never executed.
    untrusted_text: dict[str, str] = Field(default_factory=dict)
    raw: dict = Field(default_factory=dict)


# ---------------------------------------------------------------- perception


class QualityOut(_Model):
    ok: bool = True
    issues: list[str] = Field(default_factory=list)
    scores: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None  # 0-100 scalar used by the ISP verify/revert rule (guardrails.quality_score)
    baseline: dict | None = None  # the camera's rolling op 2 baseline the issues were judged against (None = absolute only)


class Counts(_Model):
    opencv: int | None = None  # OpenCV 5 cloud count (detect_yolov7 on ENGINE_NEW)
    npu: int | None = None
    production: int | None = None


class NewCatch(_Model):
    """Op 4 diff vs the previous photo. sticky: `meaning` "catches" (new = flies caught since); lure: "activity"
    (new = arrivals, disappeared = departures; the count itself is the flies present now)."""

    new_count: int | None = None
    persisted_count: int | None = None
    disappeared_count: int | None = None
    card_change_suspected: bool = False
    card_change_reasons: list[str] = Field(default_factory=list)
    registration_ok: bool | None = None
    meaning: Literal["catches", "activity"] | None = None
    inliers: int | None = None  # op 3 registration inliers
    change_ratio: float | None = None
    skipped: str | None = None  # op 4 matching skipped (e.g. no boxes); registration may still have run


class Observation(_Model):
    """Vision outputs for one photo (what the planner sees)."""

    event_id: str
    camera: str
    source: Source
    trap_kind: TrapKind = "sticky"
    captured_at: datetime | None = None
    image: str | None = None
    previous_image: str | None = None
    resolution: Literal["main", "max"] | None = None
    width: int | None = None
    height: int | None = None
    quality: QualityOut = Field(default_factory=QualityOut)
    counts: Counts = Field(default_factory=Counts)
    boxes: list[list[float]] = Field(default_factory=list)
    new_catch: NewCatch | None = None
    saturation: float | None = None  # share of the card covered by insects (op 6), 0-1
    card: dict | None = None  # {found, quad_norm, coverage}
    motion: dict | None = None  # op 8: occluded_by_large_object, occluder, camera_moved, moved_reasons, shift
    privacy: dict | None = None  # person_suspected, faces
    snapshot_id: str | None = None
    battery_percent: float | None = None  # device state at capture time, when known
    untrusted_text: dict[str, str] = Field(default_factory=dict)  # already sanitized/quoted
    analyzer: str = "unknown"
    vision_ms: float | None = None
    evidence: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- agent actions


class DeviceAck(_Model):
    ok: bool
    serial: str
    method: str
    transport: str  # iot_shadow | iot_rpc | flybot_api | fake
    request_id: str | None = None
    applied: bool | None = None  # shadow: reported.version caught up
    pending: bool = False  # accepted but the result arrives later (EZVIZ capture via FlyBot, retained shadow)
    result: dict = Field(default_factory=dict)
    error: dict | None = None
    ms: float | None = None
    evidence: list[str] = Field(default_factory=list)


class ToolCall(_Model):
    index: int
    name: str
    args: dict = Field(default_factory=dict)
    status: Literal["ok", "blocked", "error", "dedup"] = "ok"
    blocked_reason: str | None = None
    result: dict = Field(default_factory=dict)
    state_before: RunStateName | None = None
    state_after: RunStateName | None = None
    started_at: datetime | None = None
    ms: float | None = None
    by: str = "planner"  # planner | guardrail (auto verification capture, ISP revert)
    decisions_before: int | None = None  # len(trace.decisions) when the call started (proposals pair calls with decisions)


class Decision(_Model):
    rule: str  # e.g. quality.isp_fix, surge.neighbour_check, disagreement.closer_look, no_op
    action: str
    reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class Approval(_Model):
    approval_id: str
    event_id: str
    kind: Literal["card_change", "treatment"]
    cameras: list[str]
    payload: dict = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    status: Literal["pending", "approved", "rejected", "expired"] = "pending"
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None
    flybot_proposal_id: str | None = None
    on_timeout: str = "no_op"


class TraceEvent(_Model):
    at: datetime
    kind: str  # planner_fallback, guardrail_block, kill_switch, isp_verified, isp_reverted, approval_expired, ...
    detail: dict = Field(default_factory=dict)


class PlannerInfo(_Model):
    name: str  # rule | llm
    backend: str | None = None  # bedrock | anthropic | fake
    model: str | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    llm_turns: int = 0
    # LLM token usage of this run, summed over every provider call: {llm_calls, input_tokens, output_tokens,
    # cache_read_input_tokens?, cache_write_input_tokens?}. None = no LLM call (rule planner) or not reported.
    usage: dict | None = None


class Trace(_Model):
    event_id: str
    loop: Literal["per_photo", "daily_per_plot", "followup", "sweep", "mcp"] = "per_photo"
    camera: str | None = None
    plot: str | None = None
    source: Source | None = None
    trap_kind: TrapKind | None = None
    created_at: datetime
    finished_at: datetime | None = None
    state: RunStateName = "INGESTED"
    opencv_outputs: dict = Field(default_factory=dict)  # camera/image -> Observation dump
    planner: PlannerInfo = Field(default_factory=lambda: PlannerInfo(name="rule"))
    reasoning_summary: str = ""
    decisions: list[Decision] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    device_ack: list[DeviceAck] = Field(default_factory=list)
    approval: Approval | None = None
    outcome: str = "pending"
    followup: dict | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    kill_switch: bool = False
    policy_version: str | None = None


# ---------------------------------------------------------------- policy


class QualityPolicy(_Model):
    recapture_on: list[str] = Field(default_factory=lambda: ["dark", "overexposed", "glare", "low_contrast", "blurry"])
    isp_fixes: dict[str, dict[str, int]] = Field(default_factory=dict)  # issue -> relative change
    isp_defaults: dict[str, int] = Field(default_factory=dict)  # assumed current values when unknown
    postpone_on: list[str] = Field(default_factory=lambda: ["ir"])
    review_on: list[str] = Field(default_factory=lambda: ["dirty"])
    roi_on: list[str] = Field(default_factory=lambda: ["partial_card"])
    trap_missing_on: list[str] = Field(default_factory=lambda: ["no_card", "no_floor"])
    max_recaptures_per_event: int = 1
    baseline_size: int = 20
    baseline_min: int = 5
    baseline_bootstrap_previous: bool = True  # until baseline_min photos: the previous photo, when itself clean


class MotionPolicy(_Model):
    occluded_recapture: bool = True  # op 8 large occluder: not counted, one recapture within budget
    moved_reviews_per_day: int = 1  # op 8 camera_moved: request_human_review at most once a day per camera


class SurgePolicy(_Model):
    days: int = 3
    ratio: float = 1.5
    min_increase: float = 3
    min_count: float = 5
    card_replaced_skip_days: int = 4  # sticky only (surgeService: SURGE_DAYS + 1)
    confirm_min_neighbours: int = 1


class CardPolicy(_Model):
    saturation_full: float = 0.6
    on_issue: list[str] = Field(default_factory=lambda: ["card_full"])
    propose_on_card_change_suspected: bool = True


class DisagreementPolicy(_Model):
    abs_min: int = 3
    rel: float = 0.2
    closer_look_resolution: Literal["main", "max"] = "max"


class TreatmentPolicy(_Model):
    min_interval_days: int = 7
    wave_window_days: int = 10
    wave_lead_days: int = 3
    rain_mm: float = 10
    post_treatment_check_days: int = 3


class QuietHours(_Model):
    start: str = "20:00"
    end: str = "05:00"


class BatteryPolicy(_Model):
    low_percent: float = 40  # below: extra-capture budget x low_factor
    low_factor: float = 0.5
    critical_percent: float = 20  # below: no extra captures, no ISP experiments (scheduled captures continue)


class BudgetPolicy(_Model):
    extra_captures_per_camera_per_day: int = 4
    neighbour_captures_per_surge: int = 3
    isp_changes_per_camera_per_hour: int = 2
    quiet_hours: QuietHours = Field(default_factory=QuietHours)
    battery: BatteryPolicy = Field(default_factory=BatteryPolicy)


class ApprovalPolicy(_Model):
    timeout_h: float = 24
    on_timeout: Literal["no_op"] = "no_op"


class SchedulePolicy(_Model):
    wave_soon_days: int = 3
    wave_every_min: int = 60
    from_: str = Field("06:00", alias="from")
    to: str = "18:00"
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TrapKindPolicy(_Model):
    count_basis: Literal["peak", "daily_average"]
    new_catch: bool
    card_changes: bool


class KillSwitchPolicy(_Model):
    enabled: bool = False  # local override: true = no actions at all
    table: str = "sentinel-devices"
    global_serial: str = "kill_switch"
    attribute: str = "kill_switch"
    cache_s: float = 30


class InjectionPolicy(_Model):
    max_len: int = 300
    patterns: list[str] = Field(default_factory=list)


class PolicyConfig(_Model):
    version: str = "1"
    timezone: str = "Asia/Ho_Chi_Minh"
    trap_kinds: dict[str, TrapKindPolicy] = Field(default_factory=dict)
    quality: QualityPolicy = Field(default_factory=QualityPolicy)
    surge: SurgePolicy = Field(default_factory=SurgePolicy)
    card: CardPolicy = Field(default_factory=CardPolicy)
    disagreement: DisagreementPolicy = Field(default_factory=DisagreementPolicy)
    treatment: TreatmentPolicy = Field(default_factory=TreatmentPolicy)
    budgets: BudgetPolicy = Field(default_factory=BudgetPolicy)
    approval: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    schedule: SchedulePolicy = Field(default_factory=SchedulePolicy)
    kill_switch: KillSwitchPolicy = Field(default_factory=KillSwitchPolicy)
    injection: InjectionPolicy = Field(default_factory=InjectionPolicy)
    motion: MotionPolicy = Field(default_factory=MotionPolicy)
    llm_max_turns: int = 12
    llm_ground_proposals: bool = True  # LLM proposals not grounded in an observation -> request_human_review
    # FlyBot proposals (sentinel/agent/proposals.py): approval needed per action, confidence per decision rule prefix
    proposal_approval: dict[str, bool] = Field(default_factory=dict)
    proposal_confidence: dict[str, float] = Field(default_factory=dict)

    def kind(self, trap_kind: str) -> TrapKindPolicy:
        return self.trap_kinds.get(trap_kind) or TrapKindPolicy(count_basis="peak", new_catch=True, card_changes=True)
