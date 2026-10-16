"""Build a Runtime for one FlyBot request (the vision Lambda) or from the environment (local runs).

    build_runtime(camera=..., history=[...], planner="rule"|"llm", llm=None, clock=None, evidence_dir=None)

One Runtime per invocation: memory stores seeded from the request (the Lambda is stateless; FlyBot keeps the
durable state), a ProposalDevice for the request's camera, a ProposalFlyBot that returns proposals in the
response, the OpenCV 5 PipelineAnalyzer, and optionally the Anthropic LLM client (deep mode / SENTINEL_PLANNER=llm).

Env (handler + local):
    SENTINEL_PLANNER          rule (default) | llm        LLM planner for every photo (deep mode always uses it)
    SENTINEL_LLM_BACKEND      anthropic (default) | bedrock | fake
    SENTINEL_ANTHROPIC_MODEL  default claude-haiku-5-5
    ANTHROPIC_API_KEY         or ANTHROPIC_SECRET_ARN / ANTHROPIC_SECRET_NAME (Secrets Manager, read once per cold start)
    SENTINEL_POLICY           policy YAML (default sentinel/agent/policy.yaml)
    SENTINEL_KILL_SWITCH      1 = analysis only, no proposals
    SENTINEL_WEATHER          off (default in the Lambda) | open-meteo   (the surge -> treatment rule needs weather)
"""

from __future__ import annotations

import os
from datetime import datetime

from sentinel.agent.data import AgentData, OpenMeteoWeather
from sentinel.agent.guardrails import StaticKillSwitch, kill_switch_from
from sentinel.agent.policy import load_policy
from sentinel.agent.proposals import ProposalDevice, ProposalFlyBot
from sentinel.agent.runtime import Runtime
from sentinel.agent.stores import MemoryApprovalStore, MemoryStateStore
from sentinel.agent.trace import JsonlTraceSink, MemoryTraceSink, MultiSink
from sentinel.agent.vision import PipelineAnalyzer
from sentinel.common.ids import parse_dt
from sentinel.devices.base import DeviceRegistry

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
_llm_cache: dict = {}


def anthropic_key_from_secret() -> bool:
    """ANTHROPIC_API_KEY from Secrets Manager (ANTHROPIC_SECRET_ARN or ANTHROPIC_SECRET_NAME), once per process.
    The env var wins when already set (local runs). Returns True when a key is available."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    ref = os.environ.get("ANTHROPIC_SECRET_ARN") or os.environ.get("ANTHROPIC_SECRET_NAME")
    if not ref:
        return False
    try:
        from sentinel.common.aws import client

        val = client("secretsmanager").get_secret_value(SecretId=ref)["SecretString"].strip()
    except Exception as e:  # noqa: BLE001 - without it the LLM planner falls back to the rule planner (traced)
        print(f'{{"msg":"anthropic secret unreadable","error":"{type(e).__name__}"}}')
        return False
    if val and not val.startswith("placeholder"):
        os.environ["ANTHROPIC_API_KEY"] = val
        return True
    return False


def llm_from_env():
    """The LLM client (cached per process), or None when no backend / key is configured."""
    backend = (os.environ.get("SENTINEL_LLM_BACKEND") or "anthropic").lower()
    if backend in _llm_cache:
        return _llm_cache[backend]
    from sentinel.agent import llm as llm_mod

    if backend == "anthropic":
        if not anthropic_key_from_secret():
            return None
        os.environ.setdefault("SENTINEL_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        import anthropic

        client = anthropic.Anthropic(timeout=float(os.environ.get("SENTINEL_ANTHROPIC_TIMEOUT_S", "30")), max_retries=1)
        _llm_cache[backend] = llm_mod.AnthropicLLM(client=client)
    else:
        _llm_cache[backend] = llm_mod.from_env(backend)
    return _llm_cache[backend]


def build_runtime(*, camera: dict, history: list[dict] | None = None, planner: str = "rule", llm=None,
                  clock=None, evidence_dir: str | None = None, policy=None, analyzer=None, trace_jsonl: str | None = None,
                  weather: str | None = None) -> Runtime:
    """`camera`: {id, name?, trapKind, platform?, location{lat,lng}?, battery?, isp?, alertThreshold?}.
    `history`: [{capturedAt, count}] (up to 14 days) -> the camera's daily rollups for the surge rule."""
    policy = policy or load_policy()
    state = MemoryStateStore()
    serial = str(camera["id"])
    loc = camera.get("location") or {}
    seed = {"serial": serial, "source": camera.get("platform") or "ezviz", "trap_kind": camera.get("trapKind") or "lure",
            "name": camera.get("name"), "lat": loc.get("lat"), "lon": loc.get("lng", loc.get("lon")),
            "alert_threshold": camera.get("alertThreshold"), "battery_percent": camera.get("battery"),
            "farm_id": camera.get("farmId"), "plot_id": camera.get("plotId")}
    seed = {k: v for k, v in seed.items() if v is not None}
    weather = weather or os.environ.get("SENTINEL_WEATHER", "off")
    data = AgentData(state, policy, [seed], OpenMeteoWeather() if weather == "open-meteo" else None)
    for h in history or []:
        at, count = parse_dt(h.get("capturedAt")), h.get("count")
        if at is not None and isinstance(count, int | float) and not isinstance(count, bool):
            data.record_count(serial, at, int(count), h.get("photoId") or h.get("snapshotId"))
    device = ProposalDevice(serial, source=seed["source"], isp=camera.get("isp"), battery=camera.get("battery"))
    traces = MultiSink(MemoryTraceSink(), JsonlTraceSink(trace_jsonl) if trace_jsonl else None)
    kill = kill_switch_from(policy, StaticKillSwitch(False))
    if planner == "llm" and llm is None:
        llm = llm_from_env()
    rt = Runtime(policy=policy, devices=DeviceRegistry({serial: device}), analyzer=analyzer or PipelineAnalyzer(evidence_dir),
                 data=data, state=state, approvals=MemoryApprovalStore(), traces=traces, kill_switch=kill,
                 flybot=ProposalFlyBot(), llm=llm, planner=planner if llm is not None else "rule",
                 post_analysis=False)
    if clock is not None:
        rt.clock = clock if callable(clock) else (lambda c=clock: c)
    return rt


def runtime_from_env() -> Runtime:
    """A runtime for ad-hoc local runs (no request): one camera from SENTINEL_CAMERA (default 'local')."""
    return build_runtime(camera={"id": os.environ.get("SENTINEL_CAMERA", "local"),
                                 "trapKind": os.environ.get("SENTINEL_TRAP_KIND", "lure")},
                         planner=os.environ.get("SENTINEL_PLANNER", "rule"), trace_jsonl=os.environ.get("SENTINEL_TRACE_JSONL"))


def clock_at(dt: datetime | None):
    """A clock returning `dt` (the photo's capturedAt) so budgets / quiet hours are judged at capture time."""
    if dt is None:
        return None
    return lambda: dt
