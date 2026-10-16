"""LLM provider abstraction for the LLM planner: `bedrock` | `anthropic` | `fake`.

    SENTINEL_LLM_BACKEND        bedrock (default) | anthropic | fake
    SENTINEL_BEDROCK_MODEL      default global.anthropic.claude-sonnet-4-6 (Converse API)
    SENTINEL_BEDROCK_REGION     default ap-southeast-2;  profile: SENTINEL_AWS_PROFILE / AWS_PROFILE
    ANTHROPIC_API_KEY           anthropic backend (official SDK); never in code
    SENTINEL_ANTHROPIC_MODEL    default claude-opus-5-5
    SENTINEL_ANTHROPIC_EFFORT   default medium (output_config.effort)
    SENTINEL_ANTHROPIC_FALLBACKS default "default" = server-side refusal fallback (beta
                                server-side-fallback-2026-07-01); "off" disables
    SENTINEL_LLM_MAX_TOKENS     default 4096
    SENTINEL_ANTHROPIC_CACHE    default on: prompt caching (system + tools breakpoints, automatic conversation cache)
    SENTINEL_BEDROCK_CACHE      default on: Converse cachePoint after tools and system

Each backend keeps its own native message list in a `Conversation`, so provider-specific blocks (e.g.
Anthropic thinking blocks) are echoed back unchanged. Any provider failure (throttling, auth, network,
refusal, max_tokens) raises `LLMError`; the planner turns that into a RulePlanner fallback. Tests only
ever use `FakeLLM`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BEDROCK_MODEL = "global.anthropic.claude-sonnet-4-6"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


class LLMError(RuntimeError):
    def __init__(self, message: str, *, backend: str = "", code: str = "error"):
        super().__init__(message)
        self.backend, self.code = backend, code


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class LLMTurn:
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=dict)


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_write_input_tokens")


def usage_of(input_tokens=None, output_tokens=None, cache_read=None, cache_write=None) -> dict:
    """Provider usage -> the trace's normalized keys (missing values left out)."""
    vals = dict(zip(USAGE_KEYS, (input_tokens, output_tokens, cache_read, cache_write), strict=True))
    return {k: int(v) for k, v in vals.items() if isinstance(v, int | float) and not isinstance(v, bool)}


class Conversation:
    """`usage` accumulates every provider call of the conversation, including a call that then fails on its
    stop reason (refusal, max_tokens): those tokens were spent too."""

    usage: dict

    def record_usage(self, turn_usage: dict) -> None:
        if not hasattr(self, "usage") or self.usage is None:
            self.usage = {"llm_calls": 0}
        self.usage["llm_calls"] = self.usage.get("llm_calls", 0) + 1
        for k, v in turn_usage.items():
            self.usage[k] = self.usage.get(k, 0) + v

    def step(self) -> LLMTurn:
        raise NotImplementedError

    def add_results(self, results: list[tuple[ToolUse, dict, bool]]) -> None:
        """[(tool_use, result_json, is_error)] -> one user message with every tool result."""
        raise NotImplementedError


class LLMClient:
    backend: str
    model: str

    def start(self, system: str, tools: list[dict], user: str) -> Conversation:
        raise NotImplementedError


def _cache_on(env: str) -> bool:
    return os.environ.get(env, "on").strip().lower() not in ("off", "0", "false", "no")


def _max_tokens() -> int:
    return int(os.environ.get("SENTINEL_LLM_MAX_TOKENS", "4096"))


# ---------------------------------------------------------------- Bedrock (Converse)


class BedrockLLM(LLMClient):
    backend = "bedrock"

    def __init__(self, model: str | None = None, client=None, region: str | None = None):
        self.model = model or os.environ.get("SENTINEL_BEDROCK_MODEL") or DEFAULT_BEDROCK_MODEL
        self._client = client
        self.region = region or os.environ.get("SENTINEL_BEDROCK_REGION") or "ap-southeast-2"

    @property
    def client(self):
        if self._client is None:
            from sentinel.common.aws import session

            self._client = session().client("bedrock-runtime", region_name=self.region)
        return self._client

    def start(self, system, tools, user):
        return _BedrockConversation(self, system, tools, user)


class _BedrockConversation(Conversation):
    """Converse prompt caching (SENTINEL_BEDROCK_CACHE, default on): `cachePoint` blocks after the tool list and
    after the system prompt, the Converse equivalent of Anthropic's breakpoints. Not exercised against real
    Bedrock (0 quota on this account); set SENTINEL_BEDROCK_CACHE=off if a model rejects cachePoint."""

    def __init__(self, llm: BedrockLLM, system: str, tools: list[dict], user: str):
        self.llm = llm
        cache = _cache_on("SENTINEL_BEDROCK_CACHE")
        point = {"cachePoint": {"type": "default"}}
        self.system = [{"text": system}, *([point] if cache else [])]
        specs = [{"toolSpec": {"name": t["name"], "description": t["description"],
                               "inputSchema": {"json": t["input_schema"]}}} for t in tools]
        self.tool_config = {"tools": [*specs, *([point] if cache and specs else [])]}
        self.messages: list[dict] = [{"role": "user", "content": [{"text": user}]}]

    def step(self) -> LLMTurn:
        try:
            resp = self.llm.client.converse(modelId=self.llm.model, system=self.system, messages=self.messages,
                                            toolConfig=self.tool_config,
                                            inferenceConfig={"maxTokens": _max_tokens()})
        except Exception as e:  # botocore ClientError (ThrottlingException, AccessDenied...), network
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            raise LLMError(f"bedrock converse: {type(e).__name__}: {e}"[:500], backend="bedrock",
                           code=code or type(e).__name__) from e
        u = resp.get("usage") or {}
        usage = usage_of(u.get("inputTokens"), u.get("outputTokens"), u.get("cacheReadInputTokens"),
                         u.get("cacheWriteInputTokens"))
        self.record_usage(usage)
        msg = resp["output"]["message"]
        self.messages.append(msg)
        turn = LLMTurn(stop_reason=resp.get("stopReason", "end_turn"), usage=usage)
        for block in msg.get("content", []):
            if "text" in block:
                turn.text += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                turn.tool_uses.append(ToolUse(tu["toolUseId"], tu["name"], tu.get("input") or {}))
        if turn.stop_reason in ("max_tokens", "guardrail_intervened", "content_filtered"):
            raise LLMError(f"bedrock stop reason {turn.stop_reason}", backend="bedrock", code=turn.stop_reason)
        return turn

    def add_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"toolResult": {"toolUseId": tu.id, "content": [{"json": res}], "status": "error" if err else "success"}}
            for tu, res, err in results]})


# ---------------------------------------------------------------- Anthropic API (official SDK)


class AnthropicLLM(LLMClient):
    backend = "anthropic"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.environ.get("SENTINEL_ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
        self.effort = os.environ.get("SENTINEL_ANTHROPIC_EFFORT", "medium")
        self.fallbacks = os.environ.get("SENTINEL_ANTHROPIC_FALLBACKS", "default")
        self.cache = _cache_on("SENTINEL_ANTHROPIC_CACHE")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMError("ANTHROPIC_API_KEY not set", backend="anthropic", code="no_credentials")
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def start(self, system, tools, user):
        return _AnthropicConversation(self, system, tools, user)


class _AnthropicConversation(Conversation):
    """Prompt caching (SENTINEL_ANTHROPIC_CACHE, default on). Render order is tools -> system -> messages:
    - a breakpoint on the last tool and one on the system prompt: the ~9k-token policy + 14 tool schemas are
      byte-identical for every photo, so the 2nd+ photo reads them from cache (5 min TTL, refreshed on use);
    - top-level automatic caching: the growing conversation of this photo is cached turn to turn.
    Everything that varies (the photo, now) is in the messages, after the stable prefix."""

    def __init__(self, llm: AnthropicLLM, system: str, tools: list[dict], user: str):
        self.llm = llm
        if llm.cache:
            self.system: Any = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            self.tools = [dict(t) for t in tools]
            if self.tools:
                self.tools[-1]["cache_control"] = {"type": "ephemeral"}
        else:
            self.system, self.tools = system, tools
        self.messages: list[dict] = [{"role": "user", "content": user}]

    def step(self) -> LLMTurn:
        import anthropic

        kwargs: dict[str, Any] = {"model": self.llm.model, "max_tokens": _max_tokens(), "system": self.system,
                                  "tools": self.tools, "messages": self.messages,
                                  "output_config": {"effort": self.llm.effort}}
        if self.llm.cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        try:
            client = self.llm.client
            if self.llm.fallbacks and self.llm.fallbacks != "off":
                resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                   fallbacks=self.llm.fallbacks, **kwargs)
            else:
                resp = client.messages.create(**kwargs)
        except LLMError:
            raise
        except anthropic.APIStatusError as e:
            raise LLMError(f"anthropic HTTP {e.status_code}: {e.message}"[:500], backend="anthropic",
                           code=str(e.status_code)) from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"anthropic connection: {e}"[:500], backend="anthropic", code="connection") from e
        u = getattr(resp, "usage", None)
        usage = usage_of(getattr(u, "input_tokens", None), getattr(u, "output_tokens", None),
                         getattr(u, "cache_read_input_tokens", None), getattr(u, "cache_creation_input_tokens", None))
        self.record_usage(usage)
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"anthropic stop reason {resp.stop_reason}", backend="anthropic", code=resp.stop_reason)
        self.messages.append({"role": "assistant", "content": resp.content})
        turn = LLMTurn(stop_reason=resp.stop_reason or "end_turn", usage=usage)
        for block in resp.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "tool_use":
                turn.tool_uses.append(ToolUse(block.id, block.name, dict(block.input or {})))
        return turn

    def add_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(res, default=str), "is_error": err}
            for tu, res, err in results]})


# ---------------------------------------------------------------- Fake (tests, replay)


class FakeLLM(LLMClient):
    """Scripted turns. Each script item is one of:
    - dict {"text": str, "tool_uses": [{"name", "input"}]}     ("input" may be a callable(results) -> dict)
    - a callable(results_so_far) -> such a dict
    - an Exception instance (raised as LLMError, like a provider failure)
    `results_so_far` is the list of every tool result returned to the model, in order."""

    backend = "fake"

    def __init__(self, script: list[Any], model: str = "fake-planner-1"):
        self.script, self.model = list(script), model
        self.conversations: list[_FakeConversation] = []

    def start(self, system, tools, user):
        conv = _FakeConversation(self, system, tools, user)
        self.conversations.append(conv)
        return conv


class _FakeConversation(Conversation):
    def __init__(self, llm: FakeLLM, system: str, tools: list[dict], user: str):
        self.llm, self.system, self.tools, self.user = llm, system, tools, user
        self.results: list[dict] = []
        self.i = 0

    def step(self) -> LLMTurn:
        if self.i >= len(self.llm.script):
            return LLMTurn(text="(script ended)")
        item = self.llm.script[self.i]
        self.i += 1
        if isinstance(item, BaseException):
            raise LLMError(f"fake: {type(item).__name__}: {item}", backend="fake", code=type(item).__name__) from item
        if callable(item):
            item = item(self.results)
        uses = []
        for n, u in enumerate(item.get("tool_uses") or []):
            inp = u.get("input") or {}
            if callable(inp):
                inp = inp(self.results)
            uses.append(ToolUse(f"fake-{self.i}-{n}", u["name"], inp))
        return LLMTurn(text=item.get("text", ""), tool_uses=uses, stop_reason="tool_use" if uses else "end_turn")

    def add_results(self, results):
        self.results.extend(r for _, r, _ in results)


def from_env(backend: str | None = None) -> LLMClient:
    backend = (backend or os.environ.get("SENTINEL_LLM_BACKEND") or "bedrock").lower()
    if backend == "bedrock":
        return BedrockLLM()
    if backend == "anthropic":
        return AnthropicLLM()
    if backend == "fake":
        return FakeLLM([{"text": "fake backend: no decision"}])
    raise ValueError(f"unknown SENTINEL_LLM_BACKEND {backend!r}")


__all__ = ["AnthropicLLM", "BedrockLLM", "FakeLLM", "LLMClient", "LLMError", "LLMTurn", "ToolUse", "from_env"]
