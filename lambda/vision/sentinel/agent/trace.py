"""Trace sinks: in memory (one invocation) and JSONL on disk (local runs, replay). The Lambda returns the
trace to FlyBot, which stores it (MongoDB); nothing is persisted on the AWS side.
"""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.common.schemas import Trace


def dump(trace: Trace) -> dict:
    return trace.model_dump(mode="json")


class TraceSink:
    def write(self, trace: Trace) -> None:
        raise NotImplementedError

    def get(self, event_id: str) -> Trace | None:
        return None


class MemoryTraceSink(TraceSink):
    def __init__(self):
        self.traces: dict[str, dict] = {}
        self.order: list[str] = []

    def write(self, trace: Trace) -> None:
        if trace.event_id not in self.traces:
            self.order.append(trace.event_id)
        self.traces[trace.event_id] = dump(trace)

    def get(self, event_id: str) -> Trace | None:
        d = self.traces.get(event_id)
        return Trace.model_validate(d) if d else None

    def all(self) -> list[Trace]:
        return [Trace.model_validate(self.traces[e]) for e in self.order]


class JsonlTraceSink(TraceSink):
    """Append-only; the last line for an event_id wins (an update is a new line)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, trace: Trace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dump(trace), ensure_ascii=False, separators=(",", ":")) + "\n")

    def get(self, event_id: str) -> Trace | None:
        if not self.path.exists():
            return None
        found = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if f'"event_id":"{event_id}"' in line:
                d = json.loads(line)
                if d.get("event_id") == event_id:
                    found = d
        return Trace.model_validate(found) if found else None


class MultiSink(TraceSink):
    def __init__(self, *sinks: TraceSink):
        self.sinks = [s for s in sinks if s is not None]

    def write(self, trace: Trace) -> None:
        for s in self.sinks:
            s.write(trace)

    def get(self, event_id: str) -> Trace | None:
        for s in self.sinks:
            t = s.get(event_id)
            if t is not None:
                return t
        return None
