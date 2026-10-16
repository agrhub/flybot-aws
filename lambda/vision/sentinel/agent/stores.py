"""Agent state (budgets, pending ISP checks, follow-ups, last photo per camera) and approvals, in memory.

The vision Lambda is stateless: one invocation = one photo; FlyBot keeps the durable state (MongoDB). The
handler seeds a fresh MemoryStateStore from the request (history) and reads the proposals back from the trace.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from sentinel.common.ids import parse_dt
from sentinel.common.schemas import Approval


class StateStore:
    def get(self, camera: str) -> dict:
        raise NotImplementedError

    def put(self, camera: str, state: dict) -> None:
        raise NotImplementedError

    def update(self, camera: str, fn) -> dict:
        state = self.get(camera)
        fn(state)
        self.put(camera, state)
        return state

    def cameras(self) -> list[str]:
        """Serials this store knows (camera rows and/or agent state)."""
        return []

    def camera_rows(self) -> list[dict]:
        """Camera facts per serial ({serial, trap_kind, lat, lon, plot_id, farm_id, source, excluded, ...})."""
        return [{"serial": s} for s in self.cameras()]


class MemoryStateStore(StateStore):
    def __init__(self):
        self._data: dict[str, dict] = {}

    def get(self, camera: str) -> dict:
        return deepcopy(self._data.get(camera, {}))

    def put(self, camera: str, state: dict) -> None:
        self._data[camera] = deepcopy(state)

    def cameras(self) -> list[str]:
        return sorted(self._data)


class ApprovalStore:
    def put(self, approval: Approval) -> None:
        raise NotImplementedError

    def get(self, approval_id: str) -> Approval | None:
        raise NotImplementedError

    def pending(self) -> list[Approval]:
        raise NotImplementedError


class MemoryApprovalStore(ApprovalStore):
    def __init__(self):
        self._data: dict[str, dict] = {}

    def put(self, approval: Approval) -> None:
        self._data[approval.approval_id] = approval.model_dump(mode="json")

    def get(self, approval_id: str) -> Approval | None:
        d = self._data.get(approval_id)
        return Approval.model_validate(d) if d else None

    def pending(self) -> list[Approval]:
        return [Approval.model_validate(d) for d in self._data.values() if d["status"] == "pending"]

    def all(self) -> list[Approval]:
        return [Approval.model_validate(d) for d in self._data.values()]


def due(items: list[dict], now: datetime) -> tuple[list[dict], list[dict]]:
    """Split follow-ups [{at, ...}] into (due, later)."""
    ready, later = [], []
    for f in items:
        (ready if (parse_dt(f.get("at")) or now) <= now else later).append(f)
    return ready, later
