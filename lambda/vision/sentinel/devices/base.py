"""Device interface the agent's device tools go through.

Two implementations: `camos_iot.CamosIotDevice` (AI camera, AWS IoT Core shadow + RPC, protocol v1) and
`ezviz_via_flybot.EzvizViaFlyBot` (EZVIZ cameras, through FlyBot's Sentinel API). Each method returns a
`DeviceAck` and never raises for device-side failures (timeouts, NOT_SUPPORTED, BAD_VALUE): the guardrails
and the trace need the failure as data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel.common.schemas import DeviceAck

# protocol 8, `video.image` (firmware 0.25): integer ranges, same as the camera's web config page
ISP_RANGES: dict[str, tuple[int, int]] = {
    "brightness": (-64, 64),
    "contrast": (0, 100),
    "saturation": (0, 100),
    "sharpness": (0, 100),
    "flicker": (0, 3),
    "wdr": (0, 2),
    "orientation": (0, 3),
    "ldc": (0, 1),
}


def validate_image_params(params: dict) -> tuple[dict, list[dict]]:
    """Keep known integer fields within range; report the rest as protocol-style errors."""
    clean, errors = {}, []
    for k, v in params.items():
        if v is None:
            continue
        if k not in ISP_RANGES:
            errors.append({"path": f"video.image.{k}", "code": "NOT_SUPPORTED"})
            continue
        lo, hi = ISP_RANGES[k]
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            errors.append({"path": f"video.image.{k}", "code": "BAD_VALUE"})
            continue
        clean[k] = v
    return clean, errors


def clamp_image_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if k in ISP_RANGES and v is not None:
            lo, hi = ISP_RANGES[k]
            out[k] = max(lo, min(hi, round(v)))
    return out


def not_supported(serial: str, method: str, transport: str, why: str) -> DeviceAck:
    return DeviceAck(ok=False, serial=serial, method=method, transport=transport,
                     error={"code": "NOT_SUPPORTED", "message": why})


class Device(ABC):
    serial: str
    source: str  # camos | ezviz
    capabilities: frozenset[str] = frozenset()

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    @abstractmethod
    def capture(self, resolution: str = "main", detection_id: str | None = "trap-count") -> DeviceAck:
        """Take a photo now. CamOS: synchronous (image key in the ack). EZVIZ: `pending` (photo comes back
        through FlyBot's snapshot_analyzed webhook)."""

    @abstractmethod
    def set_image(self, params: dict) -> DeviceAck: ...

    @abstractmethod
    def get_image_settings(self) -> dict | None:
        """Current ISP values (reported.video.image), or None when unknown."""

    def battery_percent(self) -> float | None:
        """Battery level (0-100) of a battery-powered camera; None = mains powered or unknown."""
        return None

    @abstractmethod
    def set_detection_roi(self, polygon: list[list[float]], detection_id: str = "trap-count") -> DeviceAck: ...

    @abstractmethod
    def set_capture_schedule(self, every_min: int, from_: str, to: str, detection_id: str = "trap-count") -> DeviceAck: ...


class DeviceRegistry:
    """serial -> Device. A factory may build devices lazily (e.g. from the camera registry)."""

    def __init__(self, devices: dict[str, Device] | None = None, factory=None):
        self._devices = dict(devices or {})
        self._factory = factory

    def get(self, serial: str) -> Device | None:
        if serial not in self._devices and self._factory is not None:
            dev = self._factory(serial)
            if dev is not None:
                self._devices[serial] = dev
        return self._devices.get(serial)

    def add(self, device: Device) -> None:
        self._devices[device.serial] = device
