"""Normalize whatever arrives into a PhotoEvent: a PhotoEvent, or a dict with PhotoEvent fields.

The vision Lambda builds its PhotoEvent from the FlyBot request (sentinel.handler); the FlyBot-webhook and
Older queue-based event shapes are not used any more.
"""

from __future__ import annotations

from sentinel.common.schemas import PhotoEvent


def normalize_event(raw, camera_info=None) -> PhotoEvent:
    if isinstance(raw, PhotoEvent):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"unsupported event {type(raw).__name__}")
    return PhotoEvent.model_validate(raw)
