"""The agent's view of perception: `Analyzer.analyze(event, previous_image) -> Observation`.

`PipelineAnalyzer` runs `sentinel.vision.pipeline.analyze_photo` (ops 1, 2, 3, 4, 6, 8 plus op 5 through
`sentinel.vision.detector`: YOLO26n or YOLOv7-tiny on OpenCV 5 ENGINE_NEW) and maps the result to an
Observation. Tests use `sentinel.agent.fakes.FakeAnalyzer`.

Images are loaded from `s3://bucket/key` (boto3), `http(s)://` (httpx), a local path / `file://`, or the
process-wide `IMAGE_CACHE` (the Lambda handler decodes the request's photos once, up front, so a missing
photo is reported as `photo_not_found` before the agent runs, and the same bytes are never fetched twice).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol

from sentinel.common.schemas import Counts, NewCatch, Observation, PhotoEvent, QualityOut

IMAGE_CACHE: dict[str, object] = {}  # ref -> decoded BGR np.ndarray (handler-managed, cleared per invocation)
MAX_IMAGE_BYTES = 15 * 1024 * 1024


class Analyzer(Protocol):
    name: str

    def analyze(self, event: PhotoEvent, previous_image: str | None = None) -> Observation: ...


def quality_score(quality: QualityOut | dict | None) -> float | None:
    """0-100 scalar for the ISP verify/revert rule: 100 minus 20 per issue, minus glare and clipping."""
    if quality is None:
        return None
    q = quality if isinstance(quality, dict) else quality.model_dump()
    scores = q.get("scores") or {}
    glare = float(scores.get("glare_ratio") or scores.get("clipped") or 0.0)
    clip = float(scores.get("clip_high") or 0.0)
    return round(max(0.0, min(100.0, 100.0 - 20.0 * len(q.get("issues") or []) - 50.0 * glare - 50.0 * clip)), 2)


def fetch_bytes(ref: str, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Raw bytes of an image ref. Raises FileNotFoundError when the object / URL / file does not exist."""
    if ref.startswith("s3://"):
        from sentinel.common.aws import client

        bucket, key = ref[5:].split("/", 1)
        try:
            body = client("s3").get_object(Bucket=bucket, Key=key)["Body"]
        except Exception as e:  # botocore ClientError NoSuchKey / 404 / AccessDenied
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            if code in ("NoSuchKey", "404", "NotFound", "NoSuchBucket", "AccessDenied", "403"):
                raise FileNotFoundError(f"{ref}: {code}") from e
            raise
        data = body.read(max_bytes + 1)
    elif ref.startswith(("http://", "https://")):
        import httpx

        resp = httpx.get(ref, timeout=30.0, follow_redirects=True)
        if resp.status_code in (400, 403, 404, 410):
            raise FileNotFoundError(f"{ref[:120]}: HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.content
    else:
        p = local_path(ref)
        if not p.is_file():
            raise FileNotFoundError(str(p))
        data = p.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"image larger than {max_bytes} bytes")
    return data


def local_path(ref: str) -> Path:
    """A local path or a file:// URI (Windows: file:///D:/x.jpg -> D:/x.jpg) -> Path."""
    if ref.startswith("file://"):
        from urllib.parse import unquote, urlparse
        from urllib.request import url2pathname

        return Path(url2pathname(unquote(urlparse(ref).path)))
    return Path(ref)


def decode_image(data: bytes, ref: str = ""):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image {ref[:80]}")
    return img


def load_image(ref: str):
    if ref in IMAGE_CACHE:
        return IMAGE_CACHE[ref]
    return decode_image(fetch_bytes(ref), ref)


class PipelineAnalyzer:
    name = "opencv5.pipeline"

    def __init__(self, evidence_dir: str | Path | None = None, method: str | None = None):
        self.evidence_dir = evidence_dir or os.environ.get("SENTINEL_EVIDENCE_DIR")
        self.method = method or os.environ.get("SENTINEL_REGISTER_METHOD", "sift")
        self.last_result: dict | None = None  # raw pipeline dict of the last analysis (handler: boxes with scores)

    def analyze(self, event: PhotoEvent, previous_image: str | None = None) -> Observation:
        from sentinel.vision import pipeline

        t0 = time.perf_counter()
        hour = None
        if event.captured_at is not None:
            from sentinel.agent.policy import load_policy, local

            hour = local(event.captured_at, load_policy()).hour
        res = pipeline.analyze_photo(event.image, previous=previous_image, trap_kind=event.trap_kind,
                                     out_dir=self.evidence_dir, method=self.method, hour=hour,
                                     name=event.event_id.replace(":", "_"))
        self.last_result = res
        return self.from_pipeline(event, previous_image, res, (time.perf_counter() - t0) * 1000)

    @staticmethod
    def from_pipeline(event: PhotoEvent, previous_image: str | None, res: dict, ms: float) -> Observation:
        """Adapter for sentinel.vision.pipeline output (keys read defensively; unknown keys ignored)."""
        q = res.get("quality") or {}
        det = res.get("detect") or res.get("detections") or {}
        diff = res.get("diff") or (res.get("compare") or {}).get("diff") or None
        ev = res.get("evidence") or {}
        ev_list = [v for v in (ev.values() if isinstance(ev, dict) else ev) if isinstance(v, str)]
        return Observation(
            event_id=event.event_id, camera=event.camera, source=event.source, trap_kind=event.trap_kind,
            captured_at=event.captured_at, image=event.image, previous_image=previous_image,
            resolution=event.resolution, width=res.get("width"), height=res.get("height"),
            quality=QualityOut(ok=q.get("ok", True), issues=q.get("issues", []), scores=q.get("scores", {}),
                               score=quality_score(q)),
            counts=Counts(opencv=res.get("count", det.get("count") if isinstance(det, dict) else None),
                          npu=event.npu_count, production=event.production_count),
            boxes=res.get("boxes") or [],
            new_catch=new_catch_of(diff, res.get("register"), event.trap_kind) if diff or res.get("register") else None,
            saturation=(res.get("crowd") or {}).get("saturation", res.get("saturation")),
            card=res.get("card"), motion=res.get("motion"), privacy=res.get("privacy"),
            snapshot_id=event.snapshot_id, analyzer="opencv5.pipeline",
            vision_ms=round(ms, 1), evidence=[event.image, *ev_list] if event.image else ev_list)


def new_catch_of(diff: dict | None, register: dict | None = None, trap_kind: str | None = None) -> NewCatch:
    """Op 4 diff (+ op 3 registration) -> NewCatch, for both trap kinds. A skipped diff keeps the reason and the
    registration result, so live traces show inliers even when op 4 matching could not run."""
    d = dict(diff or {})
    reg = register or {}
    meaning = d.get("meaning") or ({"sticky": "catches", "lure": "activity"}.get(trap_kind or d.get("trap_kind") or ""))
    out = NewCatch(meaning=meaning, inliers=reg.get("inliers"), change_ratio=d.get("change_ratio"),
                   registration_ok=d.get("registration_ok", reg.get("ok")))
    if d.get("status") == "skipped":
        out.skipped = str(d.get("reason") or "skipped")
        return out
    for k in ("new_count", "persisted_count", "disappeared_count"):
        if d.get(k) is not None:
            setattr(out, k, int(d[k]))
    if meaning == "catches":  # card changes only mean something on a sticky card
        out.card_change_suspected = bool(d.get("card_change_suspected"))
        out.card_change_reasons = list(d.get("card_change_reasons") or [])
    return out
