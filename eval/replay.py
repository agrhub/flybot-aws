"""Deterministic replay of the FlyBot export through the Sentinel agent loop (T2.6).

    uv run python -m eval.replay --days 30 --approve-after-hours 2
    uv run python -m eval.replay --from 2026-10-07 --to 2026-10-14 --cameras BE4089629 --approve-none
    uv run python -m eval.replay --days 7 --agent stub          # no agent: only the detector + trace plumbing

Input: data/manifest.csv minus data/excluded_cameras.csv, restricted to a date range in farm time
(UTC+7; `--from/--to` inclusive, or `--days N` ending at the manifest's last farm day), optionally to
`--cameras`. Repeated 640x640 test pictures (exact 640x640 AND the same file hash on two or more cameras;
597 of 3,000 photos in the 2026-10 export) are dropped and counted: they are not camera frames.

For each remaining photo, in (captured_at, snapshot_id) order, the harness builds a `snapshot_analyzed`
event with the **field names of FlyBot's webhook** (ezviz-manager API.md, `snapshot_analyzed`: `id`, `event`,
`createdAt`, `deviceSerial`, `severity`, `params{snapshotId, capturedAt, imageUrl, width, height, count,
counts, aiSource, aiProvider, boxFormat, boxes[{label, confidence, bbox xyxy}], boxesTotal, truncated,
quality{...issues}, previousSnapshotId, cardReplacedAt, alertThreshold, ...}`) and calls

    sentinel.agent.loops.per_photo(event, ctx)      # A2's loop, if importable; else the "no agent" stub

Differences from a live webhook (all outside `params`, under the top-level key `_replay`): `count`/`counts`
are FlyBot's production values from the manifest (Roboflow at capture time, NOT human ground truth), `boxes`
are YOLOv7-tiny boxes from `sentinel/vision/detect_yolov7.py` (production boxes are not in the export) cached in
data/cache/yolov7/<snapshot_id>.json, `imageUrl` is a file:// URI, `quality` has the manifest's `issues` only
(the export carries no brightness/sharpness numbers), `createdAt` = capturedAt + 8 s (assumed pipeline delay).

ReplayContext (`ctx`) - what simulated devices and operators answer
-------------------------------------------------------------------
    ctx.now                       datetime of the current photo's createdAt (the replay clock, no wall clock)
    ctx.cameras                   list of {serial, name, trapKind, lat, lon, farmId, plotId, alertThreshold}
    ctx.rng                       random.Random(seed), the only allowed randomness
    ctx.capture(camera, **opts)   -> {"ok", "simulated": True, "event", "snapshot_id", "shift_s", "after_isp_change", ...}
        Returns the NEXT real photo of that camera in the manifest (time-shifted: it is the photo that would
        have arrived anyway, `shift_s` later). Repeated calls before the next scheduled photo return the
        photo after that, and so on. The scheduled photo still arrives later through per_photo. Unknown
        camera or no next photo -> {"ok": False, "error": ...}. Every call counts as an extra capture.
    ctx.set_image(camera, **params) -> {"ok": True, "acked": True, "applied": False, ...}
        Acks the ISP change (shadow desired), records it, and marks the first photo of that camera returned
        afterwards (by capture or by the schedule) with `_replay.after_isp_change = {params, requested_at}`.
        The pixels are NOT changed (the photo was taken without the change): a replay cannot show an
        improved image, it can only show that the agent asked and re-captured.
    ctx.neighbours(camera, k=1)   -> other cameras by distance [{serial, distance_km, trapKind, latest}]
        (the export has two trap cameras: each one's only neighbour is the other)
    ctx.latest(camera) / ctx.history(camera, n)   last events of a camera that the schedule delivered
    ctx.propose(kind, payload, camera=None, reason="") -> {"proposal_id", "status"}
        Recorded as a pending proposal; approvals follow the policy: approve_all (approved at once),
        approve_none (never), approve_after_hours H (approved H replay-hours later, resolved before the first
        photo at or after that time). ctx.proposal_status(id), ctx.pending_proposals().
        If the agent module defines `on_approval(proposal, ctx)` it is called when a proposal is approved.
    ctx.request_review(snapshot_id, reason)   a human review request (counted)
    ctx.fallback(reason, detail="")           the agent fell back to a rule / degraded path (counted)
    ctx.log(kind, **fields)                   any other trace line

Output: data/replay/<run_id>/trace.jsonl (one line per photo, then one per action; no wall-clock fields, so a
re-run with the same inputs and seed is byte-identical), summary.json (decisions by type, extra captures,
proposals, reviews, fallbacks, window, filters, runtime, trace sha256) and a markdown table on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib
import inspect
import json
import math
import random
import struct
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FARM_TZ = timedelta(hours=7)
PIPELINE_DELAY = timedelta(seconds=8)  # assumed capture -> webhook delay (createdAt = capturedAt + this)
TEST_SIZE = (640, 640)
MAX_BOXES = 200
LABEL = "fruit fly"

ASSUMPTIONS = [
    ("capture(camera) returns the NEXT real photo of that camera in the manifest (time-shifted by shift_s); "
    "the scheduled photo still arrives later."),
    ("set_image acks and marks the next photo returned for that camera as 'after ISP change'; the pixels are "
    "not changed (replay cannot show an improved image)."),
    ("count/counts are FlyBot's production values (Roboflow at capture time), not human ground truth; boxes are "
    "YOLOv7-tiny output (the export has no production boxes)."),
    "quality carries the manifest's issues only (no brightness/sharpness in the export).",
    "createdAt = capturedAt + 8 s; previousSnapshotId is the manifest's (it can point at a dropped test picture).",
    "neighbour lookups use the camera list after exclusions (two trap cameras).",
    "approvals follow --approve-all / --approve-none / --approve-after-hours H on the replay clock.",
]


# --------------------------------------------------------------------------- time helpers

def parse_ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone(UTC)


def iso_ms(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def farm_day(iso: str) -> date:
    return (parse_ts(iso) + FARM_TZ).date()


# --------------------------------------------------------------------------- manifest + filters

def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_excluded(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["device_serial"] for r in csv.DictReader(fh) if r.get("device_serial")}


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the first SOFn marker, or None."""
    i, n = 2, len(data)
    if data[:2] != b"\xff\xd8":
        return None
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7 or m == 0xFF:
            i += 1 if m == 0xFF else 2
            continue
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


def image_info(rows: list[dict], data_dir: Path, cache_path: Path | None) -> dict[str, dict]:
    """{snapshot_id: {w, h, sha1, bytes}} for every row, cached on (snapshot_id, file size)."""
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    out, dirty = {}, False
    for r in rows:
        sid = r["snapshot_id"]
        p = data_dir / r["image_path"]
        try:
            if not r["image_path"] or not p.is_file():
                raise OSError("no image")
            size = p.stat().st_size
        except OSError:
            out[sid] = {"w": None, "h": None, "sha1": None, "bytes": None}
            continue
        c = cache.get(sid)
        if c and c.get("bytes") == size:
            out[sid] = c
            continue
        data = p.read_bytes()
        wh = jpeg_size(data)
        out[sid] = {"w": wh[0] if wh else None, "h": wh[1] if wh else None,
                    "sha1": hashlib.sha1(data).hexdigest(), "bytes": size}
        cache[sid] = out[sid]
        dirty = True
    if dirty and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return out


def find_test_pictures(rows: list[dict], info: dict[str, dict]) -> set[str]:
    """Exact 640x640 AND a file hash seen on two or more cameras: the repeated uploaded test pictures."""
    cams_by_hash: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        i = info.get(r["snapshot_id"], {})
        if (i.get("w"), i.get("h")) == TEST_SIZE and i.get("sha1"):
            cams_by_hash[i["sha1"]].add(r["device_serial"])
    bad = {h for h, cams in cams_by_hash.items() if len(cams) >= 2}
    return {r["snapshot_id"] for r in rows
            if (info.get(r["snapshot_id"], {}).get("w"), info.get(r["snapshot_id"], {}).get("h")) == TEST_SIZE
            and info[r["snapshot_id"]].get("sha1") in bad}


def select_rows(rows: list[dict], *, excluded: set[str], test_ids: set[str], date_from: date | None,
                date_to: date | None, days: int | None, cameras: list[str] | None,
                missing_image_ids: set[str] | None = None) -> tuple[list[dict], dict]:
    """Filter and time-order. Returns (rows, report) where report counts every drop and the final window."""
    report: dict[str, Any] = {"manifest_rows": len(rows)}
    kept = [r for r in rows if r["device_serial"] not in excluded]
    report["dropped_excluded_cameras"] = len(rows) - len(kept)
    report["excluded_cameras"] = sorted(excluded)
    if cameras:
        before = len(kept)
        kept = [r for r in kept if r["device_serial"] in set(cameras)]
        report["dropped_not_in_cameras_filter"] = before - len(kept)
    if missing_image_ids:
        before = len(kept)
        kept = [r for r in kept if r["snapshot_id"] not in missing_image_ids]
        report["dropped_no_image_file"] = before - len(kept)
    # the window is anchored on the manifest's last farm day among usable cameras
    if kept:
        last = max(farm_day(r["captured_at"]) for r in kept)
        first = min(farm_day(r["captured_at"]) for r in kept)
    else:
        last = first = date.today()
    d_to = date_to or last
    d_from = date_from or (d_to - timedelta(days=days - 1) if days else first)
    before = len(kept)
    kept = [r for r in kept if d_from <= farm_day(r["captured_at"]) <= d_to]
    report["dropped_outside_date_range"] = before - len(kept)
    before = len(kept)
    drop_tests = [r for r in kept if r["snapshot_id"] in test_ids]
    kept = [r for r in kept if r["snapshot_id"] not in test_ids]
    report["dropped_test_pictures_640x640"] = len(drop_tests)
    report["test_pictures_distinct_hashes"] = None  # filled by the caller (needs hashes)
    report["window"] = {"from": d_from.isoformat(), "to": d_to.isoformat(), "farm_tz": "UTC+7"}
    kept.sort(key=lambda r: (r["captured_at"], r["snapshot_id"]))
    report["photos"] = len(kept)
    return kept, report


# --------------------------------------------------------------------------- detection

class Yolov7Source:
    """Cached YOLOv7-tiny detections: `detect(row, image_path)` -> {boxes[{xywh, score}], count, ...}."""

    def __init__(self, onnx: Path, cache_dir: Path, engine: str = "new", resize: str = "letterbox",
                 conf: float = 0.25, iou: float = 0.45):
        self.onnx, self.cache_dir = Path(onnx), Path(cache_dir)
        self.engine, self.resize, self.conf, self.iou = engine, resize, conf, iou
        st = self.onnx.stat()
        self.sig = f"{self.onnx.name}:{st.st_size}:engine={engine}:resize={resize}:conf={conf}:iou={iou}"
        self._det = None
        self.hits = self.misses = 0
        self.detect_s = 0.0

    def _detector(self):
        if self._det is None:
            from sentinel.vision.detect_yolov7 import YoloV7Detector
            self._det = YoloV7Detector(self.onnx, self.engine, resize=self.resize, conf=self.conf, iou=self.iou)
        return self._det

    def detect(self, row: dict, image_path: Path) -> dict:
        f = self.cache_dir / f"{row['snapshot_id']}.json"
        if f.exists():
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
                if c.get("sig") == self.sig:
                    self.hits += 1
                    return c
            except (OSError, ValueError):
                pass
        import cv2
        t0 = time.perf_counter()
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read {image_path}")
        res = self._detector().detect(img)
        out = {"snapshot_id": row["snapshot_id"], "sig": self.sig, "width": img.shape[1], "height": img.shape[0],
               "count": res["count"], "boxes": res["boxes"], "forward_ms": res["forward_ms"]}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out), encoding="utf-8")
        self.misses += 1
        self.detect_s += time.perf_counter() - t0
        return out


class ProductionBoxSource:
    """FlyBot's own boxes from the export (data/export/boxes/<snapshot_id>.json, xyxy pixels); none = empty list.

    Same interface as Yolov7Source. Present only for photos where FlyBot stored boxes (419 of 1,603 rows in the
    2026-10-15 export), so most photos get an empty list: use it to check box-dependent ops, not as a detector.
    """

    hits = misses = 0
    detect_s = 0.0

    def __init__(self, boxes_dir: Path):
        self.boxes_dir = Path(boxes_dir)

    def detect(self, row: dict, image_path: Path) -> dict:
        f = self.boxes_dir / f"{row['snapshot_id']}.json"
        boxes = []
        if f.exists():
            for b in json.loads(f.read_text(encoding="utf-8")).get("boxes", []):
                x0, y0, x1, y1 = b["bbox"]
                boxes.append({"xywh": [x0, y0, x1 - x0, y1 - y0], "score": float(b.get("confidence") or 0.0),
                              "class_id": 0})
        self.hits += 1
        return {"snapshot_id": row["snapshot_id"], "count": len(boxes), "boxes": boxes, "forward_ms": 0.0,
                "box_source": "production"}


class CachedYolov7Source:
    """YOLOv7-tiny results read from `--cache-dir` only (no ONNX needed; the model is git-ignored).

    For the bundled sample (data/samples/yolov7/, written by Yolov7Source on OpenCV 5 ENGINE_NEW). A photo
    without a cache file is an error, never a silent empty detection.
    """

    hits = misses = 0
    detect_s = 0.0

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def detect(self, row: dict, image_path: Path) -> dict:
        f = self.cache_dir / f"{row['snapshot_id']}.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: --box-source cached needs a cached result for every photo")
        self.hits += 1
        return json.loads(f.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- events

def _quality_issues(raw: str) -> list[str]:
    return [s.strip() for s in (raw or "").replace(";", ",").split(",") if s.strip()]


def _num(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def build_event(row: dict, det: dict, info: dict, cam: dict, data_dir: Path, *, index: int, idx_total: int) -> dict:
    """A `snapshot_analyzed`-shaped event (FlyBot field names) for one manifest row."""
    captured = parse_ts(row["captured_at"])
    boxes = []
    for b in sorted(det["boxes"], key=lambda b: -b["score"]):
        x, y, w, h = b["xywh"]
        boxes.append({"label": LABEL, "confidence": round(float(b["score"]), 4),
                      "bbox": [round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)]})
    total = len(boxes)
    try:
        counts = json.loads(row.get("ai_counts_json") or "{}")
    except ValueError:
        counts = {}
    prod = int(float(row["ai_count"])) if row.get("ai_count") not in (None, "") else None
    img_path = (data_dir / row["image_path"]).resolve()
    return {
        "id": "snap_" + row["snapshot_id"],
        "event": "snapshot_analyzed",
        "createdAt": iso_ms(captured + PIPELINE_DELAY),
        "deviceSerial": row["device_serial"],
        "severity": "info",
        "params": {
            "snapshotId": row["snapshot_id"],
            "deviceSerial": row["device_serial"],
            "deviceName": cam.get("name"),
            "provider": row.get("provider") or cam.get("provider") or "ezviz",
            "trapKind": row.get("trap_kind") or cam.get("trapKind"),
            "farmId": row.get("farm_id") or None,
            "plotId": row.get("plot_id") or None,
            "latitude": _num(row.get("lat")),
            "longitude": _num(row.get("lon")),
            "capturedAt": iso_ms(captured),
            "imageUrl": img_path.as_uri(),
            "imageUrlExpiresAt": None,
            "width": info.get("w") or det.get("width"),
            "height": info.get("h") or det.get("height"),
            "count": prod,
            "counts": counts,
            "aiSource": "cloud",
            "aiProvider": "roboflow",
            "boxFormat": "xyxy_pixels",
            "boxes": boxes[:MAX_BOXES],
            "boxesTotal": total,
            "truncated": total > MAX_BOXES,
            "quality": {"brightness": None, "contrast": None, "sharpness": None, "clipped": None,
                        "coverage": None, "issues": _quality_issues(row.get("quality_issues", "")),
                        "checkedAt": iso_ms(captured + PIPELINE_DELAY)},
            "previousSnapshotId": row.get("previous_snapshot_id") or None,
            "cardReplacedAt": row.get("card_replaced_at") or None,
            "alertThreshold": cam.get("alertThreshold"),
        },
        "_replay": {
            "index": index, "of": idx_total,
            "box_source": det.get("box_source", "yolov7-tiny"), "count_source": "production_ai_count",
            "yolov7_count": total, "after_isp_change": None, "simulated_capture": False,
        },
    }


# --------------------------------------------------------------------------- context

@dataclass
class Policy:
    mode: str = "none"  # all | none | after_hours
    hours: float = 0.0

    @classmethod
    def from_args(cls, approve_all: bool, approve_none: bool, after_hours: float | None) -> Policy:
        if approve_all:
            return cls("all")
        if after_hours is not None:
            return cls("after_hours", float(after_hours))
        return cls("none")

    def label(self) -> str:
        return {"all": "approve-all", "none": "approve-none"}.get(self.mode, f"approve-after-{self.hours:g}h")


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p = math.pi / 180
    h = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


class ReplayContext:
    """Simulated devices, neighbours and operators for one replay run (see the module docstring)."""

    def __init__(self, rows: list[dict], cameras: list[dict], policy: Policy, seed: int,
                 make_event: Callable[[dict, int], dict], on_approval: Callable | None = None):
        self.cameras = cameras
        self.policy = policy
        self.seed = seed
        self.rng = random.Random(seed)
        self.now: datetime = datetime(1970, 1, 1, tzinfo=UTC)
        self.mode = "replay"
        self._make_event = make_event
        self._on_approval = on_approval
        self._rows: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            self._rows[r["device_serial"]].append(r)
        self._pos: dict[str, int] = {c: -1 for c in self._rows}  # index of the last delivered scheduled photo
        self._extra: dict[str, int] = defaultdict(int)  # captures since that photo
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._isp_pending: dict[str, dict] = {}
        self._isp_flag: dict[tuple[str, int], dict] = {}
        self.proposals: dict[str, dict] = {}
        self.records: list[dict] = []  # trace action lines of the current photo (drained by the loop)
        self.counters: Counter = Counter()
        self._seq = 0
        self._current_snapshot: str | None = None

    # -- clock / bookkeeping (used by the loop)
    def _begin_photo(self, row: dict, event: dict) -> None:
        cam = row["device_serial"]
        self._pos[cam] = self._pos[cam] + 1
        self._extra[cam] = 0
        self.now = parse_ts(event["createdAt"])
        self._current_snapshot = row["snapshot_id"]

    def _deliver(self, event: dict) -> None:
        h = self._history[event["deviceSerial"]]
        h.append(event)
        del h[:-20]

    def isp_flag_for(self, cam: str, idx: int) -> dict | None:
        """Flag the first photo returned after set_image (idempotent for the same photo)."""
        if (cam, idx) in self._isp_flag:
            return self._isp_flag[(cam, idx)]
        if cam in self._isp_pending:
            self._isp_flag[(cam, idx)] = self._isp_pending.pop(cam)
            return self._isp_flag[(cam, idx)]
        return None

    def log(self, kind: str, /, **fields) -> dict:
        self._seq += 1
        rec = {"kind": kind, "at": iso_ms(self.now), "snapshot_id": self._current_snapshot, **fields}
        self.records.append(rec)
        self.counters[kind] += 1
        return rec

    # -- device simulation
    def capture(self, camera: str, **opts) -> dict:
        rows = self._rows.get(camera)
        if not rows:
            self.log("capture", camera=camera, ok=False, error="unknown_camera", opts=_jsonable(opts))
            return {"ok": False, "error": "unknown_camera", "simulated": True}
        idx = self._pos[camera] + 1 + self._extra[camera]
        if idx >= len(rows):
            self.log("capture", camera=camera, ok=False, error="no_next_photo", opts=_jsonable(opts))
            return {"ok": False, "error": "no_next_photo", "simulated": True}
        self._extra[camera] += 1
        row = rows[idx]
        ev = self._make_event(row, idx)
        ev["_replay"]["simulated_capture"] = True
        flag = self.isp_flag_for(camera, idx)
        if flag:
            ev["_replay"]["after_isp_change"] = flag
        shift = (parse_ts(row["captured_at"]) + PIPELINE_DELAY - self.now).total_seconds()
        self.log("capture", camera=camera, ok=True, shifted_snapshot_id=row["snapshot_id"],
                 shift_s=round(shift, 1), after_isp_change=bool(flag), opts=_jsonable(opts),
                 count_production=ev["params"]["count"], count_yolov7=ev["params"]["boxesTotal"])
        return {"ok": True, "simulated": True, "camera": camera, "snapshot_id": row["snapshot_id"], "event": ev,
                "shift_s": round(shift, 1), "after_isp_change": bool(flag),
                "note": "next real photo of this camera, time-shifted"}

    def set_image(self, camera: str, **params) -> dict:
        if camera not in self._rows:
            self.log("set_image", camera=camera, ok=False, error="unknown_camera", params=_jsonable(params))
            return {"ok": False, "error": "unknown_camera"}
        mark = {"params": _jsonable(params), "requested_at": iso_ms(self.now)}
        self._isp_pending[camera] = mark
        self.log("set_image", camera=camera, ok=True, params=mark["params"])
        return {"ok": True, "acked": True, "applied": False, "simulated": True, "requested_at": mark["requested_at"],
                "note": "acked; the next photo of this camera is marked after_isp_change (pixels unchanged in replay)"}

    def neighbours(self, camera: str, k: int = 1) -> list[dict]:
        me = next((c for c in self.cameras if c["serial"] == camera), None)
        if not me:
            return []
        out = []
        for c in self.cameras:
            if c["serial"] == camera:
                continue
            d = None
            if None not in (me.get("lat"), me.get("lon"), c.get("lat"), c.get("lon")):
                d = round(haversine_km((me["lat"], me["lon"]), (c["lat"], c["lon"])), 2)
            last = self.latest(c["serial"])
            out.append({"serial": c["serial"], "name": c.get("name"), "trapKind": c.get("trapKind"),
                        "distance_km": d, "latest": last["params"] if last else None})
        out.sort(key=lambda n: (n["distance_km"] is None, n["distance_km"] or 0.0, n["serial"]))
        return out[:k]

    def latest(self, camera: str) -> dict | None:
        h = self._history.get(camera)
        return h[-1] if h else None

    def history(self, camera: str, n: int = 10) -> list[dict]:
        return list(self._history.get(camera, [])[-n:])

    # -- operator simulation
    def propose(self, kind: str, payload: dict | None = None, camera: str | None = None, reason: str = "") -> dict:
        pid = f"prop-{len(self.proposals) + 1:04d}"
        p = {"proposal_id": pid, "kind": kind, "camera": camera, "payload": _jsonable(payload or {}),
             "reason": reason, "created_at": iso_ms(self.now), "status": "pending", "decided_at": None,
             "due_at": None}
        if self.policy.mode == "all":
            p["status"], p["decided_at"] = "approved", p["created_at"]
        elif self.policy.mode == "after_hours":
            p["due_at"] = iso_ms(self.now + timedelta(hours=self.policy.hours))
        self.proposals[pid] = p
        self.log("proposal", proposal_id=pid, proposal_kind=kind, camera=camera, payload=p["payload"], reason=reason,
                 status=p["status"], due_at=p["due_at"])
        if p["status"] == "approved":
            self.log("approval", proposal_id=pid, proposal_kind=kind, decided_at=p["decided_at"], policy=self.policy.label())
            self._notify_approved(p)
        return {"proposal_id": pid, "status": p["status"]}

    def proposal_status(self, pid: str) -> str | None:
        p = self.proposals.get(pid)
        return p["status"] if p else None

    def pending_proposals(self) -> list[dict]:
        return [dict(p) for p in self.proposals.values() if p["status"] == "pending"]

    def request_review(self, snapshot_id: str, reason: str = "") -> dict:
        self.log("review", review_of=snapshot_id, reason=reason)
        return {"ok": True, "queued": True}

    def fallback(self, reason: str, detail: str = "") -> None:
        self.log("fallback", reason=reason, detail=detail)

    def _notify_approved(self, p: dict) -> None:
        if self._on_approval:
            try:
                self._on_approval(dict(p), self)
            except Exception as e:  # noqa: BLE001 - the agent's callback must not stop the replay
                self.log("fallback", reason="on_approval_exception", detail=repr(e)[:200])

    def _resolve_due(self, until: datetime) -> None:
        """Approve after-hours proposals whose due time passed (called before each photo)."""
        for p in self.proposals.values():
            if p["status"] == "pending" and p["due_at"] and parse_ts(p["due_at"]) <= until:
                p["status"], p["decided_at"] = "approved", p["due_at"]
                self.log("approval", proposal_id=p["proposal_id"], proposal_kind=p["kind"], decided_at=p["decided_at"],
                         policy=self.policy.label())
                self._notify_approved(p)


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in (sorted(x, key=str) if isinstance(x, set) else x)]
    if is_dataclass(x) and not isinstance(x, type):
        return _jsonable(asdict(x))
    for attr in ("model_dump", "to_dict", "dict"):
        f = getattr(x, attr, None)
        if callable(f):
            try:
                return _jsonable(f())
            except Exception:  # noqa: BLE001 - fall through to str()
                break
    if isinstance(x, datetime):
        return iso_ms(x)
    return str(x)


# --------------------------------------------------------------------------- agent

def stub_per_photo(event: dict, ctx: ReplayContext) -> dict:
    """Built-in 'no agent': records that no agent ran. Used when sentinel.agent.loops is not importable."""
    return {"decision": "no_agent", "reason": "sentinel.agent.loops.per_photo not available"}


def load_agent(mode: str = "auto") -> tuple[Callable, str, Callable | None]:
    """(per_photo, label, on_approval). mode: auto (loops if importable) | stub."""
    if mode != "stub":
        try:
            mod = importlib.import_module("sentinel.agent.loops")
            fn = getattr(mod, "per_photo", None)
            if callable(fn):
                return fn, "sentinel.agent.loops.per_photo", getattr(mod, "on_approval", None)
        except Exception as e:  # noqa: BLE001 - not landed yet, or its deps are missing
            return stub_per_photo, f"stub (import failed: {type(e).__name__}: {str(e)[:80]})", None
        return stub_per_photo, "stub (sentinel.agent.loops has no per_photo)", None
    return stub_per_photo, "stub (--agent stub)", None


def decision_types(result: Any) -> list[str]:
    """Decision type names out of whatever per_photo returned (dict, list, object, None)."""
    r = _jsonable(result)
    if r is None:
        return ["none"]
    if isinstance(r, str):
        return [r]
    if isinstance(r, list):
        out = []
        for it in r:
            out += decision_types(it)
        return out or ["none"]
    if isinstance(r, dict):
        for key in ("decisions", "actions"):
            if isinstance(r.get(key), list) and r[key]:
                out = []
                for it in r[key]:
                    out += decision_types(it)
                return out
        for key in ("decision", "type", "action", "kind", "outcome"):
            v = r.get(key)
            if isinstance(v, str):
                return [v]
            if isinstance(v, dict):
                return decision_types(v)
    return ["unknown"]


# --------------------------------------------------------------------------- the loop

def load_cameras(path: Path, rows: list[dict], serials: set[str]) -> list[dict]:
    meta: dict[str, dict] = {}
    if path.exists():
        try:
            for c in json.loads(path.read_text(encoding="utf-8")):
                meta[c["serial"]] = c
        except (OSError, ValueError, KeyError):
            pass
    out = []
    for s in sorted(serials):
        m = meta.get(s, {})
        r0 = next((r for r in rows if r["device_serial"] == s), {})
        out.append({"serial": s, "name": m.get("name"), "provider": m.get("provider") or r0.get("provider"),
                    "trapKind": m.get("trapKind") or r0.get("trap_kind"),
                    "lat": m.get("lat", _num(r0.get("lat"))), "lon": m.get("lon", _num(r0.get("lon"))),
                    "farmId": m.get("farmId"), "plotId": m.get("plotId"),
                    "alertThreshold": m.get("alertThreshold")})
    return out


def replay(rows: list[dict], *, data_dir: Path, detector, agent: Callable, agent_label: str,
           on_approval: Callable | None = None, cameras: list[dict] | None = None,
           info: dict[str, dict] | None = None, policy: Policy | None = None, seed: int = 0,
           out_dir: Path, run_id: str, report: dict | None = None, progress: int = 0) -> dict:
    """Replay `rows` (already filtered and time-ordered). Writes trace.jsonl + summary.json; returns the summary."""
    policy = policy or Policy()
    info = info or {}
    cameras = cameras or load_cameras(data_dir / "export" / "cameras.json", rows, {r["device_serial"] for r in rows})
    cam_by = {c["serial"]: c for c in cameras}
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    agent_s = 0.0
    det_cache: dict[str, dict] = {}
    n = len(rows)

    def detect(row: dict) -> dict:
        sid = row["snapshot_id"]
        if sid not in det_cache:
            det_cache[sid] = detector.detect(row, data_dir / row["image_path"])
        return det_cache[sid]

    index_of = {r["snapshot_id"]: i for i, r in enumerate(rows)}
    holder: dict[str, ReplayContext] = {}

    def make_event(row: dict, cam_idx: int) -> dict:
        ev = build_event(row, detect(row), info.get(row["snapshot_id"], {}), cam_by.get(row["device_serial"], {}),
                         data_dir, index=index_of[row["snapshot_id"]], idx_total=n)
        flag = holder["ctx"]._isp_flag.get((row["device_serial"], cam_idx))
        if flag:
            ev["_replay"]["after_isp_change"] = flag
        return ev

    ctx = ReplayContext(rows, cameras, policy, seed, make_event, on_approval)
    holder["ctx"] = ctx
    # camera-local index of each row (position inside that camera's own list)
    cam_pos: dict[str, int] = defaultdict(int)
    trace_path = out_dir / "trace.jsonl"
    decisions: Counter = Counter()
    per_cam: Counter = Counter()
    det_err: list[tuple[int, int]] = []
    errors = 0
    seq = 0
    sha = hashlib.sha256()

    with open(trace_path, "w", encoding="utf-8", newline="\n") as tf:
        def emit(rec: dict) -> None:
            nonlocal seq
            seq += 1
            line = json.dumps({"seq": seq, **rec}, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n"
            # Keep the trace independent of the checkout location: file:// URIs become repo-relative,
            # so the same inputs give the same trace hash on every machine (QA round 1, m1).
            line = line.replace(ROOT.as_uri() + "/", "")
            tf.write(line)
            sha.update(line.encode("utf-8"))

        for i, row in enumerate(rows):
            cam = row["device_serial"]
            cam_idx = cam_pos[cam]
            cam_pos[cam] += 1
            ctx._current_snapshot = None
            ctx._resolve_due(parse_ts(row["captured_at"]) + PIPELINE_DELAY)
            pre = ctx.records[:]
            ctx.records.clear()
            for rec in pre:
                emit(rec)

            ev = make_event(row, cam_idx)
            # the schedule delivers this photo: it is also the flagged one if set_image was pending
            flag = ctx.isp_flag_for(cam, cam_idx)
            if flag:
                ev["_replay"]["after_isp_change"] = flag
            ctx._begin_photo(row, ev)
            ctx._deliver(ev)
            det_err.append((ev["params"]["boxesTotal"], ev["params"]["count"] or 0))

            t0 = time.perf_counter()
            result, err = None, None
            try:
                result = agent(ev, ctx)
                if inspect.isawaitable(result):
                    result = asyncio.run(_await(result))
            except Exception as e:  # noqa: BLE001 - an agent crash is a fallback, not the end of the replay
                err = f"{type(e).__name__}: {str(e)[:200]}"
                errors += 1
                ctx.fallback("agent_exception", err)
            agent_s += time.perf_counter() - t0

            types = decision_types(result) if err is None else ["agent_error"]
            for t in types:
                decisions[t] += 1
            per_cam[cam] += 1
            actions = ctx.records[:]
            ctx.records.clear()
            p = ev["params"]
            emit({"kind": "photo", "snapshot_id": row["snapshot_id"], "camera": cam, "captured_at": p["capturedAt"],
                  "sim_time": iso_ms(ctx.now), "count_production": p["count"], "count_yolov7": p["boxesTotal"],
                  "quality_issues": p["quality"]["issues"], "after_isp_change": bool(ev["_replay"]["after_isp_change"]),
                  "decision": types, "agent": _jsonable(result), "error": err, "actions": len(actions)})
            for rec in actions:
                emit(rec)
            if progress and (i + 1) % progress == 0:
                print(f"  replay {i + 1}/{n} photos, {time.perf_counter() - t_start:.0f} s", file=sys.stderr, flush=True)

        # end of window: after-hours approvals that are due before the last photo's clock were resolved already;
        # the rest stay pending and are reported as such
        end_pending = ctx.pending_proposals()

    runtime = time.perf_counter() - t_start
    kinds = Counter(p["kind"] for p in ctx.proposals.values())
    status = Counter(p["status"] for p in ctx.proposals.values())
    extra_caps = ctx.counters["capture"]
    abs_err = [abs(a - b) for a, b in det_err]
    summary = {
        "run_id": run_id,
        "agent": agent_label,
        "policy": policy.label(),
        "seed": seed,
        "window": (report or {}).get("window"),
        "filters": {k: v for k, v in (report or {}).items() if k != "window"},
        "photos_processed": n,
        "photos_per_camera": dict(sorted(per_cam.items())),
        "decisions_by_type": dict(sorted(decisions.items())),
        "extra_captures": extra_caps,
        "set_image_calls": ctx.counters["set_image"],
        "proposals": {"total": len(ctx.proposals), "by_kind": dict(sorted(kinds.items())),
                      "by_status": dict(sorted(status.items())), "pending_at_end": len(end_pending)},
        "approvals": ctx.counters["approval"],
        "reviews": ctx.counters["review"],
        "fallbacks": ctx.counters["fallback"],
        "agent_errors": errors,
        "trace_lines": seq,
        "trace_sha256": sha.hexdigest(),
        "detector_vs_production": {
            "n": n,
            "mae": round(sum(abs_err) / n, 3) if n else None,
            "bias_yolov7_minus_production": round(sum(a - b for a, b in det_err) / n, 3) if n else None,
            "note": "production count is Roboflow at capture time, not human ground truth",
        },
        "detection": {"cache_hits": getattr(detector, "hits", None), "cache_misses": getattr(detector, "misses", None),
                      "detect_s": round(getattr(detector, "detect_s", 0.0), 2)},
        "runtime": {"total_s": round(runtime, 2), "agent_s": round(agent_s, 2),
                    "per_photo_ms": round(runtime * 1000 / n, 2) if n else None},
        "assumptions": ASSUMPTIONS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


async def _await(x):
    return await x


def markdown_table(s: dict) -> str:
    lines = ["| metric | value |", "|---|---|"]
    w = s.get("window") or {}
    lines += [
        f"| run_id | {s['run_id']} |",
        f"| agent | {s['agent']} |",
        f"| window (farm time) | {w.get('from')} .. {w.get('to')} |",
        f"| photos replayed | {s['photos_processed']} {s['photos_per_camera']} |",
        ("| dropped here: 640x640 test pictures / excluded cameras / no image / outside range | "
         f"{s['filters'].get('dropped_test_pictures_640x640')} / {s['filters'].get('dropped_excluded_cameras')} / "
         f"{s['filters'].get('dropped_no_image_file', 0)} / {s['filters'].get('dropped_outside_date_range')} |"),
        f"| test pictures already dropped by the export | {s['filters'].get('export_dropped_test_pictures')} |",
        f"| extra captures | {s['extra_captures']} |",
        f"| set_image calls | {s['set_image_calls']} |",
        f"| proposals ({s['policy']}) | {s['proposals']['total']} {s['proposals']['by_status']} |",
        f"| reviews | {s['reviews']} |",
        f"| fallbacks / agent errors | {s['fallbacks']} / {s['agent_errors']} |",
        f"| YOLOv7 vs production count MAE | {s['detector_vs_production']['mae']} |",
        (f"| runtime | {s['runtime']['total_s']} s (agent {s['runtime']['agent_s']} s; detector cache "
         f"{s['detection']['cache_hits']} hit / {s['detection']['cache_misses']} miss) |"),
        f"| trace | {s['trace_lines']} lines, sha256 {s['trace_sha256'][:16]} |",
        "",
        "| decision | photos |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in s["decisions_by_type"].items()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--excluded", type=Path, default=ROOT / "data" / "excluded_cameras.csv")
    ap.add_argument("--onnx", type=Path, default=ROOT / "models" / "yolov7-tiny" / "fruitfly_deploy.onnx")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "yolov7")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "replay")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--from", dest="date_from", type=date.fromisoformat, default=None)
    ap.add_argument("--to", dest="date_to", type=date.fromisoformat, default=None)
    ap.add_argument("--days", type=int, default=None, help="last N farm days of the manifest")
    ap.add_argument("--cameras", nargs="+", default=None)
    pol = ap.add_mutually_exclusive_group()
    pol.add_argument("--approve-all", action="store_true")
    pol.add_argument("--approve-none", action="store_true", help="default")
    pol.add_argument("--approve-after-hours", type=float, default=None, metavar="H")
    ap.add_argument("--box-source", choices=["yolov7", "production", "cached"], default="yolov7",
                    help="boxes in the events: YOLOv7-tiny (default), FlyBot's exported boxes (sparse), or "
                         "YOLOv7-tiny results from --cache-dir only (no model needed: data/samples)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", choices=["auto", "stub"], default="auto")
    ap.add_argument("--limit", type=int, default=0, help="only the first N photos (debug)")
    ap.add_argument("--engine", default="new", choices=["new", "classic", "auto"])
    ap.add_argument("--resize", default="letterbox", choices=["letterbox", "stretch"])
    ap.add_argument("--progress", type=int, default=200)
    a = ap.parse_args(argv)

    all_rows = load_manifest(a.manifest)
    data_dir = a.manifest.parent
    t0 = time.perf_counter()
    info = image_info(all_rows, data_dir, data_dir / "cache" / "imginfo.json")
    test_ids = find_test_pictures(all_rows, info)
    missing = {sid for sid, i in info.items() if i["bytes"] is None}
    rows, report = select_rows(all_rows, excluded=load_excluded(a.excluded), test_ids=test_ids, date_from=a.date_from,
                               date_to=a.date_to, days=a.days, cameras=a.cameras, missing_image_ids=missing)
    export_summary = data_dir / "export" / "summary.json"
    if export_summary.exists():  # the export itself may already have dropped the test pictures
        try:
            report["export_dropped_test_pictures"] = (
                json.loads(export_summary.read_text(encoding="utf-8")).get("testPictures", {}).get("dropped"))
        except (OSError, ValueError):
            pass
    report["test_pictures_total_in_manifest"] = len(test_ids)
    report["test_pictures_distinct_hashes"] = len({info[t]["sha1"] for t in test_ids})
    report["image_info_s"] = round(time.perf_counter() - t0, 2)
    if a.limit:
        rows = rows[: a.limit]
        report["limit"] = a.limit
        report["photos"] = len(rows)
    policy = Policy.from_args(a.approve_all, a.approve_none, a.approve_after_hours)
    agent, label, on_approval = load_agent(a.agent)
    w = report["window"]
    run_id = a.run_id or f"{w['from']}_{w['to']}_{policy.label()}_seed{a.seed}" + (f"_n{a.limit}" if a.limit else "")
    print(f"replay {run_id}: {len(rows)} photos, agent = {label}", file=sys.stderr, flush=True)
    detector = (ProductionBoxSource(data_dir / "export" / "boxes") if a.box_source == "production"
                else CachedYolov7Source(a.cache_dir) if a.box_source == "cached"
                else Yolov7Source(a.onnx, a.cache_dir, a.engine, a.resize))
    cams = load_cameras(data_dir / "export" / "cameras.json", rows, {r["device_serial"] for r in rows})
    summary = replay(rows, data_dir=data_dir, detector=detector, agent=agent, agent_label=label,
                     on_approval=on_approval, cameras=cams, info=info, policy=policy, seed=a.seed,
                     out_dir=a.out_dir / run_id, run_id=run_id, report=report, progress=a.progress)
    print(markdown_table(summary))
    print(f"\nwrote {a.out_dir / run_id / 'trace.jsonl'} and summary.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
