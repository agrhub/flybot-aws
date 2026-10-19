"""FlyBot -> vision Lambda handler (docs/CONTRACTS.md section 1): one photo in, counts + quality + diff +
proposals + trace out.

    handler(event, context) -> response dict            AWS Lambda entry point (CMD in lambda/vision/Dockerfile)
    analyze(request: dict) -> response dict             the same without Lambda (sentinel.lambda_local, eval/replay_sample.py)

Flow per request: validate -> fetch + decode the photo(s) (photo_not_found / bad_request before anything runs)
-> OpenCV 5 pipeline (op 5 detector, op 2 quality, ops 1/3/4 diff, op 6 crowd, op 8 motion) inside the agent's
`analyze_trap_photo` tool -> planner (RulePlanner; LLMPlanner for `deep` mode or SENTINEL_PLANNER=llm) drives
the guarded tools -> every accepted action is a proposal for FlyBot -> response. The Lambda stores no picture:
FlyBot keeps the photo and draws the boxes; when a person is suspected the response lists the face regions to
blur (`privacy.regions`). An annotated JPEG to S3 `evidence/<cameraId>/<capturedAt>.annotated.jpg` is written only
when the request asks for it (`"evidence": true`, e.g. the benchmark), or to EVIDENCE_DIR on local runs.

Errors raise `VisionError` subclasses: Lambda reports a FunctionError whose `errorMessage` is the contract's
`{"v": 1, "error": {"code", "message"}}` JSON (errorType = the code in CamelCase).

Env: PHOTOS_BUCKET (requested evidence uploads when the photo did not come from S3; the photo's own bucket wins), EVIDENCE_DIR (local runs:
write the annotated JPEG there), DETECTOR / DETECTOR_ONNX / YOLO26_URL (sentinel.vision.detector), SENTINEL_PLANNER,
ANTHROPIC_SECRET_ARN / ANTHROPIC_API_KEY (sentinel.agent.wiring), SENTINEL_KILL_SWITCH, STAGE.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from sentinel.common.ids import iso, parse_dt, ulid

CONTRACT_VERSION = 1
MODES = ("detect", "deep")
TRAP_KINDS = ("sticky", "lure")
PLATFORMS = ("ezviz", "camos", "kinesis")
QUALITY_FLAGS = {"blurry": "blurry", "dark": "dark", "overexposed": "overexposed", "glare": "overexposed",
                 "no_card": "trap_missing"}
GRID = (4, 4)
SAFE = re.compile(r"[^A-Za-z0-9_.:-]")
MAX_BOXES = 300


class VisionError(Exception):
    code = "internal"
    status = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def payload(self) -> dict:
        return {"v": CONTRACT_VERSION, "error": {"code": self.code, "message": self.message[:500]}}

    def __str__(self) -> str:  # Lambda puts str(e) in errorMessage: FlyBot parses the contract JSON from it
        return json.dumps(self.payload(), ensure_ascii=False)


class BadRequest(VisionError):
    code, status = "bad_request", 400


class PhotoNotFound(VisionError):
    code, status = "photo_not_found", 404


class ModelUnavailable(VisionError):
    code, status = "model_unavailable", 503


class Internal(VisionError):
    code, status = "internal", 500


def _log(msg: str, **kw) -> None:
    print(json.dumps({"msg": msg, **kw}, default=str, separators=(",", ":"), ensure_ascii=False))


# ---------------------------------------------------------------- request


def photo_ref(photo, field: str) -> str:
    if not isinstance(photo, dict):
        raise BadRequest(f"{field} must be an object with s3Bucket+s3Key or url")
    if photo.get("s3Bucket") and photo.get("s3Key"):
        return f"s3://{photo['s3Bucket']}/{photo['s3Key']}"
    url = photo.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://", "file://")):
        return url
    if isinstance(url, str) and url and Path(url).is_file():  # local dev: a path
        return url
    raise BadRequest(f"{field} needs s3Bucket+s3Key or an http(s)/file url")


def clean_history(history: list) -> list[dict]:
    """FlyBot's `history` rows ({capturedAt, count, photoId?}); anything malformed is dropped, never a bad_request."""
    out = []
    for row in history:
        if not isinstance(row, dict):
            continue
        try:
            at = parse_dt(row.get("capturedAt"))
        except (ValueError, TypeError):
            continue
        count = row.get("count")
        if at is None or isinstance(count, bool) or not isinstance(count, int | float):
            continue
        out.append(row)
    return out[:400]


def parse_request(event) -> dict:
    if isinstance(event, (bytes, str)):
        try:
            event = json.loads(event)
        except ValueError as e:
            raise BadRequest(f"body is not JSON: {e}") from e
    if not isinstance(event, dict):
        raise BadRequest("request must be a JSON object")
    v = event.get("v", CONTRACT_VERSION)
    if v != CONTRACT_VERSION:
        raise BadRequest(f"unsupported contract version {v!r} (expected {CONTRACT_VERSION})")
    mode = event.get("mode") or "detect"
    if mode not in MODES:
        raise BadRequest(f"mode must be one of {MODES}")
    if "photo" not in event:
        raise BadRequest("photo is required")
    cam = event.get("camera") or {}
    if not isinstance(cam, dict) or not cam.get("id"):
        raise BadRequest("camera.id is required")
    kind = cam.get("trapKind") or "lure"
    if kind not in TRAP_KINDS:
        raise BadRequest(f"camera.trapKind must be one of {TRAP_KINDS}")
    platform = cam.get("platform") or "ezviz"
    if platform not in PLATFORMS:
        raise BadRequest(f"camera.platform must be one of {PLATFORMS}")
    captured = event.get("capturedAt")
    try:
        captured_at = parse_dt(captured) if captured else None
    except ValueError as e:
        raise BadRequest(f"capturedAt: {e}") from e
    history = event.get("history") or []
    if not isinstance(history, list):
        raise BadRequest("history must be a list")
    locale = event.get("locale") or "vi"
    if locale not in ("vi", "en"):
        raise BadRequest("locale must be vi or en")
    question = event.get("question")
    if mode == "deep" and not question:
        raise BadRequest("deep mode needs a question")
    return {"mode": mode, "photo": photo_ref(event["photo"], "photo"),
            "prev": photo_ref(event["prevPhoto"], "prevPhoto") if event.get("prevPhoto") else None,
            "camera": {**cam, "id": str(cam["id"]), "trapKind": kind, "platform": platform},
            "capturedAt": captured_at, "history": clean_history(history),
            "question": question, "locale": locale,
            "photoBucket": (event["photo"].get("s3Bucket") if isinstance(event.get("photo"), dict) else None),
            "photoId": event.get("photoId") or (event["photo"].get("s3Key") if isinstance(event.get("photo"), dict) else None),
            "prevRequested": bool(event.get("prevPhoto")),
            "evidence": event.get("evidence") is True}


# ---------------------------------------------------------------- images


def load_images(req: dict) -> dict:
    """Fetch and decode the photo and the previous photo into sentinel.agent.vision.IMAGE_CACHE."""
    from sentinel.agent import vision

    out = {}
    for key in ("photo", "prev"):
        ref = req.get(key)
        if not ref:
            continue
        t0 = time.perf_counter()
        try:
            img = vision.load_image(ref)
        except FileNotFoundError as e:
            if key == "prev":  # a missing previous photo is not fatal: analyze without the diff
                _log("prevPhoto not found", ref=ref[:120], error=str(e)[:200])
                req["prev"] = None
                continue
            raise PhotoNotFound(f"photo not found: {e}") from e
        except ValueError as e:
            if key == "prev":
                _log("prevPhoto unreadable", ref=ref[:120], error=str(e)[:200])
                req["prev"] = None
                continue
            raise BadRequest(str(e)) from e
        except Exception as e:
            raise Internal(f"cannot load {key}: {type(e).__name__}: {e}"[:300]) from e
        vision.IMAGE_CACHE[ref] = img
        out[key] = {"ms": round((time.perf_counter() - t0) * 1000, 1), "width": int(img.shape[1]), "height": int(img.shape[0])}
    return out


# ---------------------------------------------------------------- response pieces


def contract_boxes(detections: list[dict] | None, boxes_xyxy: list, width: int, height: int) -> list[dict]:
    """Detector boxes -> normalised {x, y, w, h, score, cls} (top-left corner, fractions of the photo)."""
    out = []
    if detections:
        for d in detections[:MAX_BOXES]:
            x, y, w, h = d["xywh"]
            out.append({"x": round(x / width, 4), "y": round(y / height, 4), "w": round(w / width, 4), "h": round(h / height, 4),
                        "score": round(float(d.get("score", 0.0)), 4), "cls": "fruit_fly"})
        return out
    for b in (boxes_xyxy or [])[:MAX_BOXES]:
        x0, y0, x1, y1 = b[:4]
        out.append({"x": round(x0 / width, 4), "y": round(y0 / height, 4), "w": round((x1 - x0) / width, 4),
                    "h": round((y1 - y0) / height, 4), "score": None, "cls": "fruit_fly"})
    return out


def density_grid(boxes: list[dict], rows: int = GRID[0], cols: int = GRID[1]) -> dict:
    cells = [[0] * cols for _ in range(rows)]
    for b in boxes:
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        r = min(rows - 1, max(0, int(cy * rows)))
        c = min(cols - 1, max(0, int(cx * cols)))
        cells[r][c] += 1
    return {"gridRows": rows, "gridCols": cols, "cells": cells}


def quality_block(obs) -> dict:
    issues = list(obs.quality.issues)
    flags = list(dict.fromkeys(QUALITY_FLAGS[i] for i in issues if i in QUALITY_FLAGS))
    mo = obs.motion or {}
    if mo.get("occluded_by_large_object"):
        flags.append("occluded")
    if mo.get("camera_moved"):
        flags.append("camera_moved")
    sc = obs.quality.scores or {}
    # blur: 0 = sharp, 1 = blurry, from the op 2 Laplacian variance against the "blurry" threshold (quality_thresholds.yaml)
    blur = exposure = None
    try:
        from sentinel.vision.quality import load_thresholds

        t = load_thresholds()
        sharp = sc.get("sharpness")
        if sharp is not None:
            floor = float(t["blurry"]["sharpness_min"])
            blur = round(max(0.0, min(1.0, 1.0 - float(sharp) / (2.0 * floor))), 3) if floor > 0 else None
        if sc.get("brightness") is not None:
            exposure = round(float(sc["brightness"]) / 255.0, 3)
    except Exception as e:  # thresholds file missing: scores still returned raw  # noqa: BLE001
        _log("quality thresholds unavailable", error=type(e).__name__)
    return {"ok": not flags and obs.quality.ok, "flags": flags, "issues": issues, "blur": blur, "exposure": exposure,
            "score": obs.quality.score, "scores": {k: v for k, v in sc.items() if v is not None}}


def diff_block(obs) -> dict | None:
    nc = obs.new_catch
    if nc is None:
        return None
    out = {"newCount": nc.new_count, "goneCount": nc.disappeared_count, "persistedCount": nc.persisted_count,
           "cardMoved": bool((obs.motion or {}).get("camera_moved")) or bool(nc.card_change_suspected),
           "cardChangeSuspected": nc.card_change_suspected, "meaning": nc.meaning, "registrationOk": nc.registration_ok,
           "inliers": nc.inliers, "changeRatio": nc.change_ratio}
    if nc.skipped:
        out["skipped"] = nc.skipped
    return out


def trace_block(trace, obs, timings: dict, planner_name: str) -> dict:
    steps = [{"tool": "load", "ms": timings.get("loadMs")}]
    for c in trace.tool_calls:
        step = {"tool": c.name, "ms": c.ms, "status": c.status, "by": c.by, "index": c.index}
        if c.status in ("blocked", "error"):
            step["reason"] = c.blocked_reason or (c.result or {}).get("error") or (c.result or {}).get("reason")
        if c.name == "analyze_trap_photo" and timings.get("dnnMs") is not None:
            step["dnnMs"] = timings["dnnMs"]
        steps.append(step)
    usage = trace.planner.usage or {}
    tokens = {"in": int(usage.get("input_tokens") or 0), "cachedIn": int(usage.get("cache_read_input_tokens") or 0),
              "out": int(usage.get("output_tokens") or 0), "llmCalls": int(usage.get("llm_calls") or 0)}
    return {"planner": planner_name, "steps": steps, "tokens": tokens, "ms": timings.get("totalMs"),
            "model": trace.planner.model, "fallback": trace.planner.fallback, "fallbackReason": trace.planner.fallback_reason,
            "llmTurns": trace.planner.llm_turns, "state": trace.state, "outcome": trace.outcome,
            "reasoning": trace.reasoning_summary[:2000],
            "decisions": [d.model_dump(mode="json") for d in trace.decisions],
            "events": [e.model_dump(mode="json") for e in trace.events][:50],
            "toolCalls": [_compact_call(c) for c in trace.tool_calls],
            "killSwitch": trace.kill_switch, "policyVersion": trace.policy_version, "eventId": trace.event_id}


def _compact_call(c) -> dict:
    d = c.model_dump(mode="json")
    r = d.get("result") or {}
    if "observation" in r:
        r = {k: v for k, v in r.items() if k != "observation"}
    d["result"] = _truncate(json.loads(json.dumps(r, default=str)))
    return d


def _truncate(value, depth: int = 0):
    if isinstance(value, dict):
        return {k: _truncate(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [_truncate(v, depth + 1) for v in value[:20]]
    if isinstance(value, str):
        return value[:300]
    return value


def _rejected(res: dict | None) -> dict:
    """Boxes the detector's NMS and box rules removed (non-zero reasons only), for the `analyzed` log line."""
    post = ((res or {}).get("detect") or {}).get("postprocess") or {}
    out = {k: v for k, v in (post.get("rejected") or {}).items() if v}
    if post.get("nms_removed"):
        out["nms"] = post["nms_removed"]
    return out


# ---------------------------------------------------------------- evidence


def annotate(img, boxes: list[dict], obs, model: dict, locale: str):
    import cv2

    from sentinel.vision.evidence import fit
    from sentinel.vision.motion import blur_faces

    out, _ = fit(img.copy(), 1280)
    h, w = out.shape[:2]
    if (obs.privacy or {}).get("person_suspected"):
        out, _ = blur_faces(out)
    for b in boxes:
        x0, y0 = int(b["x"] * w), int(b["y"] * h)
        x1, y1 = int((b["x"] + b["w"]) * w), int((b["y"] + b["h"]) * h)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
    label = (f"{obs.camera}  {len(boxes)} {'ruoi' if locale == 'vi' else 'flies'}  {model.get('name')} / {model.get('engine')}"
             + (f"  issues: {','.join(obs.quality.issues)}" if obs.quality.issues else ""))
    cv2.rectangle(out, (0, 0), (min(w, 12 + 9 * len(label)), 28), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def privacy_block(obs, img) -> dict:
    """obs.privacy plus, when a person is suspected, the face regions FlyBot blurs on the photo it shows:
    `regions` [{x, y, w, h}] normalised (YuNet), or `blurAll` when the face model is unavailable."""
    out = dict(obs.privacy or {"person_suspected": False})
    out.pop("faces", None)
    if not out.get("person_suspected") or img is None:
        return out
    from sentinel.vision.motion import detect_faces

    h, w = img.shape[:2]
    faces = detect_faces(img)
    if faces is None:
        return {**out, "blurAll": True, "method": "unavailable"}
    regions = []
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        regions.append({"x": round(max(0.0, x1 / w), 4), "y": round(max(0.0, y1 / h), 4),
                        "w": round(min(1.0, (x2 - x1) / w), 4), "h": round(min(1.0, (y2 - y1) / h), 4)})
    return {**out, "regions": regions, "method": "yunet"}


def store_evidence(img, req: dict, boxes: list[dict], obs, model: dict) -> dict:
    """Annotated JPEG to s3://<bucket>/evidence/<cameraId>/<capturedAt>.annotated.jpg (or EVIDENCE_DIR locally)."""
    import cv2

    t0 = time.perf_counter()
    try:
        annotated = annotate(img, boxes, obs, model, req["locale"])
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return {"annotatedS3Key": None, "crops": [], "error": "jpeg encode failed"}
        stamp = iso(req["capturedAt"] or datetime.now(UTC)).replace(":", "-")
        key = f"evidence/{SAFE.sub('_', req['camera']['id'])}/{stamp}.annotated.jpg"
        # The bucket the photo came from (FlyBot signs evidence keys against its own bucket), else the stack's
        bucket = (req.get("photoBucket") or os.environ.get("PHOTOS_BUCKET")) if req.get("evidence") else None
        local_dir = os.environ.get("EVIDENCE_DIR")
        out: dict = {"annotatedS3Key": None, "crops": []}
        if bucket and not os.environ.get("EVIDENCE_LOCAL_ONLY"):
            from sentinel.common.aws import client

            client("s3").put_object(Bucket=bucket, Key=key, Body=buf.tobytes(), ContentType="image/jpeg",
                                    Metadata={"camera": req["camera"]["id"], "model": str(model.get("name"))})
            out.update(annotatedS3Key=key, bucket=bucket)
        if local_dir:
            p = Path(local_dir) / key
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(buf.tobytes())
            out["annotatedPath"] = p.as_posix()
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    except Exception as e:  # noqa: BLE001 - evidence is best effort; the analysis still returns
        _log("evidence failed", error=f"{type(e).__name__}: {e}"[:300])
        return {"annotatedS3Key": None, "crops": [], "error": f"{type(e).__name__}: {e}"[:200]}


# ---------------------------------------------------------------- the run


def analyze(request, *, llm=None, analyzer=None, clock=None, store: bool = True) -> dict:
    """The contract: request dict -> response dict. Raises VisionError subclasses."""
    from sentinel import require_opencv5
    from sentinel.agent import loops, vision
    from sentinel.agent.proposals import proposals_from_trace, summary_text
    from sentinel.agent.wiring import build_runtime
    from sentinel.common.schemas import PhotoEvent
    from sentinel.vision.detector import DetectorUnavailable, get_detector, model_info

    t_all = time.perf_counter()
    req = parse_request(request)
    cv = require_opencv5()
    try:
        det = get_detector()
    except DetectorUnavailable as e:
        raise ModelUnavailable(str(e)) from e
    model = model_info(det)
    loaded = load_images(req)
    try:
        planner = "llm" if req["mode"] == "deep" else os.environ.get("SENTINEL_PLANNER", "rule")
        rt = build_runtime(camera=req["camera"], history=req["history"], planner=planner, llm=llm, analyzer=analyzer,
                           clock=clock or (lambda: req["capturedAt"] or datetime.now(UTC)))
        trace_id = f"tr_{(req['capturedAt'] or datetime.now(UTC)).strftime('%Y%m%d_%H%M%S')}_{ulid()[-4:].lower()}"
        ev = PhotoEvent(event_id=trace_id, camera=req["camera"]["id"], source=req["camera"]["platform"],
                        trap_kind=req["camera"]["trapKind"], captured_at=req["capturedAt"], image=req["photo"],
                        snapshot_id=req.get("photoId"), battery_percent=req["camera"].get("battery"),
                        alert_threshold=req["camera"].get("alertThreshold"),
                        untrusted_text={k: str(v) for k, v in {"camera_name": req["camera"].get("name")}.items() if v},
                        raw={"question": req["question"], "locale": req["locale"], "mode": req["mode"]})
        if req["prev"]:
            rt.state.update(ev.camera, lambda s: s.__setitem__("last_image", req["prev"]))
        t0 = time.perf_counter()
        trace = loops.per_photo(ev, rt)
        agent_ms = round((time.perf_counter() - t0) * 1000, 1)
        obs = next((o for o in rt.analyzer_observations(trace) if o.camera == ev.camera), None) \
            if hasattr(rt, "analyzer_observations") else None
        if obs is None:
            from sentinel.common.schemas import Observation

            raw_obs = next(iter(trace.opencv_outputs.values()), None)
            obs = Observation.model_validate(raw_obs) if raw_obs else None
        if obs is None:
            failed = next((c for c in trace.tool_calls if c.name == "analyze_trap_photo"), None)
            why = (failed.result or {}).get("error") if failed else "analysis did not run"
            raise Internal(f"analysis failed: {why}")
        res = getattr(rt.analyzer, "last_result", None) or {}
        width, height = obs.width or loaded["photo"]["width"], obs.height or loaded["photo"]["height"]
        detections = (res.get("detect") or {}).get("detections") if res else None
        boxes = contract_boxes(detections, obs.boxes, width, height)
        if obs.counts.opencv is None and not res.get("detect"):
            raise ModelUnavailable("detector produced no result")
        proposals = proposals_from_trace(trace, rt.policy, req["locale"])
        img = vision.IMAGE_CACHE.get(req["photo"])
        wanted = req["evidence"] or bool(os.environ.get("EVIDENCE_DIR"))
        evidence = store_evidence(img, req, boxes, obs, model) if (store and wanted and img is not None) else {"annotatedS3Key": None, "crops": []}
        privacy = privacy_block(obs, img)
        dnn_ms = (res.get("detect") or {}).get("forward_ms") if res else None
        timings = {"loadMs": loaded.get("photo", {}).get("ms"), "dnnMs": dnn_ms, "visionMs": obs.vision_ms,
                   "agentMs": agent_ms, "evidenceMs": evidence.get("ms"),
                   "totalMs": round((time.perf_counter() - t_all) * 1000, 1)}
        answer = None
        if req["mode"] == "deep":
            answer = trace.reasoning_summary if (trace.planner.name == "llm" and not trace.planner.fallback) else None
            answer = answer or summary_text(obs, trace, proposals, req["locale"])
        out = {
            "v": CONTRACT_VERSION, "traceId": trace_id, "model": model,
            "counts": {"total": int(obs.counts.opencv or 0), "byClass": {"fruit_fly": int(obs.counts.opencv or 0)}},
            "boxes": boxes, "quality": quality_block(obs), "diff": diff_block(obs),
            "density": density_grid(boxes), "proposals": proposals,
            "trace": trace_block(trace, obs, timings, trace.planner.name),
            "evidence": evidence, "timings": timings, "opencv": cv, "mode": req["mode"],
            "crowd": {"saturation": obs.saturation} if obs.saturation is not None else None,
            "privacy": privacy, "photo": {"width": width, "height": height},
        }
        if answer is not None:
            out["answer"] = answer
        if req["prev"] is None and req["prevRequested"]:
            out["diff"] = None
            out["warnings"] = ["prevPhoto not found or unreadable: diff skipped"]
        _log("analyzed", traceId=trace_id, camera=ev.camera, mode=req["mode"], count=out["counts"]["total"],
             flags=out["quality"]["flags"], proposals=[p["action"] for p in proposals], planner=trace.planner.name,
             fallback=trace.planner.fallback, totalMs=timings["totalMs"], dnnMs=dnn_ms, model=model.get("name"),
             rejected=_rejected(res))
        return json.loads(json.dumps(out, default=str))
    finally:
        vision.IMAGE_CACHE.clear()


def handler(event, context=None):
    """AWS Lambda entry point. Raises VisionError (-> FunctionError with the contract JSON in errorMessage)."""
    if isinstance(event, dict) and "body" in event and isinstance(event.get("body"), str) and "photo" not in event:
        event = event["body"]  # function URL / API Gateway shape, if ever used
    try:
        return analyze(event)
    except VisionError as e:
        _log("error", code=e.code, message=e.message[:300])
        raise
    except Exception as e:
        _log("internal error", error=f"{type(e).__name__}: {e}"[:300], trace=traceback.format_exc(limit=4)[-1500:])
        raise Internal(f"{type(e).__name__}: {e}"[:300]) from e
