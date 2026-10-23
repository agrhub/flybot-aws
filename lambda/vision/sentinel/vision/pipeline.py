"""Per-photo vision pipeline: OpenCV 5 ops 1, 2, 3, 4, 6, 8 on one trap photo (+ the camera's previous photo).

STABLE ENTRY POINT (A6 replay harness, A2 agent). Keep this signature and the top-level keys stable;
add keys, never rename or change the meaning of existing ones.

    analyze(image_path, prev_image_path=None, trap_kind="sticky", boxes_cur=None, boxes_prev=None, *,
            out_dir=None, method="sift", hour=None, baseline=None, name=None) -> dict

    image_path, prev_image_path  local file path (str / Path) or an already decoded BGR np.ndarray
    trap_kind                    "sticky" | "lure" (FlyBot Camera.trapKind)
    boxes_cur, boxes_prev        detector boxes [[x1, y1, x2, y2], ...] in each photo's own pixels
                                 (e.g. detect_yolov7); None = no boxes -> op 4 matching is skipped
    out_dir                      evidence directory (small JPEGs, privacy-safe); None = no evidence files
    method                       op 3 registration: "sift" (default, ~0.2 s/pair on a laptop CPU) or
                                 "aliked" (OpenCV 5 ALIKED + LightGlueMatcher, ~6 s/pair; close-up checks)
    hour                         farm-local hour of the capture (op 2 "dark" applies in daylight only)
    baseline                     the camera's op 2 baseline (quality.Baseline().get()), optional
    name                         evidence file prefix (default: image file stem)

Returns a JSON-serializable dict:

    schema "sentinel.vision.pipeline/1", image, prev_image, trap_kind, width, height, method,
    card        op 1 on the current photo: found, roi_source, roi (camos ROI polygons), quad_norm, coverage,
                partial, occlusion, warped_size
    card_prev   op 1 on the previous photo (same keys) or None
    quality     op 2: ok, issues, scores
    register    op 3 (or None without a previous photo): ok, method, inliers, matches, inlier_ratio, reproj_px,
                H (prev canonical -> cur canonical), frame_source [prev, cur]
    diff        op 4: trap_kind, meaning ("catches" sticky | "activity" lure), new_count, persisted_count,
                disappeared_count, new, disappeared, registration_ok, card_change_suspected (sticky only),
                card_change_reasons, change_ratio; or {status: "skipped", reason, change_ratio}
    crowd       op 6: sticky -> saturation_pct, saturation (0-1), blob_count, split_count, card_full ...;
                lure -> status "not_applicable"
    motion      op 8 (or None without a previous photo): occluded_by_large_object, occluder, camera_moved,
                moved_reasons, shift, global_change, changed_share
    boxes, count  boxes_cur as given and their number (None when no boxes were given)
    privacy     person_suspected (large occluder seen), faces (blur_faces info when evidence was sanitized)
    ops_run, timings_ms {op1_card, op2_quality, op3_register, op4_diff, op6_crowd, op8_motion, total},
    errors {op: message} (an op that raises is recorded here, the others still run),
    evidence {key: path} (flat)

`analyze_photo(image, previous=None, trap_kind=..., out_dir=None, boxes=None, prev_boxes=None, detect=True)`
is the adapter sentinel.agent.vision.PipelineAnalyzer calls: same result plus `detect` (op 5 summary: count,
detections with scores, engine_used, forward_ms) when it ran the detector itself because no boxes were given.

CLI:
    uv run python -m sentinel.vision.pipeline --image CUR.jpg [--prev PREV.jpg] [--trap-kind sticky|lure]
        [--boxes CUR_BOXES.json] [--prev-boxes PREV_BOXES.json] [--out DIR] [--method sift|aliked]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from sentinel.vision.card import find_card, roi_polygon_px
from sentinel.vision.crowd import estimate_crowd
from sentinel.vision.diff import _canonical, change_heatmap, diff_detections, transform_boxes
from sentinel.vision.motion import blur_faces, detect_motion
from sentinel.vision.quality import assess_quality
from sentinel.vision.register import register

SCHEMA = "sentinel.vision.pipeline/1"
METHODS = {"sift": "sift_flann", "aliked": "aliked_lightglue"}
CARD_KEYS = ("found", "trap_kind", "roi_source", "roi", "quad_norm", "coverage", "partial", "occlusion", "warped_size")


def _load(ref) -> tuple[np.ndarray, str | None]:
    if isinstance(ref, np.ndarray):
        return ref, None
    img = cv2.imread(str(ref))
    if img is None:
        raise FileNotFoundError(f"cannot read image {ref}")
    return img, Path(ref).as_posix()


def _card_view(c: dict | None) -> dict | None:
    return None if c is None else {k: c.get(k) for k in CARD_KEYS}


def _boxes(b) -> list[list[float]] | None:
    if b is None:
        return None
    return [[float(v) for v in box[:4]] for box in b]


def analyze(
    image_path,
    prev_image_path=None,
    trap_kind: str = "sticky",
    boxes_cur=None,
    boxes_prev=None,
    *,
    out_dir: str | Path | None = None,
    method: str = "sift",
    hour: int | None = None,
    baseline: dict | None = None,
    name: str | None = None,
    cards: tuple[dict | None, dict | None] | None = None,
) -> dict:
    """Ops 1, 2, 3, 4, 6, 8 on one photo (see module docstring for the contract). `cards` = op 1's results for both
    photos when the caller already has them (analyze_photo, for the detector's ROI)."""
    t_all = time.perf_counter()
    if trap_kind not in ("sticky", "lure"):
        raise ValueError(f"trap_kind must be 'sticky' or 'lure', not {trap_kind!r}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, not {method!r}")
    reg_method = METHODS[method]
    cur, cur_ref = _load(image_path)
    prev, prev_ref = _load(prev_image_path) if prev_image_path is not None else (None, None)
    name = name or (Path(cur_ref).stem if cur_ref else "photo")
    out = Path(out_dir) if out_dir is not None else None
    bc, bp = _boxes(boxes_cur), _boxes(boxes_prev)

    res: dict = {
        "schema": SCHEMA, "image": cur_ref, "prev_image": prev_ref, "trap_kind": trap_kind,
        "width": int(cur.shape[1]), "height": int(cur.shape[0]), "method": reg_method,
        "card": None, "card_prev": None, "quality": None, "register": None, "diff": None, "crowd": None,
        "motion": None, "boxes": bc or [], "count": None if bc is None else len(bc),
        "privacy": {"person_suspected": False, "faces": None},
        "ops_run": [], "timings_ms": {}, "errors": {}, "evidence": {},
    }
    ev = res["evidence"]

    def run(key: str, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - a failing op must not lose the other ops' results
            res["errors"][key] = f"{type(e).__name__}: {e}"
            res["errors"][key + "_trace"] = traceback.format_exc(limit=3)
            return None
        finally:
            res["timings_ms"][key] = round((time.perf_counter() - t0) * 1000.0, 2)
            res["ops_run"].append(key)

    # op 1: trap surface (sticky card / lure floor) on both photos
    def op1():
        if cards is not None:
            return cards
        c = find_card(cur, trap_kind=trap_kind, out_dir=out, name=f"{name}_card")
        p = find_card(prev, trap_kind=trap_kind) if prev is not None else None
        return c, p

    cards = run("op1_card", op1)
    card, card_prev = cards if cards else (None, None)
    res["card"], res["card_prev"] = _card_view(card), _card_view(card_prev)
    if card:
        ev.update({f"card_{k}": v for k, v in card["evidence"].items()})

    # op 2: quality gate
    q = run("op2_quality", lambda: assess_quality(cur, card=card, baseline=baseline, hour=hour, out_dir=out,
                                                   name=f"{name}_quality", trap_kind=trap_kind))
    if q:
        res["quality"] = {k: q[k] for k in ("ok", "issues", "scores")}
        ev.update({f"quality_{k}": v for k, v in q["evidence"].items()})

    # ops 3 + 4: register the canonical surfaces, then match boxes
    canon = None
    if prev is not None and card is not None and card_prev is not None:
        def op3():
            a, Ma, src_a = _canonical(prev, card_prev, 800)
            b, Mb, src_b = _canonical(cur, card, 800)
            full = "full_frame" in (src_a, src_b)
            r = register(a, b, method=reg_method, out_dir=out, name=f"{name}_reg", pixelate_evidence=full)
            return r, (a, Ma, src_a, b, Mb, src_b, full)

        got = run("op3_register", op3)
        if got:
            reg, canon = got
            res["register"] = {**{k: reg[k] for k in ("ok", "method", "inliers", "matches", "inlier_ratio", "reproj_px",
                                                       "H", "fallback_used", "fallback_reason")},
                               "frame_source": [canon[2], canon[5]]}
            ev.update({f"register_{k}": v for k, v in reg["evidence"].items()})

            def op4():
                a, Ma, _, b, Mb, _, full = canon
                if bc is None or bp is None:
                    ratio, _, _ = change_heatmap(a, b, np.asarray(reg["H"]) if reg["ok"] and reg["H"] else None)
                    return {"status": "skipped", "reason": "no boxes: pass boxes_cur and boxes_prev",
                            "trap_kind": trap_kind, "change_ratio": round(ratio, 4)}
                d = diff_detections(transform_boxes(bp, Ma), transform_boxes(bc, Mb), reg["H"],
                                    registration_ok=reg["ok"], prev_img=a, cur_img=b, out_dir=out,
                                    name=f"{name}_diff", pixelate_evidence=full, trap_kind=trap_kind)
                ev.update({f"diff_{k}": v for k, v in d["evidence"].items()})
                return {k: v for k, v in d.items() if k != "evidence"} | {"status": "ok", "box_frame": "canonical"}

            res["diff"] = run("op4_diff", op4)

    # op 6: crowd / saturation (sticky only)
    cr = run("op6_crowd", lambda: estimate_crowd(cur, trap_kind=trap_kind, card=card if trap_kind == "sticky" else None,
                                                 out_dir=out, name=name))
    if cr:
        res["crowd"] = {k: v for k, v in cr.items() if k != "evidence"}
        ev.update({f"crowd_{k}": v for k, v in cr["evidence"].items()})

    # op 8: occlusion / camera moved between the two photos
    if prev is not None:
        mo = run("op8_motion", lambda: detect_motion(prev, cur, method=reg_method, out_dir=out, name=f"{name}_motion"))
        if mo:
            res["motion"] = {k: v for k, v in mo.items() if k != "evidence"}
            ev.update({f"motion_{k}": v for k, v in mo["evidence"].items()})
            res["privacy"]["person_suspected"] = bool(mo["occluded_by_large_object"])

    # privacy: a large occluder may be a person -> blur faces in every stored evidence image
    if out is not None and res["privacy"]["person_suspected"]:
        t0 = time.perf_counter()
        infos = {}
        for k, path in ev.items():
            if not str(path).lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(str(path))
            if img is None or img.ndim != 3:
                continue
            clean, info = blur_faces(img)
            if info["method"] != "yunet" or info["faces"]:
                cv2.imwrite(str(path), clean, [cv2.IMWRITE_JPEG_QUALITY, 80])
            infos[k] = {"method": info["method"], "faces": info["faces"]}
        res["privacy"]["faces"] = infos
        res["timings_ms"]["privacy_blur"] = round((time.perf_counter() - t0) * 1000.0, 2)

    res["timings_ms"]["total"] = round((time.perf_counter() - t_all) * 1000.0, 2)
    return res


def detect_boxes(img: np.ndarray, onnx_path: str | Path | None = None, trap_kind: str | None = None,
                 card: dict | None = None) -> dict | None:
    """Op 5 (YOLO26n or YOLOv7-tiny on the OpenCV 5 new DNN engine, see sentinel.vision.detector) ->
    {boxes xyxy, detections [{xywh, score, class_id}], count, engine_used, forward_ms, source, postprocess (NMS,
    box-rule rejections and tiles, YOLO26 only)};
    None when no detector ONNX is available (DetectorUnavailable is swallowed here: the other ops still run).
    `trap_kind` and the photo's op 1 result `card` reach a detector that tiles or filters by region (YOLO26)."""
    from sentinel.vision.detector import DetectorUnavailable, build_detector, detector_kind, get_detector

    try:
        det = build_detector(detector_kind(), Path(onnx_path)) if onnx_path else get_detector()
    except DetectorUnavailable:
        return None
    if getattr(det, "tiling", None) is not None:
        res = det.detect(img, trap_kind=trap_kind, roi=roi_polygon_px(card))
    else:
        res = det.detect(img)
    boxes = [[b["xywh"][0], b["xywh"][1], b["xywh"][0] + b["xywh"][2], b["xywh"][1] + b["xywh"][3]] for b in res["boxes"]]
    engine = getattr(det, "engine_used", "unknown")
    return {"boxes": boxes, "detections": res["boxes"], "count": res["count"], "engine_used": engine,
            "forward_ms": res.get("forward_ms"), "source": Path(str(getattr(det, "onnx_path", "fake"))).name,
            "model": getattr(det, "name", type(det).__name__), "postprocess": res.get("postprocess")}


def _detector_wants_roi(trap_kind: str) -> bool:
    """True when the configured detector filters boxes by the trap surface for this trap kind (DETECTOR_ROI)."""
    from sentinel.vision.detector import DetectorUnavailable, get_detector

    try:
        regions = getattr(get_detector(), "regions", None)
    except DetectorUnavailable:
        return False
    return regions is not None and regions.wants_roi(trap_kind)


def analyze_photo(image, previous=None, trap_kind: str = "sticky", out_dir=None, boxes=None, prev_boxes=None,
                  detect: bool = True, **kw) -> dict:
    """Adapter for sentinel.agent.vision.PipelineAnalyzer: analyze() plus op 5 when no boxes are given.

    `image` / `previous` may also be s3:// or http(s) references (loaded with sentinel.agent.vision.load_image).
    With `detect` and no boxes, the configured detector (sentinel.vision.detector: YOLO26n or YOLOv7-tiny) gives
    boxes for both photos;
    its summary is under `detect` and its time under timings_ms.op5_detect.
    """
    def load(ref):
        if ref is None or isinstance(ref, np.ndarray):
            return ref
        from sentinel.agent.vision import IMAGE_CACHE, load_image, local_path

        if ref in IMAGE_CACHE:
            return IMAGE_CACHE[ref]
        local = local_path(str(ref)) if not str(ref).startswith(("s3://", "http://", "https://")) else None
        if local is not None and local.is_file():
            return _load(local)[0]
        return load_image(ref)

    cur, prev = load(image), load(previous)
    det, t5, cards, t1 = None, None, None, None
    if detect and boxes is None:
        if _detector_wants_roi(trap_kind):  # op 1 first: the detector counts only on the trap surface
            t0 = time.perf_counter()
            try:
                cards = (find_card(cur, trap_kind=trap_kind, out_dir=out_dir, name=f"{kw.get('name') or 'photo'}_card"),
                         find_card(prev, trap_kind=trap_kind) if prev is not None else None)
                t1 = round((time.perf_counter() - t0) * 1000.0, 2)
            except Exception:  # noqa: BLE001 - analyze() runs op 1 again and records the error; count without ROI
                cards = None
        t0 = time.perf_counter()
        det = detect_boxes(cur, trap_kind=trap_kind, card=cards[0] if cards else None)
        if det is not None:
            boxes = det["boxes"]
            if prev is not None and prev_boxes is None:
                prev_boxes = detect_boxes(prev, trap_kind=trap_kind, card=cards[1] if cards else None)["boxes"]
        t5 = round((time.perf_counter() - t0) * 1000.0, 2)
    res = analyze(cur, prev, trap_kind, boxes, prev_boxes, out_dir=out_dir, cards=cards, **kw)
    if t1 is not None:
        res["timings_ms"]["op1_card"] = t1
    if isinstance(image, (str, Path)):
        res["image"] = str(image)
    if isinstance(previous, (str, Path)):
        res["prev_image"] = str(previous)
    res["detect"] = None if det is None else {k: v for k, v in det.items() if k != "boxes"}
    if t5 is not None:
        res["timings_ms"]["op5_detect"] = t5
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--prev", default=None)
    ap.add_argument("--trap-kind", default="sticky", choices=["sticky", "lure"])
    ap.add_argument("--boxes", default=None, help="JSON file: [[x1, y1, x2, y2], ...] for --image")
    ap.add_argument("--prev-boxes", default=None, help="JSON file: boxes for --prev")
    ap.add_argument("--out", default=None, help="evidence directory")
    ap.add_argument("--method", default="sift", choices=sorted(METHODS))
    ap.add_argument("--hour", type=int, default=None)
    a = ap.parse_args(argv)

    def boxes(p):
        return None if p is None else json.loads(Path(p).read_text(encoding="utf-8"))

    try:
        res = analyze(a.image, a.prev, a.trap_kind, boxes(a.boxes), boxes(a.prev_boxes), out_dir=a.out,
                      method=a.method, hour=a.hour)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
