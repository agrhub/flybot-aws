"""Run the vision handler on every photo of a SYNTHETIC set (eval/synth) and record its answer next to the known truth.

    uv run --project lambda/vision python -m eval.synth.check [--manifest data/synth/demo-30d/manifest.json]
        [--url http://localhost:9001/invoke] [--limit N] [--no-prev] [--tiles] [--out DIR]
    uv run --project lambda/vision python -m eval.synth.report      # the write-ups, from the check.jsonl files

The handler is the Lambda's code path: YOLO26n on OpenCV 5 DNN (ENGINE_NEW), trap ROI, box rules, quality and motion
ops, rule planner, proposals. Each request is what FlyBot would send for that photo: the camera's platform and trap
kind, `capturedAt`, `prevPhoto` = the trap's latest photo without an injected defect (a damaged photo is a one-off,
not the new normal), and `history` = the trap's earlier counts as the handler gave them (photos it did not count
left out), so the surge rule sees what it would see in production.

Writes check.jsonl beside the manifest (git-ignored; `--out` puts it and check.meta.json in another folder, e.g. for
a second run with other detector settings), one record per photo: true and visible count, detected count,
box matches, the injected defect and its expected action, the response's quality flags, decision rules, proposal
actions, whether the photo was counted, the agent's surge decision and the surge the true counts imply. With
`--tiles`, also the detector alone on the whole frame and on 2x2 tiles. Every photo's JPEG label is verified first:
a file without the synthetic markers stops the run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from eval.synth import ROOT, SCHEMA
from eval.synth.composite import iou, read_label

DEFAULT_MANIFEST = ROOT / "data" / "synth" / "demo-30d" / "manifest.json"


def request_for(p: dict, base: Path, prev: dict | None, history: list[dict]) -> dict:
    req = {"v": 1, "mode": "detect", "photo": {"url": (base / p["file"]).resolve().as_uri()},
           "camera": {"id": p["trap"], "trapKind": p["trapKind"], "platform": p.get("platform") or "ezviz"},
           "history": history, "locale": "en", "photoId": p["photoId"]}
    if p.get("capturedAt"):
        req["capturedAt"] = p["capturedAt"]
    if prev is not None:
        req["prevPhoto"] = {"url": (base / prev["file"]).resolve().as_uri()}
    return req


def call_http(url: str, req: dict) -> dict:
    r = urllib.request.Request(url, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read())


def match_pairs(gt: list[list[float]], pred: list[list[float]], thr: float) -> set[int]:
    """Greedy one-to-one matching by IoU (highest first); returns the matched ground-truth indices at IoU >= thr."""
    pairs = sorted(((iou(g, q), i, j) for i, g in enumerate(gt) for j, q in enumerate(pred)), reverse=True)
    used_g, used_p = set(), set()
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
    return used_g


def match(gt: list[list[float]], pred: list[list[float]], thr: float) -> int:
    return len(match_pairs(gt, pred, thr))


def gt_boxes(p: dict, visible_only: bool = True) -> list[list[float]]:
    return [[b["px"][0], b["px"][1], b["px"][0] + b["px"][2], b["px"][1] + b["px"][3]] for b in p["boxes"]
            if not (visible_only and b.get("hidden"))]


def pred_boxes(res: dict, W: int, H: int) -> list[list[float]]:
    return [[b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W, (b["y"] + b["h"]) * H] for b in res.get("boxes") or []]


def tiled_boxes(det, img, grid: int = 2, overlap: float = 0.12) -> list[list[float]]:
    """Detector only (no trap ROI, no box rules): `grid` x `grid` overlapping tiles, each letterboxed to the
    detector's input, boxes merged with class-agnostic NMS. Each fly is seen `grid` times larger than full frame."""
    from sentinel.vision.box_rules import nms

    H, W = img.shape[:2]
    th, tw = int(H / grid * (1 + overlap)), int(W / grid * (1 + overlap))
    dets = []
    for gy in range(grid):
        for gx in range(grid):
            y0 = min(H - th, max(0, int(gy * H / grid - overlap * H / grid / 2)))
            x0 = min(W - tw, max(0, int(gx * W / grid - overlap * W / grid / 2)))
            for d in det.detect(img[y0:y0 + th, x0:x0 + tw])["boxes"]:
                x, y, w, h = d["xywh"]
                dets.append({**d, "xywh": [x + x0, y + y0, w, h]})
    return [[d["xywh"][0], d["xywh"][1], d["xywh"][0] + d["xywh"][2], d["xywh"][1] + d["xywh"][3]] for d in nms(dets, 0.4)]


def full_boxes(det, img) -> list[list[float]]:
    """Detector only on the whole frame (no trap ROI, no box rules): the handler's detector step alone."""
    return [[d["xywh"][0], d["xywh"][1], d["xywh"][0] + d["xywh"][2], d["xywh"][1] + d["xywh"][3]]
            for d in det.detect(img)["boxes"]]


class SurgeTruth:
    """The surge rule (sentinel.agent.domain.detect_surge, FlyBot's surgeService) applied to the TRUE counts of a trap."""

    def __init__(self):
        from sentinel.agent.policy import load_policy

        self.policy = load_policy()
        self.daily: dict[str, dict[str, dict]] = {}
        self.cards: dict[str, list[str]] = {}

    def add(self, p: dict) -> bool | None:
        from sentinel.agent.domain import detect_surge, with_today

        if not p.get("capturedAt") or not p.get("localDate"):
            return None  # pool photos have no time: no series, no surge
        day = p["localDate"]
        if p.get("cardChanged"):
            self.cards.setdefault(p["trap"], []).append(day)
        d = with_today(self.daily.get(p["trap"], {}), day, int(p["trueCount"]))
        self.daily[p["trap"]] = d
        return detect_surge(d, day, self.policy, p["trapKind"], self.cards.get(p["trap"])) is not None


def previous_clean(p: dict, by_id: dict[str, dict]) -> dict | None:
    """The trap's latest earlier photo without an injected defect (following previousPhotoId)."""
    seen = set()
    q = by_id.get(p.get("previousPhotoId") or "")
    while q is not None and q.get("defect") and q["photoId"] not in seen:
        seen.add(q["photoId"])
        q = by_id.get(q.get("previousPhotoId") or "")
    return q


def run(manifest_path: Path, url: str | None = None, limit: int | None = None, with_prev: bool = True,
        tiles: bool = False, log=print, out_dir: Path | None = None) -> dict:
    os.environ.setdefault("DETECTOR", "yolo26")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA or m.get("synthetic") is not True:
        raise SystemExit(f"{manifest_path} is not a synthetic manifest ({SCHEMA})")
    base = manifest_path.parent
    photos = m["photos"][:limit] if limit else m["photos"]
    by_id = {p["photoId"]: p for p in m["photos"]}
    for p in photos:  # every image must carry the label before anything is measured on it
        lab = read_label((base / p["file"]).read_bytes())
        if not (p.get("synthetic") is True and lab["exif"] and lab["xmp_synthetic"] and Path(p["file"]).name.startswith("SYNTH_")):
            raise SystemExit(f"{p['file']}: synthetic label missing in the manifest, filename, EXIF or XMP")
    if not url:
        from sentinel import handler as h
    if tiles:
        import cv2

        from sentinel.vision.detect_yolo26 import Yolo26Detector
        from sentinel.vision.detector import resolve_onnx

        raw_det = Yolo26Detector(resolve_onnx("yolo26"))  # conf 0.25, NMS 0.45, no box rules
    truth = SurgeTruth()
    history: dict[str, list[dict]] = {}
    n_ok, errors, model = 0, [], None
    t_run = time.perf_counter()
    out_dir = Path(out_dir) if out_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "check.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, p in enumerate(photos, 1):
            surge_truth = truth.add(p)
            prev = previous_clean(p, by_id) if with_prev else None
            req = request_for(p, base, prev, history.get(p["trap"], [])[-200:])
            try:
                res = call_http(url, req) if url else h.analyze(req, store=False)
            except Exception as e:  # noqa: BLE001 - one bad photo must not stop the run
                errors.append({"photoId": p["photoId"], "error": str(e)[:300]})
                log(f"{i:4} {p['photoId']} ERROR {str(e)[:160]}")
                continue
            model = model or res.get("model")
            pred = pred_boxes(res, p["width"], p["height"])
            gt = gt_boxes(p)
            hit = match_pairs(gt, pred, 0.3)
            scale = 640.0 / max(p["width"], p["height"])
            vis = [b for b in p["boxes"] if not b.get("hidden")]
            tr = res.get("trace") or {}
            rules = [d.get("rule") for d in tr.get("decisions") or []]
            events = [e.get("kind") for e in tr.get("events") or []]
            counted = "photo_not_counted" not in events and not any(
                r and r.split(".")[0] in ("occlusion", "trap_missing") or r == "quality.postpone" for r in rules)
            rec = {"photoId": p["photoId"], "set": m.get("name"), "trap": p["trap"], "trapKind": p["trapKind"],
                   "platform": p.get("platform"), "date": p.get("localDate"), "slot": p.get("slot"),
                   "true": p["trueCount"], "visible": p.get("visibleCount", p["trueCount"]),
                   "det": int(res["counts"]["total"]), "npred": len(pred), "tp30": len(hit), "tp50": match(gt, pred, 0.5),
                   "gtSizes640": [[round(max(b["px"][2], b["px"][3]) * scale, 1), k in hit] for k, b in enumerate(vis)],
                   "defect": (p.get("defect") or {}).get("type"), "expected": (p.get("defect") or {}).get("expected"),
                   "prevSent": prev["photoId"] if prev else None, "background": p["background"]["file"],
                   "flags": res["quality"]["flags"], "issues": res["quality"].get("issues", []), "rules": rules,
                   "actions": [x.get("action") for x in res.get("proposals") or []], "events": events,
                   "counted": counted, "surgeAgent": any(r and r.startswith("surge") for r in rules),
                   "surgeTruth": surge_truth, "totalMs": res["timings"].get("totalMs"),
                   "dnnMs": res["timings"].get("dnnMs")}
            if tiles:
                img = cv2.imread(str(base / p["file"]), cv2.IMREAD_COLOR)
                for key, boxes in (("full", full_boxes(raw_det, img)), ("tiles2x2", tiled_boxes(raw_det, img))):
                    rec[key], rec[f"{key}_tp30"] = len(boxes), match(gt, boxes, 0.3)
            if counted and p.get("capturedAt"):  # what FlyBot keeps and sends back as history
                history.setdefault(p["trap"], []).append({"capturedAt": p["capturedAt"], "count": rec["det"],
                                                          "photoId": p["photoId"]})
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
            if i % 25 == 0 or i == len(photos):
                log(f"{i:4}/{len(photos)} {p['trap']} true={rec['true']:3} det={rec['det']:3} "
                    f"defect={rec['defect'] or '-'} actions={','.join(rec['actions']) or '-'} ({time.perf_counter() - t_run:.0f} s)")
    meta = {"manifest": manifest_path.resolve().relative_to(ROOT).as_posix() if manifest_path.resolve().is_relative_to(ROOT)
            else str(manifest_path), "set": m.get("name"), "seed": m.get("seed"), "photos": len(photos), "ok": n_ok,
            "errors": errors, "generator_git": m["generator"]["git"], "model": model, "tiles": tiles,
            "prev_photo": with_prev, "transport": url or "in-process sentinel.handler.analyze (store=False)",
            "flyScale": {t["id"]: t.get("flyScale", m["config"].get("flyScale")) for t in m["config"].get("traps", [])}
            or m["config"].get("flyScale"), "fly_crops": m["credits"]["flyBank"],
            "run_s": round(time.perf_counter() - t_run, 1), "at": datetime.now().astimezone().isoformat(timespec="seconds")}
    (out_dir / "check.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--url", default=None, help="POST to a running `make lambda-local` instead of calling the handler")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-prev", action="store_true", help="send no prevPhoto (op 8 and the diff are skipped)")
    ap.add_argument("--tiles", action="store_true", help="also run the detector alone on the whole frame and on 2x2 tiles")
    ap.add_argument("--out", type=Path, default=None, help="folder for check.jsonl / check.meta.json (default: the manifest's)")
    a = ap.parse_args(argv)
    meta = run(a.manifest, a.url, a.limit, not a.no_prev, a.tiles, out_dir=a.out)
    print(f"\n{meta['ok']}/{meta['photos']} photos, {len(meta['errors'])} errors, {meta['run_s']} s -> "
          f"{(a.out or a.manifest.parent) / 'check.jsonl'}")
    return 1 if meta["errors"] and not meta["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
