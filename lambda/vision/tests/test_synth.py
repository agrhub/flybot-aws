"""Synthetic trap photo generator (eval/synth): fly cut-outs, compositing, labels, count curves, defects, day mode,
pool, the check's matching and the automatic agent evaluation's verdicts.

The end-to-end tests build a tiny fake fly dataset (dark ellipses on a white disc, YOLO labels) in tmp_path and
uses the bundled empty-trap photos (data/samples), so it needs neither the Roboflow export nor the detector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]  # repo root (lambda/vision/tests)
sys.path.insert(0, str(ROOT))

from eval.synth import (
    FILE_PREFIX,
    LABEL,
    SCHEMA,
    curve,
    defects,
)
from eval.synth import composite as C
from eval.synth.check import match, previous_clean
from eval.synth.flies import FlyCrop, cut_out, load_bank, read_labels
from eval.synth.report import judge, summarize


def _fly_image(n: int = 3, rng=None) -> tuple[np.ndarray, list[list[float]]]:
    """640x640 white disc picture with `n` well separated dark 'flies' and their xyxy boxes."""
    rng = rng or np.random.default_rng(1)
    img = np.full((640, 640, 3), 235, np.uint8)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 2, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    boxes = []
    for i in range(n):
        cx, cy = 120 + 190 * (i % 3), 140 + 200 * (i // 3)
        cv2.ellipse(img, (cx, cy), (16, 9), 30 * i, 0, 360, (30, 50, 90), -1)
        cv2.ellipse(img, (cx + 3, cy), (12, 18), 30 * i + 90, 0, 360, (150, 160, 165), 1)  # wings
        boxes.append([cx - 22.0, cy - 22.0, cx + 22.0, cy + 22.0])
    return img, boxes


def _fake_dataset(root: Path, images: int = 4) -> Path:
    for split in ("test", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for k in range(images):
            img, boxes = _fly_image(6, np.random.default_rng(k))
            cv2.imwrite(str(root / split / "images" / f"fly_{k}.jpg"), img)
            rows = [f"0 {(b[0] + b[2]) / 1280:.6f} {(b[1] + b[3]) / 1280:.6f} {(b[2] - b[0]) / 640:.6f} {(b[3] - b[1]) / 640:.6f}"
                    for b in boxes]
            (root / split / "labels" / f"fly_{k}.txt").write_text("\n".join(rows), encoding="utf-8")
    return root


def _dark_fly() -> FlyCrop:
    ratio = np.ones((40, 40, 3), np.float32)
    alpha = np.zeros((40, 40), np.float32)
    cv2.ellipse(alpha, (20, 20), (12, 6), 0, 0, 360, 1.0, -1)
    ratio[alpha > 0] = 0.2
    return FlyCrop(id="t/x.jpg#0", ratio=ratio, alpha=alpha, box=(8.0, 14.0, 32.0, 26.0))


# ---------------------------------------------------------------- flies


def test_read_labels_boxes_and_polygons(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.1 0.2\n0 0.1 0.1 0.2 0.1 0.2 0.3 0.1 0.3\n", encoding="utf-8")
    boxes = read_labels(p, 640, 640)
    assert boxes[0] == pytest.approx([288, 256, 352, 384])
    assert boxes[1] == pytest.approx([64, 64, 128, 192])  # polygon -> its bounding box


def test_cut_out_clean_fly_and_busy_background():
    img, boxes = _fly_image(1)
    fc = cut_out(img, boxes[0])
    assert fc is not None
    h, w = fc.alpha.shape
    assert fc.alpha[h // 2, w // 2] > 0.9 and fc.alpha[0, 0] == 0.0
    assert fc.ratio.max() <= 1.0 and fc.ratio[h // 2, w // 2].mean() < 0.5
    busy = np.random.default_rng(0).integers(0, 255, img.shape, dtype=np.uint8)
    assert cut_out(busy, boxes[0]) is None  # no clean ring: rejected


def test_load_bank_is_deterministic(tmp_path):
    ds = _fake_dataset(tmp_path / "ds", images=2)
    a, b = load_bank(ds, max_crops=5, seed=3), load_bank(ds, max_crops=5, seed=3)
    assert len(a) == 5 and [f.id for f in a] == [f.id for f in b]
    assert all(f.id.split("/")[0] in ("test", "valid") for f in a)


# ---------------------------------------------------------------- compositing


def test_paste_fly_darkens_and_returns_box():
    img = np.full((300, 400, 3), 200.0, np.float32)
    box = C.paste_fly(img, _dark_fly(), (200, 150), long_side=48, angle=0)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 < 200 < x1 and y0 < 150 < y1 and 30 < x1 - x0 < 60
    assert img[150, 200].mean() < 80 and img[10, 10].mean() == 200.0
    assert C.paste_fly(img, _dark_fly(), (5, 5), long_side=48, angle=0) is None  # does not fit the frame


def test_paste_fly_seamless_mode():
    img = np.full((300, 400, 3), 200.0, np.float32)
    box = C.paste_fly(img, _dark_fly(), (200, 150), long_side=48, angle=45, blend="seamless")
    assert box is not None and img[150, 200].mean() < 170
    with pytest.raises(ValueError):
        C.paste_fly(img, _dark_fly(), (200, 150), long_side=48, angle=0, blend="nope")


def test_placement_mask_keeps_margins_and_stamp_corner():
    img = np.zeros((432, 768, 3), np.float32)
    poly = [[100, 20], [700, 20], [700, 420], [100, 420]]
    m = C.placement_mask(img, poly, margin=20)
    assert m[200, 400] == 255
    assert m[200, 105] == 0 and m[25, 400] == 0  # within the margin of the polygon edge
    assert m[425, 150] == 0  # stamp corner (bottom-left)


def test_jpeg_label_roundtrip():
    img = np.full((120, 160, 3), 128, np.uint8)
    data = C.encode_jpeg(img, 85, {"synthetic": "true", "photoId": "synth-x", "trueCount": 3}, "test")
    lab = C.read_label(data)
    assert lab["exif"] == LABEL and lab["xmp_synthetic"] is True
    assert lab["comment"].startswith(LABEL) and "photoId=synth-x" in lab["comment"]
    dec = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert dec is not None and dec.shape == (120, 160, 3)
    plain = cv2.imencode(".jpg", img)[1].tobytes()
    assert C.read_label(plain) == {"exif": None, "xmp_synthetic": False, "comment": None}


def test_erase_osd_and_stamp():
    img = np.full((432, 768, 3), 120, np.uint8)
    cv2.putText(img, "05-10-2026 11:30:10", (35, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    clean = C.erase_osd(img.astype(np.float32))
    assert clean[400:425, 30:190].max() < 200  # the white date text is gone
    out = C.stamp(np.clip(clean, 0, 255).astype(np.uint8), "SIMULATED 2026-09-19 09:00 demo")
    assert out[410:425, 10:200].min() < 40 and out[410:425, 10:200].max() > 200


# ---------------------------------------------------------------- curves


def test_wave_peaks_and_levels_are_seeded():
    w = curve.Wave(baseline=1, amplitude=20, peak_day=10, rise_days=3, decline_days=6, noise=0.0)
    means = [w.mean(d) for d in range(30)]
    assert int(np.argmax(means)) == 10 and means[0] < 2 and means[29] < means[10]
    assert means[14] > means[6]  # the decline is slower than the rise
    a = curve.daily_levels(curve.Wave(noise=0.3), 30, np.random.default_rng(5))
    b = curve.daily_levels(curve.Wave(noise=0.3), 30, np.random.default_rng(5))
    assert a == b and len(a) == 30 and min(a) > 0


def test_slot_weights_and_counts():
    wts = curve.slot_weights([6.5, 9, 11.5, 14, 16.5])
    assert sum(wts) / len(wts) == pytest.approx(1.0) and wts[1] > wts[0]
    rng = np.random.default_rng(0)
    lure = curve.lure_counts([100.0] * 3, wts, rng, cap=50)
    assert all(c <= 50 for day in lure for c in day)
    new = curve.sticky_new([10.0] * 3, wts, rng)
    assert all(c >= 0 for day in new for c in day)


# ---------------------------------------------------------------- check helpers


def test_match_and_summarize():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    assert match(gt, [[1, 1, 11, 11], [100, 100, 110, 110]], 0.3) == 1
    assert match(gt, [[0, 0, 10, 10], [0, 0, 10, 10]], 0.3) == 1  # one-to-one
    rows = [{"trap": "a", "trapKind": "lure", "true": 4, "det": 3, "npred": 3, "tp30": 3, "tp50": 2},
            {"trap": "a", "trapKind": "lure", "true": 0, "det": 1, "npred": 1, "tp30": 0, "tp50": 0}]
    s = summarize(rows)
    assert s["mae"] == 1.0 and s["bias"] == 0.0
    assert s["iou30"]["recall"] == 0.75 and s["iou30"]["precision"] == 0.75
    assert s["empty_photos"] == {"photos": 1, "with_detections": 1, "false_flies": 1}


# ---------------------------------------------------------------- end to end


def _series_cfg(**kw) -> dict:
    cfg = {"name": "t", "seed": 11, "startDate": "2026-09-19", "days": 2, "utcOffset": "+07:00",
           "schedule": ["09:00", "16:00"], "flyScale": 0.11, "flyBank": {"maxCrops": 20},
           "traps": [{"id": "lure-x", "name": "L", "trapKind": "lure", "backgroundCamera": "BE4089629", "cap": 8,
                      "wave": {"baseline": 4, "amplitude": 0, "noise": 0.01}},
                     {"id": "sticky-x", "name": "S", "trapKind": "sticky", "backgroundCamera": "BE4089629",
                      "cardDays": 1, "flyScale": 0.04, "wave": {"baseline": 12, "amplitude": 0, "noise": 0.01}}]}
    cfg.update(kw)
    return cfg


def test_generate_end_to_end(tmp_path):
    from eval.synth.generate import generate

    ds = _fake_dataset(tmp_path / "ds")
    cfg = _series_cfg()
    m = generate(cfg, tmp_path / "out", ds, log=lambda *_: None)
    assert m["schema"] == SCHEMA and m["synthetic"] is True and len(m["photos"]) == 8
    schema = json.loads((ROOT / "eval" / "synth" / "manifest.schema.json").read_text(encoding="utf-8"))
    assert set(schema["required"]) <= set(m)
    assert all(set(schema["properties"]["photos"]["items"]["required"]) <= set(p) for p in m["photos"])
    for p in m["photos"]:
        f = tmp_path / "out" / p["file"]
        assert p["synthetic"] is True and f.name.startswith(FILE_PREFIX) and p["photoId"].startswith("synth-")
        assert p["trueCount"] == len(p["boxes"]) and p["capturedAt"].endswith("Z")
        lab = C.read_label(f.read_bytes())
        assert lab["exif"] == LABEL and lab["xmp_synthetic"]
        assert all(0 <= b["x"] <= 1 and 0 <= b["y"] <= 1 for b in p["boxes"])
    sticky = [p for p in m["photos"] if p["trap"] == "sticky-x"]
    assert sticky[1]["trueCount"] >= sticky[0]["trueCount"]  # flies stay on the card
    assert sticky[2]["cardChanged"] is True and sticky[2]["trueCount"] == 0  # day 2: a clean card
    assert len(m["traps"][1]["cards"]) == 2 and m["traps"][0]["cards"] == []
    lure = [p for p in m["photos"] if p["trap"] == "lure-x"]
    assert all(p["newSincePrevious"] == p["trueCount"] for p in lure)
    saved = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert saved["photos"][0]["photoId"] == m["photos"][0]["photoId"]
    assert (tmp_path / "out" / "manifest.csv").read_text(encoding="utf-8").startswith("photoId,synthetic")
    assert "CC BY 4.0" in (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    again = generate(cfg, tmp_path / "out2", ds, log=lambda *_: None)
    assert [p["sha256"] for p in again["photos"]] == [p["sha256"] for p in m["photos"]]  # same seed, same bytes


def test_one_day_alone_equals_the_bulk_run(tmp_path):
    """The daily service generates one trap-day at a time: same ids and bytes as the bulk run, sticky card replayed."""
    from eval.synth.generate import generate

    ds = _fake_dataset(tmp_path / "ds")
    cfg = _series_cfg(days=3, defects={"rate": 0.5})
    cfg["traps"][1]["cardDays"] = 2  # day 3 starts a new card, day 2 continues the first one
    bulk = generate(cfg, tmp_path / "bulk", ds, log=lambda *_: None)
    for day in ("2026-09-20", "2026-09-21"):
        from datetime import date

        one = generate(cfg, tmp_path / day, ds, log=lambda *_: None, first=date.fromisoformat(day),
                       last=date.fromisoformat(day))
        want = [p for p in bulk["photos"] if p["localDate"] == day]
        assert [p["photoId"] for p in one["photos"]] == [p["photoId"] for p in want]
        assert [p["sha256"] for p in one["photos"]] == [p["sha256"] for p in want]
        assert [p["trueCount"] for p in one["photos"]] == [p["trueCount"] for p in want]
    ids = [p["photoId"] for p in bulk["photos"]]
    assert len(set(ids)) == len(ids) and all(i.endswith("Z") and i.count("-") >= 3 for i in ids)
    assert any(p["defect"] for p in bulk["photos"])  # rate 0.5 over 12 photos


def test_defects_change_the_photo_and_say_what_to_expect():
    img = np.full((432, 768, 3), 200, np.uint8)
    cv2.circle(img, (384, 216), 150, (150, 160, 170), -1)
    for kind in defects.TYPES:
        out, params, hidden, M = defects.apply(img, kind, np.random.default_rng(1))
        assert out.shape == img.shape and out.dtype == np.uint8 and "severity" in params
        assert not np.array_equal(out, img), kind
        assert (hidden is not None) == (kind in ("occluded_hand", "occluded_leaf", "no_trap")), kind
        assert (M is not None) == (kind == "camera_moved"), kind
        e = defects.expected_for(kind, "ezviz")
        assert e["flag"] and e["actions"] and e["needsPrev"] == (kind in defects.NEEDS_PREV)
    assert defects.expected_for("dark", "camos")["actions"] == ["set_isp"]
    assert defects.expected_for("dark", "ezviz")["actions"] == ["capture_now"]
    assert defects.expected_for("no_trap", "ezviz")["counted"] is False
    M = np.float32([[1, 0, 10], [0, 1, -5]])
    assert defects.move_box([100, 100, 20, 10], M) == [110.0, 95.0, 20.0, 10.0]


def test_judge_and_previous_clean():
    exp = defects.expected_for("occluded_hand", "ezviz")
    ok = {"expected": exp, "flags": ["occluded"], "issues": [], "rules": ["occlusion.recapture"],
          "actions": ["capture_now"], "counted": False}
    assert judge(ok)["pass"] is True
    wrong_reason = {**ok, "flags": ["trap_missing"], "rules": ["trap_missing.recapture"]}
    j = judge(wrong_reason)
    assert j["outcome"] is True and j["pass"] is False
    counted = {**ok, "counted": True}
    assert judge(counted)["outcome"] is False
    by_id = {"a": {"photoId": "a", "defect": None}, "b": {"photoId": "b", "defect": {"type": "blur"},
                                                          "previousPhotoId": "a"},
             "c": {"photoId": "c", "defect": None, "previousPhotoId": "b"}}
    assert previous_clean(by_id["c"], by_id)["photoId"] == "a"  # the damaged photo is skipped
    assert previous_clean(by_id["a"], by_id) is None


def test_pool_names_no_trap_and_no_time(tmp_path):
    from eval.synth.pool import build

    ds = _fake_dataset(tmp_path / "ds")
    cfg = {"name": "pool", "seed": 5, "startDate": "2026-01-01", "schedule": ["09:00", "16:00"], "flyScale": 0.11,
           "flyBank": {"maxCrops": 20},
           "lure": {"max": 10, "bucket": 5, "perBucket": 1,
                    "traps": [{"id": "pool-lure-a", "backgroundCamera": "BE4089629", "smallMaxCount": 40}]},
           "sticky": {"backgroundCamera": "BE4089629", "cardDays": 2, "dailyCatch": [6]},
           "defects": {"lure": 1, "sticky": 0}}
    m = build(cfg, tmp_path / "pool", ds, log=lambda *_: None)
    lure = [p for p in m["photos"] if p["trapKind"] == "lure" and not p["defect"]]
    sticky = [p for p in m["photos"] if p["trapKind"] == "sticky"]
    assert [p["curveNew"] for p in lure] == sorted(p["curveNew"] for p in lure) and len(lure) == 3
    assert len(sticky) == 4 and [p["dayInCard"] for p in sticky] == [1, 1, 2, 2]
    assert all(b >= a for a, b in zip([p["trueCount"] for p in sticky], [p["trueCount"] for p in sticky][1:]))
    assert sum(1 for p in m["photos"] if p["defect"]) == len(defects.TYPES)
    for p in m["photos"]:
        assert p["capturedAt"] is None and p["photoId"].startswith("synth-pool-") and p["synthetic"] is True
        assert C.read_label((tmp_path / "pool" / p["file"]).read_bytes())["xmp_synthetic"]


def test_trainset_skips_excluded_backgrounds_and_writes_yolo_labels(tmp_path):
    import csv
    import zipfile

    from eval.synth.pool import build
    from eval.synth.trainset import export_yolo

    ds = _fake_dataset(tmp_path / "ds")
    with open(ROOT / "data" / "samples" / "samples_manifest.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    lure_bgs = [r["snapshot_id"] for r in rows if r["device_serial"] == "BE4089629"]
    keep = lure_bgs[0]
    cfg = {"name": "trainset", "seed": 9, "startDate": "2026-01-01", "schedule": ["09:00"], "flyScale": 0.11,
           "flyBank": {"maxCrops": 20}, "excludeBackgrounds": lure_bgs[1:],
           "lure": {"max": 10, "bucket": 5, "perBucket": 2,
                    "traps": [{"id": "train-lure-a", "backgroundCamera": "BE4089629", "smallMaxCount": 40}]},
           "sticky": {"backgroundCamera": "BE4089629", "cardDays": 1, "dailyCatch": []}, "defects": {}}
    m = build(cfg, tmp_path / "ts", ds, log=lambda *_: None)
    assert {Path(p["background"]["file"]).stem for p in m["photos"]} == {keep}
    info = export_yolo(m, tmp_path / "ts", tmp_path / "ts-yolo.zip")
    assert info["photos"] == len(m["photos"]) and info["boxes"] == sum(p["visibleCount"] for p in m["photos"])
    with zipfile.ZipFile(tmp_path / "ts-yolo.zip") as z:
        p = max(m["photos"], key=lambda q: q["visibleCount"])
        stem = "trainset_" + p["photoId"].removeprefix("synth-")
        lines = z.read(f"labels/{stem}.txt").decode().split()
        assert f"images/{stem}.jpg" in z.namelist() and "README.txt" in z.namelist()
    vals = [float(v) for v in lines]
    assert vals[0::5] == [0.0] * p["visibleCount"] and all(0.0 <= v <= 1.0 for v in vals)
    x, y, w, h = p["boxes"][0]["px"]
    assert vals[1:5] == pytest.approx([(x + w / 2) / p["width"], (y + h / 2) / p["height"], w / p["width"],
                                       h / p["height"]], abs=1e-5)
