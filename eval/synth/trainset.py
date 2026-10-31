"""Labelled synthetic photos for fine-tuning the fruit-fly detector (SIMULATED DATA), in the YOLO format.

    uv run --project lambda/vision python -m eval.synth.trainset [--config eval/synth/trainset.json] [--out data/synth/trainset]
    -> <out>/ (manifest.json, images/...) and <out>-yolo.zip: images/*.jpg + labels/*.txt, class 0 = fruit_fly

The pool generator (eval.synth.pool) with its own seed and fly bank. The flies are cut from the dataset's `train` split,
so the test and valid splits the detector is scored on, and the flies of the evaluation photos (data/synth/demo-30d
and pool use test and valid crops), stay unseen. The background photos that are in eval.detector_ab's real check set
(every 4th exported photo) are left out, so that check of false flies stays a check. No damaged photos.

`models/colab_train_yolo26.ipynb` adds the zip to the training split when `SYNTH_ZIP` names it (validation and test
stay the dataset's own). The labels hold class 0 only: `fruit_fly` is class 0 in v3 and v4 of camos/fruit-fly-wmcqi.
The fly bank takes crops of 18-90 px at 640 (eval.synth.flies), which leaves out the tiny boxes v4 relabelled as
ants or removed as debris; the v3 labels it reads are not otherwise cleaned.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from eval.synth import ROOT
from eval.synth.flies import DEFAULT_DATASET
from eval.synth.pool import build

DEFAULT_CONFIG = Path(__file__).resolve().parent / "trainset.json"
NOTICE = ("SIMULATED DATA: real photos of empty traps with fruit flies pasted in by OpenCV (FlyBot eval/synth). "
          "Flies from camos/fruit-fly-wmcqi (Roboflow Universe, CC BY 4.0). Not field data.\n")


def real_check_ids() -> list[str]:
    from eval.detector_ab import real_items

    return sorted(it["id"] for it in real_items(4))


def yolo_rows(photo: dict) -> list[str]:
    W, H = photo["width"], photo["height"]
    rows = []
    for b in photo["boxes"]:
        if b.get("hidden"):
            continue
        x, y, w, h = b["px"]
        x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
        if x1 > x0 and y1 > y0:
            rows.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return rows


def export_yolo(manifest: dict, out: Path, zip_path: Path) -> dict:
    """Images and one label file per photo (an empty file for a photo without a fly) in `zip_path`."""
    photos = [p for p in manifest["photos"] if not p.get("defect")]
    boxes = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # JPEGs do not compress
        z.writestr("README.txt", NOTICE)
        for p in photos:
            stem = "trainset_" + p["photoId"].removeprefix("synth-")
            rows = yolo_rows(p)
            boxes += len(rows)
            z.write(out / p["file"], f"images/{stem}.jpg")
            z.writestr(f"labels/{stem}.txt", "\n".join(rows) + ("\n" if rows else ""))
    kinds = {k: sum(p["trapKind"] == k for p in photos) for k in ("lure", "sticky")}
    return {"zip": str(zip_path), "photos": len(photos), "boxes": boxes, **kinds}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synth" / "trainset")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    cfg["excludeBackgrounds"] = real_check_ids()
    manifest = build(cfg, a.out, a.dataset)
    info = export_yolo(manifest, a.out, a.out.with_name(a.out.name + "-yolo.zip"))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
