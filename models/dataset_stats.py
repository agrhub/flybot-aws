"""Count the fruit-fly dataset per split (T0.4). Main project env, no torch.

    uv run python models/dataset_stats.py ../camos-ai-camera/models/fruit-fly-v3 --out models/dataset_stats.json

Every number in models/dataset.md comes from the JSON this writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_stem(name: str) -> str:
    """Roboflow names augmented copies `<source>_jpg.rf.<hash>.jpg`: the part before `.rf.`."""
    return name.split(".rf.")[0]


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def split_stats(root: Path, split: str) -> dict:
    img_dir, lbl_dir = root / split / "images", root / split / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    labels = sorted(lbl_dir.glob("*.txt"))
    label_stems = {p.stem for p in labels}
    sizes: Counter[str] = Counter()
    classes: Counter[int] = Counter()
    per_image: list[int] = []
    box_w: list[float] = []
    box_h: list[float] = []
    bad_lines = 0
    polygon_lines = 0
    polygon_files = 0
    mixed_files = 0
    missing_label = 0
    for img_path in images:
        h, w = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).shape[:2]
        sizes[f"{w}x{h}"] += 1
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if img_path.stem not in label_stems:
            missing_label += 1
            per_image.append(0)
            continue
        n = n_box = n_poly = 0
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 5:
                bw, bh = float(parts[3]), float(parts[4])
                n_box += 1
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                # polygon row: cls x1 y1 x2 y2 ... (counted as its bounding box, as ultralytics does)
                xy = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
                bw, bh = float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))
                n_poly += 1
            else:
                bad_lines += 1
                continue
            classes[int(parts[0])] += 1
            box_w.append(bw * w)
            box_h.append(bh * h)
            n += 1
        polygon_lines += n_poly
        polygon_files += n_poly > 0
        mixed_files += n_poly > 0 and n_box > 0
        per_image.append(n)
    sources = {source_stem(p.name) for p in images}
    return {
        "images": len(images),
        "labels": len(labels),
        "images_without_label_file": missing_label,
        "images_with_zero_boxes": sum(1 for n in per_image if n == 0),
        "boxes": int(sum(per_image)),
        "boxes_per_image": {
            "mean": round(float(np.mean(per_image)), 2) if per_image else None,
            "median": pct(per_image, 50),
            "p95": pct(per_image, 95),
            "max": int(max(per_image)) if per_image else None,
            "images_over_300": sum(1 for n in per_image if n > 300),
        },
        "class_ids": {str(k): v for k, v in sorted(classes.items())},
        "polygon_label_lines": polygon_lines,
        "label_files_with_polygons": polygon_files,
        "label_files_mixing_boxes_and_polygons": mixed_files,
        "malformed_label_lines": bad_lines,
        "image_sizes": dict(sizes),
        "box_px_at_native_size": {
            "w_median": pct(box_w, 50), "h_median": pct(box_h, 50),
            "w_p5": pct(box_w, 5), "h_p5": pct(box_h, 5),
            "w_p95": pct(box_w, 95), "h_p95": pct(box_h, 95),
        },
        "source_images": len(sources),
        "_sources": sources,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset folder holding data.yaml and train/valid/test")
    ap.add_argument("--zip", type=Path, help="the Roboflow export zip, to record its sha256")
    ap.add_argument("--out", type=Path, help="write JSON here")
    args = ap.parse_args()

    data = yaml.safe_load((args.root / "data.yaml").read_text())
    result: dict = {"data_yaml": data, "splits": {}}
    for split in ("train", "valid", "test"):
        result["splits"][split] = split_stats(args.root, split)
    src = {s: result["splits"][s].pop("_sources") for s in result["splits"]}
    result["source_overlap_between_splits"] = {
        "train&valid": len(src["train"] & src["valid"]),
        "train&test": len(src["train"] & src["test"]),
        "valid&test": len(src["valid"] & src["test"]),
    }
    result["total_images"] = sum(s["images"] for s in result["splits"].values())
    result["total_boxes"] = sum(s["boxes"] for s in result["splits"].values())
    if args.zip:
        result["zip"] = {"name": args.zip.name, "bytes": args.zip.stat().st_size,
                         "sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest()}
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
