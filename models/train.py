"""Train the fruit-fly detector (YOLO11n, YOLOv8n fallback) and export ONNX for OpenCV 5 DNN.

Runs in the training env under models/ (its own uv project; the main project never installs torch):

    cd models
    uv sync
    # CPU smoke run (pipeline check only, numbers are NOT model quality):
    uv run python train.py --data ../../camos-ai-camera/models/fruit-fly.v3i.yolov8.zip \
        --epochs 1 --imgsz 320 --fraction 0.05 --device cpu --name smoke-yolo11n --export-onnx
    # full run on a GPU (Colab T4 / AWS):
    uv run python train.py --data fruit-fly.v3i.yolov8.zip --epochs 80 --imgsz 640 --device 0 \
        --name fruitfly-yolo11n --export-onnx

--data takes the Roboflow YOLOv8 zip, a dataset folder, or a data.yaml. The dataset is staged
(copied/extracted) into models/datasets/<stem>/ with absolute paths in a new data.yaml, so the
source folder is never written to (ultralytics writes labels.cache next to the labels).
Staging also rewrites polygon label rows as their bounding box (see models/dataset.md: 3 train
files mix box and polygon rows, which ultralytics 8.4 rejects as corrupt and would drop).

Outputs: models/runs/<name>/ (weights, results.csv, plots, test_metrics.json, run_info.json) and,
with --export-onnx, models/<name>.onnx plus models/<name>.onnx.json (metadata). All git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
RUNS_DIR = MODELS_DIR / "runs"
DATASETS_DIR = MODELS_DIR / "datasets"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Keep ultralytics' settings file inside models/runs (git-ignored), not in the user's profile,
# and turn off its usage analytics. Must be set before ultralytics is imported.
# ultralytics falls back to /tmp when the folder's parent is missing, so create it first.
if "YOLO_CONFIG_DIR" not in os.environ:
    (RUNS_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(RUNS_DIR / ".ultralytics")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="Roboflow YOLOv8 zip, dataset folder, or data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights (default yolo11n.pt)")
    ap.add_argument("--fallback-model", default="yolov8n.pt",
                    help="used when --model cannot be loaded (default yolov8n.pt); '' disables")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640, help="training image size (dataset is 640x640)")
    ap.add_argument("--batch", type=int, default=16, help="batch size; -1 = ultralytics auto-batch (GPU)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the train split to use")
    ap.add_argument("--device", default=None, help="'cpu', '0', '0,1' ... (default: 0 if CUDA else cpu)")
    ap.add_argument("--workers", type=int, default=None, help="dataloader workers (default: 2 cpu, 8 gpu)")
    ap.add_argument("--patience", type=int, default=30, help="early-stopping patience in epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="false", choices=["false", "ram", "disk"], help="image cache")
    ap.add_argument("--name", default=None, help="run name (default: <model stem>-e<epochs>-<imgsz>)")
    ap.add_argument("--project", type=Path, default=RUNS_DIR, help="runs folder (default models/runs)")
    ap.add_argument("--skip-test-eval", action="store_true", help="do not evaluate best.pt on the test split")
    ap.add_argument("--export-onnx", action="store_true", help="export best.pt to models/<name>.onnx")
    ap.add_argument("--export-only", type=Path, default=None, metavar="WEIGHTS",
                    help="skip training; export this .pt (needs --name for the output file)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--export-imgsz", type=int, default=640, help="ONNX input size (static by default)")
    ap.add_argument("--dynamic", action="store_true", help="dynamic batch/height/width (default static)")
    ap.add_argument("--no-simplify", action="store_true", help="skip onnxslim simplification")
    ap.add_argument("--onnx-out", type=Path, default=None, help="default models/<name>.onnx")
    args = ap.parse_args(argv)
    if args.export_only is None and args.data is None:
        ap.error("--data is required unless --export-only is given")
    if args.export_only is not None and not args.name:
        ap.error("--export-only needs --name")
    if not 0 < args.fraction <= 1:
        ap.error("--fraction must be in (0, 1]")
    return args


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def polygon_rows_to_boxes(label_dir: Path) -> dict:
    """Rewrite `cls x1 y1 x2 y2 ...` polygon rows as `cls cx cy w h` (their bounding box)."""
    files = rows = 0
    for txt in label_dir.glob("*.txt"):
        lines = txt.read_text().splitlines()
        out, changed = [], False
        for line in lines:
            parts = line.split()
            if len(parts) >= 7 and len(parts) % 2 == 1:
                xs = [float(v) for v in parts[1::2]]
                ys = [float(v) for v in parts[2::2]]
                x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
                y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
                out.append(f"{parts[0]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
                rows += 1
                changed = True
            else:
                out.append(line)
        if changed:
            txt.write_text("\n".join(out) + "\n")
            files += 1
    return {"files_rewritten": files, "polygon_rows_to_boxes": rows}


def stage_dataset(data: Path) -> tuple[Path, dict]:
    """Copy/extract the dataset to models/datasets/<stem>/ and write a data.yaml with absolute paths."""
    import yaml

    data = data.resolve()
    if data.is_file() and data.suffix == ".zip":
        stem = data.name.removesuffix(".zip")
        src_root = None
    elif data.is_file() and data.suffix in {".yaml", ".yml"}:
        src_root, stem = data.parent, data.parent.name
    elif data.is_dir():
        src_root, stem = data, data.name
    else:
        raise SystemExit(f"--data: not a zip, folder or yaml: {data}")

    dst = DATASETS_DIR / stem
    marker = dst / ".staged.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        print(f"[stage] reusing {dst}")
        return dst / "data.yaml", info
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    if src_root is None:
        with zipfile.ZipFile(data) as zf:
            zf.extractall(dst)
        source = {"zip": str(data), "zip_sha256": sha256(data)}
    else:
        for split in ("train", "valid", "test"):
            if (src_root / split).is_dir():
                shutil.copytree(src_root / split, dst / split)
        for extra in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
            if (src_root / extra).exists():
                shutil.copy2(src_root / extra, dst / extra)
        source = {"folder": str(src_root)}

    meta = yaml.safe_load((dst / "data.yaml").read_text()) if (dst / "data.yaml").exists() else {}
    names = meta.get("names", ["fruit_fly"])
    fixed = {split: polygon_rows_to_boxes(dst / split / "labels")
             for split in ("train", "valid", "test") if (dst / split / "labels").is_dir()}
    counts = {split: len([p for p in (dst / split / "images").iterdir() if p.suffix.lower() in IMG_EXT])
              for split in ("train", "valid", "test") if (dst / split / "images").is_dir()}
    staged = {
        "path": dst.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images" if "test" in counts else None,
        "nc": len(names),
        "names": names,
    }
    if staged["test"] is None:
        staged.pop("test")
    if "roboflow" in meta:
        staged["roboflow"] = meta["roboflow"]
    (dst / "data.yaml").write_text(yaml.safe_dump(staged, sort_keys=False))
    info = {"source": source, "images": counts, "label_fix": fixed, "names": names}
    marker.write_text(json.dumps(info, indent=2))
    print(f"[stage] {dst}: {counts}; label fix: {fixed}")
    return dst / "data.yaml", info


def pick_device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def load_model(model: str, fallback: str):
    from ultralytics import YOLO

    try:
        return YOLO(model), model
    except Exception as exc:  # any load or download failure triggers the documented fallback
        if not fallback:
            raise
        print(f"[model] could not load {model} ({type(exc).__name__}: {exc}); falling back to {fallback}")
        return YOLO(fallback), fallback


def export_onnx(weights: Path, args: argparse.Namespace, name: str, extra: dict) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    t0 = time.time()
    produced = Path(model.export(format="onnx", opset=args.opset, imgsz=args.export_imgsz,
                                 dynamic=args.dynamic, simplify=not args.no_simplify, nms=False,
                                 batch=1, device="cpu"))
    out = (args.onnx_out or MODELS_DIR / f"{name}.onnx").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out)

    import onnx

    m = onnx.load(str(out))

    def dims(v):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]

    meta = {
        "onnx": out.name,
        "sha256": sha256(out),
        "bytes": out.stat().st_size,
        "from_weights": str(weights),
        "weights_sha256": sha256(weights),
        "opset": next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
        "imgsz": args.export_imgsz,
        "dynamic": args.dynamic,
        "simplify": not args.no_simplify,
        "nms_in_graph": False,
        "inputs": {i.name: dims(i) for i in m.graph.input},
        "outputs": {o.name: dims(o) for o in m.graph.output},
        "names": dict(model.names),
        "preprocess": "letterbox to imgsz (pad 114), BGR->RGB, /255, NCHW float32",
        "output_layout": "[1, 4+nc, anchors]: cx, cy, w, h (input pixels), class scores; NMS not included",
        "export_seconds": round(time.time() - t0, 1),
        **extra,
    }
    out.with_suffix(".onnx.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[export] {out} ({meta['bytes']} bytes) inputs={meta['inputs']} outputs={meta['outputs']}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    import ultralytics
    from ultralytics import settings

    settings.update({"sync": False})  # no usage analytics from this repo's runs

    if args.export_only is not None:
        export_onnx(args.export_only.resolve(), args, args.name, {"smoke_run": None})
        return 0

    data_yaml, stage_info = stage_dataset(args.data)
    device = pick_device(args.device)
    workers = args.workers if args.workers is not None else (2 if device == "cpu" else 8)
    model, used = load_model(args.model, args.fallback_model)
    name = args.name or f"{Path(used).stem}-e{args.epochs}-{args.imgsz}"
    cache = {"false": False, "ram": "ram", "disk": "disk"}[args.cache]

    t_train = time.time()
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                fraction=args.fraction, device=device, workers=workers, patience=args.patience,
                seed=args.seed, deterministic=True, cache=cache, project=str(args.project.resolve()),
                name=name, exist_ok=True, plots=True, verbose=True)
    train_seconds = time.time() - t_train
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"

    smoke = args.epochs <= 3 or args.fraction < 0.5
    info = {
        "smoke_run": smoke,
        "note": "SMOKE RUN: pipeline check only, metrics are not model quality" if smoke else "",
        "model_requested": args.model,
        "model_used": used,
        "ultralytics": ultralytics.__version__,
        "device": device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "fraction": args.fraction,
        "seed": args.seed,
        "dataset": stage_info,
        "train_seconds": round(train_seconds, 1),
        "best_weights": str(best),
    }

    if not args.skip_test_eval and "test" in stage_info["images"]:
        t_val = time.time()
        metrics = evaluate_test(best, data_yaml, args.imgsz, device, run_dir)
        metrics["smoke_run"] = smoke
        metrics["eval_seconds"] = round(time.time() - t_val, 1)
        (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        info["test_metrics"] = metrics
        print(f"[test] {metrics}")

    if args.export_onnx:
        onnx_path = export_onnx(best, args, name, {"smoke_run": smoke, "run_dir": str(run_dir)})
        info["onnx"] = str(onnx_path)

    info["total_seconds"] = round(time.time() - t_start, 1)
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"[done] {run_dir} in {info['total_seconds']} s")
    return 0


def evaluate_test(weights: Path, data_yaml: Path, imgsz: int, device: str, run_dir: Path) -> dict:
    """mAP of `weights` on the test split (ultralytics val, conf 0.001, iou 0.7)."""
    from ultralytics import YOLO

    r = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=imgsz, device=device, batch=8,
                               project=str(run_dir), name="test", exist_ok=True, plots=False)
    return {
        "split": "test",
        "weights": str(weights),
        "imgsz": imgsz,
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
    }


if __name__ == "__main__":
    sys.exit(main())
