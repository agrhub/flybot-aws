"""Convert a YOLOv7-tiny checkpoint (WongKinYiu/yolov7, e.g. the AMB82 `fruitfly_deploy.pt`) to ONNX
for OpenCV 5 DNN.

Runs in the training env (torch) with the official yolov7 source on sys.path (the checkpoint pickles
yolov7's own `models.yolo.Model`, so the source must be importable):

    cd models
    uv run python export_yolov7.py yolov7-tiny/fruitfly_deploy.pt            # -> yolov7-tiny/fruitfly_deploy.onnx
    uv run python export_yolov7.py yolov7-tiny/fruitfly_deploy.pt --end2end  # + NMS inside the graph

First run clones https://github.com/WongKinYiu/yolov7 at YOLOV7_COMMIT into models/third_party/yolov7
(git-ignored, GPL-3.0) and installs the few pure-python deps it imports that the train env lacks
(pandas, seaborn, tqdm, scipy) into models/third_party/pydeps with `uv pip install --target`, so the
train env's uv.lock is untouched.

Default graph (what the repo's detector uses): Detect head decoded in the graph ("--grid" in yolov7's
export.py), concatenated, **no NMS**: input `images` [1, 3, S, S] float32 RGB /255 letterboxed (pad
114), output `output` [1, N, 5 + nc] = cx, cy, w, h (input pixels), objectness, class scores
(N = 25200 at S = 640). Score = obj * cls, NMS with cv2.dnn.NMSBoxes (sentinel/vision/detect_yolov7.py).

`--end2end`: yolov7's End2End wrapper with its onnxruntime NMS (`max_wh` set), output
`[n, 7]` = batch, x0, y0, x1, y1, class, score. Kept only to test what OpenCV can load.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
THIRD_PARTY = MODELS_DIR / "third_party"
YOLOV7_DIR = THIRD_PARTY / "yolov7"
PYDEPS_DIR = THIRD_PARTY / "pydeps"
YOLOV7_REPO = "https://github.com/WongKinYiu/yolov7.git"
YOLOV7_COMMIT = "a207844b1ce82d204ab36d87d496728d3d2348e7"  # main on 2026-10-15
# Imported by yolov7's models/ and utils/ but not needed by the train env itself.
PYDEPS = ["pandas", "seaborn", "tqdm", "scipy", "pytz", "tzdata"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_yolov7(src: Path) -> str:
    """Clone yolov7 at the pinned commit if `src` is missing; return the checked-out commit."""
    if not (src / "models" / "yolo.py").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", YOLOV7_REPO, str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", "-q", YOLOV7_COMMIT], check=True)
    patch_numpy2(src)
    out = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def patch_numpy2(src: Path) -> int:
    """NumPy 2 dropped np.int/np.float/np.bool and np.trapz, which yolov7 still uses (camos Colab patch + trapz)."""
    import re

    changed = 0
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new = re.sub(r"\bnp\.int\b(?![\d_a-z])", "int", text)
        new = re.sub(r"\bnp\.float\b(?![\d_a-z])", "float", new)
        new = re.sub(r"\bnp\.bool\b(?![\d_a-z])", "bool", new)
        new = re.sub(r"\bnp\.trapz\(", "np.trapezoid(", new)  # removed in NumPy 2.x (utils/metrics.py)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    return changed


def ensure_pydeps() -> None:
    missing = [m for m in ("pandas", "seaborn", "tqdm", "scipy") if importlib.util.find_spec(m) is None]
    if missing and not (PYDEPS_DIR / "seaborn").exists():
        uv = shutil.which("uv") or "uv"
        subprocess.run([uv, "pip", "install", "--python", sys.executable, "--target", str(PYDEPS_DIR),
                        "--no-deps", *PYDEPS], check=True)
    # Appended, not prepended: the env's own numpy/torch must win over anything in pydeps.
    sys.path.append(str(PYDEPS_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path, help="YOLOv7 .pt (training best.pt or reparameterized deploy .pt)")
    ap.add_argument("--imgsz", type=int, default=640, help="static square input size (default 640)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--end2end", action="store_true", help="NMS in the graph (onnxruntime NMS variant)")
    ap.add_argument("--max-det", type=int, default=300, help="--end2end: max boxes kept")
    ap.add_argument("--iou", type=float, default=0.45, help="--end2end: NMS IoU")
    ap.add_argument("--conf", type=float, default=0.25, help="--end2end: score threshold")
    ap.add_argument("--no-simplify", action="store_true", help="skip onnxslim")
    ap.add_argument("--out", type=Path, default=None, help="default <weights stem>[-end2end].onnx next to weights")
    ap.add_argument("--yolov7-dir", type=Path, default=YOLOV7_DIR)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    weights = args.weights.resolve()
    commit = ensure_yolov7(args.yolov7_dir.resolve())
    ensure_pydeps()
    sys.path.insert(0, str(args.yolov7_dir.resolve()))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import models.common  # yolov7's
    import torch
    from torch import nn
    from utils.activations import Hardswish, SiLU  # yolov7's export-friendly activations

    t0 = time.time()
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    model = ckpt if not isinstance(ckpt, dict) else (ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"])
    model = model.float().eval()
    if hasattr(model, "fuse"):
        model = model.fuse()  # Conv+BN fusion; no-op on an already reparameterized deploy model
    for m in model.modules():
        if isinstance(m, models.common.Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    head = model.model[-1]
    head_type = type(head).__name__
    head.export = False  # decode the grid in the graph (yolov7 export.py --grid)
    head.concat = True  # one output tensor [1, N, 5 + nc] instead of (z, raw maps)
    names = getattr(model, "names", None) or [str(i) for i in range(head.nc)]
    stride = [int(s) for s in model.stride.tolist()]
    if args.imgsz % max(stride):
        raise SystemExit(f"--imgsz {args.imgsz} is not a multiple of the max stride {max(stride)}")

    img = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        y = model(img)
    n_anchors = int(y.shape[1])

    export_model = model
    output_names = ["output"]
    if args.end2end:
        from models.experimental import End2End

        head.end2end = True
        export_model = End2End(model, args.max_det, args.iou, args.conf, max_wh=args.imgsz, device="cpu",
                               n_classes=head.nc)

    suffix = "-end2end" if args.end2end else ""
    out = (args.out or weights.with_name(f"{weights.stem}{suffix}.onnx")).resolve()
    kwargs = {"dynamo": False}  # the TorchScript exporter: yolov7 code is not dynamo-traceable as is
    torch.onnx.export(export_model, img, str(out), opset_version=args.opset, input_names=["images"],
                      output_names=output_names, do_constant_folding=True, **kwargs)

    import onnx

    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    simplified = False
    if not args.no_simplify:
        try:
            import onnxslim

            m = onnxslim.slim(m)
            onnx.save(m, str(out))
            simplified = True
        except Exception as exc:  # noqa: BLE001 - keep the unsimplified graph, record why
            print(f"[export] onnxslim failed ({type(exc).__name__}: {exc}); keeping the raw graph")

    def dims(v):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]

    anchors = head.anchor_grid.reshape(head.nl, -1).tolist() if hasattr(head, "anchor_grid") else None
    meta = {
        "onnx": out.name,
        "sha256": sha256(out),
        "bytes": out.stat().st_size,
        "family": "yolov7",
        "from_weights": str(weights),
        "weights_sha256": sha256(weights),
        "yolov7_commit": commit,
        "torch": torch.__version__,
        "head": head_type,
        "nc": int(head.nc),
        "names": list(names) if not isinstance(names, dict) else [names[k] for k in sorted(names)],
        "stride": stride,
        "anchors_px": anchors,
        "opset": next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
        "imgsz": args.imgsz,
        "dynamic": False,
        "simplify": simplified,
        "nms_in_graph": bool(args.end2end),
        "inputs": {i.name: dims(i) for i in m.graph.input},
        "outputs": {o.name: dims(o) for o in m.graph.output},
        "anchors_total": n_anchors,
        "preprocess": "letterbox to imgsz (pad 114), BGR->RGB, /255, NCHW float32",
        "output_layout": ("[n, 7]: batch, x0, y0, x1, y1, class, score (NMS done)" if args.end2end else
                          "[1, N, 5+nc]: cx, cy, w, h (input pixels), objectness, class scores; "
                          "score = obj*cls; NMS not included"),
        "license": "GPL-3.0 (WongKinYiu/yolov7 code and weights derived from yolov7-tiny.pt)",
        "export_seconds": round(time.time() - t0, 1),
    }
    out.with_suffix(".onnx.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("onnx", "bytes", "head", "nc", "names", "opset", "inputs",
                                            "outputs", "nms_in_graph", "simplify")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
