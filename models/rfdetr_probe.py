"""RF-DETR (Apache-2.0) export + OpenCV 5 DNN load probe (T0.9 license alternative).

Two steps, two environments:

    # 1. export the COCO-pretrained checkpoint to ONNX (train env: torch; rfdetr installed with
    #    `uv pip install --target third_party/rfdetr-deps --no-deps rfdetr==1.11.2 ...`, see LICENSE_OPTIONS.md)
    cd models && uv run python rfdetr_probe.py export --size nano
    # 2. load + forward with OpenCV 5 (main project env), new and classic engines
    uv run python models/rfdetr_probe.py load models/rfdetr/rfdetr-nano-coco.onnx IMAGE

Writes models/rfdetr/<name>.onnx (git-ignored) and appends the result of each step to
models/rfdetr_probe.json (errors recorded verbatim). Pretrained COCO weights only: this checks the
graph, not fruit-fly accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
OUT_DIR = MODELS_DIR / "rfdetr"
LOG = MODELS_DIR / "rfdetr_probe.json"
DEPS = MODELS_DIR / "third_party" / "rfdetr-deps"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def record(entry: dict) -> None:
    log = json.loads(LOG.read_text()) if LOG.exists() else []
    log.append(entry)
    LOG.write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(entry, indent=2))


def export(size: str, opset: int) -> int:
    sys.path.append(str(DEPS))
    sys.path.append(str(MODELS_DIR / "third_party" / "pydeps"))  # tqdm, scipy, pandas (export_yolov7.py)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RF_HOME", str(OUT_DIR / "weights"))  # checkpoint download stays on this drive
    entry = {"step": "export", "size": size, "opset": opset, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import rfdetr
        import torch

        entry.update(rfdetr=getattr(rfdetr, "__version__", "?"), torch=torch.__version__)
        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall}[size]
        t0 = time.time()
        model = cls()
        entry["resolution"] = int(model.model_config.resolution)
        produced = Path(model.export(output_dir=str(OUT_DIR), opset_version=opset, verbose=False))
        onnx_path = produced if produced.suffix == ".onnx" else next(OUT_DIR.glob("*.onnx"))
        final = OUT_DIR / f"rfdetr-{size}-coco.onnx"
        if onnx_path != final:
            onnx_path.replace(final)
        import onnx

        m = onnx.load(str(final))

        def dims(v):
            return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]

        entry.update(ok=True, onnx=str(final.relative_to(MODELS_DIR)), bytes=final.stat().st_size,
                     inputs={i.name: dims(i) for i in m.graph.input},
                     outputs={o.name: dims(o) for o in m.graph.output},
                     ops=sorted({n.op_type for n in m.graph.node}), seconds=round(time.time() - t0, 1))
    except Exception as exc:  # noqa: BLE001 - any failure is the probe result, recorded verbatim
        entry.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-3000:])
    record(entry)
    return 0 if entry.get("ok") else 1


def load(onnx_path: Path, image: Path | None, conf: float) -> int:
    import cv2
    import numpy as np

    sys.path.insert(0, str(MODELS_DIR.parent))
    from sentinel import require_opencv5
    from sentinel.vision.detect_smoke import engine_of

    cvv = require_opencv5()
    meta = json.loads(LOG.read_text()) if LOG.exists() else []
    res = next((e.get("resolution") for e in reversed(meta) if e.get("onnx") and Path(e["onnx"]).name == onnx_path.name), None)
    size = int(res or 384)
    img = cv2.imread(str(image)) if image else np.full((size, size, 3), 127, np.uint8)
    rgb = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    outs = {}
    for engine, flag in (("new", cv2.dnn.ENGINE_NEW), ("classic", cv2.dnn.ENGINE_CLASSIC)):
        entry = {"step": "load", "onnx": str(onnx_path), "engine_requested": engine, "opencv": cvv,
                 "image": str(image) if image else None, "resolution": size}
        try:
            t0 = time.perf_counter()
            net = cv2.dnn.readNet(str(onnx_path), "", "", flag)
            entry["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            names = list(net.getUnconnectedOutLayersNames())
            net.setInput(blob)
            t0 = time.perf_counter()
            res = net.forward(names)
            entry["forward_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            entry["engine_used"] = engine_of(net)
            entry["outputs"] = {n: list(r.shape) for n, r in zip(names, res)}
            outs[engine] = dict(zip(names, res))
            logits = next((r for r in res if r.shape[-1] > 4), None)
            if logits is not None:
                p = 1 / (1 + np.exp(-logits[0]))
                entry["queries_above_conf"] = int((p.max(axis=1) >= conf).sum())
                entry["top_classes"] = [int(c) for c in p.max(axis=0).argsort()[::-1][:3]]
            entry["finite"] = bool(all(np.isfinite(r).all() for r in res))
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001 - recorded verbatim
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        record(entry)
    if len(outs) == 2:
        diffs = {n: float(np.abs(outs["new"][n] - outs["classic"][n]).max()) for n in outs["new"] if n in outs["classic"]}
        record({"step": "parity", "onnx": str(onnx_path), "max_abs_diff_new_vs_classic": diffs})
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--size", choices=["nano", "small"], default="nano")
    e.add_argument("--opset", type=int, default=17)
    lo = sub.add_parser("load")
    lo.add_argument("onnx", type=Path)
    lo.add_argument("image", type=Path, nargs="?")
    lo.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args(argv)
    return export(a.size, a.opset) if a.cmd == "export" else load(a.onnx, a.image, a.conf)


if __name__ == "__main__":
    sys.exit(main())
