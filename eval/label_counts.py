"""Count ground-truth labeling tool (T0.7b): a local web page to type the true fruit-fly count.

    uv run python eval/select_gt_sample.py                         # once: data/labels/gt_sample.csv
    uv run python eval/label_counts.py --labeler alice             # http://127.0.0.1:8765
    uv run python eval/label_counts.py --labeler alice --onnx models/yolov7-tiny/fruitfly_deploy.onnx

Shows each sample photo with pre-drawn boxes (the YOLOv7 detector run with OpenCV 5 when --onnx is
given, cached in data/labels/prelabel__<onnx stem>.json; or production boxes from a --boxes JSONL,
one line {"snapshot_id": ..., "bbox": [[x0, y0, x1, y1], ...]} in pixels or 0-10000). Production
`ai_count` is shown next to the photo. The labeler fixes the count:

- click a box to reject it (false positive), click empty image to add a dot (missed fly),
  right-click a dot to remove it; the count follows boxes kept + dots, or type it directly;
- flags: card full (too crowded to count), glare, person in view, no trap in view, staged
  (picture / test rig, not a field trap), unusable;
- Enter = save and next, N / P = next / previous, B = boxes on/off, Alt+1-6 = flags.

Saves to data/labels/counts.csv (git-ignored), one row per (snapshot_id, labeler); saving again
replaces that row. Binds to 127.0.0.1 only and serves only the sample's images.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
TOOL_VERSION = "1"
FLAGS = ["card_full", "glare", "person", "no_trap", "staged", "unusable"]
FIELDS = ["snapshot_id", "device_serial", "captured_at", "ai_count", "true_count", *FLAGS,
          "n_prelabel", "n_rejected", "n_added", "prelabel_source", "note", "labeler", "labeled_at",
          "tool_version"]


class Store:
    """counts.csv, keyed by (snapshot_id, labeler); rewritten atomically on every save."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[tuple[str, str], dict] = {}
        if path.exists():
            for r in csv.DictReader(path.open(encoding="utf-8")):
                self.rows[(r["snapshot_id"], r["labeler"])] = r

    def save(self, row: dict) -> None:
        with self.lock:
            self.rows[(row["snapshot_id"], row["labeler"])] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for r in self.rows.values():
                    w.writerow({k: r.get(k, "") for k in FIELDS})
            tmp.replace(self.path)


class Prelabels:
    """Boxes to pre-draw, as [[x, y, w, h, score], ...] in image pixels."""

    def __init__(self, onnx: Path | None, boxes_jsonl: Path | None, cache_dir: Path, resize: str):
        self.source = "none"
        self.cache: dict[str, list] = {}
        self.detector = None
        self.cache_path = None
        self.lock = threading.Lock()
        if boxes_jsonl:
            self.source = f"boxes:{boxes_jsonl.name}"
            for line in boxes_jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["snapshot_id"]] = d["bbox"]  # normalized later, per image size
            self.raw = True
        elif onnx:
            sys.path.insert(0, str(ROOT))
            from sentinel.vision.detect_yolov7 import YoloV7Detector

            self.detector = YoloV7Detector(onnx, "new", resize=resize)
            self.source = f"onnx:{onnx.name}:{resize}:{self.detector.engine}"
            self.cache_path = cache_dir / f"prelabel__{onnx.stem}__{resize}.json"
            if self.cache_path.exists():
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.raw = False
        else:
            self.raw = False

    def get(self, snapshot_id: str, image_path: Path) -> list:
        if self.raw:
            boxes = self.cache.get(snapshot_id, [])
            import cv2

            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            norm = boxes and max(max(b[2], b[3]) for b in boxes) > max(w, h)  # 0-10000 coordinates
            out = []
            for b in boxes:
                x0, y0, x1, y1 = (b[0] * w / 1e4, b[1] * h / 1e4, b[2] * w / 1e4, b[3] * h / 1e4) if norm else b[:4]
                out.append([x0, y0, x1 - x0, y1 - y0, None])
            return out
        if self.detector is None:
            return []
        with self.lock:
            if snapshot_id not in self.cache:
                import cv2

                res = self.detector.detect(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
                self.cache[snapshot_id] = [[*b["xywh"], b["score"]] for b in res["boxes"]]
                self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
            return self.cache[snapshot_id]


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Fly count labels</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--acc:#1f6feb;--line:#d8d8d2}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecece8;--mut:#a3a39c;--card:#232321;--acc:#58a6ff;--line:#3a3a37}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:8px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
main{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;padding:16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
#wrap{position:relative;background:#000;min-height:200px}
#cv{width:100%;height:auto;display:block;cursor:crosshair}
aside{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
label{display:block;margin:4px 0}
input[type=number]{font-size:28px;width:120px}
button{font-size:14px;padding:6px 12px;margin:4px 4px 0 0}
.mut{color:var(--mut)} .big{font-size:20px} kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px}
#prog{flex:1;min-width:200px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
#bar{height:100%;background:var(--acc);width:0}
</style></head><body>
<header><b>Fly count labels</b><span id="who" class="mut"></span><div id="prog"><div id="bar"></div></div>
<span id="pos"></span></header>
<main><div id="wrap"><canvas id="cv"></canvas></div>
<aside>
<div class="mut" id="meta"></div>
<p class="big">Production count: <b id="ai"></b><br><span class="mut">pre-drawn boxes: <span id="np"></span>
(kept <span id="nk"></span>, added <span id="na"></span>)</span></p>
<label>True fruit-fly count<br><input type="number" id="count" min="0" step="1"></label>
<div id="flags"></div>
<label>Note<br><input type="text" id="note" style="width:100%"></label>
<button id="save">Save and next (Enter)</button><button id="prev">Prev (P)</button><button id="next">Next (N)</button>
<button id="tog">Boxes (B)</button><button id="nextun">Next unlabeled</button>
<p class="mut" id="status"></p>
<p class="mut">Click a box: reject / restore. Click elsewhere: add a missed fly. Right-click a dot: remove.
Flags: <kbd>Alt</kbd>+<kbd>1</kbd>-<kbd>6</kbd>.</p>
</aside></main>
<script>
const FLAGS = __FLAGS__;
const FLAG_TEXT = {card_full:"card full (too crowded)",glare:"glare",person:"person in view",
  no_trap:"no trap in view",staged:"staged (picture / test rig)",unusable:"unusable (blur, fog, covered)"};
let items=[], i=0, boxes=[], rejected=new Set(), dots=[], showBoxes=true, typed=false, img=new Image(), scale=1;
const $=id=>document.getElementById(id), cv=$("cv"), ctx=cv.getContext("2d");
$("flags").innerHTML = FLAGS.map((f,k)=>`<label><input type="checkbox" id="f_${f}"> ${k+1}. ${FLAG_TEXT[f]}</label>`).join("");
function kept(){return boxes.length-rejected.size+dots.length}
function draw(){
  ctx.drawImage(img,0,0);
  const lw=Math.max(2,img.width/500);
  if(showBoxes) boxes.forEach((b,k)=>{ctx.lineWidth=lw;ctx.strokeStyle=rejected.has(k)?"rgba(160,160,160,.8)":"#ff3030";
    ctx.setLineDash(rejected.has(k)?[6,4]:[]);ctx.strokeRect(b[0],b[1],b[2],b[3]);});
  ctx.setLineDash([]);
  dots.forEach(d=>{ctx.fillStyle="#20e040";ctx.beginPath();ctx.arc(d[0],d[1],lw*3,0,7);ctx.fill();});
  $("np").textContent=boxes.length;$("nk").textContent=boxes.length-rejected.size;$("na").textContent=dots.length;
  if(!typed) $("count").value=kept();
}
let loading=false, token=0;
async function load(k){
  i=Math.max(0,Math.min(items.length-1,k)); const it=items[i], my=++token;
  loading=true; $("save").disabled=true; $("count").value=""; $("status").textContent="loading...";
  const r=await (await fetch("/api/item/"+it.snapshot_id)).json();
  if(my!==token) return;  // a newer load started meanwhile
  const lab=r.label, next=new Image();
  next.onload=()=>{ if(my!==token) return;
    img=next; boxes=r.boxes; rejected=new Set(); dots=[]; typed=false;
    cv.width=img.width;cv.height=img.height;
    if(lab){ typed=true; $("count").value=lab.true_count; } draw(); $("count").focus(); $("count").select();
    loading=false; $("save").disabled=false;
    $("status").textContent=lab?("labeled "+lab.labeled_at):"not labeled yet";};
  next.src="/img/"+it.snapshot_id;
  FLAGS.forEach(f=>$("f_"+f).checked=!!(lab&&lab[f]==="1")); $("note").value=lab?lab.note:"";
  $("ai").textContent=it.ai_count;
  $("meta").textContent=`#${it.order} · ${it.device_serial} · ${it.local_time} (${it.time_of_day}) · bucket ${it.count_bucket}`+(it.quality_issues?` · ${it.quality_issues}`:"");
  $("pos").textContent=`${i+1} / ${items.length} · labeled ${items.filter(x=>x.labeled).length}`;
  $("bar").style.width=(100*items.filter(x=>x.labeled).length/items.length)+"%";
}
function pt(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height]}
cv.addEventListener("click",e=>{const [x,y]=pt(e);typed=false;
  const k=showBoxes?boxes.findIndex(b=>x>=b[0]&&x<=b[0]+b[2]&&y>=b[1]&&y<=b[1]+b[3]):-1;
  if(k>=0){rejected.has(k)?rejected.delete(k):rejected.add(k)} else dots.push([x,y]); draw();});
cv.addEventListener("contextmenu",e=>{e.preventDefault();const [x,y]=pt(e);const r=Math.max(12,img.width/80);
  const k=dots.findIndex(d=>Math.hypot(d[0]-x,d[1]-y)<r); if(k>=0){dots.splice(k,1);typed=false;draw();}});
$("count").addEventListener("input",()=>typed=true);
async function save(){
  if(loading){$("status").textContent="still loading, wait";return}
  const v=$("count").value; if(v===""||+v<0||!Number.isInteger(+v)){$("status").textContent="type a whole number";return}
  const it=items[i], body={snapshot_id:it.snapshot_id,true_count:+v,note:$("note").value,
    n_prelabel:boxes.length,n_rejected:rejected.size,n_added:dots.length};
  FLAGS.forEach(f=>body[f]=$("f_"+f).checked?1:0);
  const r=await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(!r.ok){$("status").textContent="save failed: "+await r.text();return}
  it.labeled=true; load(i+1);
}
$("save").onclick=save;$("next").onclick=()=>load(i+1);$("prev").onclick=()=>load(i-1);
$("tog").onclick=()=>{showBoxes=!showBoxes;draw()};
$("nextun").onclick=()=>{const k=items.findIndex((x,j)=>j>i&&!x.labeled);load(k<0?items.findIndex(x=>!x.labeled):k)};
document.addEventListener("keydown",e=>{const t=e.target.id;
  if(e.key==="Enter"){e.preventDefault();save();return}
  if(e.altKey&&/^Digit[1-6]$/.test(e.code)){e.preventDefault();const c=$("f_"+FLAGS[+e.code.slice(5)-1]);c.checked=!c.checked;return}
  if(t==="note") return;
  if(e.key==="n"||e.key==="N"){load(i+1)} else if(e.key==="p"||e.key==="P"){load(i-1)}
  else if(e.key==="b"||e.key==="B"){showBoxes=!showBoxes;draw()}
  });
(async()=>{const r=await (await fetch("/api/items")).json(); items=r.items; $("who").textContent="labeler: "+r.labeler+" · pre-labels: "+r.prelabel_source;
  const k=items.findIndex(x=>!x.labeled); load(k<0?0:k);})();
</script></body></html>"""


def make_handler(sample: list[dict], by_id: dict, data_root: Path, store: Store, pre: Prelabels, labeler: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, PAGE.replace("__FLAGS__", json.dumps(FLAGS)).encode(), "text/html; charset=utf-8")
            if path == "/api/items":
                items = [{**{k: r[k] for k in ("order", "snapshot_id", "device_serial", "local_time", "time_of_day",
                                               "count_bucket", "ai_count", "quality_issues")},
                          "labeled": (r["snapshot_id"], labeler) in store.rows} for r in sample]
                return self._json({"items": items, "labeler": labeler, "prelabel_source": pre.source})
            if path.startswith("/api/item/"):
                r = by_id.get(path.rsplit("/", 1)[1])
                if not r:
                    return self._json({"error": "not in sample"}, 404)
                boxes = pre.get(r["snapshot_id"], data_root / r["image_path"])
                return self._json({"boxes": boxes, "label": store.rows.get((r["snapshot_id"], labeler))})
            if path.startswith("/img/"):
                r = by_id.get(path.rsplit("/", 1)[1])
                if not r:
                    return self._send(404, b"not in sample", "text/plain")
                f = (data_root / r["image_path"]).resolve()
                if data_root.resolve() not in f.parents or not f.is_file():
                    return self._send(404, b"no image", "text/plain")
                return self._send(200, f.read_bytes(), "image/jpeg")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
                r = by_id[d["snapshot_id"]]
                count = int(d["true_count"])
                if count < 0:
                    raise ValueError("negative count")
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                return self._send(400, f"bad request: {exc}".encode(), "text/plain")
            store.save({
                "snapshot_id": r["snapshot_id"], "device_serial": r["device_serial"], "captured_at": r["captured_at"],
                "ai_count": r["ai_count"], "true_count": count,
                **{f: 1 if d.get(f) else 0 for f in FLAGS},
                "n_prelabel": int(d.get("n_prelabel", 0)), "n_rejected": int(d.get("n_rejected", 0)),
                "n_added": int(d.get("n_added", 0)), "prelabel_source": pre.source,
                "note": str(d.get("note", ""))[:500].replace("\n", " "), "labeler": labeler,
                "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"), "tool_version": TOOL_VERSION,
            })
            return self._json({"ok": True})

    return H


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeler", required=True, help="your name or initials (stored with every label)")
    ap.add_argument("--sample", type=Path, default=ROOT / "data" / "labels" / "gt_sample.csv")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "labels" / "counts.csv")
    ap.add_argument("--onnx", type=Path, default=None, help="pre-label with this YOLOv7 ONNX (OpenCV 5)")
    ap.add_argument("--resize", default="stretch", choices=["stretch", "letterbox"])
    ap.add_argument("--boxes", type=Path, default=None, help="production boxes JSONL instead of --onnx")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if not args.sample.exists():
        raise SystemExit(f"{args.sample} missing: run eval/select_gt_sample.py first")
    sample = sorted(csv.DictReader(args.sample.open(encoding="utf-8")), key=lambda r: int(r["order"]))
    by_id = {r["snapshot_id"]: r for r in sample}
    store = Store(args.out)
    pre = Prelabels(args.onnx, args.boxes, args.out.parent, args.resize)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sample, by_id, args.data_root, store, pre, args.labeler))
    print(f"labeling {len(sample)} photos as '{args.labeler}' -> {args.out}\nopen http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
