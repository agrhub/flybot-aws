# Bundled sample (for judges without hardware, AWS or the detector)

24 real photos from the two orchard EZVIZ lure-trap cameras, 12 consecutive photos per camera, copied unchanged
from the FlyBot export (`data/export/images/`, git-ignored):

| Camera | Window (UTC) | Size | Production count (FlyBot, Roboflow RF-DETR) |
|---|---|---|---|
| BE4089629 | 2026-10-05 04:30 .. 07:35 | 768x432, 22-24 KB | 0-6 |
| BE4111159 | 2026-10-07 04:30 .. 08:00 | 2304x1296, 142-176 KB | 0 |

- `samples_manifest.csv`: the columns of `data/manifest.csv` (data/README.md). `image_path` is relative to this
  folder, `trap_kind` is `lure` (confirmed on site; the export says `sticky`), `previous_snapshot_id` points inside
  the sample, `s3_key` is left empty (FlyBot's storage path is not needed here). Counts are FlyBot's production
  values, **not human ground truth**.
- `yolov7/<snapshot_id>.json`: the YOLOv7-tiny result of each photo, computed by `eval/replay.py`
  (`sentinel/vision/detect_yolov7.py`, OpenCV 5.0.0 `ENGINE_NEW`, letterbox 640, conf 0.25, NMS 0.45). The ONNX
  model is git-ignored, so `--box-source cached` reads these files instead of running the network.
- No person in any photo: checked by eye on contact sheets of all 24 photos and with the repo's own op 8
  (`sentinel.vision.motion.detect_motion` on each consecutive pair: no large occluder, no camera move) and YuNet
  (`detect_faces`: 0 faces on every photo). AC6527557 (faces a building) is not in the sample.

Run: `make replay-sample` (or the command in the Makefile without make). Expected on 15 Oct 2026: 24 photos,
decisions `no_op` 18 and `request_human_review` 6 (BE4089629: production 5-6 flies vs YOLOv7 0, spread above the
limit of 3), 0 extra captures, 0 proposals, trace sha256 `d2f45242e65513de…`, under 1 s of agent time.

Photos: AGRHUB's orchard cameras (FlyBot), see NOTICE. Detector trained on camos/fruit-fly-wmcqi
(Roboflow Universe, CC BY 4.0).
