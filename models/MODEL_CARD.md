# Model card: FlyBot fruit-fly detector

**Primary detector = YOLO26n** (Ultralytics, NMS-free end-to-end head) trained on Colab
(`models/colab_train_yolo26.ipynb`) and running on OpenCV 5 DNN, new engine (`DETECTOR=yolo26`). **YOLOv7-tiny**
(the AI camera's AMB82 NPU model) stays as the baseline and the edge model; its section follows. Every number cites its
source file; "not measured" means not measured.

## YOLO26n (primary)

### Deployed since 2026-10-29: v4, `fruit_fly` + `other_insect`

Trained with the same notebook on camos/fruit-fly-wmcqi **v4** (CC BY 4.0): the v3 images after a review of the tiny
boxes (29 Oct: 391 ants relabelled `other_insect`, 569 debris or wing boxes removed; same export settings and
splits as v3, plus 42 empty BE4111159 photos). The Lambda counts `fruit_fly` only; `other_insect` boxes come back
under `others` (`sentinel/vision/detect_yolo26.py`).

| Item | Value | Source |
|---|---|---|
| ONNX | `fruitfly_yolo26n.onnx`, 9,865,025 bytes, sha256 `b900161002d3b04e866acf616e17602f1519d98f496414abc34d0d23ac4addd6`, 2 classes, same graph shape as below | `eval/results/yolo26n_v4_onnx_sidecar.json` |
| Training | Colab T4, Ultralytics 8.4.177, 60 epochs, 0.384 h | `eval/results/yolo26n_v4_train_results.csv` |
| mAP@50 / mAP@50-95, test split (80 images) | `fruit_fly` **0.964** / 0.713; `other_insect` 0.604 / 0.311 (29 ants on 8 images) | `eval/results/yolo26n_v4_test_metrics.json` |
| v3 vs v4 through the Lambda's detector and box rules, v4 test split | `fruit_fly` F1 0.897 -> **0.921** (false boxes 264 -> 151), count MAE per photo 2.04 -> 1.80, ants counted as fruit flies 18 of 29 -> **0**, flies on the 4 empty BE4111159 photos 2 -> 0; valid split F1 0.952 -> 0.956 | `eval/results/yolo26_v3_vs_v4.json` |
| OpenCV 5 vs ONNX Runtime | Colab: 20 test images, both graphs ok; laptop: 20 sample photos, 0 mismatches | `eval/results/yolo26n_v4_opencv_check_colab.json`, `eval/results/yolo26_v4_parity.json` |
| 24 sample photos through the Lambda handler | 24 ok, 0 flies, MAE 1.54 vs production (as v3) | `eval/results/replay_sample_yolo26_v4.json` |

Both models are scored on v4's cleaned labels, which favours v4 slightly: the removed debris boxes and the ants
now count against v3 (at most 57 of v3's 113 extra false boxes). The table below is the v3 model (2026-10-16).

Where the weights are: v4 at `models/yolo26/` in the public model bucket (`YOLO26_URL` of the Lambda), v3 kept for a
rollback at `models/yolo26-v3/` (`.onnx`, `.onnx.json` and `-o2m.onnx`, uploaded 2026-10-31). The bucket has no
versioning, so a new upload to `models/yolo26/` replaces the file there. The deploy bakes the local
`models/yolo26/fruitfly_yolo26n.onnx` (+ `.onnx.json`) into the Lambda image and uploads it to `models/yolo26/`. To roll
back, put the v3 files there (download them from `models/yolo26-v3/`, sha256
`fb4da5c0512088b32f447fc8fa14bd0fd4ed164768ed57d5690617ca2be7f176`) and run `./deploy.sh`. The weights are kept out of
git: each retrain adds about 10 MB per file, and git would keep every version forever.

| Item | Value | Source |
|---|---|---|
| Architecture | YOLO26n, 2,389,735 parameters, 5.4 GFLOPs, 1 class `fruit_fly` | training log (`eval/results/yolo26n_train_results.csv`) |
| Training | Colab T4, Ultralytics 8.4.177, from `yolo26n.pt`, 60 epochs, 640, batch 32, AdamW auto, 0.387 h | notebook output |
| ONNX | `fruitfly_yolo26n.onnx`, 9,863,554 bytes, sha256 `fb4da5c0512088b32f447fc8fa14bd0fd4ed164768ed57d5690617ca2be7f176`, opset 17, static 640, output `[1, 300, 6]` (x1, y1, x2, y2, score, class), no NMS in the graph (the Lambda applies NMS and box rules, see Post-processing) | `eval/results/yolo26n_onnx_sidecar.json` |
| Fallback graph | `fruitfly_yolo26n-o2m.onnx`, output `[1, 5, 8400]` (one-to-many head); not needed | same |
| mAP@50 (test split, 76 images, 2,349 boxes) | **0.953** (YOLOv7-tiny: 0.958) | `eval/results/yolo26n_test_metrics.json` |
| mAP@50-95 | **0.700** (YOLOv7-tiny: 0.67) | same |
| Precision / recall | 0.913 / 0.898 (YOLOv7-tiny: 0.928 / 0.908) | same |
| OpenCV 5 (new engine) vs ONNX Runtime, Colab | 20 test images: same detections at conf 0.25 (IoU 1.0, score diff < 4e-5, count diff 0) for both graphs | `eval/results/yolo26n_opencv_parity_colab.json` |
| OpenCV 5 vs ONNX Runtime, laptop | 20 FlyBot sample photos: 0 mismatches, worst box diff 0.0 px | `eval/results/yolo26_parity.json` |
| 24 sample photos through the Lambda handler (laptop CPU) | 0 flies, MAE 1.54 vs production RF-DETR with the box rules (before them: 15 flies, all of them the trap's lens on BE4111159, MAE 2.17, like YOLOv7-tiny's 15); DNN p50 116.0 ms on a laptop, 21 Oct (YOLOv7-tiny 189.1 ms in an earlier run), not a benchmark | `eval/results/replay_sample_yolo26.json`, `replay_sample_yolov7.json`, `yolo26_box_rules.json` |

The raw-tensor check in the notebook reported "mismatch" (max abs diff 421): the end-to-end head's TopK orders
tied low-score rows differently between engines. Detections above the threshold are identical, so the check now
compares detections, not raw rows. The sample photos are mostly empty lure traps; field accuracy is not measured.

### Post-processing (Lambda)

`sentinel/vision/detect_yolo26.py` + `sentinel/vision/box_rules.py`, the same for both heads:

1. rows with score >= `DETECTOR_CONF` (0.25);
2. class-agnostic NMS (`cv2.dnn.NMSBoxes`) at `DETECTOR_IOU` (0.45). Until this change the end-to-end head had no NMS
   in the Lambda; the one-to-one head still left 173 (valid) and 117 (test) pairs of boxes at IoU >= 0.5;
3. box rules that reject what cannot be one fruit fly, without retraining. Side = sqrt(w h) / sqrt(W H) of the photo;
   the "median" rules use the photo's remaining boxes when there are at least 3.

| Rule (variable, default) | Evidence (`eval/results/yolo26_box_rules.json`) |
|---|---|
| `DETECTOR_IOU` 0.45 | labelled flies overlapping a neighbour at IoU >= 0.45: 26 of 73,980 (train), 4 of 2,838 (valid), 2 of 2,349 (test) |
| `DETECTOR_MIN_SCORE` 0.35 | valid F1 after NMS by score limit: 0.25 0.9523, 0.30 0.9527, **0.35 0.9526**, 0.40 0.9500 (0.35: higher precision for the same F1); 2.4 % (valid) and 6.2 % (test) of the true positives score below it. The owner's ESP32 reports of the lens at 26-36 % fall under it up to 0.35; the lens size on ESP32 photos is not measured |
| `DETECTOR_MAX_SIDE` 0.12 | 167 of 73,980 train flies are larger (close-up pictures; p99.9 0.1326); the lens of the lure trap on the 12 BE4111159 sample photos measures 0.149-0.158 (scores 0.43-0.66). A lizard is many times a fly |
| `DETECTOR_MIN_SIDE` 0.004 | smallest labelled fly 0.0086 (train) |
| `DETECTOR_MAX_MEDIAN_RATIO` 3.0 | 11 of 73,834 train flies are above 3x their photo's median side |
| `DETECTOR_MIN_MEDIAN_RATIO` 0.25 | 24 of 73,834 train flies are below |
| `DETECTOR_MAX_ASPECT` 4.0 | 165 of 73,980 train flies are longer than 4:1; valid F1 by limit: 3.0 0.9497, **4.0 0.9526**, 5.0 0.9513, none 0.9509 |
| `DETECTOR_MEDIAN_MIN_BOXES` 3, `DETECTOR_BOX_RULES` on | `off` keeps steps 1-2 only |

The score and aspect limits were chosen on the valid split; the test split is held out. OpenCV 5.0.0, new engine,
letterbox 640, IoU 0.5 matching (`eval/detector_box_rules.py`):

| Split | Post-processing | Precision | Recall | F1 |
|---|---|---|---|---|
| valid (93 images, 2,838 flies) | before (score >= 0.25, no NMS) | 0.8862 | 0.9690 | 0.9258 |
| | NMS 0.45 | 0.9352 | 0.9658 | 0.9503 |
| | NMS + rules | **0.9528** | 0.9524 | **0.9526** |
| test (76 images, 2,349 flies) | before | 0.8376 | 0.9242 | 0.8788 |
| | NMS 0.45 | 0.8769 | 0.9221 | 0.8989 |
| | NMS + rules | **0.9025** | 0.8987 | **0.9006** |
| 24 bundled lure-trap photos | before: 15 flies (BE4111159: 15, production 0), MAE 2.167 vs production | | | |
| | NMS + rules: 0 flies, MAE 1.542; removed: 12 lens boxes (too large), 3 boxes at the top edge (score 0.26-0.31) | | | |

These numbers come from the OpenCV path the Lambda runs, so they differ from the Ultralytics validation above
(precision 0.913 / recall 0.898 on test, its own decoding and threshold). Not done: suppressing a box that stays at
the same place across a camera's photos. The contract's `history` carries counts only, and in a lure trap dead
flies stay where they fell, so a persistence rule would remove real catches; the diff block already reports new
and persisted boxes. Limitation seen on the same photos: the flies on the lid of a full trap view are small specks
once the frame is letterboxed to 640 (size not measured) and are not detected (BE4089629: production 37 flies over
12 photos, YOLO26 0 before and after); the rules remove false boxes, they add no recall.

## YOLOv7-tiny (baseline, AMB82 NPU edge model)

## Model

| Item | Value |
|---|---|
| Task | object detection, 1 class `fruit_fly` (id 0) |
| Architecture | **YOLOv7-tiny**, reparameterized deploy graph (Realtek `reparam_yolov7-tiny.py`), head `Detect`, strides 8/16/32, default anchors (`--noautoanchor`) |
| Parameters | 6,014,038 (checkpoint stored as float16; exported as float32) |
| Weights | `models/yolov7-tiny/fruitfly_deploy.pt` (git-ignored), sha256 `487516c7896b4471a2e1e49f12e463b5d467c66352a0893718fd89d8ff942a45` |
| Trained by | AGRHUB, Colab (the AI camera project's training notebook `train_fruitfly_yolov7tiny.ipynb`): WongKinYiu/yolov7 `train.py`, from `yolov7-tiny.pt`, 640x640, batch 32, `hyp.scratch.tiny.yaml`, 150 epochs planned. Epochs actually run: not recorded in the checkpoint (`epoch: -1` after reparameterization) |
| Runtime | OpenCV 5.0.0 `cv2.dnn`, `ENGINE_NEW` passed explicitly, CPU; NMS with `cv2.dnn.NMSBoxes` (`sentinel/vision/detect_yolov7.py`) |

## ONNX interface (`models/export_yolov7.py`)

| Item | Value |
|---|---|
| File | `models/yolov7-tiny/fruitfly_deploy.onnx`, 24,146,362 bytes, sha256 `0cf45768…a16f` (`fruitfly_deploy.onnx.json`) |
| Download | `make models` (`models/fetch_models.py`, SHA-256 checked) from `https://sentinel-685527430355-models-public.s3.ap-southeast-2.amazonaws.com/models/yolov7-tiny/fruitfly_deploy.onnx` (public bucket, GetObject only) or `$SENTINEL_MODELS_BASE_URL` (e.g. a GitHub release); full sha256 `0cf45768bafd58180c91007270815a8634598cedb280f60f274bf0002b31a16f` |
| Opset / shape | 17, static `images [1, 3, 640, 640]` float32, RGB /255 |
| Output | `output [1, 25200, 6]`: cx, cy, w, h (input pixels), objectness, class score; score = obj x cls |
| NMS | not in the graph. An `--end2end` variant (`[n, 7]`, NMS inside) runs on the new engine but fails on the classic one (LICENSE_OPTIONS.md) |
| Simplified | onnxslim |
| Engines | new: yes; classic: yes. New vs classic on 3 real photos: max abs diff 0.0036 px (boxes), 6.1e-6 (scores) (`eval/results/count_vs_production__fruitfly_deploy.json`) |
| vs onnxruntime 1.31.0 | max abs diff 0.00085 px (boxes), 5.0e-6 (scores), same 315 anchors above 0.25, one export photo (`eval/results/yolov7_onnx_parity.csv`) |

## Data

camos/fruit-fly-wmcqi **v3** (Roboflow, **CC BY 4.0**), YOLOv8 export, zip sha256
`6372b7a6f7c8f9b74a6c1f4dd3c3c528f3699fbb67e6d2a621e3f8fca8d7b877`; splits in [`dataset.md`](dataset.md):
train 1,605 / valid 93 / test 76 images. The test split has no augmented twin of a training image.

## Metrics

| Metric | Data | Value | Source |
|---|---|---|---|
| mAP@50 | test split, 76 images, 2,349 boxes | **0.958** | `eval/results/yolov7_tiny_test_metrics.csv` (yolov7 `test.py`, 640, conf 0.001, IoU 0.65, CPU, on `fruitfly_deploy.pt`) |
| mAP@50-95 | same | 0.67 | same |
| Precision / recall | same | 0.928 / 0.908 | same |
| Count vs **production** `ai_count` (Roboflow RF-DETR Small, **not human**) | 93 photos of the 2 trap cameras (gt_sample) | MAE 1.91, bias +1.01 (letterbox); MAE 1.86, bias +1.19 (stretch) | `eval/results/count_vs_production__fruitfly_deploy__trap-cameras.{csv,json}` |
| Count vs production, all 3 cameras | first 48 of gt_sample | MAE 13.92, bias -8.71 (letterbox) | `eval/results/count_vs_production__fruitfly_deploy.{csv,json}`: dominated by AC6527557, which has no trap in view (both models count bricks and leaves) |
| Count MAE vs **human** counts | 150-photo GT sample | not measured (labels pending, `eval/GROUND_TRUTH.md`) | |
| CPU latency, new vs classic engine | | not measured as a benchmark: forward times in the results files were taken on a shared, busy laptop | `bench/REPORT_COOL.md` (EC2 benchmark) |

The acceptance target (mAP@50 >= 0.90 on the 76 test images) is met by this checkpoint. It was trained outside this
repository, so its training curves (`results.txt`) are not available here.

## Preprocessing

`letterbox` (keep aspect, pad 114: yolov7's own inference, used for the test-split mAP) and
`stretch` (resize to 640x640 like the Roboflow export) give the same count agreement on the trap
cameras (table above). Default: letterbox.

## Limitations

- The export's photos with flies are mostly a staged scene (a picture of flies seen by the camera,
  before about 4-7 Oct), the rest an empty lure trap, and one camera (AC6527557) shows a building,
  not a trap. No field photo with a real catch was seen; field accuracy is not measured.
- Max 100 boxes per training image; very full cards are not covered.
- Small flies (median ~35x48 px at 640 in the dataset); lower input sizes not evaluated.

## Fallbacks

| Model | State |
|---|---|
| RF-DETR Nano (Apache-2.0) | COCO checkpoint exported and run on the new engine (`rfdetr_probe.json`); fruit-fly training notebook ready, not run |
| YOLO11n (AGPL-3.0) | pipeline + smoke runs only (`smoke_run.json`); `colab_train.ipynb` not run |

## License

YOLOv7 code and `yolov7-tiny.pt` are **GPL-3.0**; the fine-tuned weights and their ONNX are
derivatives. Dataset CC BY 4.0 (credit: CamOS, Roboflow Universe camos/fruit-fly-wmcqi).
Options and recommendation: [`LICENSE_OPTIONS.md`](LICENSE_OPTIONS.md).
