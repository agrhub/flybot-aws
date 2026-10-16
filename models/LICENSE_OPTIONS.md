# Detector license options (T0.9)

Checked 2026-10-15. Evidence: `models/rfdetr_probe.json`, `models/yolov7-tiny/fruitfly_deploy.onnx.json`,
`eval/results/count_vs_production__fruitfly_deploy*.{csv,json}`, `eval/results/yolov7_tiny_test_metrics.csv`,
`models/smoke_run.json` (YOLO11n).

## Candidates

| Detector | Code / weights license | Trained fruit-fly weights we have | ONNX on OpenCV 5.0.0 new engine | Classic engine |
|---|---|---|---|---|
| **YOLOv7-tiny** (WongKinYiu/yolov7) | **GPL-3.0** (code; weights fine-tuned from `yolov7-tiny.pt`, same repo) | **yes**: owner's `fruitfly_deploy.pt` (AMB82 NPU model, reparameterized) | loads and runs; `[1,25200,6]`; matches onnxruntime 1.31.0 to 0.00085 px / 5e-6 score | loads and runs (no-NMS export); `--end2end` export fails |
| RF-DETR Nano/Small (Roboflow `rfdetr`) | **Apache-2.0** (code and COCO checkpoints) | **no**: the production RF-DETR Small lives in Roboflow; the plan does not include weight export ("Your current plan does not have weights export included", owner, Roboflow UI). Needs our own training (`models/colab_train_rfdetr.ipynb`) | Nano COCO checkpoint: loads and runs; `dets [1,300,4]`, `labels [1,300,91]`; matches onnxruntime to 0.00023 (boxes) / 0.00054 (logits) | **fails** (error below) |
| YOLO11n (Ultralytics) | **AGPL-3.0** (or Enterprise license) | no (only smoke runs) | loads and runs (T0.4) | loads (static export) |

The production model (Roboflow RF-DETR Small, `fruit-fly-wmcqi/2`) keeps running inside FlyBot
through the Roboflow workflow; nothing here changes that.

## Exact errors

RF-DETR Nano (COCO), classic engine, `cv2.dnn.readNet(path, "", "", cv2.dnn.ENGINE_CLASSIC)`:

```
OpenCV(5.0.0) ...\modules\dnn\src\onnx\onnx_importer.cpp:1114: error: (-2:Unspecified error) in function 'cv::dnn::dnn5_v20260605::ONNXImporter::handleNode'
> Node [Slice@ai.onnx]:(onnx_node!/transformer/decoder/layers.0/cross_attn/Slice_5) parse error: OpenCV(5.0.0) ...\modules\dnn\src\layers\slice_layer.cpp:220: error: (-215:Assertion failed) end < 0 || end > start in function 'cv::dnn::SliceLayerImpl::SliceLayerImpl'
```

YOLOv7-tiny `--end2end` export (NMS in graph, onnxruntime NMS variant), classic engine, forward:

```
cv2.error: OpenCV(5.0.0) ...\modules\dnn\src\layers\gather_layer.cpp:102: error: (-2:Unspecified error) in function 'void __cdecl cv::dnn::GatherLayerImpl::forward_impl<__int64>(...)'
> must be equal to
>     'outer_dims * indices_total * inner_size' is 177
```

The same end2end file runs on the new engine (37 boxes on the check image, equal to the no-NMS
export + `cv2.dnn.NMSBoxes`). The repo uses the no-NMS export.

D-FINE / RT-DETR were not tried: RF-DETR (same DETR family, the production family) already loads
on the new engine.

## What each choice means

1. **YOLOv7-tiny, repo under GPL-3.0** (owner's decision: primary detector). The repo's own code may
   stay MIT/Apache-compatible only if the GPL parts are kept separate; simplest is to license the
   whole repo GPL-3.0. GPL-3.0 (unlike AGPL) has no network clause: running it as a service does not
   oblige publishing changes, distributing it does. The yolov7 source is not vendored
   (`models/third_party/` is git-ignored; `export_yolov7.py` clones a pinned commit), and the
   runtime path (`sentinel/vision/detect_yolov7.py`) needs only OpenCV + the ONNX file. The ONNX
   weights are a derivative of GPL weights: ship them with GPL-3.0 notice and the CC BY 4.0 dataset
   credit. Same model family as the AMB82 camera NPU, which is a good story for judges.
2. **RF-DETR Nano, Apache-2.0**: cleanest license and the production family, loads on the new
   engine, but needs a 1-2 h Colab run (not done) and has no classic-engine fallback, and is
   heavier (Nano ONNX 107.8 MB, fp32, 384 px) than YOLOv7-tiny (24.1 MB, 640 px).
3. YOLO11n, AGPL-3.0: no reason left to prefer it; kept as the last fallback.

## Recommendation

Ship **YOLOv7-tiny under GPL-3.0** now (it is trained, on the test split it scores mAP@50 0.958,
and it runs on the new engine with onnxruntime parity), add a `LICENSE` (GPL-3.0) plus a
`NOTICE` crediting WongKinYiu/yolov7 (GPL-3.0) and camos/fruit-fly-wmcqi (CC BY 4.0). Train the
RF-DETR Nano notebook only if the owner wants an Apache-2.0 release later; the code path for it
(`models/rfdetr_probe.py`) is already checked. Decision owner: lead / owner.
