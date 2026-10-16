# Colab runbook: YOLO26n fruit-fly detector (owner)

The vision Lambda runs **YOLO26n** (Ultralytics, NMS-free end-to-end head) on OpenCV 5 DNN. Until its weights
exist the Lambda runs the **YOLOv7-tiny** baseline (`DETECTOR=yolov7`, the AMB82 NPU model converted by
`export_yolov7.py`, fetched by `make models`). This runbook produces the YOLO26 weights in about an hour on a
Colab T4 and switches the Lambda to them.

## 1. Train + export on Colab (~1 h)

1. colab.research.google.com > File > Upload notebook > `models/colab_train_yolo26.ipynb`.
2. Runtime > Change runtime type > **T4 GPU**.
3. Settings cell: paste your Roboflow API key in `ROBOFLOW_API_KEY` (universe.roboflow.com > Settings > API
   key) so the notebook downloads camos/fruit-fly-wmcqi v3 itself; without a key it asks you to upload
   `fruit-fly.v3i.yolov8.zip` (local copy `../camos-ai-camera/models/fruit-fly.v3i.yolov8.zip`).
   `EPOCHS = 60` is the ~1 h budget (not measured for YOLO26: the first epochs print the time per epoch; lower
   `EPOCHS` if the ETA passes 1 h). Set `SMOKE_TEST = True` first for a 3-minute end-to-end check of the notebook.
4. Runtime > Run all. The last cell downloads (keep the error text verbatim if a cell fails):

| File | What |
|---|---|
| `fruitfly_yolo26n.onnx` | the detector for OpenCV 5 (opset 17, 640 static, batch 1): end-to-end head `[1, 300, 6]`, or the one-to-many head `[1, 5, 8400]` when the notebook's OpenCV check failed on the end-to-end graph |
| `fruitfly_yolo26n.onnx.json` | sidecar: imgsz, head, version, SHA-256, test metrics, OpenCV check status (read by the detector and the model card) |
| `fruitfly_yolo26n-test_metrics.json`, `-results.csv` | mAP50 / mAP50-95 / precision / recall on the 76-image test split; per-epoch curves |
| `fruitfly_yolo26n-opencv_check.json` | cv2.dnn (OpenCV 5.0.0.93, new engine) vs onnxruntime on 20 test images, both graphs |
| `fruitfly_yolo26n-o2m.onnx` | one-to-many fallback graph (downloaded only when the end-to-end one passed and was deployed) |
| `fruitfly_yolo26n-run.zip` | weights (best.pt), plots, logs |

## 2. On the laptop: verify, upload, switch the Lambda

```
mkdir -p models/yolo26 && mv ~/Downloads/fruitfly_yolo26n* models/yolo26/        # git-ignored
cd lambda/vision && uv run --with onnxruntime python ../../models/verify_yolo26.py ../../models/yolo26/fruitfly_yolo26n.onnx
    # exit 0 = OpenCV 5 new engine matches onnxruntime on 20 sample photos (writes eval/results/yolo26_parity.json)
    # exit 2 = OpenCV cannot load/run the end-to-end graph: use fruitfly_yolo26n-o2m.onnx instead (rename it to
    #          fruitfly_yolo26n.onnx, set "head": "one_to_many" in the .onnx.json) and run the check again
cd ../.. && DETECTOR=yolo26 make replay-sample      # 24 sample photos through the handler with YOLO26
bash models/upload_yolo26.sh                        # -> s3://sentinel-685527430355-models-public/yolo26/ (profile sentinel)
DETECTOR=yolo26 make deploy                         # Lambda env DETECTOR=yolo26 + YOLO26_URL/SHA256 from models/yolo26/*.onnx.json
```

The Lambda fetches the ONNX from the public bucket at cold start (SHA-256 checked, about 10 MB) unless the image
was built while `models/yolo26/` existed (`make build-lambda` bakes it in). If `curl -I` of the uploaded URL is not
200, the bucket policy only allows `models/*`: add `yolo26/*` to its public GetObject statement (or upload under
`models/yolo26/` and set `MODELS_PREFIX=models/yolo26`).

## 3. Record the result

- `models/MODEL_CARD.md`: the test-split numbers from `fruitfly_yolo26n-test_metrics.json` (cite the file), the
  OpenCV check outcome from `eval/results/yolo26_parity.json`, the head that was deployed.
- `docs/DECISIONS.md`: one line "YOLO26 end-to-end head on OpenCV 5 DNN: ok | fallback one-to-many (reason)".
- `bench/`: add the YOLO26 column with `make bench`.

## Fallback notebooks (kept from the first build)

`colab_train.ipynb` (YOLO11n) and `colab_train_rfdetr.ipynb` (RF-DETR Nano) take the same dataset zip; their
outputs are not wired to the Lambda.
