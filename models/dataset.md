# Dataset: camos/fruit-fly v3 (YOLOv8 export)

**Credit:** "fruit-fly" dataset by CamOS on Roboflow Universe,
https://universe.roboflow.com/camos/fruit-fly-wmcqi/dataset/3, licensed **CC BY 4.0**.

Every number below was counted on 2026-10-15 by `models/dataset_stats.py` (main project env, OpenCV 5,
no torch) from the extracted export, not copied from the Roboflow README. Source of truth:
[`models/dataset_stats.json`](dataset_stats.json).

```
uv run python models/dataset_stats.py ../camos-ai-camera/models/fruit-fly-v3 \
    --zip ../camos-ai-camera/models/fruit-fly.v3i.yolov8.zip --out models/dataset_stats.json
```

## Where it is

| Item | Value |
|---|---|
| Zip | `../camos-ai-camera/models/fruit-fly.v3i.yolov8.zip`, 57,291,462 bytes, sha256 `6372b7a6f7c8f9b74a6c1f4dd3c3c528f3699fbb67e6d2a621e3f8fca8d7b877` |
| Extracted | `../camos-ai-camera/models/fruit-fly-v3/` (read only for this repo; `train.py` stages its own copy in `models/datasets/`) |
| Zip contents | same split counts as the extracted folder (`unzip -Z1`: 1605/93/76 images and labels) |
| Roboflow | workspace `camos`, project `fruit-fly-wmcqi`, version 3, exported 2026-10-14 |

## data.yaml (as exported)

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: 1
names: ['fruit_fly']
```

The relative paths do not resolve from the dataset folder in ultralytics; `train.py` and the Colab
notebook write a new `data.yaml` with `path:` set to the absolute staged folder.

## Counts per split

| Split | Images | Label files | Boxes | Boxes / image (mean / median / p95 / max) | Images with 0 boxes | Distinct source images |
|---|---:|---:|---:|---|---:|---:|
| train | 1,605 | 1,605 | 75,018 | 46.74 / 47 / 90 / 100 | 2 | 1,404 |
| valid | 93 | 93 | 2,838 | 30.52 / 14 / 80.6 / 94 | 0 | 93 |
| test | 76 | 76 | 2,349 | 30.91 / 33.5 / 67.5 / 90 | 0 | 76 |
| **total** | **1,774** | **1,774** | **80,205** | | 2 | 1,573 |

- One class, id 0 = `fruit_fly` in all splits (no other class id appears).
- Image size: every image is **640x640** (Roboflow preprocessing "Resize to 640x640 (Stretch)").
- "Distinct source images" = distinct file-name prefixes before `.rf.` (Roboflow names augmented copies
  `<source>_jpg.rf.<hash>.jpg`). No source image appears in two splits (overlap train/valid,
  train/test, valid/test = 0), so the test split has no augmented twin of a training image.
- The 1,573 distinct sources match the "1,573 images" in MASTER_PLAN. The plan's 85,307 boxes do not
  match this export (80,205 counted here); the plan number is from another source and was not checked.
- No image has more than 300 boxes (max 100), so ultralytics' default `max_det=300` does not clip.

## Box sizes (pixels on the 640x640 images)

| Split | width median (p5-p95) | height median (p5-p95) |
|---|---|---|
| train | 33.0 (14.5-49.5) | 44.0 (20.0-69.0) |
| valid | 37.5 (24.5-50.6) | 50.0 (32.0-72.0) |
| test | 35.5 (14.0-50.0) | 48.5 (17.5-73.0) |

Flies are small (median ~35x48 px at 640). Training at imgsz 320 halves that (p5 ~7x9 px), so the real
run trains and exports at **640**; 320 is only used for the CPU smoke run.

## Label quirks (handled in staging)

- **1,038 polygon rows** (`cls x1 y1 x2 y2 ...`) in **31 train label files**; valid and test have none.
- **3 of those files mix polygon and box rows.** ultralytics 8.4.177 asserts
  `labels mix segment and detection rows` and drops such an image as corrupt.
- `train.py` (staging) and the Colab notebook rewrite every polygon row as its bounding box
  (min/max of the points, clipped to [0, 1]), the same conversion ultralytics applies to
  polygon-only files.
- Checked with ultralytics' own `verify_image_label` on **all** images (models/smoke_run.json,
  `ultralytics_label_check_all_images`):
  - original export: train 3 corrupt (`image_02267_…`, `image_02354_…`, `image_04258_…`),
    74,891 boxes kept; valid/test 0 corrupt.
  - staged copy: 0 corrupt in every split, train 75,016 boxes kept (ultralytics also removes 2
    duplicate rows from the 75,018 counted above).
- 2 train images have an empty label file (backgrounds); kept.
