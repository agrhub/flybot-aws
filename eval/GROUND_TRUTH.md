# Count ground truth (T0.7b): protocol

Goal: a human fruit-fly count for 150 exported photos, so detector counts (production RF-DETR,
our YOLOv7-tiny) can be scored against people instead of against each other. Target: one AgriHub
engineer, under an hour (about 20 s per photo).

## Read this first: what the export actually contains (checked 2026-10-15)

Looking at the photos of the 3,000-photo export (`data/manifest.csv`) changes what "ground truth"
means here:

| Camera | What it shows | Production `ai_count` |
|---|---|---|
| AC6527557 "Gravity01" | a residential building facade and terrace, **no trap in view**, on every photo looked at (19 photos spread over 29 Sep to 13 Oct; not all 1,118 were viewed) | counts up to 210: every box is a false positive on bricks, leaves and the timestamp text |
| BE4089629 "Củ Chi" | until about 4 to 7 Oct: a round picture of flies (the same few frames repeat, e.g. `ai=13` three times), most likely a test rig rather than a field trap; after that: the inside of a lure trap, empty | 13 to 36 on the picture, 0 on the empty trap |
| BE4111159 "Bình Dương" | same pattern as BE4089629 | same |

So: (1) AC6527557 counts must not be used as pest data at all; (2) most photos with flies are a
staged scene; (3) no field photo with a real catch has been seen in the sample. The labels still
matter: they measure counting on these images, and the flags below separate staged, empty and
no-trap photos so nobody mixes them up later. Since AC6527557 points at a home, check its photos
for people (flag `person`), and raise it with the camera owner.

## The sample

`eval/select_gt_sample.py` picks 150 photos, deterministic (seed `20261015`), stratified over
camera x time of day (camera local time UTC+7: night 18-06, morning 06-10, midday 10-14,
afternoon 14-18) x production count bucket (0-5, 6-20, 21-50, >50). 39 of the 48 strata have photos;
each gets 3 or 4 (fewer when the stratum is smaller). Rows are ordered round-robin over strata, so
stopping early still leaves a balanced set. The allocation is written to
`data/labels/gt_sample.json`.

```
uv run python eval/select_gt_sample.py            # -> data/labels/gt_sample.csv (+ .json)
```

## Labeling

```
uv run python eval/label_counts.py --labeler <your initials> \
    --onnx models/yolov7-tiny/fruitfly_deploy.onnx        # pre-drawn boxes (optional)
# open http://127.0.0.1:8765
```

Without `--onnx` there are no boxes, only the photo and the production count. The export has no
production box coordinates (FlyBot stores `aiResults.insects[].bbox` but
`export_sentinel_dataset.js` does not write them); `--boxes <jsonl>` takes them once it does.

For each photo:

1. Count every **fruit fly** (oriental fruit fly, *Bactrocera*, "ruồi vàng") you can see on the
   card or in the trap, including flies at the edge or partly hidden. Do not count other insects,
   dirt, glare spots, grid lines, or flies outside the trap.
2. Use the boxes as a start: **click a red box** that is not a fruit fly (it turns grey and dashed),
   **click on a fly without a box** (green dot), right-click a dot to remove it. The count follows
   (kept boxes + dots). If boxes are hopeless, type the number instead.
3. Tick what applies (keyboard Alt+1 to Alt+6):
   - `card full`: too crowded to count reliably (give your best estimate anyway);
   - `glare`: reflection or overexposure hides part of the card;
   - `person`: a person or part of one is visible (these photos are later blurred or dropped);
   - `no trap in view`: the camera is not looking at a trap (count 0);
   - `staged`: a picture, screen or test rig rather than a field trap (count what is shown);
   - `unusable`: blurred, fogged or covered; count 0 or your best estimate, it will be left out.
4. Enter saves and moves on. N / P move without saving, B hides the boxes.

The production count is shown on purpose (it is what FlyBot showed to farmers); do not copy it.
When unsure between two numbers, write the lower one and a note.

Labels go to `data/labels/counts.csv` (git-ignored): one row per photo and labeler, with the
count, flags, how many pre-drawn boxes were kept / rejected / added, the pre-label source, time and
tool version. Saving the same photo again replaces your row. A second labeler on 20 photos
(`--labeler <other>`) gives an agreement figure.

Rows with `labeler=agent-test` are 5 tool checks made by an agent, not ground truth: every
script that reads `counts.csv` ignores them.

## Using the labels

```
uv run python eval/count_vs_production.py models/yolov7-tiny/fruitfly_deploy.onnx --n 150 \
    --labels data/labels/counts.csv
```

adds `vs_human` (our detector) and `production_vs_human` (RF-DETR) blocks to
`eval/results/count_vs_production__*.json`. Report them separately for trap photos and for
`staged` / `no_trap` photos.

## Already measured without humans (not ground truth)

`eval/count_vs_production.py` against the production count, OpenCV 5.0.0 new engine, conf 0.25,
NMS IoU 0.45 (source: `eval/results/count_vs_production__fruitfly_deploy*.{csv,json}`):

| Photos | n | MAE letterbox | bias letterbox | MAE stretch | bias stretch |
|---|---:|---:|---:|---:|---:|
| first 48 of the sample (all 3 cameras) | 48 | 13.92 | -8.71 | 12.29 | -3.96 |
| trap cameras only (BE4089629, BE4111159) | 93 | 1.91 | +1.01 | 1.86 | +1.19 |

Bias = mean(YOLOv7-tiny - production). Agreement between two models says nothing about which
one is right; the human labels are what decides.
