# Synthetic trap photos (SIMULATED DATA)

The orchard traps have caught no fly since about 7 Oct 2026 (the lure was not replaced) and the ESP32 trap camera is
not in the field yet, so there is no 30-day history with flies. This package makes **clearly labelled simulated
photos** for the FlyBot demo farm: real photos of *empty* traps with fruit flies composited in by OpenCV 5, a plausible
fly wave per trap, a share of photos damaged on purpose, and the exact truth for every photo (flies, defect, the action
the policy expects). It never replaces field data and is never mixed into a real camera's history.

```
make synth           # showcase: 4 traps, 5 photos a day, 2026-09-19..10-18 -> data/synth/demo-30d/ (git-ignored)
make synth-ahead     # the same traps continued 2026-10-19..11-10 -> data/synth/demo-ahead/
make synth-pool      # pool for the demo's virtual traps (no trap, no time in the stamp) -> data/synth/pool/
make synth-check     # the vision handler on demo-30d and the pool, then both write-ups (eval/SYNTHETIC_*.md)
cd lambda/vision && uv run pytest -q tests/test_synth.py   # generator tests (no dataset or detector needed)
```

Without make, from the repository root (`UV="uv run --project lambda/vision python -m"`):

```
$UV eval.synth.generate [--config eval/synth/demo_30d.json] [--from D] [--to D] [--day D] [--traps a,b] [--out DIR]
$UV eval.synth.pool [--config eval/synth/pool.json]
$UV eval.synth.check [--manifest data/synth/demo-30d/manifest.json] [--tiles] [--url http://localhost:9001/invoke]
$UV eval.synth.report [--sets data/synth/demo-30d data/synth/pool]
```

## Daily mode

Every random draw comes from a stream keyed by (seed, trap id, day, slot), so one trap-day generated alone gives the
same photo ids and bytes as the bulk run (`--day 2026-10-25 --traps demo-lure-a --out DIR`; a sticky card is replayed
from its first photo). Photo ids come from the scheduled slot: `synth-<trap>-<YYYYmmddTHHMM>Z` (UTC). A daily service
can therefore generate, or re-generate, any day of any configured trap without touching the others.

## How a photo is made

1. **Flies** (`flies.py`): single flies cut out of the camos/fruit-fly v3 dataset, `test` and `valid` splits only
   (the detector never trained on them). Kept: isolated, fully visible, on a clean bright ring, not blurred. Each fly
   is stored as a transmittance map (crop / local disc colour) plus a soft alpha mask.
2. **Background**: a real empty lure-trap photo from `data/samples` (24 photos, checked free of people; cameras in
   `data/excluded_cameras.csv` are never used). The trap floor comes from the pipeline's own floor finder
   (`sentinel.vision.card.find_lure_floor`), so flies land where the detector counts. The camera's date overlay is
   inpainted away. Sticky traps get a synthetic yellow card (`sentinel.vision.synth.make_card`) filling most of the
   frame, as the ESP32 trap assembly guide asks, shaded with the frame's light.
3. **Compositing** (`composite.py`): random position on the floor or card, size, any rotation; multiply blend under
   the soft mask (default; `"blend": "seamless"` uses `cv2.seamlessClone`), slight blur to match the camera; then
   photo-level brightness / contrast jitter, sensor noise and JPEG quality 80-92.
4. **Counts** (`curve.py`): per trap a wave (baseline, rise, peak, slower decline) with correlated day-to-day noise,
   and daytime activity weights per capture slot. Lure traps: each photo is independent (the count resets). Sticky
   traps: new flies stay on the card, same pose, until the card is changed every `cardDays` days. Not tied to
   temperature or degree-days.
5. **Defects** (`defects.py`, `"defects": {"rate": 0.1}` in the config): blur, darkness, glare, a dirty lens, a hand
   or a leaf over the trap, a camera that moved, no trap in view. Blur, glare, darkness and the empty view are the
   failure-injection harness's functions (eval/failure_injection.py), the hand is op 8's own synthetic occluder. Each
   damaged photo carries `defect.expected`: the flag, decision rule and proposal actions the policy prescribes, and
   whether it must not be counted. Flies under an occluder or out of view are marked `hidden` (`visibleCount`).

Fly size is a setting: lure 0.11 x the trap floor width, from FlyBot's production boxes on BE4089629 photos of
2026-09-19..28 when the trap still caught flies (long side median 63 px on 768 px photos; RF-DETR boxes, not human
labels); sticky 0.04 x the card's long side (a ~1 cm fly on the 15 x 25 cm card).

## Pool

`pool.py` (config `pool.json`) makes photos for many virtual traps: lure photos with 0..60 flies (about 10 per bucket
of 5), 7-day sticky cards at several catch rates (every slot, flies accumulating), and per defect type a few damaged
lure and sticky photos. The stamp says only `SIMULATED` and `capturedAt` is null: the demo assigns traps and times.

## Labels: a composite can never pass as a field photo

- file name `SYNTH_...jpg`, photo id `synth-...`;
- EXIF ImageDescription and Software, XMP (`flybot:synthetic="true"`, photo id, trap, time, true count, defect),
  JPEG comment;
- a visible `SIMULATED` stamp (series: with local time and trap) where the camera date was;
- `synthetic: true` on every manifest row and on the manifest; the output README repeats it with the credits.

`eval.synth.check` refuses to measure a file that lacks any of these.

## Output

`manifest.json` follows [`manifest.schema.json`](manifest.schema.json): config and seed, credits, traps (demo
positions, platform, cards, daily curve levels) and one row per photo with `trueCount`, `visibleCount`,
`newSincePrevious`, `defect`, the fly boxes (contract-style fractions and pixels, with the dataset crop each fly came
from), the background photo and the render settings. `manifest.csv` flattens the rows. [`sample/`](sample/) holds a
few photos and their manifest rows.

## Checks

`eval.synth.check` sends every photo through `sentinel.handler.analyze` (the Lambda code path: YOLO26n on OpenCV 5
DNN `ENGINE_NEW`, trap ROI, box rules, quality and motion ops, rule planner, proposals; or `make lambda-local` with
`--url`) as FlyBot would: with the trap's latest clean photo as `prevPhoto` and its earlier counts as `history`. It
writes `check.jsonl` beside the manifest. `eval.synth.report` turns those into two write-ups, both kept apart from the
real-data results in [`../RESULTS.md`](../RESULTS.md):

- [`../SYNTHETIC_RESULTS.md`](../SYNTHETIC_RESULTS.md): detected vs true count and boxes on the clean photos (with
  `--tiles`, also the detector alone on the whole frame and on 2x2 tiles);
- [`../SYNTHETIC_AGENT_EVAL.md`](../SYNTHETIC_AGENT_EVAL.md): the agent's decisions vs the generated truth (defect vs
  expected action, false alarms on clean photos, surges vs the surge rule on the true counts). This is an automatic
  evaluation on synthetic photos, **not an expert review**: no person gave these verdicts. The human expert review is
  [`../EXPERT_REVIEW.md`](../EXPERT_REVIEW.md).

## Credits

- Flies: "fruit-fly" dataset by CamOS on Roboflow Universe,
  https://universe.roboflow.com/camos/fruit-fly-wmcqi/dataset/3, licensed CC BY 4.0 (see `models/dataset.md`).
  The composites are an adaptation: flies were cut out, scaled, rotated, recoloured by the target light and pasted.
- Backgrounds: empty lure-trap photos from AGRHUB's orchard cameras (FlyBot), `data/samples`.
