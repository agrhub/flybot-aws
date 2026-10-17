# Op 1 hand-check: `lure` trap profile on 50 real lure-trap photos

Run: `uv run python -m sentinel.vision.card_check --trap-kind lure --out docs/evidence/vision/card_check_lure`
(2026-10-15). Photos: 50 real camera frames of the two lure-trap cameras (BE4089629 25, BE4111159 25;
768x432 and 2304x1296; the 640x640 uploaded pictures are left out), one per farm day spread over each
camera's days and hours (`card_check.pick`). Overlays: [card_check_lure/](card_check_lure/) (cyan = floor
polygon ROI, green = 4-corner fit, everything outside the polygon pixelated). Raw results:
[card_check_lure/results.json](card_check_lure/results.json). Verdicts are mine, from looking at every photo
unpixelated (scratch contact sheet, not stored).

## Result

| | Count | Note |
|---|---:|---|
| floor found (`found: true`) | 50 / 50 | `whole_frame` fallback used 0 times |
| floor polygon ROI correct | 49 / 50 | #1 wrong: the camera was tilted up, the lid ceiling merged with the floor (ROI 0.83 of the frame instead of ~0.65) |
| 4-corner fit kept (`floor_quad`, rectified crop) | 38 / 50 | 34 ok, 4 approximate (#26, #35, #46, #47: one corner a few % into the wall or a sliver of floor left out) |
| 4-corner fit rejected, polygon only (`floor_polygon`) | 12 / 50 | 7 of them were visibly wrong fits (slanted side cutting a floor corner off, or a corner sticking out of the floor: #32, #34, #37, #39, #42, #48, #50), correctly rejected |
| time per photo (find_lure_floor, this laptop) | median 93 ms | resize to 640 px included, JPEG decode not |

**Tuning disclosure.** The corner-angle gate (`LURE["quad_min_angle"] = 78`) was chosen on this set:
every visibly wrong 4-corner fit had a corner angle of 63-75 deg, every kept fit >= 80. To check it was not
overfitted, a second set of 50 other real lure frames (random, seed 20261015, none in the set above) was
run without changes ([card_check_lure/holdout_results.json](card_check_lure/holdout_results.json)):
floor found 50 / 50, `floor_quad` 45, `floor_polygon` 5; by eye all 50 polygons
are correct and 2 of the 45 kept quads are approximate (slightly slanted left side, holdout #18 and #42).

**Not measured:** IoU against hand-drawn floor outlines (none drawn); lure photos with flies on the floor
(since 2026-10-07 the trap is empty; earlier frames show few flies), photos taken through a dirty lens.

`found` semantics (card.py docstring): for `lure`, `found` means a plausible floor region (coverage
0.15-0.97, solidity >= 0.85, median V >= 70, median S <= 70); `quad` exists only when the 4-corner fit
also passes (region fills >= 0.9 of the quad, quad covers >= 0.95 of the region, corners inside the frame
+- 2 %, every corner 90 +- 12 deg). When `found` is false the ROI is the whole frame, never empty.

## Sticky profile unchanged

`find_card(img)` (default `trap_kind="sticky"`) gives byte-identical JSON (all fields except `time_ms`,
`evidence`) before and after this change on 140 cases: 20 synthetic cards x {yellow, auto} and 50 export
photos x {yellow, auto} (scratch comparison, 2026-10-15); `tests/test_vision_card.py` pins it.

## Per-photo verdicts

| # | camera | captured_at (UTC) | size | scene (visual) | roi_source | coverage | polygon verdict | 4-corner verdict | evidence |
|---|---|---|---|---|---|---:|---|---|---|
| 1 | BE4089629 | 2026-10-03T03:30 | 2304x1296 | trap interior, camera tilted up (lid + floor) | floor_polygon | 0.83 | wrong: camera pointed up at the lid, lid ceiling merged with the floor (ROI too large) | rejected (cover 0.9987); polygon used | [jpg](card_check_lure/01_BE4089629_6ac076c04c3227d11cfbb1f8.jpg) |
| 2 | BE4089629 | 2026-10-03T07:30 | 2304x1296 | inside white lure trap (floor, bait port) | floor_quad | 0.70 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/02_BE4089629_6ac0af034c3227d11cfbb3dc.jpg) |
| 3 | BE4089629 | 2026-10-04T03:30 | 2304x1296 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/03_BE4089629_6ac1c8434c3227d11cfbc16f.jpg) |
| 4 | BE4089629 | 2026-10-05T02:27 | 2304x1296 | inside white lure trap (floor, bait port) | floor_quad | 0.67 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/04_BE4089629_6ac30af4db4317640470a02e.jpg) |
| 5 | BE4089629 | 2026-10-05T03:00 | 2304x1296 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/05_BE4089629_6ac312b958107cdf5156490c.jpg) |
| 6 | BE4089629 | 2026-10-05T03:40 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/06_BE4089629_6ac31c2e9e6c6fdc7f8fcfc4.jpg) |
| 7 | BE4089629 | 2026-10-06T03:25 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/07_BE4089629_6ac46a20dc089e0be8cd24ab.jpg) |
| 8 | BE4089629 | 2026-10-07T09:30 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/08_BE4089629_6ac611289d7333beb125be2a.jpg) |
| 9 | BE4089629 | 2026-10-07T12:00 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.68 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/09_BE4089629_6ac6345a9d7333beb125be39.jpg) |
| 10 | BE4089629 | 2026-10-07T16:00 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.68 | ok (floor between the ribbed walls) | rejected (cover 0.8434); polygon used | [jpg](card_check_lure/10_BE4089629_6ac66c919d7333beb125be4e.jpg) |
| 11 | BE4089629 | 2026-10-08T09:56 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/11_BE4089629_6ac768b6d4e3b04cff431989.jpg) |
| 12 | BE4089629 | 2026-10-09T02:46 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/12_BE4089629_6ac8557265765d9381efa90f.jpg) |
| 13 | BE4089629 | 2026-10-09T04:00 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/13_BE4089629_6ac866d765ed4059e7da7f4e.jpg) |
| 14 | BE4089629 | 2026-10-09T07:00 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/14_BE4089629_6ac8910165ed4059e7da7f56.jpg) |
| 15 | BE4089629 | 2026-10-10T16:00 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.68 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/15_BE4089629_6aca61209d7333beb125bffd.jpg) |
| 16 | BE4089629 | 2026-10-11T14:35 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.68 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/16_BE4089629_6acb9e9ad4427f155b3ed859.jpg) |
| 17 | BE4089629 | 2026-10-10T18:00 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/17_BE4089629_6aca7d399d7333beb125c006.jpg) |
| 18 | BE4089629 | 2026-10-10T21:30 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/18_BE4089629_6acaae779d7333beb125c021.jpg) |
| 19 | BE4089629 | 2026-10-11T20:35 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.68 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/19_BE4089629_6acbf2f935da959c78fefc55.jpg) |
| 20 | BE4089629 | 2026-10-12T21:35 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.70 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/20_BE4089629_6acd528ad7f580a4573c998c.jpg) |
| 21 | BE4089629 | 2026-10-13T02:05 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.68 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/21_BE4089629_6acd91d1d7f580a4573c99a8.jpg) |
| 22 | BE4089629 | 2026-10-13T05:35 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/22_BE4089629_6acdc309d7f580a4573c99c1.jpg) |
| 23 | BE4089629 | 2026-10-14T16:05 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.68 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/23_BE4089629_6acfa8312edf1794f1ef61c6.jpg) |
| 24 | BE4089629 | 2026-10-15T00:05 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/24_BE4089629_6ad018b22edf1794f1ef6209.jpg) |
| 25 | BE4089629 | 2026-10-14T23:05 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.69 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/25_BE4089629_6ad00aa12edf1794f1ef6202.jpg) |
| 26 | BE4111159 | 2026-10-07T04:30 | 2304x1296 | inside white lure trap (floor, bait port) | floor_quad | 0.66 | ok (floor between the ribbed walls) | approx: top-right corner runs into the wall | [jpg](card_check_lure/26_BE4111159_6ac5cace9d7333beb125be0c.jpg) |
| 27 | BE4111159 | 2026-10-07T07:00 | 2304x1296 | inside white lure trap (floor, bait port) | floor_polygon | 0.59 | ok (floor between the ribbed walls) | rejected (cover 0.8329); polygon used | [jpg](card_check_lure/27_BE4111159_6ac5edfb9d7333beb125be19.jpg) |
| 28 | BE4111159 | 2026-10-08T01:30 | 2304x1296 | inside white lure trap (floor, bait port) | floor_quad | 0.64 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/28_BE4111159_6ac6f21c844de1a8095cba8f.jpg) |
| 29 | BE4111159 | 2026-10-08T02:00 | 2304x1296 | inside white lure trap (floor, bait port) | floor_quad | 0.64 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/29_BE4111159_6ac6f923cb181161dbb29096.jpg) |
| 30 | BE4111159 | 2026-10-08T02:45 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.64 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/30_BE4111159_6ac703ad92c896770d1521d4.jpg) |
| 31 | BE4111159 | 2026-10-09T05:30 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.63 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/31_BE4111159_6ac87bde9d7333beb125bf36.jpg) |
| 32 | BE4111159 | 2026-10-09T07:30 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.61 | ok (floor between the ribbed walls) | rejected, correctly (4-corner fit cut a floor corner off or stuck out; angle 69.5) | [jpg](card_check_lure/32_BE4111159_6ac897fd65ed4059e7da7f58.jpg) |
| 33 | BE4111159 | 2026-10-09T09:30 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.63 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/33_BE4111159_6ac8b41e65ed4059e7da7f65.jpg) |
| 34 | BE4111159 | 2026-10-10T07:30 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.61 | ok (floor between the ribbed walls) | rejected, correctly (4-corner fit cut a floor corner off or stuck out; angle 68.52) | [jpg](card_check_lure/34_BE4111159_6ac9e97d9d7333beb125bfcd.jpg) |
| 35 | BE4111159 | 2026-10-10T08:08 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.59 | ok (floor between the ribbed walls) | approx: left side slanted, a floor sliver left out | [jpg](card_check_lure/35_BE4111159_6ac9f2646b2d368d8b350666.jpg) |
| 36 | BE4111159 | 2026-10-10T08:25 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.60 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/36_BE4111159_6ac9f693141af6f516da9481.jpg) |
| 37 | BE4111159 | 2026-10-11T07:30 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.62 | ok (floor between the ribbed walls) | rejected, correctly (4-corner fit cut a floor corner off or stuck out; angle 71.4) | [jpg](card_check_lure/37_BE4111159_6acb3af99d7333beb125c057.jpg) |
| 38 | BE4111159 | 2026-10-11T09:58 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.64 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/38_BE4111159_6acb5de3b9e739178a14e319.jpg) |
| 39 | BE4111159 | 2026-10-11T11:30 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.65 | ok (floor between the ribbed walls) | rejected, correctly (4-corner fit cut a floor corner off or stuck out; angle 63.16) | [jpg](card_check_lure/39_BE4111159_6acb733ddb98d7a19b9d7655.jpg) |
| 40 | BE4111159 | 2026-10-12T04:00 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.66 | ok (floor between the ribbed walls) | rejected (cover 0.8488); polygon used | [jpg](card_check_lure/40_BE4111159_6acc5b4535da959c78fefc91.jpg) |
| 41 | BE4111159 | 2026-10-12T08:00 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.59 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/41_BE4111159_6acc9385d7f580a4573c9933.jpg) |
| 42 | BE4111159 | 2026-10-12T12:30 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.64 | ok (floor between the ribbed walls) | rejected, correctly (4-corner fit cut a floor corner off or stuck out; angle 75.17) | [jpg](card_check_lure/42_BE4111159_6accd2cdd7f580a4573c994a.jpg) |
| 43 | BE4111159 | 2026-10-13T01:00 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.64 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/43_BE4111159_6acd8295d7f580a4573c99a1.jpg) |
| 44 | BE4111159 | 2026-10-13T05:30 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.65 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/44_BE4111159_6acdc1dcd7f580a4573c99c0.jpg) |
| 45 | BE4111159 | 2026-10-13T09:00 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.61 | ok (floor between the ribbed walls) | rejected (angle 72.68); polygon used | [jpg](card_check_lure/45_BE4111159_6acdf31ad7f580a4573c99d8.jpg) |
| 46 | BE4111159 | 2026-10-14T11:30 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.65 | ok (floor between the ribbed walls) | approx: bottom-right corner on the wall base (fisheye bulge) | [jpg](card_check_lure/46_BE4111159_6acf67bcd7f580a4573c9a96.jpg) |
| 47 | BE4111159 | 2026-10-14T16:30 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.65 | ok (floor between the ribbed walls) | approx: bottom-right corner on the wall base (fisheye bulge) | [jpg](card_check_lure/47_BE4111159_6acfae0d2edf1794f1ef61c7.jpg) |
| 48 | BE4111159 | 2026-10-13T22:00 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.64 | ok (floor between the ribbed walls) | rejected, correctly (4-corner fit cut a floor corner off or stuck out; angle 75.42) | [jpg](card_check_lure/48_BE4111159_6acea9e4d7f580a4573c9a42.jpg) |
| 49 | BE4111159 | 2026-10-14T22:30 | 768x432 | inside white lure trap (floor, bait port) | floor_quad | 0.66 | ok (floor between the ribbed walls) | ok | [jpg](card_check_lure/49_BE4111159_6ad0026d2edf1794f1ef61fa.jpg) |
| 50 | BE4111159 | 2026-10-14T19:30 | 768x432 | inside white lure trap (floor, bait port) | floor_polygon | 0.64 | ok (floor between the ribbed walls) | rejected, correctly (4-corner fit cut a floor corner off or stuck out; angle 74.99) | [jpg](card_check_lure/50_BE4111159_6acfd83d2edf1794f1ef61da.jpg) |
