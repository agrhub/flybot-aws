# Op 1 hand-check: card finder on 50 real export photos

Run: `uv run python -m sentinel.vision.card_check` (2026-10-15). Photos: 50 from `data/manifest.csv`,
17 / 17 / 16 over AC6527557 / BE4089629 / BE4111159, one per farm day spread over the camera's days and
hours (`card_check.pick`). Overlays: [card_check/](card_check/) (outside the detected quad pixelated:
the AC6527557 camera looks at a building where people can appear). Raw results:
[card_check/results.json](card_check/results.json). Verdicts are mine, from looking at every photo
unpixelated (scratch contact sheet, not stored).

## Main finding: no photo in the export shows a yellow sticky card

| Scene (visual) | Photos in this set | Photos in the whole export |
|---|---:|---:|
| AC6527557: building facade, no trap in view | 17 | 1,118 (1280x720, plus 8 at 640x640) |
| BE4089629 / BE4111159: inside a white bait trap (white floor, bait port, stains) | 21 | 1,293 (768x432 and 2304x1296) |
| 640x640 fly-disc pictures (Roboflow-style uploads, a few pictures repeated, not camera frames) | 12 | 597 |
| **Yellow sticky card** | **0** | **0** (HSV yellow share > 10 % of the frame: 0 of 3,000) |

So the acceptance target "card found correctly on >= 95 % of 200 real photos" **cannot be measured on this
export**: there is no real photo with a card. It is measured on synthetic composites instead
([card_synth.json](card_synth.json)), and needs real card photos (new trap set-up or the AI camera) to
be checked for real.

## Hit rate

| Profile | Correct | Wrong | Note |
|---|---:|---:|---|
| `yellow` (default, the plan's card) | **50 / 50** correct "no card" | 0 false cards | before the plausibility tests (fill >= 0.8, solidity >= 0.85, quad overflow <= 1.12, corner angle >= 40 deg) it took bait stains for a card in 188 of the 3,000 export photos; after them 0 of 3,000 |
| `auto` (yellow, then white trap surface) | 14 / 17 building photos correctly empty | 3 building photos: a white wall/awning taken as trap | white fallback is **not** usable as a card finder |
| `auto` on fly-disc pictures | 12 / 12 found the disc, quad inscribed in a round disc (approximate) | - | a disc is not a quad; the quad is only a crop |
| `auto` inside the bait trap | 0 / 21 meaningful | 21 arbitrary partial quads | the white floor fills the frame, there is no card edge to find |

Decision: `find_card` defaults to `yellow`; when no card is found, ops 3-4 work on the whole frame
(`frame_source: full_frame`). `white` stays opt-in for experiments.

## Per-photo verdicts


| # | camera | captured_at (UTC) | size | scene (visual) | yellow card in photo? | `yellow` result | yellow verdict | `auto` (white fallback) result | auto verdict | evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AC6527557 | 2026-09-29T01:22 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/01_AC6527557_6abb12c875e50bc9cc8a8a57.jpg) |
| 2 | AC6527557 | 2026-09-29T20:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/02_AC6527557_6abc1fd3c11d5ca72cfbfadc.jpg) |
| 3 | AC6527557 | 2026-10-01T00:00 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/03_AC6527557_6abda28bc11d5ca72cfc0fc4.jpg) |
| 4 | AC6527557 | 2026-10-02T03:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/04_AC6527557_6abf2546c11d5ca72cfc2390.jpg) |
| 5 | AC6527557 | 2026-10-03T07:00 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | found cov=0.21 partial | wrong (wall/sky taken as trap) | [jpg](card_check/05_AC6527557_6ac0a8034c3227d11cfbb382.jpg) |
| 6 | AC6527557 | 2026-10-04T10:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | found cov=0.27 partial | wrong (wall/sky taken as trap) | [jpg](card_check/06_AC6527557_6ac22abc4c3227d11cfbc607.jpg) |
| 7 | AC6527557 | 2026-10-05T06:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | found cov=0.20 partial | wrong (wall/sky taken as trap) | [jpg](card_check/07_AC6527557_6ac343fb4c3227d11cfbd141.jpg) |
| 8 | AC6527557 | 2026-10-06T03:05 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/08_AC6527557_6ac4656abae070f7ce0531d2.jpg) |
| 9 | AC6527557 | 2026-10-07T09:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/09_AC6527557_6ac611329d7333beb125be2b.jpg) |
| 10 | AC6527557 | 2026-10-08T08:47 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/10_AC6527557_6ac758910691025f82d3df27.jpg) |
| 11 | AC6527557 | 2026-10-09T16:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/11_AC6527557_6ac916b09d7333beb125bf79.jpg) |
| 12 | AC6527557 | 2026-10-10T08:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/12_AC6527557_6ac9f79f9d7333beb125bfd4.jpg) |
| 13 | AC6527557 | 2026-10-11T02:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/13_AC6527557_6acaf4bc9d7333beb125c03f.jpg) |
| 14 | AC6527557 | 2026-10-12T16:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/14_AC6527557_6acd0b0ed7f580a4573c9961.jpg) |
| 15 | AC6527557 | 2026-10-12T21:00 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/15_AC6527557_6acd4a54d7f580a4573c9982.jpg) |
| 16 | AC6527557 | 2026-10-13T21:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/16_AC6527557_6acea2dcd7f580a4573c9a3b.jpg) |
| 17 | AC6527557 | 2026-10-14T20:30 | 1280x720 | building facade (no trap in view) | no | no card | ok (true negative) | no card | ok (nothing found) | [jpg](card_check/17_AC6527557_6acfe64b2edf1794f1ef61e1.jpg) |
| 18 | BE4089629 | 2026-09-29T01:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.38 partial | approx (quad inscribed in round disc) | [jpg](card_check/18_BE4089629_6abb149f75e50bc9cc8a8a79.jpg) |
| 19 | BE4089629 | 2026-09-29T20:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.38 partial | approx (quad inscribed in round disc) | [jpg](card_check/19_BE4089629_6abc1fcdc11d5ca72cfbfac0.jpg) |
| 20 | BE4089629 | 2026-10-01T00:00 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.36 partial | approx (quad inscribed in round disc) | [jpg](card_check/20_BE4089629_6abda285c11d5ca72cfc0f9f.jpg) |
| 21 | BE4089629 | 2026-10-02T03:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.38 partial | approx (quad inscribed in round disc) | [jpg](card_check/21_BE4089629_6abf253ec11d5ca72cfc2374.jpg) |
| 22 | BE4089629 | 2026-10-03T07:00 | 2304x1296 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.56 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/22_BE4089629_6ac0a7fa4c3227d11cfbb379.jpg) |
| 23 | BE4089629 | 2026-10-04T10:30 | 2304x1296 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.40 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/23_BE4089629_6ac22ab34c3227d11cfbc5ff.jpg) |
| 24 | BE4089629 | 2026-10-05T06:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.58 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/24_BE4089629_6ac33ceb4c3227d11cfbd0f4.jpg) |
| 25 | BE4089629 | 2026-10-06T03:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.57 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/25_BE4089629_6ac46435def56a2e9df7b08e.jpg) |
| 26 | BE4089629 | 2026-10-07T12:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.33 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/26_BE4089629_6ac6345a9d7333beb125be39.jpg) |
| 27 | BE4089629 | 2026-10-08T09:35 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.65 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/27_BE4089629_6ac763f08771d4439135f1e9.jpg) |
| 28 | BE4089629 | 2026-10-09T01:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.60 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/28_BE4089629_6ac83ca09d7333beb125bf1e.jpg) |
| 29 | BE4089629 | 2026-10-10T10:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.60 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/29_BE4089629_6aca0cbe9d7333beb125bfdc.jpg) |
| 30 | BE4089629 | 2026-10-11T08:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.83 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/30_BE4089629_6acb421e9d7333beb125c05c.jpg) |
| 31 | BE4089629 | 2026-10-11T23:05 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.59 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/31_BE4089629_6acc162035da959c78fefc6a.jpg) |
| 32 | BE4089629 | 2026-10-13T02:05 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.59 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/32_BE4089629_6acd91d1d7f580a4573c99a8.jpg) |
| 33 | BE4089629 | 2026-10-14T09:35 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.62 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/33_BE4089629_6acf4cc9d7f580a4573c9a88.jpg) |
| 34 | BE4089629 | 2026-10-14T20:35 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.32 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/34_BE4089629_6acfe7792edf1794f1ef61e6.jpg) |
| 35 | BE4111159 | 2026-09-29T01:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.29 partial | approx (quad inscribed in round disc) | [jpg](card_check/35_BE4111159_6abb1499c11d5ca72cfbe957.jpg) |
| 36 | BE4111159 | 2026-09-29T20:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.38 partial | approx (quad inscribed in round disc) | [jpg](card_check/36_BE4111159_6abc1fc8c11d5ca72cfbfaa4.jpg) |
| 37 | BE4111159 | 2026-10-01T00:00 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.36 partial | approx (quad inscribed in round disc) | [jpg](card_check/37_BE4111159_6abda280c11d5ca72cfc0f7a.jpg) |
| 38 | BE4111159 | 2026-10-02T03:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.36 partial | approx (quad inscribed in round disc) | [jpg](card_check/38_BE4111159_6abf2539c11d5ca72cfc234f.jpg) |
| 39 | BE4111159 | 2026-10-03T07:00 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.29 partial | approx (quad inscribed in round disc) | [jpg](card_check/39_BE4111159_6ac0a7f04c3227d11cfbb36b.jpg) |
| 40 | BE4111159 | 2026-10-04T10:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.38 partial | approx (quad inscribed in round disc) | [jpg](card_check/40_BE4111159_6ac22aa94c3227d11cfbc5e3.jpg) |
| 41 | BE4111159 | 2026-10-05T13:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.36 partial | approx (quad inscribed in round disc) | [jpg](card_check/41_BE4111159_6ac3a6594c3227d11cfbd88e.jpg) |
| 42 | BE4111159 | 2026-10-06T01:30 | 640x640 | uploaded fly-disc picture (640x640, not a camera frame) | no | no card | ok (true negative) | found cov=0.29 partial | approx (quad inscribed in round disc) | [jpg](card_check/42_BE4111159_6ac44f184c3227d11cfbde58.jpg) |
| 43 | BE4111159 | 2026-10-08T08:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.64 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/43_BE4111159_6ac74d879d7333beb125beb3.jpg) |
| 44 | BE4111159 | 2026-10-08T18:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.29 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/44_BE4111159_6ac7da249d7333beb125bee1.jpg) |
| 45 | BE4111159 | 2026-10-10T08:25 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.88 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/45_BE4111159_6ac9f693141af6f516da9481.jpg) |
| 46 | BE4111159 | 2026-10-11T07:30 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.72 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/46_BE4111159_6acb3af99d7333beb125c057.jpg) |
| 47 | BE4111159 | 2026-10-11T19:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.29 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/47_BE4111159_6acbdcb635da959c78fefc4b.jpg) |
| 48 | BE4111159 | 2026-10-13T06:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.43 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/48_BE4111159_6acdc8e5d7f580a4573c99c3.jpg) |
| 49 | BE4111159 | 2026-10-14T08:30 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.63 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/49_BE4111159_6acf3d8dd7f580a4573c9a82.jpg) |
| 50 | BE4111159 | 2026-10-14T22:00 | 768x432 | inside white bait trap (floor, bait port) | no | no card | ok (true negative) | found cov=0.28 partial | not meaningful (floor fills frame; arbitrary partial quad) | [jpg](card_check/50_BE4111159_6acffb652edf1794f1ef61f6.jpg) |