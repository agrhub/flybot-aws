# Demo video script (≤ 5:00)

The cut as built: 4:51, 1920x1080, 30 fps, English voice-over with burned-in English subtitles (a Vietnamese
subtitle file and a Vietnamese-subtitled render come from the same command), unlisted YouTube. The competition asks
for: team, the app running, architecture, OpenCV 5 usage, AWS integration, main results. Every number on screen comes
from a file named in [REPORT.md](REPORT.md); anything replayed, synthetic or simulated says so on screen.

The film is built from sources in the FlyBot repository (`tools/media/film`, see its README): screen recordings of
the live app (`record-app.mjs`, `shots.mjs`), motion graphics on one timeline (`film.html`), the spoken lines and
subtitles (`script.js`), the cut of the recordings (`edit.js`) and one render command (`render-film.mjs`). The voice
is synthesized offline (Kokoro, open weights) and can be replaced line by line by a recorded voice.

The app part follows the path a judge can take today: **Guest access → Dashboard → Cameras → photo → Sentinel trace →
approvals → assistant**, recorded in a guest session. The ESP32-S3 camera, the camera log and the firmware pages are
recorded in the owner's account (labelled on screen). The hardware part is an animation of the trap camera's parts
and wiring, from the BOM and the 4G wiring sheet, followed by the team's own footage of the trap in the orchard.

## Shot list

| Time | Scene | What is on screen | Voice-over (short) |
|---|---|---|---|
| 0:00-0:10 | Title | FlyBot logo, flies, "FlyBot on AWS · OpenCV 5 · AWS Lambda · IoT Core · Kinesis Video Streams · AGRHUB" | "Farms count fruit flies on traps. FlyBot photographs the traps and counts the flies, with OpenCV 5 on AWS." |
| 0:10-0:22 | The problem | The glare sample (`data/samples/images/BE4089629/6ac34afa4c3227d11cfbd154.jpg`) next to the strong-blur failure-injection frame | "A count alone can't tell a bad photo from a catch …" |
| 0:22-0:32 | The idea | Trap camera → OpenCV 5 on Lambda → Sentinel agent → a person confirms, and the loop | "FlyBot Sentinel checks every photo, decides, proposes; nothing reaches a camera until a person confirms." |
| 0:32-0:45 | Bill of materials | The parts, cut out of the team's photo of the real board, with the prices read on 17 Oct 2026 (camos-esp32-trap `docs/BOM.md`) | "Our open trap camera: ESP32-S3, 4G modem, GPS, an 18650 cell and a small solar panel." |
| 0:45-1:10 | Wiring | The 4G build wired step by step (cell and charger, 5 V boost to the board, modem from the cell on GPIO41/42, GPS on 3V3 and GPIO14, battery divider on GPIO1), as in the 4G wiring sheet | One line per step |
| 1:10-1:19 | The real build | The team's photo of the perfboard with each module labelled; the trap lid and body | "This is the real board. It goes into the lid of a solar trap housing." |
| 1:19-1:49 | In the orchard | **The team's footage of the wired trap installed in the orchard (30 s)**; a placeholder card until it is dropped in | "In the orchard, the trap hangs at fruit height …" |
| 1:49-2:01 | Guest access | https://flybot.agrhub.com/auth/login, **Guest access**, the guest and "simulated data" banners | "Anyone can try FlyBot: one click on Guest access …" |
| 2:01-2:16 | Dashboard | The density card: the two real EZVIZ traps, then a simulated farm | "Two real orchard traps, and a clearly labelled simulated network of 114 traps on 47 farms." |
| 2:16-2:27 | Cameras | Real / simulated, sticky / lure filters, 2 to 5 columns, search | "The camera list searches and filters …" |
| 2:27-2:40 | A counted photo | A simulated sticky trap's history ("Drawn: N" next to the count) and its photo with the OpenCV boxes and the SIMULATED stamp | "Every photo, real or synthetic, goes through the same OpenCV 5 Lambda …" |
| 2:40-2:59 | Sentinel trace | A trace opened: perception, decision (rule `surge.unconfirmed`), the agent's steps, evidence pictures | "What OpenCV saw, which rule and why, every step …" |
| 2:59-3:11 | Approvals | The proposals waiting for a person (guest view: guests cannot decide) | "Proposals wait for a person; only after the owner confirms does a camera act." |
| 3:11-3:23 | Assistant | "How many flies on the traps this week?" and the answer | "Growers can ask the assistant in plain words …" |
| 3:23-3:45 | ESP32-S3 camera (owner's account) | Live view at JPEG 640x480, the camera log card, Admin > Firmware (releases per model and channel) and the camera fleet | "Streams live through Kinesis Video Streams; log to the console; firmware over the air through IoT Core, alpha / beta / stable." |
| 3:45-3:54 | Field map, reports | The simulated network on the map with the heat map; a report | "Traps on a field map with a heat map; reports per farm." |
| 3:54-4:08 | Architecture | `docs/diagrams/architecture.png`, panned: input, OpenCV 5, perception, agent, AWS, human approval | "Two camera families, one vision Lambda, one agent, a person on every action; one SAM stack in Sydney." |
| 4:08-4:26 | OpenCV 5 | The `readNet(..., ENGINES[engine])` line; "YOLO26n test mAP@50 0.953"; "2 x 2 tiles on synthetic sticky cards: recall 0.07 → 0.82"; "warm p50 1,091 ms on Lambda" | "YOLO26n runs on OpenCV 5's new DNN engine …" |
| 4:26-4:38 | Results | "Failure injection 38/38, 0 false actuations (earlier build, fake devices)" · "≈ 17 USD a month" · "≈ 0.054 USD per 1,000 photos" · "COOL on Graviton4: 4.6 % slower per photo" | "Measured, including what did not go our way." |
| 4:38-4:51 | Limits and end | "0 human labels and 0/50 expert verdicts yet · sticky cards measured on synthetic photos only · no autonomous spraying"; "flybot.agrhub.com → Guest access · github.com/agrhub/flybot-aws · AGRHUB · FlyBot team" | "Honest limits, and a person always decides. Try it with Guest access." |

Not in this cut: team members on camera (no person appears in the film unless they agree; the end card names the
team instead) and a live approval in the owner's account (the console shows the approvals; the guest session cannot
decide). Either can be added as a recorded scene if the team wants it, within the 5:00.

## Recording checklist

- [ ] Guest access and the simulated farm are on (Admin > System > Demo access); the stack is healthy (alarms green)
      and the Lambda is warm before the take (one call with a sample photo).
- [ ] Never on screen: `deploy.env`, `.env`, Secrets Manager values, AWS access keys, the GitHub token, the Telegram bot
      token, device tokens, Wi-Fi passwords (the camera's setup page shows the SoftAP password: crop it).
- [ ] Never on screen: the real ESP32-S3 camera's serial and verify code (where one must appear, use the sample identity
      FlyBot's claim page shows), its label QR, its GPS position or a map around it, and the owner's home address.
- [ ] No photo of the excluded camera AC6527557 (building facade) and no face of anyone who did not agree to appear;
      only team members and the advisor on camera.
- [ ] Synthetic, simulated, replayed or staged material is labelled on screen (the simulated farm, "Drawn: N", the
      24-photo replay, the failure injection, fake devices), and the owner-account approval shot says so.
- [ ] Numbers on screen checked against `models/MODEL_CARD.md`, `eval/RESULTS.md`, `eval/SYNTHETIC_RESULTS.md`,
      `bench/REPORT_COOL.md`, `bench/results/lambda_latency.json` and `infra/COST.md` of the submitted commit.
- [ ] Orchard footage only from FlyBot's own commercial (AGRHUB) or new footage shot by the team; no person in it.
- [ ] Every recording and the orchard clip checked with `tools/media/film/peoplecheck.py` (counts only) before anyone
      looks at it; frames it flags are cut.
- [ ] 1920x1080, 30 fps, English voice-over (synthesized, or recorded line by line), English subtitles burned in;
      total ≤ 5:00 (`render-film.mjs` refuses a longer film).
- [ ] Diagrams: `docs/diagrams/*.png`; rebuild with `uv run --no-project python docs/diagrams/src/build_diagrams.py`
      and `uv run --no-project --with markdown==3.7 python docs/diagrams/src/render.py --png`.
- [ ] Upload as unlisted on YouTube; put the link in `README.md` and on Devpost.
