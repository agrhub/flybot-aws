# Demo video script (≤ 5:00)

The cut as built: 4:59, 1920x1080, 30 fps, English voice-over with burned-in English subtitles (a Vietnamese
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
| 0:00-0:20 | Why it matters | "Fruit flies: the most damaging pest of fruit worldwide", then three figures with their sources on screen: "US$2 billion lost every year to fruit flies in Africa alone" (icipe, fruit fly integrated pest management); "10-40 % of a mango harvest" (ICAR-IIHR, via ICRISAT ISSCA); "closed markets: infested fruit is turned away at export" (De Meyer et al., ZooKeys 2015; icipe); the line that tephritid fruit flies are "by far the most important group of horticultural pests worldwide" (De Meyer, Clarke, Vera, Hendrichs, ZooKeys 2015). No worldwide yearly figure: none specific to fruit flies could be checked | "Fruit flies are the most damaging pest of fruit in the world … In Africa alone they cost two billion dollars a year …" |
| 0:20-0:31 | The farmer's problem | Walk the orchard → count by eye → act too late, then FlyBot: a camera on every trap | "Farmers walk to every trap and count the flies by eye. It is slow, the numbers come late …" |
| 0:31-0:43 | Title | FlyBot logo, flies, "FlyBot on AWS · OpenCV 5 · AWS Lambda · IoT Core · Kinesis Video Streams · AGRHUB" | "FlyBot does that work for them: a camera on each trap takes the photos, and OpenCV 5 on AWS counts the flies and warns the farmer in time." |
| 0:43-1:12 | How it works | One photo's path, animated, in three areas (the orchard, AWS in Sydney, the farmer): a fly enters the trap and the camera in the lid flashes; the photo goes over 4G (HTTPS, the camera's own token) to FlyBot on EC2, into Amazon S3 and MongoDB; AWS Lambda runs OpenCV 5 and YOLO26n boxes and counts the flies on it; the Sentinel agent ticks its rules and proposes "take a new photo"; the count goes into the dashboard and reports; a person confirms on the phone, and the command goes back to the camera through AWS IoT Core. Few words: the service names and one line each (docs/SYSTEM_ARCHITECTURE.md, section 2.1). Replaces the panned architecture diagram | "Here is how it works … AWS Lambda runs OpenCV 5: YOLO26n counts the flies and checks the photo … any action waits for a person to confirm it." |
| 1:12-1:25 | Bill of materials | The hanging trap's parts (ESP32-S3 N16R8 CAM board, OV5640 with the wide-angle lens, A7680C, ATGM336H, TP4056, mini 5 V boost, LiPo pouch cell 103450, solar trap housing and lure), cut out of the team's photos of the board, with the prices of camos-esp32-trap docs/BOM.md ("field camera in the hanging solar trap", read on 17, 24 and 25 Oct 2026): "about US$39 per trap camera": board US$6.91, camera US$9.20, 4G modem US$6.49, GPS US$6.65 (both with antennas), charger US$0.24, 5 V boost US$0.22, LiPo cell US$2.76, housing US$3.00 + lure US$0.34; the 4G power switch, resistors, capacitors, prototype board and terminals US$3.45; VND prices at 26,050 VND/USD (Vietcombank, 21 Oct 2026); total ≈ US$39.27 (944,760 VND + US$3.00 housing), SIM not included; no sticky cards in this trap | "Our open trap camera: ESP32-S3, 4G modem, GPS, a flat lithium-polymer cell and a small solar panel." |
| 1:25-1:47 | Wiring | The 4G build wired step by step as in camos-esp32-trap docs/WIRING.md (25 Oct 2026): the panel and the LiPo on the charger; the 5 V boost (set to 5 V, measured 5.0-5.2 V) to the board's 5V pin, the board making its own 3.3 V; the modem from the cell through its power switch (AO3401A + C1815) on GPIO47, UART on GPIO41/42; the GPS also from the cell, its ground switched by a C1815 on GPIO21, its UART on GPIO14/GPIO3; the battery divider on GPIO1. Only the wiring is shown, no firmware setting | One line per step |
| 1:47-1:54 | The real build | The team's photo (25 Oct 2026) of the perfboard in the trap lid, with the wide-angle lens on the camera and the 5 V boost, each module labelled; the finished trap and the lid's solar panel | "This is the real board. It goes into the lid of a solar trap housing." |
| 1:54-2:16 | In the orchard | **The team's footage of the wired trap installed in the orchard** (28.8 s, shot 24 Oct 2026; the last frame holds to 30 s; no person found by `peoplecheck.py`) | "In the orchard, the trap hangs at fruit height …" |
| 2:16-2:26 | Guest access | https://flybot.agrhub.com/auth/login, **Guest access**, the guest and "simulated data" banners | "Anyone can try FlyBot: one click on Guest access …" |
| 2:26-2:39 | Dashboard | The density card: the two real EZVIZ traps, then a simulated farm | "Two real orchard traps, and a clearly labelled simulated network of 114 traps on 47 farms." |
| 2:39-2:48 | Cameras | Real / simulated, sticky / lure filters, 2 to 5 columns, search | "The camera list searches and filters …" |
| 2:48-3:01 | A counted photo | A simulated sticky trap's history ("Drawn: N" next to the count) and its photo with the OpenCV boxes and the SIMULATED stamp | "Every photo, real or synthetic, goes through the same OpenCV 5 Lambda …" |
| 3:01-3:20 | Sentinel trace | A trace opened: perception, decision (rule `surge.unconfirmed`), the agent's steps, evidence pictures | "What OpenCV saw, which rule and why, every step …" |
| 3:20-3:32 | Approvals | The proposals waiting for a person (guest view: guests cannot decide) | "Proposals wait for a person; only after the owner confirms does a camera act." |
| 3:32-3:42 | Assistant | "How many flies on the traps this week?" and the answer | "Growers can ask the assistant in plain words …" |
| 3:42-4:02 | ESP32-S3 camera (owner's account) | Live view at JPEG 640x480, the camera log card, Admin > Firmware (releases per model and channel) and the camera fleet | "Streams live through Kinesis Video Streams; log to the console; firmware over the air through IoT Core, alpha / beta / stable." |
| 4:02-4:11 | Field map, reports | The simulated network on the map with the heat map; a report | "Traps on a field map with a heat map; reports per farm." |
| 4:11-4:29 | OpenCV 5 | The `readNet(..., ENGINES[engine])` line; "YOLO26n test mAP@50 0.953"; "2 x 2 tiles on synthetic sticky cards: recall 0.07 → 0.82"; "warm p50 980 ms on Lambda" | "YOLO26n runs on OpenCV 5's new DNN engine …" |
| 4:29-4:41 | Results | "Failure injection 38/38, 0 false actuations (earlier build, fake devices)" · "≈ 17 USD a month" · "≈ 0.046 USD per 1,000 photos" · "COOL on Graviton4: 4.6 % slower per photo" | "Measured, including what did not go our way." |
| 4:41-4:59 | Expert review, limits and end | "Expert review: 43 of 49 agent decisions policy-correct (87.8 %); the 6 wrong ones were needless recaptures" (`eval/results/expert_review.json`, REPORT section 6.7) · "0 human count labels yet · sticky cards measured on synthetic photos only · no autonomous spraying"; "flybot.agrhub.com → Guest access · github.com/agrhub/flybot-aws · AGRHUB · FlyBot team" | "Our agronomy advisor judged 43 of 49 agent decisions correct. Honest limits, and a person always decides. Try it with Guest access." |

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
- [x] Upload as unlisted on YouTube; put the link in `README.md` and on Devpost: https://youtu.be/mfCE2EEFPqg (26 Oct 2026).
