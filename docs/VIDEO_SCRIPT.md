# Demo video script (≤ 5:00)

Target 4:45, 1920x1080, English voice-over with burned-in English subtitles, unlisted YouTube. The competition asks
for: team, the app running, architecture, OpenCV 5 usage, AWS integration, main results. Every number on screen comes
from a file named in [REPORT.md](REPORT.md); anything replayed, synthetic or simulated says so on screen.

The app part follows the path a judge can take today: **Guest access → Dashboard → Cameras → Sentinel trace →
approval → assistant**. The guest account is read-only, so the approval is shown from the owner's account (labelled on
screen); everything else is recorded in the guest session.

## Shot list

| Time | Shot (what to capture) | On-screen text | Voice-over (short) |
|---|---|---|---|
| 0:00-0:12 | Orchard and trap footage from FlyBot's own commercial (AGRHUB), title card | FlyBot on AWS · OpenCV 5 + AWS · AGRHUB | "Fruit flies are counted on traps. FlyBot photographs the traps and counts them." |
| 0:12-0:27 | Team on camera, each member with first name and role; the ESP32-S3 trap camera on the desk | *[name, role]* for each member (owner fills in); the agronomy advisor (MSc Trương Quốc Ánh, IAS) may be named or appear | "We are the FlyBot team at AGRHUB, with our agronomy advisor." |
| 0:27-0:42 | The problem: the sample photo with glare (`data/samples/images/BE4089629/6ac34afa4c3227d11cfbd154.jpg`) next to the strong-blur failure-injection frame (`docs/evidence/eval/failure_injection/6ac33ceb4c3227d11cfbd0f4__blur_strong.jpg`) | "A count can't tell a bad photo from a catch" | "A count alone counts bad photos too, and the camera never reacts." |
| 0:42-0:55 | **Guest access:** https://flybot.agrhub.com/auth/login, press **Guest access**; the two-hour guest banner and the "Simulated" banner appear | "Guest access: read-only, two hours, no sign-up" | "Anyone can try it: one click opens a read-only demo account." |
| 0:55-1:20 | **Dashboard:** the density card filtered by farm and trap kind; the two real EZVIZ traps first, then the simulated farm network ("SIM ·" trap names, "(simulated)" farms) | "Real traps: 2 EZVIZ lure traps" · "SIMULATED: 114 traps on 47 farms, synthetic photos counted by the real Lambda" | "Two real orchard traps, and a clearly labelled simulated network so you can see FlyBot at scale." |
| 1:20-1:45 | **Cameras:** a real EZVIZ trap's latest photo with its boxes and quality flags; a simulated sticky trap whose count sits beside "Drawn: N" | "OpenCV 5 on Lambda: warm p50 1,091 ms" · "Drawn: N = flies pasted on the synthetic photo" | "Every photo, real or synthetic, goes through the same OpenCV 5 Lambda." |
| 1:45-2:15 | **Sentinel trace:** open one trace: perception (count, quality, diff), decision (rule), action (proposal); the photo with the boxes FlyBot draws | "Perception → Decision → Action" · "The Lambda stores no picture" | "For each photo the console shows what OpenCV saw, what the agent decided and why." |
| 2:15-2:35 | **Approval** (owner's account, labelled): the `capture_now` card in the chat panel and the same card on Telegram, the reason in the viewer's language; press **Confirm**; the new photo arrives with its own trace | "Nothing reaches a camera until a person confirms" · "Owner account: guests cannot approve" | "The agent proposes; a person confirms; the new photo starts the loop again." |
| 2:35-3:00 | **Assistant** (back in the guest session): ask "How many flies on the traps this week?", then "Why was this photo not counted?" about a flagged photo; the answer cites the quality numbers | "Deep mode: Claude Haiku 5.5 on OpenCV's numbers, never the pixels" | "Growers can ask about a photo; the agent answers from what OpenCV measured." |
| 3:00-3:20 | **ESP32-S3 camera** (owner's account): the camera page with Live view (Medium quality) and a snapshot; Admin > Firmware with the alpha / beta / stable channels and an update's progress; the camera log streaming | "Live view: Kinesis Video Streams WebRTC" · "Firmware updates over IoT Core, alpha / beta / stable" | "Our open trap camera streams live, updates over the air and sends its log through AWS IoT Core." |
| 3:20-3:40 | `docs/diagrams/architecture.png`, pan along the six columns; pause on HUMAN APPROVAL, COOL and the guest box | "Input → OpenCV 5 → Perception → Agent → AWS → Action" | "Two camera families, one vision Lambda, one agent, and a person on every action." |
| 3:40-4:05 | **OpenCV 5:** the line `cv2.dnn.readNet(..., ENGINES[engine])` in `lambda/vision/sentinel/vision/detect_yolo26.py` and the engine in a response; a synthetic sticky card counted whole-frame vs with 2 x 2 tiles; `docs/diagrams/trace_example.png` | "New DNN engine: YOLO26n, test mAP@50 0.953" · "Tiles on sticky cards: recall 0.07 → 0.82 (synthetic)" | "YOLO26n runs on OpenCV 5's new DNN engine; tiling the frame finds the small flies on a sticky card." |
| 4:05-4:20 | **AWS integration:** `./deploy.sh` summary; the CloudFormation stack; Lambda monitor; an IoT Core thing shadow and the log topic; a KVS signaling channel; CloudWatch alarms | "One SAM stack · Sydney · one command" · "~17 USD/month demo, ~0.054 USD per 1,000 photos" | "Everything is one stack in Sydney, deployed with one command." |
| 4:20-4:35 | Result cards: detector table (`models/MODEL_CARD.md`), failure injection (`eval/RESULTS.md` section 4), synthetic checks (`eval/SYNTHETIC_RESULTS.md`), COOL table (`bench/REPORT_COOL.md`) | "Failure injection 38/38, 0 false actuations (earlier build, fake devices)" · "Synthetic: labelled, not field data" · "COOL on Graviton4: 4.6 % slower per photo than pip" | "We measured everything, including what did not go our way." |
| 4:35-4:45 | Limits card, then end card with the URLs | "0 human labels and 0/50 expert verdicts yet · sticky cards measured on synthetic photos only · no autonomous spraying" · "flybot.agrhub.com → Guest access · github.com/agrhub/flybot-aws" | "Honest limits, and a person always decides. Try it with Guest access." |

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
- [ ] Orchard footage only from FlyBot's own commercial (AGRHUB) or new footage shot by the team.
- [ ] 1920x1080, 30 fps, voice-over recorded separately, English subtitles burned in; total ≤ 5:00.
- [ ] Diagrams: `docs/diagrams/*.png`; rebuild with `uv run --no-project python docs/diagrams/src/build_diagrams.py`
      and `uv run --no-project --with markdown==3.7 python docs/diagrams/src/render.py --png`.
- [ ] Upload as unlisted on YouTube; put the link in `README.md` and on Devpost.
