# Demo video script (≤ 5:00)

Target 4:45, 1920x1080, English voice-over with burned-in English subtitles, unlisted YouTube. The competition asks
for: team, the app running, architecture, OpenCV 5 usage, AWS integration, main results. Every number on screen comes
from a file named in [REPORT.md](REPORT.md); anything replayed, staged or simulated says so on screen.

## Shot list

| Time | Shot (what to capture) | On-screen text | Voice-over (short) |
|---|---|---|---|
| 0:00-0:12 | Orchard and trap footage from FlyBot's own commercial (AGRHUB), title card | FlyBot on AWS · OpenCV 5 + AWS · AGRHUB | "Fruit flies are counted on traps. FlyBot photographs the traps and counts them." |
| 0:12-0:27 | Team on camera, each member with first name and role; the ESP32-S3 trap camera on the desk | *[name, role]* for each member (owner fills in); the agronomy advisor (MSc Trương Quốc Ánh, IAS) may be named or appear | "We are the FlyBot team at AGRHUB." |
| 0:27-0:45 | The problem: the sample photo with glare (`data/samples/images/BE4089629/6ac34afa4c3227d11cfbd154.jpg`) next to the strong-blur failure-injection frame (`docs/evidence/eval/failure_injection/6ac33ceb4c3227d11cfbd0f4__blur_strong.jpg`) | "A count can't tell a bad photo from a catch" | "A count alone counts bad photos too, and the camera never reacts." |
| 0:45-1:40 | **App running** at https://flybot.agrhub.com: camera list, one EZVIZ trap photo with its boxes, quality flags and trace (perception → decision → action); the confirm card for `capture_now` in the chat panel and the same card on Telegram; press **Confirm**; the new photo arrives and gets its own trace | "Live on AWS" · "OpenCV 5 on Lambda: warm p50 1,091 ms" | "Each photo goes to an OpenCV 5 Lambda. The agent proposes the next step; a person confirms; the new photo starts the loop again." |
| 1:40-2:10 | **Chatbot deep mode:** ask "Why is this photo not counted?" about a flagged photo; the answer cites the quality numbers; the chatbot offers a `propose_action` card | "Deep mode: Claude Haiku 5.5 on OpenCV's numbers, never the pixels" | "Growers can ask about a photo. The agent answers from what OpenCV measured and proposes an action." |
| 2:10-2:40 | **ESP32-S3 trap camera:** scan the QR on its setup page → Add camera → claimed; open the camera page, Live view (Medium quality), take a snapshot; the photo is analysed | "Zero-touch registration via AWS IoT Core" · "Live view: Kinesis Video Streams WebRTC" | "Our open trap camera registers itself through IoT Core and streams live over Kinesis Video Streams." |
| 2:40-3:05 | `docs/diagrams/architecture.png`, pan along the six columns; pause on the HUMAN APPROVAL and COOL markers | "Input → OpenCV 5 → Perception → Agent → AWS → Action" | "Two camera families, one vision Lambda, one agent, and a person on every action." |
| 3:05-3:40 | **OpenCV 5:** the line `cv2.dnn.readNet(..., ENGINES[engine])` in `lambda/vision/sentinel/vision/detect_yolo26.py` and `engine_used: new` in a response; an annotated evidence picture; SIFT vs ALIKED + LightGlue matches on the sample pair (`cd lambda/vision && uv run python -m sentinel.vision.register ../../data/samples/images/BE4089629/6ac34afa4c3227d11cfbd154.jpg ../../data/samples/images/BE4089629/6ac34f1fe2041e0cbbd42bc8.jpg --method sift_flann --out <dir>`, then `--method aliked_lightglue`); `docs/diagrams/trace_example.png` | "New DNN engine: YOLO26n, test mAP@50 0.953" · "Glare → capture_now; clean → no_op" | "YOLO26n runs on OpenCV 5's new DNN engine; the quality gate, registration and occlusion checks decide whether a photo counts." |
| 3:40-4:05 | **AWS integration:** `./deploy.sh` summary; CloudFormation stack `flybot-dev`; Lambda monitor graphs; IoT Core thing and its shadow; KVS signaling channel `cam-<serial>`; CloudWatch alarms | "One SAM stack · Sydney · one command" · "~17 USD/month demo, ~0.054 USD per 1,000 photos" | "Everything is one stack in Sydney, deployed with one command." |
| 4:05-4:30 | Result cards: detector table (`models/MODEL_CARD.md`), failure-injection table (`eval/RESULTS.md` section 4), COOL table (`bench/REPORT_COOL.md`) | "Failure injection 38/38, 0 false actuations (earlier build, fake devices)" · "COOL on Graviton4: 4.6 % slower per photo than pip" | "We measured everything, including what did not go our way." |
| 4:30-4:45 | Limits card, then end card with the URLs and `make replay-sample` | "0 human labels yet · traps empty in October · small flies missed on full traps · no autonomous spraying" · "github.com/agrhub/flybot-aws" | "Honest limits, and a person always decides. Clone it and replay real orchard photos in one command." |

## Recording checklist

- [ ] The stack is healthy (`./deploy.sh --dry-run` clean, alarms green) and the Lambda is warm before the take (one
      call with a sample photo).
- [ ] Never on screen: `deploy.env`, `.env`, Secrets Manager values, AWS access keys, the GitHub token, Telegram bot
      token, device tokens, Wi-Fi passwords (the camera's setup page shows the SoftAP password: crop it), the claim
      verify code of a camera you keep.
- [ ] No photo of the excluded camera AC6527557 (building facade) and no face of anyone who did not agree to appear;
      only team members on camera.
- [ ] Replayed, staged or simulated material is labelled on screen (the 24-photo replay, the failure injection, fake
      devices).
- [ ] Numbers on screen checked against `models/MODEL_CARD.md`, `eval/RESULTS.md`, `bench/REPORT_COOL.md`,
      `bench/results/lambda_latency.json` and `infra/COST.md` of the submitted commit.
- [ ] Orchard footage only from FlyBot's own commercial (AGRHUB) or new footage shot by the team.
- [ ] 1920x1080, 30 fps, voice-over recorded separately, English subtitles burned in; total ≤ 5:00.
- [ ] Diagrams: `docs/diagrams/*.png`; rebuild with `uv run --no-project python docs/diagrams/src/build_diagrams.py`
      and `uv run --no-project --with markdown==3.7 python docs/diagrams/src/render.py --png`.
- [ ] Upload as unlisted on YouTube; put the link in `README.md` and on Devpost.
