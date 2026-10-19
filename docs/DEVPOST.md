# Devpost submission text

Paste-ready sections for opencv26.devpost.com. The main text (Inspiration to Built with) stays under 900 words. Every
number comes from a file named in [REPORT.md](REPORT.md); "not measured" means nobody measured it.

**Project name:** FlyBot on AWS

**Tagline:** Trap cameras that look before they count: OpenCV 5 and a guarded agent on AWS for fruit-fly monitoring.

**Team and advisor:** AGRHUB, makers of FlyBot. Agronomy advisor: MSc Trương Quốc Ánh, Principal Researcher and Deputy Head of the Biotechnology Department, Institute of Agricultural Science for Southern Vietnam (IAS, under VAAS), Ho Chi Minh City; research in plant breeding genetics, plant biotechnology and DNA barcoding, and applied projects on smart agriculture and IoT for pest control (https://iasvn.org/bo-mon-cong-nghe-sinh-hoc/). The advisor is not a Devpost team member.

**Links:** live app https://flybot.agrhub.com · code https://github.com/agrhub/flybot-aws (submodules
https://github.com/agrhub/flybot and https://github.com/agrhub/camos-esp32-trap) · video: *(unlisted YouTube link,
added after the upload)*

**Images for the gallery:** `docs/diagrams/architecture.png`, `docs/diagrams/agent_workflow.png`,
`docs/diagrams/trace_example.png`, plus screenshots of FlyBot (camera page, confirm card, chatbot answer).

## Inspiration

We make FlyBot, an app that photographs fruit-fly traps in orchards and counts the flies with AI. Our own data showed
what a count alone misses: one camera filmed a building facade and every "fly" it reported was false; glare and blur
were counted like any other photo; the camera never reacted to a bad picture. Growers do not need more numbers. They
need a system that notices when a photo is wrong, asks for a better one, and only bothers them when it matters.

## What it does

Every trap photo, from EZVIZ cloud cameras or from our open ESP32-S3 trap camera, goes to an OpenCV 5 Lambda on
AWS. It counts the flies with YOLO26n on OpenCV 5's new DNN engine, grades the image quality, finds the trap, registers
the photo with the previous one to separate new flies from old ones, measures crowding, and detects hands, people and
a moved camera (faces are blurred in stored evidence). A small agent then decides what should happen next under
guardrails: take another photo, change the camera's image settings or schedule, ask a person to review, or alert the
grower. FlyBot shows each decision as a confirm card in the app and on Telegram; nothing reaches a camera until a
person presses Confirm, and the new photo starts the loop again. Growers can also ask FlyBot's chatbot about a photo
("why did the count jump?"): it calls the same Lambda in deep mode, where a Claude Haiku 5.5 planner answers from
OpenCV's numbers, never from the pixels. ESP32-S3 cameras join on their own (zero-touch registration through AWS IoT
Core), are claimed by QR code, and stream live view over Kinesis Video Streams WebRTC.

## How we built it

- **OpenCV 5:** the new DNN engine (YOLO26n ONNX, NMS, box rules; the engine actually used is recorded), imgproc
  (HSV, contours, perspective, Laplacian, histograms, adaptive threshold, watershed), features (SIFT + FLANN + USAC
  MAGSAC; ALIKED + LightGlue on demand), MOG2, phaseCorrelate, and YuNet face blurring.
- **Agent:** a deterministic rule planner for every photo and an LLM planner for questions, 14 typed tools, guardrails
  (run states, budgets, quiet hours, battery, evidence-grounded proposals, untrusted-text filter, kill switch) and a
  trace for every photo.
- **AWS (one SAM stack, Sydney):** Lambda arm64 container built by CodeBuild into ECR, EC2 t4g.micro running FlyBot
  (Node, Vue, MongoDB, Caddy TLS), IoT Core (thing type, per-thing policy, shadow, RPC, presence), Kinesis Video
  Streams WebRTC, S3, Secrets Manager, CloudWatch alarms. One command deploys it: `./deploy.sh`.
- **Device:** ESP-IDF firmware for a Freenove ESP32-S3 camera with Wi-Fi or 4G, GPS, solar and deep sleep, a setup
  portal and a web console.

## Challenges we ran into

We rebuilt the entry mid-contest: a first version duplicated FlyBot with its own console and queues; the final one is
FlyBot itself on AWS with a stateless vision Lambda, which is simpler to run and to judge. YOLO26n's end-to-end head
still left duplicate boxes and counted the trap's lens, so we added NMS and fruit-fly box rules. Live video from a
microcontroller without a hardware H.264 encoder needed the camera's own JPEG frames over the WebRTC data channel. Our orchard traps were empty for
most of October, so surges could only be tested on staged replays. And COOL turned out slower than the plain pip wheel
on our DNN-heavy workload.

## Accomplishments that we're proud of

- YOLO26n on the OpenCV 5 DNN engine: test mAP@50 0.953 and mAP@50-95 0.700, with detections identical to ONNX
  Runtime; the box rules raised F1 on the held-out test split from 0.8788 to 0.9006.
- 38 of 38 failure-injection runs passed with zero false actuations: blur, glare, darkness, IR, occluders, prompt
  injection in pixels and metadata, LLM timeouts and refusals, offline cameras, RPC timeouts.
- A one-command AWS deployment at about 17 USD a month for the demo and about 0.054 USD per 1,000 photos; warm
  Lambda p50 1,091 ms per photo pair.
- An honest COOL benchmark on 500 real photo pairs.
- A judge can run 24 real orchard photos through the Lambda's handler with one command, no hardware and no AWS.

## What we learned

A vision system earns trust by refusing to count bad photos and by showing its evidence. `ENGINE_AUTO` can fall back
to the classic DNN engine silently, so the engine must be checked, not assumed. Benchmarks on the real workload beat
headline speed-ups: the pip arm64 wheel already ships KleidiCV. And keeping a person on every action made the agent
simpler to build and easier to trust.

## What's next

Label 150 trap photos and have our agronomy advisor review 50 agent decisions (both ready, 0 done); improve recall on small
flies in full lure-trap views; run the ESP32-S3 camera in the orchard on 4G and solar and measure its battery; add
yellow sticky cards; and try OpenCV 5's in-library VLM for second opinions on doubtful photos.

## Built with

opencv, opencv-5, python, numpy, onnx, yolo26, ultralytics, aws, aws-lambda, amazon-ecr, aws-codebuild, amazon-ec2,
aws-graviton, aws-iot-core, amazon-kinesis-video-streams, webrtc, amazon-s3, aws-secrets-manager, amazon-cloudwatch,
amazon-sns, aws-sam, cloudformation, cool, kleidicv, anthropic, claude, gemini, vertex-ai, node.js, express, vue.js,
mongodb, caddy, telegram, esp32-s3, esp-idf, mqtt, roboflow

## Special award opt-in: Best Use of COOL

We ran the full per-photo vision call (decode, the detector on OpenCV 5's new DNN engine on two photos, trap finding,
quality, SIFT registration, diff, occlusion) on 500 fixed real trap-photo pairs, 5 runs plus a parallel pass, on the
COOL Graviton4 AMI (ami-0a393ffc6f5d928e2, OpenCV 5.1.0-dev, KleidiCV 0.7.0, c8g.xlarge) and on the pip wheel on c8g,
c7g and x86 c7i, one throwaway instance each, same script. Proof that COOL ran it: `cv2.__file__` under `/opt/cool`
and the build-info dump. Result: COOL is 3 % faster on the OpenCV image ops but 11 % slower on the DNN forward, so
4.6 % slower per photo and USD 0.0197 vs 0.0174 per 1,000 photos; we serve with the pip wheel on Lambda arm64. The
benchmark used the earlier YOLOv7-tiny detector; COOL with YOLO26n is not measured. Method, evidence and rerun
commands: `bench/REPORT_COOL.md`.

## Special award opt-in: Agentic Vision

OpenCV 5 output decides the next action: on twelve consecutive real photos of one trap, clean photos get `no_op` and
the photo where the quality gate measures glare gets a `capture_now` proposal; in failure injection, blur leads to a
recapture, glare on our own camera to an image-setting change and a verification photo, IR or an occluder to "not
counted". Every action is a proposal that a person confirms in FlyBot or on Telegram; the confirmed capture produces
the next photo. Guardrails enforce run states, budgets, quiet hours, battery limits, evidence-grounded proposals and a
kill switch; LLM failures fall back to the rule planner. Evaluation: 9/9 scenario steps and 38/38 failure-injection
runs with 0 false actuations (earlier build, fake devices), a real trace on the current Lambda handler, traces stored
with every photo. Not measured yet: expert verdicts and real approval times. Evidence: `docs/REPORT.md` section 8,
`docs/evidence/agent/trace_example.json`, `eval/RESULTS.md`.
