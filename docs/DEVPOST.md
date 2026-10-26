# Devpost submission text

Paste-ready sections for opencv26.devpost.com. The main text (Inspiration to Built with) is about 1,000 words. Every
number comes from a file named in [REPORT.md](REPORT.md); "not measured" means nobody measured it.

**Project name:** FlyBot on AWS

**Tagline:** Trap cameras that look before they count: OpenCV 5 and a guarded agent on AWS for fruit-fly monitoring.

**Team and advisor:** AGRHUB, makers of FlyBot. Agronomy advisor: MSc Trương Quốc Ánh, Principal Researcher and Deputy Head of the Biotechnology Department, Institute of Agricultural Science for Southern Vietnam (IAS, under VAAS), Ho Chi Minh City; research in plant breeding genetics, plant biotechnology and DNA barcoding, and applied projects on smart agriculture and IoT for pest control (https://iasvn.org/bo-mon-cong-nghe-sinh-hoc/). The advisor is not a Devpost team member.

**Try it:** https://flybot.agrhub.com/auth/login, press **Guest access**: a shared, read-only demo account for two
hours (the two real traps, a labelled simulated farm network, the Sentinel console and the chatbot).

**Links:** live app https://flybot.agrhub.com · code https://github.com/agrhub/flybot-aws (submodules
https://github.com/agrhub/flybot and https://github.com/agrhub/camos-esp32-trap) · video: https://youtu.be/mfCE2EEFPqg (4:59, unlisted)

**Images for the gallery:** `docs/diagrams/architecture.png`, `docs/diagrams/agent_workflow.png`,
`docs/diagrams/trace_example.png`, screenshots of FlyBot (guest dashboard, Sentinel approvals and traces, the ESP32-S3
camera live), and four stills of the video: how one photo goes through FlyBot, the trap camera's parts (about US$39),
its wiring as built and the real build.

## Inspiration

Fruit flies are "by far the most important group of horticultural pests worldwide" (De Meyer et al., ZooKeys 2015); in
Africa alone they cost about US$2 billion a year (icipe). We make FlyBot, which photographs fruit-fly traps in orchards and counts the flies with AI. Our data showed what a
count misses: one camera filmed a building and every "fly" it reported was false; glare and blur were counted like any
photo; the camera never reacted. Growers need a system that notices a bad photo, asks for a better one, and bothers
them only when it matters.

## What it does

Every trap photo, from EZVIZ cloud cameras or from our open ESP32-S3 trap camera, goes to an OpenCV 5 Lambda on
AWS. It counts the flies with YOLO26n on OpenCV 5's new DNN engine, grades the image quality, finds the trap, registers
the photo with the previous one to separate new flies from old ones, measures crowding, and detects hands, people and
a moved camera (the Lambda stores no picture; FlyBot blurs the face regions it returns). A small agent then decides what should happen next under
guardrails: take another photo, change the camera's image settings or schedule, ask a person to review, or alert the
grower. FlyBot shows each decision as a confirm card in the app and on Telegram; nothing reaches a camera until a
person presses Confirm, and the new photo starts the loop again. Growers can also ask FlyBot's chatbot about a photo
("why did the count jump?"): it calls the same Lambda in deep mode, where a Claude Haiku 5.5 planner answers from
OpenCV's numbers, never from the pixels. ESP32-S3 cameras join on their own (zero-touch registration through AWS IoT
Core), are claimed by QR code, stream live view over Kinesis Video Streams WebRTC, and are managed as a fleet
(firmware channels, OTA rollouts with a battery guard, remote log, crash reports). A battery camera that sleeps between photos still takes orders: FlyBot keeps a live request, a GPS fix or a setting for
its next wake and shows when that is; Wi-Fi or 4G is chosen remotely. Judges get Guest access, a labelled simulated
farm counted by the real Lambda, and reports of their own.

## How we built it

- **OpenCV 5:** the new DNN engine (YOLO26n ONNX, NMS, box rules; the engine actually used is recorded), imgproc
  (HSV, contours, perspective, Laplacian, histograms, adaptive threshold, watershed), features (SIFT + FLANN + USAC
  MAGSAC; ALIKED + LightGlue on demand), MOG2, phaseCorrelate, YuNet face regions, and 2 x 2 tiles for small flies on
  sticky cards.
- **Agent:** a deterministic rule planner for every photo and an LLM planner for questions, 14 typed tools, guardrails
  (run states, budgets, quiet hours, battery, evidence-grounded proposals, untrusted-text filter, kill switch) and a
  trace for every photo.
- **AWS (one SAM stack, Sydney):** Lambda arm64 container built by CodeBuild into ECR, EC2 t4g.micro running FlyBot
  (Node, Vue, MongoDB, Caddy TLS), IoT Core (thing type, per-thing policy, shadow, RPC, presence), Kinesis Video
  Streams WebRTC, S3, Secrets Manager, CloudWatch alarms. One command deploys it: `./deploy.sh`.
- **Device:** ESP-IDF firmware for an ESP32-S3 board with an OV5640 wide-angle lens, 4G, GPS and a solar-charged
  LiPo cell in a hanging trap (about US$39 of parts, camos-esp32-trap docs/BOM.md); deep sleep, a setup portal, a web console, OTA updates and a remote log over IoT Core.

## Challenges we ran into

YOLO26n's end-to-end head still left duplicate boxes and counted the trap's lens, so we added NMS and fruit-fly box
rules. Live video from a microcontroller without a hardware H.264 encoder needed the camera's own JPEG frames over the
WebRTC data channel. Our orchard traps were empty for most of October, so we generated labelled synthetic photos with
known counts; they showed that flies on a sticky card shrink below 20 px when the photo is letterboxed to 640, and
tiling the frame took recall from 0.07 to 0.82 without retraining. And COOL turned out slower than the plain pip wheel
on our DNN-heavy workload.

## Accomplishments that we're proud of

- YOLO26n on the OpenCV 5 DNN engine: test mAP@50 0.953 and mAP@50-95 0.700, with detections identical to ONNX
  Runtime; the box rules raised F1 on the held-out test split from 0.8788 to 0.9006.
- 38 of 38 failure-injection runs passed with zero false actuations (image defects, prompt injection, LLM and camera
  failures).
- Our agronomy advisor judged 43 of the 49 agent decisions he could decide correct (87.8 %, 95 % interval
  75.8-94.3 %, 1 unsure); the 6 he rejected are needless recaptures on one camera. We acted on it: lure traps now get
  their own glare limit, and on the same photos 1 of those 7 recaptures remains (the unsure one) while none of the 43
  correct decisions changed.
- One-command AWS deployment: about 17 USD a month for the demo, 0.046 USD per 1,000 photos, warm Lambda p50
  980 ms.
- An honest COOL benchmark on 500 real photo pairs.
- Guest access with a labelled simulated network of 114 traps on 47 farms; its 2,852 seed photos went through the
  real Lambda.
- A judge can run 24 real orchard photos through the Lambda's handler with one command, no hardware and no AWS.

## What we learned

A vision system earns trust by refusing to count bad photos and showing its evidence. `ENGINE_AUTO` can silently
fall back to the classic DNN engine, so check it. Synthetic photos with a known truth find weaknesses fast, but they
are not field data. A person on every action keeps the agent simple and trusted.

## What's next

Ask our agronomy advisor to review the quality gate's new glare limit on fresh photos (we re-scored his photos, he has
not re-reviewed them); label 150 trap photos (ready, 0 done); improve recall on small flies in full lure-trap views; measure the ESP32-S3 trap's battery life in the orchard and fix its weaker Wi-Fi on battery; add
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
runs with 0 false actuations (earlier build, fake devices), a real trace on the current Lambda handler, an automatic
check on 147 synthetic photos damaged on purpose (the policy's action on 111; simulated data, not an expert review),
traces stored with every photo. Our agronomy advisor, MSc Trương Quốc Ánh (IAS), judged 50 of the agent's decisions:
43 correct, 6 incorrect, 1 unsure, a policy-correct rate of 87.8 % (95 % interval 75.8-94.3 %). Every proposal, human
review and guardrail decision he saw was right; all six errors are needless recapture requests after false blur or
glare flags on one camera. The blur cause was already fixed; a glare limit for lure tray floors leaves 1 of the 7
rejected or unsure recaptures on the same photos, with none of the 43 correct decisions changed. Not measured yet: real approval times. Evidence: `docs/REPORT.md` sections 6.7 and 8,
`docs/evidence/agent/trace_example.json`, `eval/RESULTS.md`, `eval/SYNTHETIC_AGENT_EVAL.md`.
