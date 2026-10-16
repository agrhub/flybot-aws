# FlyBot on AWS

FlyBot counts fruit flies on orchard traps from camera photos and advises growers. This repository is the OpenCV AI Competition 2026 entry: FlyBot deployed on AWS, with

- **OpenCV 5 on Lambda** replacing the hosted detector (YOLO26n exported to ONNX, OpenCV 5 DNN new engine), plus the planning agent that proposes camera actions and answers deep-analysis questions in FlyBot's chatbot;
- **AWS IoT Core** as the camera hub (shadow, RPC, OTA) and **Kinesis Video Streams WebRTC** for live view;
- an open **ESP32-S3 trap camera** (Wi-Fi or 4G, solar + battery, GPS) in `firmware-esp32/`.

Status (16 Oct 2026): the vision Lambda with YOLO26n, the FlyBot integration (private submodule), the ESP32 firmware and the one-command deploy are in. Contracts between the parts: docs/CONTRACTS.md; design decisions: docs/DECISIONS.md.

Layout: `flybot/` (**private** submodule: the FlyBot app, see below), `firmware-esp32/` (public submodule, the trap camera firmware), `lambda/vision/` (OpenCV 5 + agent Lambda, Python 3.12 / uv), `infra/` (SAM template + scripts), `bench/`, `eval/`, `models/`, `docs/`.

## Open-core split

| Part | Where | Access |
|---|---|---|
| This repo: vision Lambda + agent (`lambda/vision/`), infra (`infra/`, `./deploy.sh`), bench, eval, docs, YOLO26n weights on the public model bucket | agrhub/flybot-aws | public, AGPL-3.0 |
| ESP32-S3 trap camera firmware (`firmware-esp32/`) | agrhub/camos-esp32-trap | public, its own license |
| FlyBot web app (`flybot/`: API, Vue client, chatbot, camera platforms) | agrhub/flybot | **private**, proprietary (not covered by this repo's license) |
| AMB82 firmware, production datasets, production model weights | AGRHUB | private |

Judges get read access to `agrhub/flybot` on request: the owner adds their GitHub accounts as read-only collaborators (or runs a
judge-only deployment, https://flybot.agrhub.com). Everything in the "Judge entry" below except FlyBot's own UI works without it:
`make test`, `make replay-sample` and `make lambda-local` only use `lambda/vision/`, `data/samples/` and `models/`.

```
git clone https://github.com/agrhub/flybot-aws.git && cd flybot-aws
git submodule update --init firmware-esp32        # public
git submodule update --init flybot                # only with read access to agrhub/flybot
```

(`git clone --recurse-submodules` stops with an authentication error on `flybot/` without access; the clone itself and the
public submodule are still usable, finish with the two commands above.) The deployed EC2 reads the private submodule with a
read-only deploy key that `./deploy.sh` creates and asks you to add on GitHub (or a fine-grained token, `FLYBOT_REPO_TOKEN`).

## Judge entry

Everything below runs from a clean clone with `uv` (Python 3.12) and `make`; AWS steps need the AWS CLI v2 + SAM CLI and the account profile `sentinel` (`aws login --profile sentinel`, region ap-southeast-2 only).

### 1. Run locally (10 min)

```
make setup                 # uv sync in lambda/vision (OpenCV 5.0.0, numpy, pydantic, anthropic; no torch)
make models                # YOLOv7-tiny baseline ONNX (24 MB, SHA-256 checked) into models/yolov7-tiny/
make test                  # 185 tests (vision ops on real trap fixtures, agent guardrails, the FlyBot contract with a fake detector + S3 stub); ~5 min
make replay-sample         # 24 real lure-trap photos through the Lambda handler -> data/replay/sample/{responses.jsonl,summary.json,evidence/}
make lambda-local          # POST http://localhost:9001/invoke, the same JSON as Lambda.Invoke (docs/CONTRACTS.md 1); GET /health
```

`make replay-sample` prints one line per photo (count, quality flags, proposals, DNN ms) and a summary table; the annotated
JPEGs (boxes, flags, faces blurred when a person is suspected) land in `data/replay/sample/evidence/<camera>/`. A request by hand:

```
curl -s localhost:9001/invoke -d '{"v":1,"mode":"detect","photo":{"url":"file:///ABS/PATH/flybot-aws/data/samples/images/BE4089629/6ac327d44c3227d11cfbd084.jpg"},"camera":{"id":"BE4089629","trapKind":"lure"},"capturedAt":"2026-10-05T04:30:12Z","locale":"en"}' | python -m json.tool
```

Deep mode (`"mode":"deep","question":"Why did the count jump today?"`) answers in markdown through the LLM planner (Anthropic key in
`.env`, see `.env.example`); without a key it falls back to the rule planner and a templated answer. FlyBot itself (web app, chatbot,
cameras) runs with `docker compose up` (needs the private `flybot/` submodule, branch `feat/aws`); set
`AI_PROVIDER=opencv-lambda` and `VISION_LAMBDA_URL=http://host.docker.internal:9001/invoke` to use the local Lambda.

### 2. Deploy on AWS in one command (~35 min the first time)

```
cp deploy.env.example deploy.env      # once: fill it (domain, emails, EZVIZ keys, Telegram/SMTP, Google key path); git-ignored
aws login --profile sentinel          # 12-hour session
./deploy.sh                           # or: make deploy
```

`deploy.env` holds every setting (Lambda, EC2, FlyBot app secrets); `./deploy.sh` does the rest, idempotently, and can be re-run
after any change:

1. **preflight**: finds `aws` / `sam` / Python even off PATH, checks the session (expired: prints `aws login --profile ...`, exit 2),
   account, region (Sydney only), that CloudFormation, IoT Core, KVS, Lambda, ECR, SSM, Secrets Manager, SNS, S3 and IAM answer,
   the default VPC, `t4g.small` in the AZs, the Elastic IP and vCPU quotas, and that GitHub `main` is what you have locally
   (the EC2 clones GitHub);
2. **discover**: IoT Data-ATS endpoint, newest AL2023 arm64 AMI (then pinned: a new AMI would replace the instance), your public
   IP for SSH, an SNS topic with a confirmed email to reuse, the photos bucket, the secrets, the stack state;
3. **models**: uploads `models/yolo26/` to the public model bucket under `models/yolo26/` only when the sha256 differs, checks the
   anonymous GET (`DETECTOR=yolo26` falls back to yolov7 when the files are missing);
4. **secrets**: creates or updates `flybot-<stage>-app` (JWT and VAPID generated once), `flybot-<stage>-gcp` (the service-account
   JSON) and, when needed, the Anthropic key, only when the content changed; values are never printed;
5. **stack** (bucket, ECR, IoT thing type + device policy, roles, EC2 + Elastic IP; the Lambda is kept), then a highlighted box
   with the **DNS record to add** while the build runs; **image**: the Lambda image is built on the EC2 through SSM only when its
   inputs changed (tag = hash of the code + models); **lambda**: the function on that image + alarms;
6. **bootstrap**: re-runs the host's current user-data (git pull, `.env`, secrets, `docker compose up -d --build`);
7. **health**: waits until 1.1.1.1 and 8.8.8.8 resolve the domain to the Elastic IP (up to 30 min) and HTTPS answers with a valid
   certificate; **smoke**: one contract call to the Lambda with a sample photo; **coldstart**: cold/warm latency into
   `bench/results/lambda_latency.json`; **summary**: URL, IP, Lambda, detector + sha, IoT endpoint, KVS role, what is left.

`./deploy.sh --dry-run` runs preflight and discovery for real and prints the plan without changing anything; `--only <step>` re-runs
one step; `--show-ip` prints the Elastic IP and the DNS record; logs go to `infra/.build/deploy-<stage>.log`. `./destroy.sh --yes
[--keep-photos] [--keep-secrets]` removes it again (the model bucket and `sentinel-dev` are never touched). Left for a human: the DNS
record at Namecheap (printed as soon as the IP exists) and registering the first account (it becomes the admin). Details and expected
output: `infra/DEPLOY_CHECKLIST.md`; costs: `infra/COST.md`; `make host CMD='...'` runs a command on the EC2 through SSM.

### 3. Detector: YOLO26n on OpenCV 5

`models/colab_train_yolo26.ipynb` trains YOLO26n on the Roboflow camos/fruit-fly dataset (~1 h on a T4), exports the NMS-free
end-to-end ONNX (`[1, 300, 6]`), checks it on OpenCV 5 DNN against ONNX Runtime, and `models/verify_yolo26.py` repeats the check on
20 sample photos on the laptop; `models/COLAB_RUNBOOK.md` is the owner's step list, `models/MODEL_CARD.md` the numbers with their
source files. Until those weights exist the Lambda runs the YOLOv7-tiny baseline (`DETECTOR=yolov7`).

### 4. Reproduce the bench and the evaluation

`make bench` (Roboflow vs OpenCV 5 Lambda vs YOLOv7-tiny vs COOL on Graviton; `bench/`) and `eval/` (replay harness,
failure injection, ground-truth labels) are being adapted to this build; `bench/REPORT_COOL.md` and `eval/RESULTS.md` carry the
first build's measured numbers with their source files.

License: this repository is AGPL-3.0 (see LICENSE and NOTICE; the YOLO26n weights derive from Ultralytics YOLO26, AGPL-3.0). The
`flybot/` submodule points at a private, proprietary repository that is not licensed by this repo. Dataset: camos fruit-fly
dataset, CC BY 4.0.
